"""Model construction helpers around ``openunreid.models.build_model``."""

import os
import os.path as osp
import shutil

import torch
from openunreid.models import build_model as _build_model
from openunreid.models.backbones import resnet, resnet_ibn_a
from torch.nn.parallel import DataParallel, DistributedDataParallel

__all__ = ["BackboneTap", "build_model", "unwrap"]


def unwrap(model):
    """Strip a (Distributed)DataParallel wrapper."""
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        return model.module
    return model


def _seed_hub_cache(arch, local_path):
    """Make a local ImageNet checkpoint visible to ``torch.hub``.

    OpenUnReID fetches ImageNet weights with ``load_state_dict_from_url``, which
    looks into the hub cache before downloading. Compute nodes often have no
    internet access, so a local file is linked into the cache instead.
    """
    urls = dict(resnet.model_urls)
    urls.update(resnet_ibn_a.model_urls)
    if arch not in urls:
        raise KeyError(f"no ImageNet checkpoint is known for backbone '{arch}'")
    if not osp.isfile(local_path):
        raise FileNotFoundError(f"MODEL.backbone_path not found: {local_path}")

    cache_dir = osp.join(torch.hub.get_dir(), "checkpoints")
    cached = osp.join(cache_dir, osp.basename(urls[arch]))
    if osp.exists(cached):
        return cached
    os.makedirs(cache_dir, exist_ok=True)
    try:
        os.symlink(osp.abspath(local_path), cached)
    except FileExistsError:  # another process of the same job was faster
        pass
    except OSError:  # file systems without symlinks
        shutil.copyfile(local_path, cached)
    return cached


def build_model(cfg, num_classes, init=None):
    """Build a re-ID model, optionally from a local ImageNet checkpoint.

    ``cfg.MODEL.backbone_path`` (optional) points to the ImageNet weights of the
    backbone; without it they are downloaded by torch.hub as in OpenUnReID.
    ``init`` is the source-domain pre-trained re-ID checkpoint.
    """
    backbone_path = cfg.MODEL.get("backbone_path", None)
    if backbone_path and cfg.MODEL.imagenet_pretrained:
        _seed_hub_cache(cfg.MODEL.backbone, str(backbone_path))
    return _build_model(cfg, num_classes, init=init)


class BackboneTap:
    """Record the backbone feature maps of a teacher-student model.

    The SP and AT distillation baselines work on the last convolutional
    activations, which the OpenUnReID models do not return. Forward hooks give
    access to them without touching the model code.
    """

    def __init__(self, model):
        model = unwrap(model)
        self.student = None
        self.teacher = None
        self._handles = [
            model.net.backbone.register_forward_hook(self._save("student")),
            model.mean_net.backbone.register_forward_hook(self._save("teacher")),
        ]

    def _save(self, name):
        def hook(module, inputs, output):
            setattr(self, name, output)

        return hook

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
