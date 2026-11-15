"""Selection of the source support set.

After each task, S2P looks for the source identities that are the most similar
to the target data just seen. Their images form the support set on which the
feature similarities are preserved while the model adapts to the next task.
"""

from collections import OrderedDict, namedtuple

import torch
import torch.nn.functional as F

from .data import make_train_loader, subset_by_pids

__all__ = ["SupportSelection", "nearest_source", "select_support_set"]

SupportSelection = namedtuple(
    "SupportSelection", ["pids", "pairs", "num_target", "num_source"]
)
SupportSelection.__doc__ = """Result of the support-set selection.

pids: sorted source identities that make up the support set.
pairs: (target index, source index, cosine similarity) for every target sample,
    with indices into the target / source datasets given to the selection.
num_target, num_source: number of distinct images that were compared.
"""


def nearest_source(target_features, source_features):
    """For each target feature, the most similar source feature.

    Both inputs are ``OrderedDict`` (dataset index -> L2-normalised feature).
    Returns a list of ``(target index, source index, cosine similarity)``.
    """
    target_idx = list(target_features.keys())
    source_idx = list(source_features.keys())
    target = torch.stack([target_features[i] for i in target_idx])
    source = torch.stack([source_features[i] for i in source_idx])
    similarity, nearest = (target @ source.t()).max(dim=1)
    return [
        (t, source_idx[n], s)
        for t, n, s in zip(target_idx, nearest.tolist(), similarity.tolist())
    ]


@torch.no_grad()
def _sample_features(cfg, model, target_set, source_set):
    """Features of randomly drawn (target, source) training batches.

    ``cfg.TRAIN.iters`` joint batches are drawn with the training sampler and
    the training augmentations, as in the experiments of the paper, and embedded
    by ``model`` in evaluation mode (i.e. with the target-domain batch norm).
    """
    loader = make_train_loader(cfg, [target_set, source_set], joint=True)
    was_training = model.training
    model.eval()
    loader.new_epoch(0)

    features = (OrderedDict(), OrderedDict())
    for _ in range(len(loader)):
        batch = loader.next()  # [target sub-batch, source sub-batch]
        for domain, sub_batch in enumerate(batch):
            images = sub_batch["img"]
            if isinstance(images, (list, tuple)):  # mutual transform: first view
                images = images[0]
            outputs = F.normalize(model(images.cuda()), p=2, dim=-1).cpu()
            for index, feature in zip(sub_batch["ind"].tolist(), outputs):
                features[domain][index] = feature

    model.train(was_training)
    return features


def select_support_set(cfg, model, target_set, source_set):
    """Pick the source identities closest to ``target_set``.

    Every sampled target image votes for its nearest sampled source image; the
    support set is made of *all* the images of the identities that received a
    vote.

    Args:
        model: network used to embed the images (its teacher / mean network is
            used when it has one, since it runs in evaluation mode).
        target_set: labelled view of the target images of the task that was
            just learned. NOTE: the labels are only used by the identity
            sampler that draws the batches, see docs/AUDIT.md (R1).
        source_set: the labelled source dataset.

    Returns:
        (support dataset, SupportSelection)
    """
    target_features, source_features = _sample_features(
        cfg, model, target_set, source_set
    )
    pairs = nearest_source(target_features, source_features)
    pids = sorted({source_set.data[source_index][1] for _, source_index, _ in pairs})
    selection = SupportSelection(
        pids=pids,
        pairs=pairs,
        num_target=len(target_features),
        num_source=len(source_features),
    )
    return subset_by_pids(source_set, pids), selection
