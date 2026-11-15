import os

import pytest
import torch
import yaml
from synthetic import make_market1501, make_msmt17

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def cpu_only():
    """Let the test-suite run without a GPU.

    OpenUnReID moves tensors with ``.cuda()`` everywhere. On a machine without
    CUDA these calls are turned into no-ops, so the same code path is exercised
    on CPU. Nothing is patched when a GPU is available.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if torch.cuda.is_available():
        yield
        return
    patch = pytest.MonkeyPatch()
    patch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    patch.setattr(torch.nn.Module, "cuda", lambda self, *args, **kwargs: self)
    patch.setattr(torch.cuda, "device_count", lambda: 1)
    yield
    patch.undo()


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets")
    make_market1501(str(root))
    make_msmt17(str(root))
    return root


@pytest.fixture(scope="session")
def source_checkpoint(tmp_path_factory, cpu_only):
    """Stand-in for a source-domain pre-trained model (random weights)."""
    from openunreid.models.builder import ReIDBaseModel

    torch.manual_seed(0)
    model = ReIDBaseModel("resnet50", 0, pooling="gem", pretrained=False)
    path = tmp_path_factory.mktemp("pretrained") / "source.pth"
    torch.save({"state_dict": model.state_dict(), "epoch": 0, "best_mAP": 0.0}, path)
    return path


def tiny_config(method, data_root, logs_root, source_checkpoint, tmp_path, no_source=False, **s2p):
    """The shipped config, shrunk to run on CPU with the synthetic datasets."""
    with open(os.path.join(REPO, "configs", f"{method}_msmt17_to_market1501.yaml")) as f:
        config = yaml.safe_load(f)
    config["DATA_ROOT"] = str(data_root)
    config["LOGS_ROOT"] = str(logs_root)
    config["OUDA"]["n_tasks"] = 3
    config["S2P"].update(s2p)
    config["MODEL"]["imagenet_pretrained"] = False
    config["MODEL"]["source_pretrained"] = None if no_source else str(source_checkpoint)
    config["DATA"]["height"], config["DATA"]["width"] = 64, 32
    config["TRAIN"]["epochs"], config["TRAIN"]["iters"] = 1, 2
    config["TRAIN"]["val_freq"] = 1
    config["TRAIN"]["LOADER"]["workers_per_gpu"] = 0
    config["TRAIN"]["PSEUDO_LABELS"]["search_type"] = 3  # faiss on CPU
    config["TEST"]["LOADER"]["workers_per_gpu"] = 0
    config["TEST"]["dist_cuda"] = torch.cuda.is_available()
    path = tmp_path / f"{method}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path
