"""Configuration: OpenUnReID yaml config + the S2P / OUDA sections."""

from pathlib import Path

from easydict import EasyDict
from openunreid.utils.config import cfg, cfg_from_list, cfg_from_yaml_file

from .losses import KD_INPUTS

__all__ = ["METHODS", "load_config"]

METHODS = ("mmt", "spcl")

_DEFAULTS = {
    "METHOD": "mmt",
    "OUDA": {
        # number of identity-disjoint subsets the target stream is split into
        "n_tasks": 5,
    },
    "S2P": {
        # distillation on the support set: s2p | lwf | sp | at
        "kd": "s2p",
        "kd_weight": 0.1,
        # MMD between target features and (teacher) source features
        "mmd_weight": 0.1,
        # log the most / least similar (target, source) pairs to TensorBoard
        "log_images": False,
    },
}


def _fill_defaults(config, defaults):
    for key, value in defaults.items():
        if isinstance(value, dict):
            if key not in config or config[key] is None:
                config[key] = EasyDict()
            _fill_defaults(config[key], value)
        elif key not in config:
            config[key] = value


def load_config(path, overrides=None, set_cfgs=None):
    """Load a yaml config into the global OpenUnReID ``cfg`` and validate it.

    Args:
        path: yaml file.
        overrides: {dotted key: value} applied after loading; ``None`` values
            are ignored (i.e. command-line options that were not given).
        set_cfgs: raw ``KEY VALUE`` pairs, see ``openunreid.utils.config``.
    """
    cfg_from_yaml_file(path, cfg)
    _fill_defaults(cfg, _DEFAULTS)

    for dotted_key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted_key.split(".")
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    if set_cfgs:
        cfg_from_list(set_cfgs, cfg)

    cfg.DATA_ROOT = Path(cfg.DATA_ROOT)
    cfg.LOGS_ROOT = Path(cfg.LOGS_ROOT)
    cfg.METHOD = str(cfg.METHOD).lower()
    cfg.S2P.kd = str(cfg.S2P.kd).lower()
    validate(cfg)
    return cfg


def validate(config):
    def check(condition, message):
        if not condition:
            raise ValueError("invalid config: " + message)

    check(config.METHOD in METHODS, f"METHOD must be one of {METHODS}")
    check(
        len(config.TRAIN.datasets) == 2
        and list(config.TRAIN.unsup_dataset_indexes) == [0],
        "TRAIN.datasets must be {target: mode, source: mode} (in this order) "
        "with TRAIN.unsup_dataset_indexes: [0]",
    )
    check(config.OUDA.n_tasks >= 1, "OUDA.n_tasks must be >= 1")
    check(
        config.MODEL.mean_net,
        "MODEL.mean_net must be True: the mean (teacher) network provides the "
        "distillation targets of S2P",
    )
    check(config.S2P.kd in KD_INPUTS, f"S2P.kd must be one of {sorted(KD_INPUTS)}")
    check(
        not (config.METHOD == "spcl" and config.S2P.kd == "lwf"),
        "S2P.kd: lwf needs a classifier, which SpCL does not have",
    )
    check(config.S2P.kd_weight >= 0 and config.S2P.mmd_weight >= 0, "weights must be >= 0")
    check(
        config.MODEL.source_pretrained is not None,
        "MODEL.source_pretrained (source-domain pre-trained checkpoint) is required; "
        "train it with OpenUnReID/tools/source_pretrain",
    )
    if config.METHOD == "spcl":
        check(
            config.TRAIN.PSEUDO_LABELS.use_outliers,
            "SpCL needs TRAIN.PSEUDO_LABELS.use_outliers: True",
        )
