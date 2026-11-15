"""Pseudo-label generation restricted to the target images of one task."""

from openunreid.core.label_generators import LabelGenerator
from openunreid.utils.dist_utils import get_dist_info
from torch import nn

__all__ = ["TaskLabelGenerator"]


class TaskLabelGenerator(LabelGenerator):
    """``LabelGenerator`` over explicit data loaders.

    The OpenUnReID generator builds its loaders from the config, i.e. over the
    whole target dataset. In the online setting only the images of the current
    task may be clustered, so the loaders are passed in.
    """

    def __init__(self, cfg, models, data_loaders, datasets, verbose=True):
        # LabelGenerator.__init__ is deliberately not called: it would build
        # loaders over the full target dataset. Its attributes are set here.
        assert (
            "PSEUDO_LABELS" in cfg.TRAIN
        ), "cannot find settings in the config file for pseudo labels"

        self.cfg = cfg
        if isinstance(models, nn.Module):
            models = [models]
        self.models = models
        self.verbose = verbose

        self.data_loaders = data_loaders
        self.datasets = [dataset.data for dataset in datasets]

        self.cluster_type = cfg.TRAIN.PSEUDO_LABELS.cluster
        self.num_classes = []
        self.indep_thres = []
        if cfg.TRAIN.PSEUDO_LABELS.cluster_num is not None:
            # for kmeans
            self.num_classes = cfg.TRAIN.PSEUDO_LABELS.cluster_num

        self.rank, self.world_size, _ = get_dist_info()
