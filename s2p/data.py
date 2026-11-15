"""Online (task-incremental) data stream and data-loader helpers.

In the OUDA setting the target identities are split into ``n_tasks`` disjoint
subsets that are seen one after the other, while the labelled source dataset is
always available. This module builds those subsets *outside* of the dataset
classes, so the OpenUnReID datasets stay untouched.
"""

import copy
import os.path as osp
import re

from openunreid.data.datasets import build_dataset
from openunreid.data.samplers import build_test_sampler, build_train_sampler
from openunreid.data.transformers import (
    build_test_transformer,
    build_train_transformer,
)
from openunreid.data.utils.base_dataset import ImageDataset
from openunreid.data.utils.dataset_wrapper import IterLoader, JointDataset
from openunreid.utils.dist_utils import get_dist_info
from torch.utils.data import DataLoader

__all__ = [
    "OUDAStream",
    "make_eval_loaders",
    "make_train_loader",
    "split_into_tasks",
    "subset_by_pids",
]

_MARKET_STYLE = re.compile(r"([-\d]+)_c(\d)")


def _pid_from_market_style_name(path, label):
    return int(_MARKET_STYLE.search(osp.basename(path)).group(1))


def _pid_from_label(path, label):
    return label


# Datasets whose training labels are re-indexed by OpenUnReID: the person id
# of the benchmark has to be read back from the file name. Every other dataset
# (msmt17, custom_*) already uses the benchmark id as its label.
_PERSON_ID_PARSERS = {
    "market1501": _pid_from_market_style_name,
    "dukemtmcreid": _pid_from_market_style_name,
    "cuhk03-np": _pid_from_market_style_name,
}


def person_id(dataset_name, path, label):
    """The benchmark person id of a sample."""
    return _PERSON_ID_PARSERS.get(dataset_name, _pid_from_label)(path, label)


def split_into_tasks(dataset_name, data, n_tasks):
    """Split ``data`` into ``n_tasks`` identity-disjoint lists.

    An identity goes to task ``person_id % n_tasks``. The returned records are
    ``(path, person_id, camid)`` tuples.
    """
    tasks = [[] for _ in range(n_tasks)]
    for path, label, camid in data:
        pid = person_id(dataset_name, path, label)
        tasks[pid % n_tasks].append((path, pid, camid))
    return tasks


def _make_dataset(records, mode, transform, keep_labels):
    if keep_labels:
        relabel = {pid: i for i, pid in enumerate(sorted({r[1] for r in records}))}
        data = [(path, relabel[pid], camid) for path, pid, camid in records]
    else:
        data = [(path, 0, camid) for path, _, camid in records]
    return ImageDataset(data, mode, transform=transform, verbose=False)


def subset_by_pids(dataset, pids):
    """A copy of ``dataset`` restricted to the identities in ``pids``."""
    pids = set(pids)
    data = [record for record in dataset.data if record[1] in pids]
    if len(data) == 0:
        raise ValueError("no sample is left after filtering the identities")
    return ImageDataset(data, dataset.mode, transform=dataset.transform, verbose=False)


class OUDAStream:
    """The target stream (one identity-disjoint subset per task) and the source.

    The training datasets must be ordered ``[target, source]`` in
    ``cfg.TRAIN.datasets``, with ``cfg.TRAIN.unsup_dataset_indexes == [0]``.
    """

    def __init__(self, cfg, n_tasks):
        names = list(cfg.TRAIN.datasets.keys())
        modes = list(cfg.TRAIN.datasets.values())
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.target_name, self.source_name = names
        self.target_mode, self.source_mode = modes

        self.train_transform = build_train_transformer(cfg)
        self.test_transform = build_test_transformer(cfg)

        target = build_dataset(
            self.target_name,
            cfg.DATA_ROOT,
            self.target_mode,
            del_labels=False,
            transform=None,
            verbose=False,
        )
        self._target_tasks = split_into_tasks(self.target_name, target.data, n_tasks)
        for task_id, records in enumerate(self._target_tasks):
            if len(records) == 0:
                raise ValueError(
                    f"task {task_id} of {self.target_name} has no image: reduce OUDA.n_tasks"
                )

        self._source = build_dataset(
            self.source_name,
            cfg.DATA_ROOT,
            self.source_mode,
            del_labels=False,
            transform=None,
            verbose=False,
        )

    def task_size(self, task_id):
        return len(self._target_tasks[task_id])

    @property
    def source_num_pids(self):
        return self._source.num_pids

    def target(self, task_id, train=True, keep_labels=False):
        """The target images of one task.

        Ground-truth labels are dropped unless ``keep_labels`` is set; images
        are always returned in the same (sorted) order, so pseudo labels
        computed on one view can be applied to another.
        """
        transform = self.train_transform if train else self.test_transform
        return _make_dataset(
            self._target_tasks[task_id], self.target_mode, transform, keep_labels
        )

    def source(self, train=True):
        transform = self.train_transform if train else self.test_transform
        return ImageDataset(
            list(self._source.data), self.source_mode, transform=transform, verbose=False
        )

    def summary(self):
        lines = [f"OUDA stream: target={self.target_name} source={self.source_name}"]
        for task_id, records in enumerate(self._target_tasks):
            lines.append(
                f"  task {task_id}: {len({r[1] for r in records}):5d} identities, {len(records):6d} images"
            )
        lines.append(
            f"  source: {self._source.num_pids:5d} identities, {len(self._source):6d} images"
        )
        return "\n".join(lines)


def _loader_sizes(cfg, section):
    _, _, dist = get_dist_info()
    batch_size = section.LOADER.samples_per_gpu
    num_workers = section.LOADER.workers_per_gpu
    if not dist:
        batch_size *= cfg.total_gpus
        num_workers *= cfg.total_gpus
    return batch_size, num_workers


def make_train_loader(cfg, datasets, joint=True, epoch=0):
    """Training loader(s) over already-built datasets.

    Same construction as ``openunreid.data.build_train_dataloader``, which can
    only build its datasets from the config.
    """
    combined = JointDataset(datasets, verbose=False) if joint else copy.deepcopy(datasets)
    samplers = build_train_sampler(cfg, combined, epoch=epoch)
    batch_size, num_workers = _loader_sizes(cfg, cfg.TRAIN)

    def loader(dataset, sampler):
        return IterLoader(
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            ),
            length=cfg.TRAIN.iters,
        )

    if joint:
        return loader(combined, samplers)
    return [loader(d, s) for d, s in zip(combined, samplers)]


def make_eval_loaders(cfg, datasets):
    """Sequential (non-shuffled) loaders, e.g. to extract features for clustering."""
    samplers = build_test_sampler(cfg, datasets)
    batch_size, num_workers = _loader_sizes(cfg, cfg.TEST)
    return [
        DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        for dataset, sampler in zip(datasets, samplers)
    ]
