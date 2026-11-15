# Added for S2P (Hamza Rami). Not part of upstream OpenUnReID.

import os.path as osp

import pandas as pd

from ..utils.base_dataset import ImageDataset


class CustomDataset(ImageDataset):
    """A re-ID dataset described by csv files (used for RandPerson in S2P).

    Use it with the dataset name ``custom_<directory>``: the files are read
    from ``<root>/<directory>/list_<mode>.csv`` with ``mode`` one of
    ``train``, ``val``, ``trainval``, ``query`` or ``gallery``.

    Each csv has the three columns ``path``, ``id`` and ``camid``. A relative
    ``path`` is resolved against ``<root>/<directory>``.
    """

    prefix = "custom_"
    dataset_url = None

    def __init__(self, root, mode, name, del_labels=False, **kwargs):
        assert name.startswith(self.prefix) and len(name) > len(self.prefix)
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, name[len(self.prefix):])
        self.del_labels = del_labels

        subsets_cfgs = ["train", "val", "trainval", "query", "gallery"]
        if mode not in subsets_cfgs:
            raise ValueError(
                "Invalid mode. Got {}, but expected to be "
                "one of [train | val | trainval | query | gallery]".format(mode)
            )

        self.list_path = osp.join(self.dataset_dir, "list_" + mode + ".csv")
        self.check_before_run([self.dataset_dir, self.list_path])

        data = self.process_dir(self.list_path)
        super(CustomDataset, self).__init__(data, mode, **kwargs)

    def process_dir(self, list_path):
        df = pd.read_csv(list_path)
        data = []
        for path, pid, camid in zip(df["path"], df["id"], df["camid"]):
            path = str(path)
            if not osp.isabs(path):
                path = osp.join(self.dataset_dir, path)
            # use 0 as labels for all images when labels are deleted
            data.append((path, 0 if self.del_labels else int(pid), int(camid)))
        return data
