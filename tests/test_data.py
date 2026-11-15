import os
import re

import pytest
import torch
from easydict import EasyDict
from openunreid.data.datasets import build_dataset

from s2p.data import OUDAStream, person_id, split_into_tasks, subset_by_pids
from s2p.support import nearest_source


def raw_pid(path):
    return int(re.match(r"(\d+)_c", os.path.basename(path)).group(1))


def stream_config(data_root, target="market1501", source="msmt17"):
    return EasyDict(
        {
            "DATA_ROOT": str(data_root),
            "DATA": {
                "height": 64,
                "width": 32,
                "norm_mean": [0.485, 0.456, 0.406],
                "norm_std": [0.229, 0.224, 0.225],
                "TRAIN": {
                    "is_autoaug": False,
                    "is_flip": True,
                    "flip_prob": 0.5,
                    "is_pad": False,
                    "is_blur": False,
                    "is_erase": False,
                    "is_mutual_transform": False,
                },
            },
            "TRAIN": {"datasets": {target: "trainval", source: "trainval"}},
        }
    )


def test_person_id_is_read_from_market_style_names():
    assert person_id("market1501", "/data/x_c9/0042_c3s1_000001_00.jpg", 7) == 42
    assert person_id("cuhk03-np", "/d/0007_c2_0001.png", 0) == 7
    # msmt17 and csv datasets: the label is the benchmark id
    assert person_id("msmt17", "/d/0003/0003_000_01.jpg", 3) == 3


def test_tasks_are_an_identity_disjoint_partition(data_root):
    full = build_dataset("market1501", data_root, "trainval", del_labels=False, verbose=False)
    tasks = split_into_tasks("market1501", full.data, 5)
    assert sum(len(t) for t in tasks) == len(full)
    seen = set()
    for task_id, records in enumerate(tasks):
        pids = {pid for _, pid, _ in records}
        assert pids and not (pids & seen)
        seen |= pids
        # same rule as the experiments of the paper: person id modulo n_tasks
        assert all(pid % 5 == task_id for pid in pids)
        assert all(raw_pid(path) == pid for path, pid, _ in records)


def test_stream_views_share_the_image_order(data_root):
    stream = OUDAStream(stream_config(data_root), n_tasks=5)
    unlabelled = stream.target(2, train=True)
    labelled = stream.target(2, train=False, keep_labels=True)
    assert [d[0] for d in unlabelled.data] == [d[0] for d in labelled.data]
    assert {d[1] for d in unlabelled.data} == {0}
    assert labelled.num_pids == len({raw_pid(d[0]) for d in labelled.data})
    assert sum(stream.task_size(t) for t in range(5)) == 120 * 8
    assert stream.source_num_pids == 20


def test_support_subset_keeps_the_selected_identities(data_root):
    """Regression test for the wrong-identity support set (docs/AUDIT.md, C1).

    With Market-1501 as the source, labels are re-indexed (0, 1, 2, ...) while
    person ids are not contiguous: selecting by label must return the images
    that carry these labels.
    """
    source = build_dataset("market1501", data_root, "trainval", del_labels=False, verbose=False)
    wanted_labels = [3, 10, 57]
    wanted_people = {raw_pid(p) for p, label, _ in source.data if label in wanted_labels}
    support = subset_by_pids(source, wanted_labels)
    assert {raw_pid(p) for p, _, _ in support.data} == wanted_people
    assert {label for _, label, _ in support.data} == set(wanted_labels)
    assert len(support) == 3 * 8
    with pytest.raises(ValueError):
        subset_by_pids(source, [10 ** 6])


def test_nearest_source():
    target = {10: torch.tensor([1.0, 0.0]), 11: torch.tensor([0.0, 1.0])}
    source = {5: torch.tensor([0.0, 1.0]), 7: torch.tensor([0.6, 0.8]), 9: torch.tensor([1.0, 0.0])}
    pairs = nearest_source(target, source)
    assert [(t, s) for t, s, _ in pairs] == [(10, 9), (11, 5)]
    assert all(abs(sim - 1.0) < 1e-6 for _, _, sim in pairs)


def test_custom_csv_dataset(tmp_path, data_root):
    import pandas as pd

    market = build_dataset("market1501", data_root, "trainval", del_labels=False, verbose=False)
    directory = tmp_path / "randperson"
    directory.mkdir()
    rows = [{"path": p, "id": 100 + label, "camid": cam} for p, label, cam in market.data[:24]]
    pd.DataFrame(rows).to_csv(directory / "list_train.csv", index=False)
    dataset = build_dataset("custom_randperson", tmp_path, "train", del_labels=False, verbose=False)
    assert len(dataset) == 24 and dataset.num_pids == 3
    assert {pid for _, pid, _ in dataset.data} == {100, 101, 102}
    assert subset_by_pids(dataset, [101]).num_pids == 1
