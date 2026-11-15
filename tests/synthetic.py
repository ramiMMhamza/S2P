"""Tiny synthetic re-ID datasets in the Market-1501 and MSMT17 layouts.

Every identity is a fixed random colour pattern plus noise, so that even a
randomly initialised network gives identity-consistent features.
"""

import os

import numpy as np
from PIL import Image


def _image(identity_seed, rng, height=64, width=32):
    base = np.random.RandomState(identity_seed).randint(0, 255, size=(4, 2, 3))
    img = np.kron(base, np.ones((height // 4, width // 2, 1))).astype(np.float32)
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_market1501(root, num_train_ids=120, images_per_id=8, num_test_ids=8):
    rng = np.random.RandomState(0)
    base = os.path.join(root, "market1501", "Market-1501-v15.09.15")
    for name in ("bounding_box_train", "query", "bounding_box_test"):
        os.makedirs(os.path.join(base, name), exist_ok=True)
    # person ids are deliberately not contiguous, as in the real dataset
    person_ids = [pid for pid in range(2, 2 * num_train_ids + 2) if pid % 7 != 0]
    for pid in person_ids[:num_train_ids]:
        for k in range(images_per_id):
            name = f"{pid:04d}_c{1 + k % 3}s1_{k:06d}_00.jpg"
            _image(pid, rng).save(os.path.join(base, "bounding_box_train", name))
    for pid in range(1001, 1001 + num_test_ids):
        for k in range(2):
            name = f"{pid:04d}_c1s1_{k:06d}_00.jpg"
            _image(pid, rng).save(os.path.join(base, "query", name))
        for k in range(4):
            name = f"{pid:04d}_c{2 + k % 2}s1_{k:06d}_00.jpg"
            _image(pid, rng).save(os.path.join(base, "bounding_box_test", name))


def make_msmt17(root, num_train_ids=20, images_per_id=8, num_test_ids=6):
    rng = np.random.RandomState(1)
    base = os.path.join(root, "msmt17", "MSMT17_V1")
    lists = {"train": [], "val": [], "query": [], "gallery": []}

    def add(split_dir, list_name, pid, k, cam):
        rel = f"{pid:04d}/{pid:04d}_{k:03d}_{cam:02d}_0303morning_0015_0.jpg"
        path = os.path.join(base, split_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _image(5000 + pid + (0 if split_dir == "train" else 3000), rng).save(path)
        lists[list_name].append(f"{rel} {pid}")

    num_val_ids = max(1, num_train_ids // 5)
    for pid in range(num_train_ids):
        for k in range(images_per_id):
            add("train", "train" if pid < num_train_ids - num_val_ids else "val", pid, k, 1 + k % 3)
    for pid in range(num_test_ids):
        for k in range(2):
            add("test", "query", pid, k, 1)
        for k in range(2, 6):
            add("test", "gallery", pid, k, 2 + k % 2)
    for name, lines in lists.items():
        with open(os.path.join(base, f"list_{name}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
