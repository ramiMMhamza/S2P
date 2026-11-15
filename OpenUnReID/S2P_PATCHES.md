# Vendored OpenUnReID

This directory is a copy of [open-mmlab/OpenUnReID](https://github.com/open-mmlab/OpenUnReID)
at commit `c3e3ada` (2021-06-14, the last upstream commit), Apache-2.0.

S2P does **not** modify the toolbox: everything specific to the paper lives in
`../s2p/` and uses the toolbox through its public API. The only changes are the
patches below, each marked with an `S2P patch` comment so they can be found with
`grep -rn "S2P patch" OpenUnReID`.

| # | File | Why |
|---|------|-----|
| P1 | `openunreid/core/utils/faiss_utils.py` | faiss >= 1.7 renamed `cast_integer_to_long_ptr` to `cast_integer_to_idx_t_ptr` |
| P2 | `openunreid/data/transformers/auto_augment.py`, `openunreid/core/metrics/rank.py` | `np.int` / `np.bool` were removed in numpy 1.24 |
| P3 | `openunreid/data/transformers/__init__.py` | torchvision >= 0.15 rejects integer interpolation codes |
| P4 | `requirements.txt` | `sklearn` and `pillow-simd` no longer install from PyPI; faiss pin relaxed; `pandas` for P5 |
| P5 | `openunreid/data/datasets/{__init__,cuhk03np,custom_reid}.py` | datasets used in the paper: CUHK03-NP and csv-described datasets (`custom_<dir>`, RandPerson) |

To move to a newer toolbox version, re-apply these five patches.
