# Code audit (November 2026)

Audit of the code released with the WACV 2024 paper (commit `1002b85`), and of
what was changed on the `feat/audit` branch.

## How the audit was done

* Read every file of the repository and diffed the vendored `OpenUnReID/` against
  upstream (commit `c3e3ada`): **43 files modified, 3 added, 4 deleted**.
* Static analysis (pyflakes / ruff).
* **Executed** both entry points (`tools/MMT/main.py`, `tools/SpCL/main.py`)
  end to end, 5 tasks, in a Python 3.10 / PyTorch 1.13.1 environment (the one in
  `docs/INSTALL.md`), on CPU, with small synthetic datasets in the Market-1501
  and MSMT17 layouts. Logs of these runs are summarised below.

Limitation: no GPU and no real dataset were available. Everything below about
*behaviour* was observed on synthetic data; the published numbers have **not**
been re-run. See "What still has to be verified".

Severity: **B** blocks running, **C** silently changes results,
**P** performance, **M** maintainability.

## 1. Blocking errors (B)

| id | Finding | Evidence |
|----|---------|----------|
| B1 | `openunreid/data/datasets/__init__.py` imports `personx`, `veri`, `vehicleid`, `vehiclex`, which are not in the repository: `import openunreid` fails, **no entry point can start** from a fresh clone. | `ModuleNotFoundError: openunreid.data.datasets.personx` |
| B2 | `tools/MMT/main.py` imports `tensorflow` (unused); TensorFlow is a 500 MB+ dependency in `requirements.txt` and its presence changes how TensorBoard writes files (a partial install crashes `SummaryWriter`). `pillow-simd` does not build on most systems. | `ModuleNotFoundError: tensorflow` |
| B3 | SpCL without a distributed launcher: `model.num_features` on a `TeacherStudentNetwork` (the attribute is on `model.net`). | `AttributeError` at `tools/SpCL/main.py:409` |
| B4 | Machine-specific paths everywhere: `/data/ramih/...`, `/data/cvlab/...`, `/out/`, `/optim/score.txt`, `./data/ramih/...` in `resnet.py`, `cd /home/ids/hrami` in `S2P_run.sh`, an HTTP proxy IP in `run_cmd_train.sh`. The `--exec` switch selects one of three hard-coded machines; `--exec 0` reads keys that do not exist in the configs (`backbone_path_local`, `source_pretrained_local`, `DATA_ROOT_local` for SpCL). Without `--exec`, logs go to `/out/`. | `KeyError` / `PermissionError` |
| B5 | `run_s2p_train.sh`: bash syntax while the README says `sh`; paths only valid when launched from the parent directory of a clone named `S2P`; `mkdir` without `-p`; `PYTHONPATH` relative to the wrong directory. | |
| B6 | 242 committed build artefacts (`__pycache__` for 3 Python versions, `build/`, `*.egg-info`, a compiled `.so`, generated `rank_cy.c`). The `.so` only loads on CPython 3.10 / x86-64 and hides a missing build step. | |

## 2. Silent correctness problems (C)

| id | Finding | Impact |
|----|---------|--------|
| **C1** | **Support set built from the wrong identities** when the source is Market-1501, DukeMTMC-reID or CUHK03-NP. The selected source ids are taken from the dataset's `id` field, which OpenUnReID *re-indexes* (0, 1, 2, ...) for these datasets, but the filter compared them with the *raw* person ids parsed from file names. | Reproduced: wanted raw ids `[10, 20, 30]` -> support set contained `[9, 19, 29]`. The similarity-preservation loss is then computed on unrelated identities. MSMT17 as source (the README's example) is **not** affected (its labels are the raw ids). |
| **C2** | **RandPerson / csv source: no support-set selection at all.** In `custom_reid.py` the filtered frame is computed and discarded (`df[df["id"].isin(filtre)]` without assignment). | The "support set" is the entire source dataset. Affects every experiment with `custom_randperson` as source (the configs as committed). |
| C3 | MMT: the classifier size is computed from the *first* task only. A later task with more images than the first can produce more clusters than classifier outputs. | Index error or truncated labels on unbalanced splits. |
| C4 | MMT never calls `set_random_seed` (SpCL does). | MMT runs are not seeded although `TRAIN.seed` / `deterministic` are set. |
| C5 | MMT: the LR scheduler created for task 1 is passed to the runners of later tasks, which have a *new* optimizer; it steps the old optimizer. | Latent (the MMT config has `lr_scheduler: null`). |
| C6 | SpCL `--KD LwF` actually trains with Attention Transfer (`KDLossAT`) on feature maps. | Mislabelled baseline. |
| C7 | The KD term at task 0 is `loss * 0`, where `loss` is whatever the previous loop iteration left: it only works because `kd_loss` is never the first key of the loss dict (otherwise `NameError`). | Fragile. |
| C8 | The "source" test (`i == 0`, e.g. MSMT17) is skipped and **mAP = 0 is written to TensorBoard**. | Misleading curves. |
| C9 | `test_reid` switches DSBN to the tested domain and does not switch back; clustering and validation that follow use whatever BN was tested last. In the committed configs the last test set is the target, so results are unaffected, but a different `TEST.datasets` order changes training. | Order-dependent. |
| C10 | Validation uses the labelled split of the target. In the original code `save()` forces `is_best=True`, so "model_best" is actually the latest model (correct for the online setting, but the naming hides it). | None, clarified. |

## 3. Performance (P)

| id | Finding |
|----|---------|
| P1 | Every training step runs inside `torch.autograd.set_detect_anomaly(True)`: a debugging mode that typically slows training several times. |
| P2 | `print(jaccard_dist)` and `print(features)` dump full N x N / N x 2048 matrices to the log at every clustering. |
| P3 | `update_labels` reads up to 22 images per cluster from disk at every epoch to build image grids that are then discarded. |
| P4 | Support-set selection: a Python double loop over all (target, source) pairs for cosine similarities, plus 4 image decodes per target image. Replaced by one matrix product. |
| P5 | Six full copies of the 383 MB checkpoint per run (`checkpoint`, `model_best`, `0..4model_best`). |

## 4. Architecture and dependency on OpenUnReID (M)

The S2P logic was written *inside* a fork of OpenUnReID:

* the signature of `build_dataset` and of **every** dataset class was changed
  (`n_tasks, task_id, filtre` as positional arguments);
* `filtre` is polymorphic (`True` = task split, list of ints = source ids, list
  of strings = paths, `False` = everything) and is interpreted differently by
  each dataset class (C1 and C2 come from there);
* `BaseRunner`, `build_model`, `ReIDBaseModel`, `DSBN`, `HybridMemory`,
  `LabelGenerator`, `extract_features`, `save_checkpoint` all got S2P-specific
  arguments, and the two `main.py` files duplicate ~400 lines each;
* a lot of dead code: `main_ouda.py` (x2), `build_train_dataloader_for_sim(_old)`,
  `build_val_dataloader_source`, `adapting_features`, `weighted_clustering`,
  `KDLoss_2` (broken), `Logm`, unused weighted-memory buffers, ...

### Decision

1. `OpenUnReID/` is now the **pristine upstream** code plus 5 small, documented
   compatibility patches (`OpenUnReID/S2P_PATCHES.md`).
2. All S2P code lives in the `s2p` package and uses the toolbox through its
   public API only:

   | S2P need | How it is done without touching the toolbox |
   |----------|---------------------------------------------|
   | target stream split in tasks | `s2p.data.OUDAStream`: builds the full dataset once with `build_dataset`, splits by person id, produces `ImageDataset`s |
   | clustering on the current task only | `s2p.label_generator.TaskLabelGenerator`, a `LabelGenerator` subclass that receives its loaders |
   | support-set batch in DSBN | `s2p.dsbn.SupportDSBN`, swapped in by class (same parameters and state-dict keys) |
   | feature maps for SP / AT | forward hooks (`s2p.models.BackboneTap`) |
   | local ImageNet weights | linked into the `torch.hub` cache (`s2p.models`) |
   | S2P losses, runners | `s2p.losses`, `s2p.runners` (subclasses of `BaseRunner`) |
   | everything machine-specific | config + command-line options, no hard-coded path |

3. Upstream has not been updated since 2021 and is not on PyPI, so it stays
   vendored. If it is ever published, `OpenUnReID/` can become a dependency.

## 5. Changes in behaviour of the refactored code

What the new code does differently from the code of the paper. Everything else
(sampling, losses, the teacher network, the number of epochs, the support-set
algorithm) was kept.

| | Original | Now |
|-|----------|-----|
| Support set (C1, C2) | wrong identities for re-indexed sources; whole dataset for csv sources | the selected identities |
| MMT classifier size (C3) | from task 1 | max over all tasks |
| Seeding (C4) | SpCL only | MMT and SpCL |
| LR scheduler (C5) | MMT: stale | rebuilt per task (as SpCL did) |
| SpCL + `lwf` (C6) | ran AT | rejected; `at` is available explicitly |
| Anomaly detection, debug prints (P1-P3) | on | off |
| Evaluation | every dataset of `TEST.datasets` except the first one | every dataset, before adaptation and after each task, saved to `results.json` |
| KD loss | Frobenius norm over an (N, N, N) expansion | same value, computed in O(N^2) memory (equality covered by a test) |
| SpCL memory | fork of `HybridMemory` with unused weighting code | upstream `HybridMemory` (identical loss: the weights were disabled) |
| Checkpoints | 6 copies | `checkpoint.pth` + `task_<k>.pth` (resumable with `--resume-from`) |

## 6. Open questions (R) — need the author's decision

* **R1 — target labels in the support-set sampling.** The support set is found
  from batches drawn by the identity sampler (P x K) over the previous task's
  target images, and this sampler uses the target *ground-truth* identities.
  They only decide which images are compared (not the training signal), but it
  is still use of target labels in an unsupervised setting. Kept as in the
  paper; a label-free alternative is to sample uniformly or use the pseudo
  labels of the last epoch.
* **R2 — the distillation teacher.** The teacher is the mean (EMA) network,
  which keeps being updated during the current task (comment in the original
  code: "take the mean as a teacher in ours framework"). A frozen copy of the
  model at the end of the previous task is the more common choice in continual
  learning. Kept as in the paper.
* **R3 — support set from the previous task only.** The original code builds a
  list of support loaders over all past tasks but only uses the last one. Kept.

## 7. What still has to be verified

* A full run on GPU with the real datasets, compared against the paper's tables
  (MSMT17 -> Market-1501 with MMT and SpCL first). C1/C2 mean that experiments
  with Market/Duke/CUHK03-NP/RandPerson as source *will* give different numbers.
* Multi-GPU (DDP) runs: the code path is the same as before, but it was only
  exercised with one process here.
