"""End-to-end runs of the full pipeline on synthetic data (CPU friendly)."""

import json

import pytest
import torch
from conftest import tiny_config

from s2p.train import main


def run(method, data_root, source_checkpoint, tmp_path, argv=(), **s2p):
    logs = tmp_path / "logs"
    config = tiny_config(method, data_root, logs, source_checkpoint, tmp_path, **s2p)
    results = main([str(config), "--work-dir", "run", *argv])
    return logs / "run", results


def check_outputs(work_dir, results, n_tasks=3, n_models=1):
    assert (work_dir / "log.txt").stat().st_size > 0
    assert (work_dir / "config.yaml").exists()
    for task_id in range(n_tasks):
        assert (work_dir / f"task_{task_id}.pth").exists()
    # source-only reference + one evaluation per task, for every model
    assert len(results) == (n_tasks + 1) * n_models
    assert [r["after_task"] for r in results[::n_models]] == list(range(-1, n_tasks))
    assert all(0.0 <= r["mAP"] <= 1.0 for r in results)
    assert json.loads((work_dir / "results.json").read_text()) == results
    log = (work_dir / "log.txt").read_text()
    assert log.count("Support set:") == n_tasks - 1
    return log


@pytest.mark.slow
def test_mmt_s2p(data_root, source_checkpoint, tmp_path):
    work_dir, results = run("mmt", data_root, source_checkpoint, tmp_path, log_images=True)
    log = check_outputs(work_dir, results, n_models=2)
    assert "kd_s2p" in log and "mmd" in log

    state = torch.load(work_dir / "task_2.pth", map_location="cpu")
    assert state["epoch"] == 3 and {"state_dict_1", "state_dict_2"} <= set(state)

    # restart from the end of the second task: only the last task is trained
    resumed = main(
        [str(tmp_path / "mmt.yaml"), "--work-dir", "resumed", "--resume-from", str(work_dir / "task_1.pth")]
    )
    assert [r["after_task"] for r in resumed] == [2, 2]


@pytest.mark.slow
def test_spcl_s2p(data_root, source_checkpoint, tmp_path):
    work_dir, results = run("spcl", data_root, source_checkpoint, tmp_path)
    log = check_outputs(work_dir, results)
    assert "kd_s2p" in log and "hybrid_memory" in log


@pytest.mark.slow
@pytest.mark.parametrize("method, kd", [("mmt", "lwf"), ("mmt", "sp"), ("spcl", "at")])
def test_distillation_baselines(method, kd, data_root, source_checkpoint, tmp_path):
    work_dir, _results = run(
        method, data_root, source_checkpoint, tmp_path, argv=["--kd", kd, "--n-tasks", "2"]
    )
    assert "kd_" + kd in (work_dir / "log.txt").read_text()


@pytest.mark.slow
def test_plain_online_baseline(data_root, source_checkpoint, tmp_path):
    """Both S2P terms switched off: the base method in the online setting."""
    work_dir, _results = run(
        "mmt", data_root, source_checkpoint, tmp_path,
        argv=["--kd-weight", "0", "--mmd-weight", "0", "--n-tasks", "2"],
    )
    log = (work_dir / "log.txt").read_text()
    # no S2P meter in the training progress lines
    train_lines = [line for line in log.splitlines() if line.lstrip("\x1b[1m").startswith("Epoch:")]
    assert train_lines
    assert not any("kd_s2p" in line or "mmd " in line for line in train_lines)


def test_config_validation(data_root, source_checkpoint, tmp_path):
    config = tiny_config("spcl", data_root, tmp_path / "logs", source_checkpoint, tmp_path)
    with pytest.raises(ValueError, match="lwf"):
        main([str(config), "--kd", "lwf"])
    with pytest.raises(ValueError, match="source_pretrained"):
        main([str(tiny_config("mmt", data_root, tmp_path / "logs", None, tmp_path, no_source=True))])
