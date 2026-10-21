"""
Tests for B0: model shapes, augmentation, scheduling helpers, and the most important
property of the trainer: stopping and resuming gives exactly the same weights as an
uninterrupted run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pandas as pd
import torch

from kneescan import TASKS
from kneescan.data.mrnet import read_labels
from kneescan.data.splits import make_cv_folds, make_inner_tune, split_for
from kneescan.data.transforms import random_affine, to_network_input
from kneescan.models.baseline_mrnet import MRNetBaseline
from kneescan.train.b0_trainer import PlaneJob, class_weights, compute_deadline, should_pause
from kneescan.train.config import B0Config, config_hash, load_b0_config
from tests.conftest import REPO_ROOT, make_fake_mrnet

TINY = B0Config(epochs=3, pretrained=False, amp=False, num_workers=0, deterministic=True)


def test_model_output_is_scalar_logit():
    model = MRNetBaseline(pretrained=False)
    assert model(torch.randn(7, 3, 64, 64)).shape == ()
    assert model.slice_features(torch.randn(7, 3, 64, 64)).shape == (7, 256)


def test_network_input_and_augmentation_shapes():
    vol = torch.randint(0, 256, (5, 64, 64), dtype=torch.uint8)
    out = random_affine(vol.float(), 25, 25, 0.5, torch.Generator().manual_seed(0))
    assert out.shape == (5, 64, 64)
    assert to_network_input(out).shape == (5, 3, 64, 64)


def test_augmentation_is_reproducible_with_same_generator():
    vol = torch.rand(4, 32, 32)
    a = random_affine(vol, 25, 25, 0.5, torch.Generator().manual_seed(3))
    b = random_affine(vol, 25, 25, 0.5, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_class_weights_favour_rare_class():
    labels = pd.DataFrame({"a": [1, 1, 1, 0], "b": [0, 0, 0, 1], "c": [1, 0, 1, 0]}).to_numpy()
    w = class_weights(labels)
    assert w[0, 0] > w[0, 1] and w[1, 1] > w[1, 0] and w[2, 0] == w[2, 1]


def test_deadline_and_pause():
    now = dt.datetime(2026, 10, 20, 7, 30)
    deadline = compute_deadline("08:00", now)
    assert deadline == dt.datetime(2026, 10, 20, 8, 0)
    assert compute_deadline("08:00", dt.datetime(2026, 10, 20, 9, 0)) == dt.datetime(2026, 10, 21, 8, 0)
    assert should_pause(now, deadline, last_epoch_seconds=40 * 60)       # 40-min epoch won't fit
    assert not should_pause(now, deadline, last_epoch_seconds=10 * 60)
    assert not should_pause(now, None, 1e9)


def test_repo_config_loads():
    cfg = load_b0_config(REPO_ROOT / "configs" / "b0_alexnet.yaml")
    assert cfg.epochs == 50 and cfg.lr == 1e-5 and len(config_hash(cfg)) == 12


def _job(tmp_path, run_name, cfg=TINY):
    root = make_fake_mrnet(tmp_path / "data", n_train=80, n_valid=4, hw=64)
    labels = read_labels(root, "train")
    folds = make_cv_folds(labels)
    split = split_for(0, folds, make_inner_tune(labels, folds))
    return PlaneJob(cfg, root, labels, split, fold=0, plane="sagittal",
                    run_dir=tmp_path / "runs" / run_name, device="cpu")


def test_resume_gives_identical_weights(tmp_path):
    straight = _job(tmp_path, "straight")
    assert straight.run() == "done"

    first = _job(tmp_path, "interrupted")
    assert first.run(max_epochs_this_call=1) == "paused"
    second = _job(tmp_path, "interrupted")       # a fresh process would build a fresh object
    assert second.run() == "done"

    a = torch.load(straight.run_dir / "last.pt", weights_only=False)["models"]
    b = torch.load(second.run_dir / "last.pt", weights_only=False)["models"]
    for task in TASKS:
        for key in a[task]:
            assert torch.equal(a[task][key], b[task][key]), f"{task}/{key} differs after resume"
    # the metrics log has exactly one row per (epoch, task), with no duplicates from the resume
    metrics = pd.read_csv(second.run_dir / "metrics.csv")
    assert len(metrics) == TINY.epochs * 3 and not metrics.duplicated(["epoch", "task"]).any()


def test_finished_job_writes_preds_without_valid(tmp_path):
    job = _job(tmp_path, "preds")
    job.run()
    preds = pd.read_csv(job.run_dir / "preds.csv", dtype={"case_id": str})
    assert set(preds["subset"]) == {"train", "tune", "oof"}
    assert preds["case_id"].astype(int).max() < 80          # valid ids start at 80 in the fake set
    assert preds["prob"].between(0, 1).all()
    assert (job.run_dir / "done.json").exists()
    assert job.run() == "done"                             # re-running a finished job is a no-op


def test_resume_refuses_changed_config(tmp_path):
    _job(tmp_path, "cfg").run(max_epochs_this_call=1)
    other = dataclasses.replace(TINY, lr=1e-3)
    try:
        _job(tmp_path, "cfg", cfg=other).run()
    except RuntimeError as e:
        assert "different config" in str(e)
    else:
        raise AssertionError("resume with a changed config should fail")
