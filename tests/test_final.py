"""
test_final.py: the one-time final-test code works BEFORE it's used for real (on fake data only; the real test sets stay untouched).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from kneescan import TASKS
from kneescan.eval.final_metrics import paired_report, task_report
from kneescan.eval.final_predict import reader_predict_mrnet
from tests.conftest import REPO_ROOT, make_fake_mrnet
from tests.test_app import write_fake_run

_spec = importlib.util.spec_from_file_location("freeze", REPO_ROOT / "scripts" / "freeze.py")
freeze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze)


def test_task_report_uses_the_frozen_threshold_and_calibrator() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0.01, 0.99)
    report = task_report(y, p, threshold=0.3, platt=(1.0, 0.0))     # identity calibrator
    assert report["t_hs"] == 0.3 and report["auc_ci"][0] <= report["auc"] <= report["auc_ci"][1]
    assert abs(report["ece_raw"] - report["ece_calibrated"]) < 1e-9
    clustered = task_report(y, p, 0.3, (1.0, 0.0), clusters=np.arange(300) // 2)
    assert clustered["auc"] == report["auc"]


def test_paired_report_of_identical_models_is_zero() -> None:
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    report = paired_report(y, p, p)
    assert report["delta_auc"] == 0 and not report["new_reliably_better"] and not report["new_reliably_worse"]


def test_reader_predicts_every_valid_exam(tmp_path: Path) -> None:
    root = make_fake_mrnet(tmp_path / "data", n_train=10, n_valid=4, hw=64)
    write_fake_run(tmp_path / "runs" / "fake")
    preds = reader_predict_mrnet(tmp_path / "runs" / "fake", root, "valid", device="cpu")
    assert len(preds) == 4 * len(TASKS) and preds["prob"].between(0, 1).all()


def test_freeze_finds_reader_and_b0_weights(tmp_path: Path) -> None:
    write_fake_run(tmp_path / "reader", n_folds=2)
    assert len(freeze.weight_files(tmp_path / "reader")) == 2
    job = tmp_path / "b0" / "fold0" / "sagittal"
    job.mkdir(parents=True)
    for task in TASKS:
        (job / f"best_{task}.pt").write_bytes(b"x")
    assert len(freeze.weight_files(tmp_path / "b0")) == 3
    assert freeze.sha256(job / "best_acl.pt") == freeze.sha256(job / "best_abnormal.pt")


def test_final_eval_hybrid_is_the_mean_logit() -> None:
    spec = importlib.util.spec_from_file_location("final_eval", REPO_ROOT / "scripts" / "final_eval.py")
    final_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final_eval)
    assert np.allclose(final_eval.mean_logit(np.array([0.9, 0.2]), np.array([0.5, 0.2])), [0.75, 0.2], atol=1e-3)
