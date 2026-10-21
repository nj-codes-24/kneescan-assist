"""Tests for the CV splits (R3, R4) and the metrics (§5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kneescan import TASKS
from kneescan.data.mrnet import read_labels
from kneescan.data.splits import make_cv_folds, make_inner_tune, split_for
from kneescan.eval.metrics import (auc, bootstrap_ci, brier, ece, paired_bootstrap_delta,
                                   sens_spec_at, threshold_for_sensitivity, wilson_ci)
from tests.conftest import make_fake_mrnet


def _labels(tmp_path, n=200):
    return read_labels(make_fake_mrnet(tmp_path, n_train=n, n_valid=2, hw=8), "train")


def test_folds_cover_every_case_once_and_are_deterministic(tmp_path):
    labels = _labels(tmp_path)
    a, b = make_cv_folds(labels), make_cv_folds(labels)
    assert a.equals(b)
    assert sorted(a["case_id"]) == sorted(labels["case_id"])
    assert set(a["fold"]) == {0, 1, 2, 3, 4}
    assert list(a.columns) == ["case_id", "fold"]          # no labels in the saved file


def test_folds_are_stratified(tmp_path):
    labels = _labels(tmp_path, n=500)
    folds = make_cv_folds(labels).merge(labels, on="case_id")
    rates = folds.groupby("fold")["acl"].mean()
    assert rates.max() - rates.min() < 0.05


def test_inner_tune_is_inside_training_part_and_disjoint(tmp_path):
    labels = _labels(tmp_path)
    folds = make_cv_folds(labels)
    inner = make_inner_tune(labels, folds)
    for f in range(5):
        s = split_for(f, folds, inner)
        assert not (set(s.train) & set(s.tune)) and not (set(s.train) & set(s.oof))
        assert not (set(s.tune) & set(s.oof))
        assert len(s.train) + len(s.tune) + len(s.oof) == len(labels)
        assert abs(len(s.tune) - 0.1 * (len(labels) - len(s.oof))) <= 1


def test_auc_matches_sklearn_and_ci_brackets_it():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    assert auc(y, p) == roc_auc_score(y, p)
    point, lo, hi = bootstrap_ci(y, p, n_boot=300)
    assert lo < point < hi


def test_cluster_bootstrap_runs():
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 2, 100), rng.random(100)
    clusters = np.repeat(np.arange(50), 2)
    point, lo, hi = bootstrap_ci(y, p, n_boot=200, clusters=clusters)
    assert lo <= point <= hi


def test_paired_delta_of_identical_models_is_zero():
    rng = np.random.default_rng(3)
    y, p = rng.integers(0, 2, 100), rng.random(100)
    assert paired_bootstrap_delta(y, p, p, n_boot=100) == (0.0, 0.0, 0.0)


def test_wilson_known_value():
    phat, lo, hi = wilson_ci(8, 10)
    assert phat == 0.8 and abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3


def test_sens_spec_and_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1, 0.1, 0.0])
    r = sens_spec_at(y, p, 0.5)
    assert r["sensitivity"][0] == 0.75 and abs(r["specificity"][0] - 5 / 6) < 1e-9
    t = threshold_for_sensitivity(y, p, 0.75)
    assert t == 0.7 and sens_spec_at(y, p, t)["sensitivity"][0] >= 0.75


def test_calibration_perfect_and_bad():
    y = np.array([0] * 50 + [1] * 50)
    assert ece(y, np.full(100, 0.5)) == 0.0
    assert abs(ece(y, np.where(y == 1, 0.0, 1.0)) - 1.0) < 1e-12
    assert brier(y, y.astype(float)) == 0.0
