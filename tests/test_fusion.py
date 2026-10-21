"""Tests for kneescan.eval.fusion on synthetic predictions (no training needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kneescan import PLANES, TASKS
from kneescan.eval.fusion import cv_report, fuse_fold


def _synthetic_preds(n_cases: int = 300, seed: int = 0) -> pd.DataFrame:
    """5 folds; each case is OOF in its own fold and train/tune in the others."""
    rng = np.random.default_rng(seed)
    case_ids = [f"{i:04d}" for i in range(n_cases)]
    labels = {t: rng.integers(0, 2, n_cases) for t in TASKS}
    fold_of = rng.integers(0, 5, n_cases)
    rows = []
    for fold in range(5):
        for i, cid in enumerate(case_ids):
            subset = "oof" if fold_of[i] == fold else ("tune" if i % 10 == 0 else "train")
            for plane in PLANES:
                for task in TASKS:
                    y = labels[task][i]
                    p = float(np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.25), 0.001, 0.999))
                    rows.append(dict(fold=fold, plane=plane, subset=subset, case_id=cid, task=task, label=int(y), prob=p))
    return pd.DataFrame(rows)


def test_fuse_fold_returns_only_that_folds_oof_cases():
    preds = _synthetic_preds()
    fused = fuse_fold(preds, 2, "acl")
    expected = set(preds[(preds.fold == 2) & (preds.subset == "oof")].case_id)
    assert set(fused.case_id) == expected and fused.prob.between(0, 1).all()


def test_fusion_beats_single_plane_on_synthetic_data():
    report = cv_report(_synthetic_preds())
    for task in TASKS:
        assert report["B0"][task]["auc"] > report["per_plane"][f"sagittal/{task}"]["auc"]
        assert report["B0"][task]["n"] == 300


def test_missing_plane_is_an_error():
    preds = _synthetic_preds()
    with pytest.raises(ValueError):
        fuse_fold(preds[preds.plane != "axial"], 0, "acl")
