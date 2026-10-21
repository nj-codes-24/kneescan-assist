"""
fusion.py: combine B0's three plane predictions per task and score the cross-validation.

What it does
    * `load_b0_predictions(runs_dir)` gathers every job's preds.csv into one table.
    * `fuse_fold(preds, fold, task, fit_on)` fits a logistic regression on the 3 plane
      probabilities (sagittal, coronal, axial) of one fold and applies it to that fold's
      OOF cases.
        fit_on="train" -> B0, faithful to the paper (LR fit on the CNNs' own training
                          predictions, which are over-confident)
        fit_on="tune"  -> B0-oof variant (LR fit on inner-tuning predictions)
    * `cv_report(preds)` pools OOF predictions over the 5 folds and computes the §5 metrics.
      This is the "beat this" number.

Why also report B0-oof?
    Fitting the fuser on training predictions is what the paper did, but those predictions
    are optimistic. Showing both tells us how much that choice matters.

Alternatives
    With more data, a stacked model with its own inner CV is cleaner still. With 1,130
    exams, the inner tuning split (~90 exams per fold) is the honest compromise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kneescan import PLANES, TASKS
from kneescan.eval.metrics import (auc, bootstrap_ci, brier, ece, sens_spec_at,
                                   threshold_for_sensitivity)


def load_b0_predictions(runs_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(runs_dir).glob("fold*/*/preds.csv"))
    if not files:
        raise FileNotFoundError(f"no preds.csv under {runs_dir}")
    return pd.concat([pd.read_csv(f, dtype={"case_id": str}) for f in files], ignore_index=True)


def _wide(preds: pd.DataFrame, fold: int, task: str, subset: str) -> pd.DataFrame:
    """One row per case, one column per plane, plus the label."""
    sel = preds[(preds["fold"] == fold) & (preds["task"] == task) & (preds["subset"] == subset)]
    wide = sel.pivot(index="case_id", columns="plane", values="prob")
    missing = set(PLANES) - set(wide.columns)
    if missing:
        raise ValueError(f"fold {fold} {task}: planes missing {sorted(missing)} (jobs not finished?)")
    labels = sel.drop_duplicates("case_id").set_index("case_id")["label"]
    return wide[list(PLANES)].join(labels)


def fit_fusion(preds: pd.DataFrame, fold: int, task: str, fit_on: str = "train") -> LogisticRegression:
    """The fold's plane-fusion logistic regression (inputs: sagittal, coronal, axial probabilities, in PLANES order)."""
    fit = _wide(preds, fold, task, fit_on)
    return LogisticRegression().fit(fit[list(PLANES)], fit["label"])


def fuse_fold(preds: pd.DataFrame, fold: int, task: str, fit_on: str = "train") -> pd.DataFrame:
    """OOF fused probabilities for one fold/task. Columns: case_id, fold, task, label, prob."""
    lr = fit_fusion(preds, fold, task, fit_on)
    oof = _wide(preds, fold, task, "oof")
    return pd.DataFrame({"case_id": oof.index, "fold": fold, "task": task, "label": oof["label"].to_numpy(),
                         "prob": lr.predict_proba(oof[list(PLANES)])[:, 1]})


def single_plane_oof(preds: pd.DataFrame, plane: str, task: str) -> pd.DataFrame:
    """OOF probabilities of one plane's CNN alone (e.g. sagittal ACL = the KneeMRI model)."""
    sel = preds[(preds["plane"] == plane) & (preds["task"] == task) & (preds["subset"] == "oof")]
    return sel[["case_id", "fold", "task", "label", "prob"]].reset_index(drop=True)


def summarise(oof: pd.DataFrame) -> dict:
    """§5 metrics on pooled OOF predictions of one task, plus the per-fold AUC spread."""
    y, p = oof["label"].to_numpy(), oof["prob"].to_numpy()
    point, lo, hi = bootstrap_ci(y, p)
    per_fold = [auc(g["label"], g["prob"]) for _, g in oof.groupby("fold")]
    t_hs = threshold_for_sensitivity(y, p, 0.90)
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "auc": point, "auc_ci": [lo, hi],
        "auc_per_fold_mean": float(np.mean(per_fold)), "auc_per_fold_sd": float(np.std(per_fold, ddof=1)),
        "at_0.5": sens_spec_at(y, p, 0.5),
        "t_hs": t_hs, "at_t_hs": sens_spec_at(y, p, t_hs),
        "ece_uniform10": ece(y, p), "ece_quantile10": ece(y, p, strategy="quantile"), "brier": brier(y, p),
    }


def cv_report(preds: pd.DataFrame) -> dict:
    folds = sorted(preds["fold"].unique())
    report: dict = {"folds": [int(f) for f in folds]}
    for variant, fit_on in (("B0", "train"), ("B0-oof", "tune")):
        report[variant] = {}
        for task in TASKS:
            oof = pd.concat([fuse_fold(preds, f, task, fit_on) for f in folds], ignore_index=True)
            report[variant][task] = summarise(oof)
    report["per_plane"] = {f"{plane}/{task}": summarise(single_plane_oof(preds, plane, task))
                           for plane in PLANES for task in TASKS}
    return report
