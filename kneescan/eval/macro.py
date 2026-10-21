"""
macro.py: the headline number. Macro AUC (the unweighted mean of the abnormal, ACL and meniscus AUCs)
with a bootstrap CI, and the paired gain of one model over another (decision D-003).

What it does
    * `wide_oof(preds)`: long table (one row per exam x finding) -> labels and probabilities as
      (n_exams, 3) arrays, rows sorted by case_id so two models line up exam by exam.
    * `macro_auc(labels, probs)`: mean of the 3 per-finding AUCs. Same definition as the Kaggle metric.
    * `macro_auc_ci(...)`: bootstrap 95% CI. Whole exams are resampled, so the 3 findings of an exam stay together.
    * `paired_macro_gain(...)`: macro AUC(new) - macro AUC(old) on the *same* resampled exams.
      If its CI excludes 0, the new model is reliably better: the "Kept?" rule in docs/experiments.md.

Why paired?
    Two models scored on the same exams share most of their luck (easy and hard exams).
    Pairing cancels that shared part, so the CI of the *difference* is much narrower than
    comparing two separate CIs. That makes it possible to detect small real gains.

Alternatives
    With only one finding, use `metrics.paired_bootstrap_delta` directly. It's the same idea.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kneescan import TASKS
from kneescan.eval.metrics import BOOTSTRAP_SEED, N_BOOTSTRAP, auc


def wide_oof(preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Long OOF table (case_id, task, label, prob) -> labels (n, 3), probs (n, 3), case_ids."""
    probs = preds.pivot(index="case_id", columns="task", values="prob")[list(TASKS)].sort_index()
    labels = preds.pivot(index="case_id", columns="task", values="label")[list(TASKS)].sort_index()
    if probs.isna().any().any():
        raise ValueError("some exams are missing a prediction for one of the findings")
    return labels.to_numpy(dtype=int), probs.to_numpy(dtype=float), list(probs.index)


def macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Unweighted mean of the per-finding AUCs; every finding counts 1/3 however rare it is."""
    return float(np.mean([auc(labels[:, k], probs[:, k]) for k in range(labels.shape[1])]))


def _has_both_classes(labels: np.ndarray) -> bool:
    """A resample is usable only if every finding has positives and negatives (else AUC is undefined)."""
    return all(len(np.unique(labels[:, k])) == 2 for k in range(labels.shape[1]))


def _percentiles(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def macro_auc_ci(labels: np.ndarray, probs: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(macro AUC, lower, upper): 95% percentile bootstrap over whole exams."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(labels), size=len(labels))
        if _has_both_classes(labels[rows]):  # skip the rare resample where a finding has no positives
            scores.append(macro_auc(labels[rows], probs[rows]))
    return (macro_auc(labels, probs), *_percentiles(scores))


def paired_macro_gain(labels: np.ndarray, probs_new: np.ndarray, probs_old: np.ndarray,
                      n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(gain, lower, upper) of macro AUC(new) - macro AUC(old), same resampled exams for both."""
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(labels), size=len(labels))
        if _has_both_classes(labels[rows]):
            gains.append(macro_auc(labels[rows], probs_new[rows]) - macro_auc(labels[rows], probs_old[rows]))
    point = macro_auc(labels, probs_new) - macro_auc(labels, probs_old)
    return (point, *_percentiles(gains))
