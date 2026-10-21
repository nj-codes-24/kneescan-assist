"""
metrics.py: every number the Honest Judge reports (EVAL_PROTOCOL.md §5).

What it does
    * `auc`: area under the ROC curve.
    * `bootstrap_ci`: percentile bootstrap CI for any metric. 2,000 resamples by default,
      seed 20261018. Optional `clusters` resamples whole groups (KneeMRI exams, R9b).
    * `paired_bootstrap_delta`: CI for metric(model A) - metric(model B) on the same cases.
    * `wilson_ci`: CI for a proportion (sensitivity, specificity, pointing rate).
    * `sens_spec_at`: sensitivity and specificity at a fixed threshold, with Wilson CIs.
    * `threshold_for_sensitivity`: the highest threshold reaching a target sensitivity.
      Chosen on OOF predictions only, never on a test set.
    * `ece`, `brier`, `reliability_bins`: calibration.

Why the bootstrap and not DeLong for AUC CIs?
    One method works for every metric (AUC, ECE, ΔAUC) and handles clustered data. DeLong is
    faster but AUC-only.

Alternatives
    For very small positive counts (< 10), use exact or Bayesian intervals. Bootstrap CIs get
    unreliable there.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261018

Metric = Callable[[np.ndarray, np.ndarray], float]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def _resample_indices(n: int, rng: np.random.Generator, clusters: np.ndarray | None) -> np.ndarray:
    if clusters is None:
        return rng.integers(0, n, size=n)
    unique = np.unique(clusters)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    return np.concatenate([members[c] for c in chosen])


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: Metric = auc, n_boot: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05,
                 clusters: np.ndarray | None = None) -> tuple[float, float, float]:
    """(point estimate, lower, upper). Resamples that contain only one class are skipped."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _resample_indices(len(y), rng, clusters)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(y, p), float(lo), float(hi)


def paired_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: Metric = auc,
                           n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED, alpha: float = 0.05,
                           clusters: np.ndarray | None = None) -> tuple[float, float, float]:
    """metric(A) - metric(B), with the same resampled cases for both models."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = _resample_indices(len(y), rng, clusters)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(y, p_a) - metric(y, p_b), float(lo), float(hi)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """(proportion, lower, upper) using the Wilson score interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return phat, float(centre - half), float(centre + half)


def sens_spec_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, tuple[float, float, float]]:
    """Sensitivity and specificity (each with a Wilson CI) when predicting positive if p >= threshold."""
    y, pred = np.asarray(y).astype(bool), np.asarray(p) >= threshold
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    tn, fp = int((~pred & ~y).sum()), int((pred & ~y).sum())
    return {"sensitivity": wilson_ci(tp, tp + fn), "specificity": wilson_ci(tn, tn + fp)}


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.90) -> float:
    """Highest threshold t with sensitivity(p >= t) >= target. Use on OOF predictions only."""
    y, p = np.asarray(y).astype(bool), np.asarray(p)
    pos = np.sort(p[y])[::-1]                   # positive scores, high to low
    k = int(np.ceil(target * len(pos)))         # need at least k positives above threshold
    return float(pos[k - 1])


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10,
                     strategy: str = "uniform") -> list[dict[str, float]]:
    """Per bin: mean predicted probability, observed positive rate, count."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = which == b
        if m.any():
            bins.append({"mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean()), "count": int(m.sum())})
    return bins


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error: count-weighted |observed - predicted| over bins."""
    bins = reliability_bins(y, p, n_bins, strategy)
    n = sum(b["count"] for b in bins)
    return float(sum(b["count"] * abs(b["frac_pos"] - b["mean_pred"]) for b in bins) / n)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))
