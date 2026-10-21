"""
final_metrics.py: the numbers of the ONE-TIME final test (EVAL_PROTOCOL §5, §6, §8), computed with FROZEN thresholds and calibrators.

What it does
    * `task_report(y, p, threshold, platt, clusters)`: AUC with a bootstrap 95% CI, sensitivity/specificity (Wilson CIs) at 0.5 and
      at the frozen T_hs, and ECE + Brier both raw and after the frozen Platt calibration. With `clusters` (KneeMRI exam IDs), the
      bootstrap resamples whole exams (R9b).
    * `paired_report(y, p_new, p_old, clusters)`: AUC(new) − AUC(old) with a paired bootstrap CI, on identical cases.
    Nothing here is fitted on the test data: thresholds and calibrators come from cross-validation and are frozen first (R2, R14).

Why a separate module from cv_report?
    CV scoring *fits* calibrators and thresholds on out-of-fold data. The final test must only *apply* frozen ones. Keeping the two
    apart makes it impossible to refit on test data by accident.

Alternatives
    With several test sites you'd also report per-site results. Here there's one internal (MRNet) and one external (KneeMRI) test.
"""

from __future__ import annotations

import numpy as np

from kneescan.eval.calibration import apply_platt
from kneescan.eval.metrics import auc, bootstrap_ci, brier, ece, paired_bootstrap_delta, sens_spec_at


def task_report(y: np.ndarray, p: np.ndarray, threshold: float, platt: tuple[float, float],
                clusters: np.ndarray | None = None) -> dict:
    """All §5 metrics for one finding on one test set."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    point, lo, hi = bootstrap_ci(y, p, auc, clusters=clusters)
    calibrated = apply_platt(p, *platt)
    return {"n": int(len(y)), "positives": int(y.sum()), "auc": point, "auc_ci": [lo, hi],
            "at_0.5": sens_spec_at(y, p, 0.5), "t_hs": threshold, "at_t_hs": sens_spec_at(y, p, threshold),
            "ece_raw": ece(y, p), "ece_calibrated": ece(y, calibrated),
            "brier_raw": brier(y, p), "brier_calibrated": brier(y, calibrated)}


def paired_report(y: np.ndarray, p_new: np.ndarray, p_old: np.ndarray, clusters: np.ndarray | None = None) -> dict:
    """AUC(new) − AUC(old) on the same cases, with a paired bootstrap CI."""
    delta, lo, hi = paired_bootstrap_delta(np.asarray(y), np.asarray(p_new), np.asarray(p_old), auc, clusters=clusters)
    return {"delta_auc": delta, "delta_ci": [lo, hi], "new_reliably_better": lo > 0, "new_reliably_worse": hi < 0}
