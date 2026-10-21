"""
test_calibration.py: Platt scaling fixes a known distortion, keeps the ranking, and cross-fitting never uses
the fold it scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kneescan.eval.calibration import apply_platt, crossfit_platt, fit_platt, logit
from kneescan.eval.metrics import auc, ece


def distorted_data(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """True probabilities t, but the 'model' reports sigmoid(2 * logit(t) + 1): over-confident and shifted up."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < true_p).astype(int)
    reported = 1 / (1 + np.exp(-(2 * logit(true_p) + 1)))
    return y, reported


def test_platt_recovers_the_distortion_and_lowers_ece() -> None:
    y, p = distorted_data()
    a, b = fit_platt(y, p)
    assert abs(a - 0.5) < 0.1 and abs(b + 0.5) < 0.15     # inverse of logit*2 + 1 is (logit - 1) / 2
    calibrated = apply_platt(p, a, b)
    assert ece(y, calibrated) < ece(y, p) / 2
    assert abs(auc(y, calibrated) - auc(y, p)) < 1e-12        # monotonic: ranking (and AUC) unchanged


def test_crossfit_uses_only_other_folds() -> None:
    y, p = distorted_data(n=1000)
    oof = pd.DataFrame({"fold": np.arange(1000) % 5, "label": y, "prob": p})
    calibrated = crossfit_platt(oof)
    a, b = fit_platt(oof.loc[oof.fold != 0, "label"], oof.loc[oof.fold != 0, "prob"])
    assert np.allclose(calibrated[oof.fold == 0], apply_platt(p[oof.fold == 0], a, b))
