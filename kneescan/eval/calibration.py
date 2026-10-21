"""
calibration.py: Platt scaling, so that "70%" from the model means "about 70 of 100 such knees have the finding".

What it does
    * `fit_platt(y, p)`: fits p_calibrated = sigmoid(a * logit(p) + b) by logistic regression. Two numbers per finding.
    * `apply_platt(p, a, b)`: uses them.
    * `crossfit_platt(oof)`: the honest check. For each fold, fit on the *other* folds' OOF predictions and apply to this
      fold. The calibrated ECE it gives was never measured on data the calibrator saw.
    The app uses one calibrator per finding, fitted on all pooled OOF predictions (EVAL_PROTOCOL R7: never on test data).

Why do we need it?
    Training with pos_weight (to stop the model ignoring rare tears) deliberately pushes probabilities up for the
    rare class, so raw outputs are not frequencies. Platt scaling is monotonic, so ranking and AUC are unchanged.
    Only the meaning of the number improves.

Alternatives
    Isotonic regression is more flexible, but needs more data than ~1,100 exams to avoid a staircase-shaped mapping.
    Temperature scaling (only `a`) is even simpler, but can't fix a shifted base rate, which pos_weight causes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6   # keeps logit() finite for probabilities of exactly 0 or 1


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(a, b) for p_cal = sigmoid(a * logit(p) + b). C is large: almost no regularisation, only 2 parameters."""
    model = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), np.asarray(y))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))


def crossfit_platt(oof: pd.DataFrame) -> np.ndarray:
    """Calibrated OOF probabilities of one finding. Each fold's calibrator is fitted on the other folds only."""
    calibrated = np.zeros(len(oof))
    for fold in sorted(oof["fold"].unique()):
        here = (oof["fold"] == fold).to_numpy()
        a, b = fit_platt(oof.loc[~here, "label"], oof.loc[~here, "prob"])
        calibrated[here] = apply_platt(oof.loc[here, "prob"].to_numpy(), a, b)
    return calibrated
