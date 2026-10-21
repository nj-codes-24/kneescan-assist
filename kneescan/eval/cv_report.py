"""
cv_report.py: the full cross-validation report of one run, and the paired comparison of two runs. Used by
scripts/score_cv.py (real runs) and scripts/smoke_test.py (tiny synthetic runs), so both are scored by identical code.

What it does
    * `score(runs_dir)`: macro AUC with a 95% CI, every EVAL_PROTOCOL §5 metric per finding, and the Platt calibrator
      with its cross-fitted effect.
    * `compare(runs_dir, other_dir)`: paired macro-AUC gain over another run on the same exams; kept = CI above 0.
    * `to_markdown(report)`: the readable version written to results/cv/<run>.md.

Why one module?
    The numbers on the resume must come from one place. If two scripts computed them separately, they could drift apart.

Alternatives
    With many runs to compare at once, an experiment tracker (MLflow) would show these tables in a dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kneescan import TASKS
from kneescan.eval.calibration import crossfit_platt, fit_platt
from kneescan.eval.fusion import summarise
from kneescan.eval.macro import macro_auc_ci, paired_macro_gain, wide_oof
from kneescan.eval.metrics import brier, ece
from kneescan.eval.oof import load_oof


def compare(runs: Path, other: Path) -> dict:
    """Paired macro-AUC gain of `runs` over `other`, on the exams both have predictions for."""
    labels, new, ids_new = wide_oof(load_oof(runs))
    labels_old, old, ids_old = wide_oof(load_oof(other))
    if ids_new != ids_old or not np.array_equal(labels, labels_old):
        raise ValueError("the two runs don't cover the same exams with the same labels")
    gain, lo, hi = paired_macro_gain(labels, new, old)
    return {"against": other.name, "gain": gain, "gain_ci": [lo, hi], "kept": lo > 0}


def calibration_report(task_oof: pd.DataFrame) -> dict:
    """Platt calibrator for the app (fitted on all OOF), and its honest effect (cross-fitted by fold)."""
    a, b = fit_platt(task_oof["label"], task_oof["prob"])
    y, crossfit = task_oof["label"].to_numpy(), crossfit_platt(task_oof)
    return {"platt_a": a, "platt_b": b, "ece_calibrated_crossfit": ece(y, crossfit),
            "brier_calibrated_crossfit": brier(y, crossfit)}


def score(runs: Path) -> dict:
    """Everything EVAL_PROTOCOL §5 asks for, on the run's pooled out-of-fold predictions."""
    oof = load_oof(runs)
    labels, probs, ids = wide_oof(oof)
    point, lo, hi = macro_auc_ci(labels, probs)
    per_task = {task: summarise(oof[oof["task"] == task]) for task in TASKS}
    calibration = {task: calibration_report(oof[oof["task"] == task].reset_index(drop=True)) for task in TASKS}
    return {"run": runs.name, "n_exams": len(ids), "macro_auc": point, "macro_auc_ci": [lo, hi],
            "per_task": per_task, "calibration": calibration}


def to_markdown(report: dict) -> str:
    """Readable summary: macro AUC, gain vs the compared run, and one row per finding."""
    lines = [f"# {report['run']}: pooled OOF on MRNet train ({report['n_exams']} exams, 5 folds)", "",
             f"**Macro AUC {report['macro_auc']:.3f}** (95% CI {report['macro_auc_ci'][0]:.3f}–{report['macro_auc_ci'][1]:.3f})", ""]
    if "comparison" in report:
        c = report["comparison"]
        lines += [f"Gain vs {c['against']}: {c['gain']:+.3f} (95% CI {c['gain_ci'][0]:+.3f} to {c['gain_ci'][1]:+.3f}) "
                  f"-> kept: {'yes' if c['kept'] else 'no'}", ""]
    lines += ["| Finding | AUC (95% CI) | Sens / Spec @0.5 | Sens / Spec @T_hs | ECE raw → calibrated | Brier raw → calibrated |",
              "|---|---|---|---|---|---|"]
    for task, s in report["per_task"].items():
        c = report["calibration"][task]
        lines.append(f"| {task} | {s['auc']:.3f} ({s['auc_ci'][0]:.3f}–{s['auc_ci'][1]:.3f}) | "
                     f"{s['at_0.5']['sensitivity'][0]:.2f} / {s['at_0.5']['specificity'][0]:.2f} | "
                     f"{s['at_t_hs']['sensitivity'][0]:.2f} / {s['at_t_hs']['specificity'][0]:.2f} | "
                     f"{s['ece_uniform10']:.3f} → {c['ece_calibrated_crossfit']:.3f} | "
                     f"{s['brier']:.3f} → {c['brier_calibrated_crossfit']:.3f} |")
    lines += ["", "T_hs = the threshold reaching 90% sensitivity on these OOF predictions. Calibrated values are cross-fitted "
              "(each fold's calibrator was fitted on the other folds)."]
    return "\n".join(lines) + "\n"
