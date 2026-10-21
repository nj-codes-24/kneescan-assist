"""
oof.py: collect the pooled out-of-fold (OOF) predictions of any finished CV run into one table.

What it does
    `load_oof(runs_dir)` returns one row per (exam, finding): case_id, fold, task, label, prob.
      * Reader runs: runs/<name>/fold*/preds.csv, rows with subset == "oof".
      * B0 runs:     runs/b0/fold*/<plane>/preds.csv, fused per task with the paper's logistic regression
                     (kneescan/eval/fusion.py), exactly as scripts/eval_b0_cv.py scores it.

Why one format?
    Every model, however it's built, is then scored and compared by the same code.

Alternatives
    If runs produced thousands of files, a small database (SQLite) would beat globbing CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kneescan import TASKS
from kneescan.eval.fusion import fuse_fold, load_b0_predictions

COLUMNS = ["case_id", "fold", "task", "label", "prob"]


def is_b0_run(runs_dir: Path) -> bool:
    """B0 runs have one sub-folder per plane inside each fold folder."""
    return any(Path(runs_dir).glob("fold*/*/preds.csv"))


def load_reader_oof(runs_dir: Path) -> pd.DataFrame:
    files = sorted(Path(runs_dir).glob("fold*/preds.csv"))
    if not files:
        raise FileNotFoundError(f"no fold*/preds.csv under {runs_dir} (is the run finished?)")
    preds = pd.concat([pd.read_csv(f, dtype={"case_id": str}) for f in files], ignore_index=True)
    return preds[preds["subset"] == "oof"][COLUMNS].reset_index(drop=True)


def load_b0_oof(runs_dir: Path) -> pd.DataFrame:
    preds = load_b0_predictions(runs_dir)
    folds = sorted(preds["fold"].unique())
    parts = [fuse_fold(preds, f, task, fit_on="train") for f in folds for task in TASKS]
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def load_oof(runs_dir: str | Path) -> pd.DataFrame:
    runs_dir = Path(runs_dir)
    return load_b0_oof(runs_dir) if is_b0_run(runs_dir) else load_reader_oof(runs_dir)
