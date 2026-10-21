"""
splits.py: the cross-validation folds of EVAL_PROTOCOL.md §2 (rules R3 and R4).

What it does
    * `make_cv_folds(labels)`: 5 stratified folds over MRNet `train`, stratified on the
      label combination "abnormal-acl-meniscus" (e.g. "110"), with seed 42.
    * `make_inner_tune(labels, folds)`: for each outer fold f, holds out 10% of the
      *training* part (folds != f) for epoch selection, with seed 42 + f.
    * `split_for(fold, folds, inner)`: gives the three case-id lists a run needs:
      train / tune / oof.

Why stratify on the combination?
    The three labels are correlated (a tear implies abnormal). Stratifying on all three
    together keeps every fold's mix of normal / ACL / meniscus / both close to the whole set.

Alternatives
    With patient IDs you would use StratifiedGroupKFold so that one patient never lands in two
    folds. MRNet doesn't release them (EVAL_PROTOCOL R5).

The saved CSVs contain case IDs and fold numbers only, never labels (licence: no redistribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from kneescan import TASKS

N_FOLDS = 5
SEED = 42
INNER_TUNE_FRACTION = 0.10


def label_combo(labels: pd.DataFrame) -> pd.Series:
    """'abnormal acl meniscus' as a 3-character string, e.g. '110'."""
    return labels[list(TASKS)].astype(int).astype(str).agg("".join, axis=1)


def make_cv_folds(labels: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with columns case_id, fold (0..n_folds-1)."""
    labels = labels.sort_values("case_id").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=labels.index)
    for f, (_, held_out) in enumerate(skf.split(labels, label_combo(labels))):
        fold.iloc[held_out] = f
    return pd.DataFrame({"case_id": labels["case_id"], "fold": fold.astype(int)})


def make_inner_tune(labels: pd.DataFrame, folds: pd.DataFrame, fraction: float = INNER_TUNE_FRACTION,
                    seed: int = SEED) -> pd.DataFrame:
    """Return columns outer_fold, case_id: the inner tuning cases for each outer fold."""
    merged = labels.merge(folds, on="case_id", validate="one_to_one")
    rows = []
    for f in sorted(folds["fold"].unique()):
        train_part = merged[merged["fold"] != f].sort_values("case_id")
        _, tune = train_test_split(train_part, test_size=fraction, random_state=seed + int(f),
                                   stratify=label_combo(train_part))
        rows += [{"outer_fold": int(f), "case_id": c} for c in sorted(tune["case_id"])]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FoldSplit:
    train: list[str]   # used for gradient steps
    tune: list[str]    # inner tuning set: epoch selection only
    oof: list[str]     # held-out fold: untouched until OOF prediction


def split_for(fold: int, folds: pd.DataFrame, inner: pd.DataFrame) -> FoldSplit:
    tune = set(inner.loc[inner["outer_fold"] == fold, "case_id"])
    oof = folds.loc[folds["fold"] == fold, "case_id"]
    rest = folds.loc[folds["fold"] != fold, "case_id"]
    return FoldSplit(train=sorted(set(rest) - tune), tune=sorted(tune), oof=sorted(oof))
