"""
plane_ensemble.py: R2, "three specialists, one answer". Combines single-plane Reader runs (sagittal, coronal, axial)
into one run by averaging their logits for each exam and finding.

What it does
    * `combine_runs(component_dirs, out_dir)`: reads every component's fold*/preds.csv (train / tune / OOF probabilities),
      averages the components' logits per (fold, subset, exam, finding), turns the mean back into a probability, and writes
      out_dir/fold*/preds.csv plus out_dir/components.yaml (which runs it was built from).
      The result looks exactly like any other run folder, so scripts/score_cv.py, the comparison with B0 and the
      failure gallery work on it unchanged.
    * `read_components(run_dir)`: the component list, or None for an ordinary single-model run (used by the app).

Why average logits, with equal weights?
    It needs no fitting, so there is nothing to overfit and nothing that could leak information from the evaluation
    folds. Averaging logits (not probabilities) treats "very sure" and "very unsure" symmetrically.

Alternatives
    With more data, a logistic regression fitted on held-out predictions (B0's stacking) could learn that one plane is
    better for a given finding. With 1,130 exams the equal vote is the honest, simple choice (D-014).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kneescan.eval.calibration import logit

KEYS = ["fold", "subset", "case_id", "task"]
COMPONENTS_FILE = "components.yaml"


def _read_component(run_dir: Path) -> pd.DataFrame:
    files = sorted(Path(run_dir).glob("fold*/preds.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"no fold*/preds.csv under {run_dir}")
    return pd.concat([pd.read_csv(f, dtype={"case_id": str}) for f in files], ignore_index=True)


def average_logits(components: list[pd.DataFrame]) -> pd.DataFrame:
    """Rows keyed by (fold, subset, case_id, task) with the label and the probability from the mean logit."""
    stacked = pd.concat([c.assign(logit=logit(c["prob"])) for c in components], ignore_index=True)
    counts = stacked.groupby(KEYS).size()
    if (counts != len(components)).any():
        raise ValueError("components don't cover the same exams, folds and subsets")
    combined = stacked.groupby(KEYS).agg(label=("label", "first"), logit=("logit", "mean")).reset_index()
    combined["prob"] = 1.0 / (1.0 + np.exp(-combined["logit"]))
    return combined.drop(columns="logit")


def combine_runs(component_dirs: list[Path], out_dir: Path) -> pd.DataFrame:
    """Write out_dir/fold*/preds.csv and out_dir/components.yaml. Returns the combined table."""
    combined = average_logits([_read_component(d) for d in component_dirs])
    for fold, part in combined.groupby("fold"):
        (out_dir / f"fold{fold}").mkdir(parents=True, exist_ok=True)
        part[["fold", "subset", "case_id", "task", "label", "prob"]].to_csv(out_dir / f"fold{fold}" / "preds.csv", index=False)
    listing = {"rule": "mean of component logits, equal weights", "components": [str(Path(d)) for d in component_dirs]}
    (out_dir / COMPONENTS_FILE).write_text(yaml.safe_dump(listing), encoding="utf-8")
    return combined


def read_components(run_dir: Path) -> list[Path] | None:
    """Component run folders of a combined run (resolved next to it), or None for an ordinary run."""
    path = Path(run_dir) / COMPONENTS_FILE
    if not path.exists():
        return None
    names = yaml.safe_load(path.read_text(encoding="utf-8"))["components"]
    return [Path(run_dir).parent / Path(name.replace("\\", "/")).name for name in names]   # Windows-written paths work on Linux too


def _oof_logit_table(component_dirs: list[Path]) -> pd.DataFrame:
    """One row per (fold, case_id, task): the label and each component's OOF logit in its own column."""
    wide = None
    for i, d in enumerate(component_dirs):
        part = _read_component(d)
        part = part[part["subset"] == "oof"].assign(**{f"logit{i}": lambda x: logit(x["prob"])})
        part = part[["fold", "case_id", "task", "label", f"logit{i}"]]
        wide = part if wide is None else wide.merge(part.drop(columns="label"), on=["fold", "case_id", "task"], validate="1:1")
    return wide


def stack_crossfit(component_dirs: list[Path]) -> pd.DataFrame:
    """Weighted vote per finding, learned without leakage: each fold's weights are fitted on the OTHER folds' OOF logits.

    Returns the OOF table (fold, subset='oof', case_id, task, label, prob) of the stacked model.
    """
    from sklearn.linear_model import LogisticRegression
    table = _oof_logit_table(component_dirs)
    features = [f"logit{i}" for i in range(len(component_dirs))]
    parts = []
    for (task, fold), here in table.groupby(["task", "fold"]):
        others = table[(table["task"] == task) & (table["fold"] != fold)]
        stacker = LogisticRegression(C=1.0).fit(others[features], others["label"])
        parts.append(here.assign(subset="oof", prob=stacker.predict_proba(here[features])[:, 1]))
    return pd.concat(parts, ignore_index=True)[["fold", "subset", "case_id", "task", "label", "prob"]]


def combine_task_runs(task_components: dict[str, list[Path]], out_dir: Path) -> pd.DataFrame:
    """Idea I5: for each finding, average only the runs trained for that finding (e.g. its 3 plane specialists).

    `task_components` maps each finding to its component run folders. The result is written like `combine_runs`.
    """
    parts = []
    for task, dirs in task_components.items():
        components = [frame[frame["task"] == task] for frame in (_read_component(d) for d in dirs)]
        parts.append(average_logits(components))
    combined = pd.concat(parts, ignore_index=True)
    for fold, part in combined.groupby("fold"):
        (out_dir / f"fold{fold}").mkdir(parents=True, exist_ok=True)
        part[["fold", "subset", "case_id", "task", "label", "prob"]].to_csv(out_dir / f"fold{fold}" / "preds.csv", index=False)
    listing = {"rule": "per finding: mean of that finding's component logits",
               "task_components": {t: [str(Path(d)) for d in dirs] for t, dirs in task_components.items()}}
    (out_dir / COMPONENTS_FILE).write_text(yaml.safe_dump(listing), encoding="utf-8")
    return combined
