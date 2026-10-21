"""
key_slice_position.py: where in a series does the model's key slice sit, and is that different when it's wrong?

What it does
    For every out-of-fold exam of a run, each plane's key slice (highest attention for a finding) is read from the fold
    model that never saw the exam. Its relative position is d = index / (slices - 1), so 0 and 1 are the first and last slice.
    A key slice is an "edge slice" if d < 0.15 or d > 0.85. The share of edge key slices is compared between exams the model
    got right and exams it got wrong (at the run's T_hs threshold), per finding and plane.

Why
    The failure gallery suggested the model often relies on edge slices when it fails (docs/failure_cases.md). This checks
    that across all 1,130 exams instead of 12, before changing the model because of it (D-015).

Usage
    .venv\\Scripts\\python.exe scripts\\key_slice_position.py --runs runs\\reader_r2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explain_cases import exam_volumes, fold_models_by_plane  # noqa: E402

from kneescan import PLANES, TASKS  # noqa: E402
from kneescan.app.predictor import load_cv_summary  # noqa: E402
from kneescan.data.mrnet import find_mrnet_root  # noqa: E402
from kneescan.data.transforms import to_25d_input  # noqa: E402
from kneescan.eval.oof import load_reader_oof  # noqa: E402

EDGE = 0.15


@torch.no_grad()
def key_positions(models: dict, volumes: dict[str, np.ndarray], device: torch.device) -> dict[tuple[str, str], float]:
    """(plane, task) -> relative position of the key slice, each plane read by its own model."""
    positions = {}
    for plane in PLANES:
        slices = to_25d_input(torch.from_numpy(volumes[plane]).to(device))
        weights = models[plane]({plane: slices}).attention[plane].float().cpu().numpy()   # (tasks, S)
        for t, task in enumerate(TASKS):
            positions[(plane, task)] = int(np.argmax(weights[t])) / max(weights.shape[1] - 1, 1)
    return positions


def collect(run_dir: Path, root: Path, device: torch.device) -> pd.DataFrame:
    oof = load_reader_oof(run_dir)
    summary = load_cv_summary(run_dir)
    rows = []
    for fold in sorted(oof["fold"].unique()):
        models = fold_models_by_plane(run_dir, int(fold), device)
        fold_oof = oof[oof["fold"] == fold]
        for case_id, exam in fold_oof.groupby("case_id"):
            positions = key_positions(models, exam_volumes(root, case_id), device)
            for _, row in exam.iterrows():
                correct = (row["prob"] >= summary[row["task"]]["threshold"]) == bool(row["label"])
                for plane in PLANES:
                    rows.append({"task": row["task"], "plane": plane, "correct": bool(correct),
                                 "position": positions[(plane, row["task"])]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = collect(args.runs, find_mrnet_root(REPO_ROOT / "data" / "mrnet"), device)
    table["edge"] = (table["position"] < EDGE) | (table["position"] > 1 - EDGE)
    summary = table.groupby(["task", "plane", "correct"])["edge"].agg(["mean", "size"]).round(3)
    overall = table.groupby("correct")["edge"].agg(["mean", "size"]).round(3)
    print(summary.to_string(), "\n\nall findings and planes:\n", overall.to_string())
    out = REPO_ROOT / "results" / "cv" / f"{args.runs.name}_key_slice_position.json"
    out.write_text(json.dumps({"edge_definition": f"d < {EDGE} or d > {1 - EDGE}",
                               "by_task_plane": {f"{t}/{p}/{'correct' if c else 'wrong'}": {"edge_share": v["mean"], "n": int(v["size"])}
                                                 for (t, p, c), v in summary.iterrows()},
                               "overall": {("correct" if c else "wrong"): {"edge_share": v["mean"], "n": int(v["size"])}
                                           for c, v in overall.iterrows()}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
