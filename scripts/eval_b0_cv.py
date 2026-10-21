"""
eval_b0_cv.py: score the finished B0 cross-validation. This is the "beat this" number.

Reads runs/b0/fold*/<plane>/preds.csv (OOF predictions only; MRNet valid is never touched)
and writes results/cv/b0_cv.json plus a readable results/cv/b0_cv.md.

Usage
    .venv\\Scripts\\python.exe scripts\\eval_b0_cv.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kneescan import TASKS
from kneescan.eval.fusion import cv_report, load_b0_predictions
from kneescan.train.run_logger import append_lab_log

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = {"abnormal": 0.937, "acl": 0.965, "meniscus": 0.847}


def to_markdown(report: dict) -> str:
    lines = ["# B0 cross-validation (pooled OOF, MRNet train, 5 folds)", "",
             "| Task | B0 AUC (95% CI) | B0-oof AUC | Sens / Spec @0.5 | ECE | Paper (hidden test) |",
             "|---|---|---|---|---|---|"]
    for task in TASKS:
        b0, b0o = report["B0"][task], report["B0-oof"][task]
        sens, spec = b0["at_0.5"]["sensitivity"][0], b0["at_0.5"]["specificity"][0]
        lines.append(f"| {task} | {b0['auc']:.3f} ({b0['auc_ci'][0]:.3f}-{b0['auc_ci'][1]:.3f}) | "
                     f"{b0o['auc']:.3f} | {sens:.2f} / {spec:.2f} | {b0['ece_uniform10']:.3f} | {PAPER[task]} |")
    lines += ["", "Per-plane CNN AUCs (no fusion):", ""]
    lines += [f"- {k}: {v['auc']:.3f} ({v['auc_ci'][0]:.3f}-{v['auc_ci'][1]:.3f})" for k, v in report["per_plane"].items()]
    lines += ["", "CV scores are not test scores, and the paper's numbers come from a different test set "
              "(EVAL_PROTOCOL §3)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "b0")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "cv")
    args = parser.parse_args()

    report = cv_report(load_b0_predictions(args.runs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.runs.name}_cv.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(report)
    (args.out / f"{args.runs.name}_cv.md").write_text(md, encoding="utf-8")
    print(md)
    append_lab_log("b0_cv_scored", runs=str(args.runs),
                   auc={t: round(report["B0"][t]["auc"], 4) for t in TASKS})


if __name__ == "__main__":
    main()
