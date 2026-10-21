"""
score_cv.py: score one finished CV run and, optionally, compare it with a previous run. This is how
every row of docs/experiments.md gets its numbers.

What it does
    1. Loads the run's pooled OOF predictions (B0 or Reader; kneescan/eval/oof.py).
    2. Macro AUC with a bootstrap 95% CI, plus every EVAL_PROTOCOL §5 metric per finding.
    3. With --compare: the paired macro-AUC gain over the other run on the same exams, and
       "kept" = the gain's CI lies entirely above 0.
    4. Writes results/cv/<run>.json and .md, and appends one row to experiments/runs.csv.
    It never touches MRNet `valid` or KneeMRI (the test sets).

Usage
    .venv\\Scripts\\python.exe scripts\\score_cv.py --runs runs\\b0 --change "B0 baseline"
    .venv\\Scripts\\python.exe scripts\\score_cv.py --runs runs\\reader_r18 --compare runs\\b0 --change "Reader R1"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

from kneescan.eval.cv_report import compare, score, to_markdown
from kneescan.train.run_logger import append_lab_log

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = REPO_ROOT / "experiments" / "runs.csv"


def append_runs_csv(report: dict, change: str, notes: str) -> None:
    c = report.get("comparison", {})
    row = {"run_id": report["run"], "date": dt.date.today().isoformat(),
           "config_path": f"runs/{report['run']}/config.yaml", "change": change,
           "macro_auc_oof": round(report["macro_auc"], 4),
           "ci_low": round(report["macro_auc_ci"][0], 4), "ci_high": round(report["macro_auc_ci"][1], 4),
           "gain_vs_prev": round(c["gain"], 4) if c else "", "gain_ci_low": round(c["gain_ci"][0], 4) if c else "",
           "gain_ci_high": round(c["gain_ci"][1], 4) if c else "", "kept": c.get("kept", "baseline"), "notes": notes}
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(row)).writerow(row)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # the report has arrows (→); the Windows console defaults to cp1252
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None, help="previous kept run to compare against")
    parser.add_argument("--change", required=True, help="one line: what this run changed")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    report = score(args.runs)
    if args.compare is not None:
        report["comparison"] = compare(args.runs, args.compare)
    out = REPO_ROOT / "results" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.runs.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(report)
    (out / f"{args.runs.name}.md").write_text(md, encoding="utf-8")
    append_runs_csv(report, args.change, args.notes)
    append_lab_log("cv_scored", run=args.runs.name, macro_auc=round(report["macro_auc"], 4),
                   comparison=report.get("comparison"))
    print(md)


if __name__ == "__main__":
    main()
