"""
make_hybrid.py: build the out-of-fold predictions of a hybrid, the mean logit of a Reader run and the B0 baseline (idea I2).

What it does
    Loads both runs' pooled OOF predictions (kneescan/eval/oof.py), averages their logits per exam and finding with equal weights
    (kneescan/eval/plane_ensemble.average_logits), and writes runs/<out>/fold*/preds.csv. scripts/score_cv.py can then score it like any
    run, which gives the hybrid its own frozen threshold and calibrator for the final test.

Why equal weights?
    Nothing is fitted, so nothing can overfit or leak. The two families are about equally accurate (docs/experiments.md, per-plane diagnosis).

Usage
    .venv\\Scripts\\python.exe scripts\\make_hybrid.py --reader runs\\reader_r2_seeds --baseline runs\\b0 --out runs\\hybrid_r2s_b0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from kneescan.eval.oof import load_oof
from kneescan.eval.plane_ensemble import average_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parts = [load_oof(run).assign(subset="oof") for run in (args.reader, args.baseline)]
    combined = average_logits(parts)
    for fold, part in combined.groupby("fold"):
        (args.out / f"fold{fold}").mkdir(parents=True, exist_ok=True)
        part[["fold", "subset", "case_id", "task", "label", "prob"]].to_csv(args.out / f"fold{fold}" / "preds.csv", index=False)
    (args.out / "hybrid.yaml").write_text(yaml.safe_dump({"rule": "mean of logits, equal weights",
                                                          "reader": str(args.reader), "baseline": str(args.baseline)}), encoding="utf-8")
    print(f"{args.out}: {combined['case_id'].nunique()} exams")


if __name__ == "__main__":
    main()
