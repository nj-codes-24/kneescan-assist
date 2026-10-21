"""
train_b0_cv.py: run the whole B0 cross-validation (5 folds x 3 planes = 15 jobs, 45 CNNs).

What it does
    Runs the jobs one after another, in fold order (fold 0's 3 planes first, so a complete
    fold is available early for a sanity check). Finished jobs are skipped and a half-done
    job resumes from its last.pt, so you can stop it at any time (Ctrl-C) and start the same
    command again later.

Exit codes
    0 = all requested jobs done, 3 = paused because of --stop-at (re-run later to continue).

Usage (from the repo root)
    .venv\\Scripts\\python.exe scripts\\train_b0_cv.py                    # everything, 24/7 mode
    .venv\\Scripts\\python.exe scripts\\train_b0_cv.py --stop-at 08:00    # night mode
    .venv\\Scripts\\python.exe scripts\\train_b0_cv.py --folds 0 --planes sagittal --epochs 1   # quick check
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd

from kneescan import PLANES
from kneescan.data.mrnet import find_mrnet_root, read_labels
from kneescan.data.splits import split_for
from kneescan.train.b0_trainer import PlaneJob
from kneescan.train.config import load_b0_config
from kneescan.train.run_logger import append_lab_log

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "b0_alexnet.yaml")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "mrnet")
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "splits")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "b0")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--planes", nargs="+", default=list(PLANES), choices=PLANES)
    parser.add_argument("--stop-at", default=None, help="HH:MM: pause cleanly before this time")
    parser.add_argument("--epochs", type=int, default=None, help="override (smoke tests only; uses a separate runs dir)")
    args = parser.parse_args()

    cfg = load_b0_config(args.config)
    runs = args.runs
    if args.epochs is not None:  # never mix smoke-test runs with real ones
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
        runs = runs.parent / f"{runs.name}_smoke_{args.epochs}ep"

    root = find_mrnet_root(args.data)
    labels = read_labels(root, "train")
    folds = pd.read_csv(args.splits / "mrnet_train_5fold.csv", dtype={"case_id": str})
    inner = pd.read_csv(args.splits / "mrnet_inner_tune.csv", dtype={"case_id": str})

    append_lab_log("b0_cv_start", folds=args.folds, planes=args.planes, stop_at=args.stop_at, runs=str(runs))
    for fold in args.folds:
        split = split_for(fold, folds, inner)
        for plane in args.planes:
            run_dir = runs / f"fold{fold}" / plane
            print(f"=== fold {fold} / {plane}: train {len(split.train)}, tune {len(split.tune)}, oof {len(split.oof)}")
            job = PlaneJob(cfg, root, labels, split, fold, plane, run_dir)
            status = job.run(resume=True, stop_at=args.stop_at)
            print(f"    -> {status} (last epoch {job.last_epoch_seconds:.0f} s)")
            if status == "paused":
                return 3
    append_lab_log("b0_cv_done", folds=args.folds, planes=args.planes, runs=str(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
