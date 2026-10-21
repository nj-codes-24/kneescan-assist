"""
train_reader_cv.py: run a Reader configuration over all 5 CV folds (one model per fold).

What it does
    Runs the folds in order. A finished fold is skipped and a half-done fold resumes from its last.pt,
    so you can stop at any time (Ctrl-C) and start the same command again later. Each run's folder gets
    a copy of its config (config.yaml), so results can always be traced to their exact settings.

Exit codes
    0 = all requested folds done, 3 = paused because of --stop-at (re-run later to continue).

Usage (from the repo root)
    .venv\\Scripts\\python.exe scripts\\train_reader_cv.py --config configs\\reader_r18.yaml
    .venv\\Scripts\\python.exe scripts\\train_reader_cv.py --config configs\\reader_r18.yaml --stop-at 08:00
    .venv\\Scripts\\python.exe scripts\\train_reader_cv.py --config configs\\reader_r18.yaml --folds 0 --epochs 1   # quick check
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

import pandas as pd

from kneescan.data.mrnet import find_mrnet_root, read_labels
from kneescan.data.splits import split_for
from kneescan.train.config import load_reader_config
from kneescan.train.reader_trainer import ReaderFoldJob
from kneescan.train.run_logger import append_lab_log

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "mrnet")
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "splits")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stop-at", default=None, help="HH:MM: pause cleanly before this time")
    parser.add_argument("--epochs", type=int, default=None, help="override (smoke tests only; uses a separate runs dir)")
    args = parser.parse_args()

    cfg = load_reader_config(args.config)
    runs = REPO_ROOT / "runs" / cfg.name
    if args.epochs is not None:  # never mix smoke-test runs with real ones
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
        runs = runs.parent / f"{cfg.name}_smoke_{args.epochs}ep"
    runs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, runs / "config.yaml")

    root = find_mrnet_root(args.data)
    labels = read_labels(root, "train")
    folds = pd.read_csv(args.splits / "mrnet_train_5fold.csv", dtype={"case_id": str})
    inner = pd.read_csv(args.splits / "mrnet_inner_tune.csv", dtype={"case_id": str})

    append_lab_log("reader_cv_start", config=cfg.name, folds=args.folds, stop_at=args.stop_at, runs=str(runs))
    for fold in args.folds:
        split = split_for(fold, folds, inner)
        print(f"=== {cfg.name} fold {fold}: train {len(split.train)}, tune {len(split.tune)}, oof {len(split.oof)}",
              flush=True)
        job = ReaderFoldJob(cfg, root, labels, split, fold, runs / f"fold{fold}")
        status = job.run(resume=True, stop_at=args.stop_at)
        print(f"    -> {status} (last epoch {job.last_epoch_seconds:.0f} s, best {job.best})", flush=True)
        if status == "paused":
            return 3
    append_lab_log("reader_cv_done", config=cfg.name, folds=args.folds, runs=str(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
