"""
sanity_overfit.py: can the Reader memorise a handful of exams? A wiring check before any long run.

What it does
    Trains a Reader config on 32 training exams, with no augmentation, dropout or plane dropout, and
    scores it on the *same* 32 exams after every epoch. A correctly wired model drives the loss towards 0
    and the AUC towards 1.0 within a few epochs. If it can't, there's a bug (wrong labels, a detached
    gradient, a learning rate that is far off), and a 6-hour CV run would only waste the GPU.

Why
    "Overfit one small batch first" is the cheapest bug detector in deep learning (Karpathy's recipe).
    It says nothing about generalisation, and its numbers are never reported as results.

Usage
    .venv\\Scripts\\python.exe scripts\\sanity_overfit.py --config configs\\reader_r18.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path

import pandas as pd

from kneescan.data.mrnet import find_mrnet_root, read_labels
from kneescan.data.splits import FoldSplit, split_for
from kneescan.train.config import load_reader_config
from kneescan.train.reader_trainer import ReaderFoldJob

REPO_ROOT = Path(__file__).resolve().parents[1]
N_EXAMS = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    cfg = dataclasses.replace(load_reader_config(args.config), epochs=args.epochs, accumulate_exams=1,
                              dropout=0.0, plane_dropout=0.0, rotate_deg=0.0, shift_px=0.0, flip_p=0.0)
    root = find_mrnet_root(REPO_ROOT / "data" / "mrnet")
    labels = read_labels(root, "train")
    folds = pd.read_csv(REPO_ROOT / "splits" / "mrnet_train_5fold.csv", dtype={"case_id": str})
    inner = pd.read_csv(REPO_ROOT / "splits" / "mrnet_inner_tune.csv", dtype={"case_id": str})
    few = split_for(0, folds, inner).train[:N_EXAMS]          # training exams of fold 0 only
    split = FoldSplit(train=few, tune=few, oof=few)           # score on the very exams it trains on

    run_dir = REPO_ROOT / "runs" / f"sanity_{cfg.name}"
    shutil.rmtree(run_dir, ignore_errors=True)                # a sanity check always starts fresh
    job = ReaderFoldJob(cfg, root, labels, split, 0, run_dir)
    job.run(resume=False)
    print(pd.read_csv(run_dir / "metrics.csv").pivot(index="epoch", columns="task", values="tune_auc").round(3))
    print(pd.read_csv(run_dir / "metrics.csv").pivot(index="epoch", columns="task", values="tune_loss").round(3))


if __name__ == "__main__":
    main()
