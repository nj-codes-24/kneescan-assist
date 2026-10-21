"""
make_splits.py: create and save the CV folds (EVAL_PROTOCOL R3, R4). Run ONCE, commit the output.

Writes
    splits/mrnet_train_5fold.csv   case_id, fold
    splits/mrnet_inner_tune.csv    outer_fold, case_id
Neither file contains labels.

Refuses to overwrite existing split files: changing folds after training started would
break the protocol. Use --force only before any training run, and log why.

Usage
    .venv\\Scripts\\python.exe scripts\\make_splits.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kneescan.data.mrnet import find_mrnet_root, read_labels
from kneescan.data.splits import label_combo, make_cv_folds, make_inner_tune
from kneescan.train.run_logger import append_lab_log

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "mrnet")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "splits")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    folds_path, inner_path = args.out / "mrnet_train_5fold.csv", args.out / "mrnet_inner_tune.csv"
    if (folds_path.exists() or inner_path.exists()) and not args.force:
        raise SystemExit("split files already exist; they are frozen (EVAL_PROTOCOL R3).")

    labels = read_labels(find_mrnet_root(args.data), "train")
    folds = make_cv_folds(labels)
    inner = make_inner_tune(labels, folds)
    args.out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(folds_path, index=False)
    inner.to_csv(inner_path, index=False)

    # Show balance per fold (printed only, never saved: labels stay out of the repo)
    combo = label_combo(labels).rename("combo")
    table = folds.join(combo).groupby(["fold", "combo"]).size().unstack(fill_value=0)
    print(table)
    print(f"inner tune sizes: {inner.groupby('outer_fold').size().to_dict()}")
    append_lab_log("splits_created", n_train=len(labels), files=[str(folds_path), str(inner_path)])


if __name__ == "__main__":
    main()
