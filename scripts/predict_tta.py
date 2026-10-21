"""
predict_tta.py: ladder step L4. Re-score a finished run's out-of-fold exams with test-time augmentation (TTA): each exam is
read as-is AND left-right mirrored, and the two logits are averaged.

What it does
    For every fold, it loads that fold's best.pt and predicts only its OOF exams, which that model never trained on, twice (original
    + horizontally flipped volume, the same flip used in training augmentation). It writes runs/<out>/fold*/preds.csv (subset "oof")
    and a copy of the config, so scripts/combine_planes.py and scripts/score_cv.py work on it unchanged.

Why mirror only?
    It's the one augmentation the model saw in training that maps a knee onto a plausible knee. Rotations and shifts change the
    framing too much to be a fair "second look". TTA is an "advanced trick" under the protocol: it stays only if its paired gain beats the CI.

Usage
    .venv\\Scripts\\python.exe scripts\\predict_tta.py --runs runs\\reader_sag_att --out runs\\reader_sag_att_tta
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explain_cases import fold_model  # noqa: E402

from kneescan import TASKS  # noqa: E402
from kneescan.data.mrnet import MRNetExamDataset, find_mrnet_root, read_labels  # noqa: E402
from kneescan.data.splits import split_for  # noqa: E402
from kneescan.data.transforms import to_25d_input  # noqa: E402
from kneescan.models.reader import KneeReader  # noqa: E402
from kneescan.train.config import load_reader_config  # noqa: E402


@torch.no_grad()
def flip_averaged_logits(model: KneeReader, volumes: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean of the logits for the exam as-is and mirrored left-right (the last image axis)."""
    original = {p: to_25d_input(v) for p, v in volumes.items()}
    mirrored = {p: to_25d_input(v.flip(-1)) for p, v in volumes.items()}
    return (model(original).logits.float() + model(mirrored).logits.float()) / 2


def predict_fold(run_dir: Path, fold: int, root: Path, labels: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    folds = pd.read_csv(REPO_ROOT / "splits" / "mrnet_train_5fold.csv", dtype={"case_id": str})
    inner = pd.read_csv(REPO_ROOT / "splits" / "mrnet_inner_tune.csv", dtype={"case_id": str})
    planes = load_reader_config(run_dir / "config.yaml").planes
    dataset = MRNetExamDataset(root, "train", labels, split_for(fold, folds, inner).oof, planes)
    model = fold_model(run_dir, fold, device)
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        volumes = {p: v.to(device).float() for p, v in item["volumes"].items()}
        probs = torch.sigmoid(flip_averaged_logits(model, volumes)).cpu().numpy()
        for k, task in enumerate(TASKS):
            rows.append({"fold": fold, "subset": "oof", "case_id": item["case_id"], "task": task,
                         "label": int(item["labels"][k]), "prob": float(probs[k])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_mrnet_root(REPO_ROOT / "data" / "mrnet")
    labels = read_labels(root, "train")
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.runs / "config.yaml", args.out / "config.yaml")
    for fold in range(5):
        (args.out / f"fold{fold}").mkdir(exist_ok=True)
        predict_fold(args.runs, fold, root, labels, device).to_csv(args.out / f"fold{fold}" / "preds.csv", index=False)
        print(f"fold {fold} done", flush=True)


if __name__ == "__main__":
    main()
