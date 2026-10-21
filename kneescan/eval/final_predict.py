"""
final_predict.py: predictions for the ONE-TIME final test (EVAL_PROTOCOL §8). B0 and the chosen Reader on MRNet `valid`
(120 exams) and on KneeMRI (917 sagittal volumes, ACL only).

What it does
    * B0 on MRNet: per fold, the 9 CNNs (3 planes × 3 findings, best checkpoints) give plane probabilities. That fold's
      logistic-regression fusion (fitted on its training predictions, as in the paper) combines them. The 5 folds' fused
      probabilities are averaged (EVAL_PROTOCOL §3).
    * B0 on KneeMRI: the mean over folds of the sagittal ACL CNN alone, with no fusion (§3, R11).
    * Reader on both: the app's Predictor (5 fold answers, each the mean logit of the members that see the planes present),
      averaged over folds. On KneeMRI only the sagittal specialist can vote.
    All probabilities are raw. Calibration is applied later with the frozen Platt parameters (kneescan/eval/final_metrics.py).

Why reuse the app's Predictor for the Reader?
    The number we report must come from exactly the code a user would run.

Alternatives
    For a larger test set you'd batch exams on the GPU. 120 + 917 exams take minutes one at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kneescan import PLANES, TASKS
from kneescan.app.predictor import Predictor
from kneescan.data.kneemri import KneeMRIDataset
from kneescan.data.mrnet import load_volume, read_labels
from kneescan.data.transforms import to_network_input
from kneescan.eval.fusion import fit_fusion, load_b0_predictions
from kneescan.models.baseline_mrnet import MRNetBaseline

N_FOLDS = 5


def load_b0_job(job_dir: Path, device: torch.device) -> dict[str, MRNetBaseline]:
    """The three task CNNs (best checkpoints) of one (fold, plane) B0 job."""
    models = {}
    for task in TASKS:
        model = MRNetBaseline(pretrained=False)
        model.load_state_dict(torch.load(job_dir / f"best_{task}.pt", map_location=device))
        models[task] = model.to(device).eval()
    return models


@torch.no_grad()
def b0_probs(models: dict[str, MRNetBaseline], volume: np.ndarray, device: torch.device) -> dict[str, float]:
    x = to_network_input(torch.from_numpy(volume).to(device))
    return {task: float(torch.sigmoid(models[task](x))) for task in models}


def b0_predict_mrnet(b0_dir: Path, root: Path, split: str, device: torch.device) -> pd.DataFrame:
    """One row per (case_id, task): label and B0's fold-averaged fused probability."""
    labels = read_labels(root, split)
    train_preds = load_b0_predictions(b0_dir)
    fused = np.zeros((len(labels), len(TASKS)))
    for fold in range(N_FOLDS):
        jobs = {p: load_b0_job(b0_dir / f"fold{fold}" / p, device) for p in PLANES}
        fusers = {t: fit_fusion(train_preds, fold, t, "train") for t in TASKS}
        for i, case_id in enumerate(labels["case_id"]):
            plane_probs = {p: b0_probs(jobs[p], np.array(load_volume(root, split, p, case_id)), device) for p in PLANES}
            for k, task in enumerate(TASKS):
                features = pd.DataFrame([[plane_probs[p][task] for p in PLANES]], columns=list(PLANES))
                fused[i, k] += fusers[task].predict_proba(features)[0, 1] / N_FOLDS
    return long_table(labels, fused)


def reader_predict_mrnet(run_dir: Path, root: Path, split: str, device: str | None = None) -> pd.DataFrame:
    labels = read_labels(root, split)
    predictor = Predictor(run_dir, device=device)
    probs = np.zeros((len(labels), len(TASKS)))
    for i, case_id in enumerate(labels["case_id"]):
        volumes = {p: np.array(load_volume(root, split, p, case_id)) for p in PLANES}
        probs[i] = predictor.fold_probabilities(volumes).mean(axis=0)
    return long_table(labels, probs)


def long_table(labels: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    rows = [{"case_id": cid, "task": task, "label": int(labels.iloc[i][task]), "prob": float(probs[i, k])}
            for i, cid in enumerate(labels["case_id"]) for k, task in enumerate(TASKS)]
    return pd.DataFrame(rows)


def b0_predict_kneemri(b0_dir: Path, dataset: KneeMRIDataset, device: torch.device) -> pd.DataFrame:
    """ACL probability per KneeMRI volume: mean over folds of B0's sagittal ACL CNN (no fusion; §3)."""
    acl_models = [load_b0_job(b0_dir / f"fold{f}" / "sagittal", device)["acl"] for f in range(N_FOLDS)]
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        prob = np.mean([b0_probs({"acl": m}, item["volume"].numpy(), device)["acl"] for m in acl_models])
        rows.append(kneemri_row(item, prob))
    return pd.DataFrame(rows)


def reader_predict_kneemri(run_dir: Path, dataset: KneeMRIDataset, device: str | None = None) -> pd.DataFrame:
    """ACL probability per KneeMRI volume from the Reader, sagittal plane only."""
    predictor = Predictor(run_dir, device=device)
    acl = TASKS.index("acl")
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        prob = float(predictor.fold_probabilities({"sagittal": item["volume"].numpy()}).mean(axis=0)[acl])
        rows.append(kneemri_row(item, prob))
    return pd.DataFrame(rows)


def kneemri_row(item: dict, prob: float) -> dict:
    return {"filename": item["filename"], "exam_id": item["exam_id"], "acl_tear": int(item["acl_tear"]),
            "acl_complete": int(item["acl_complete"]), "prob": prob}
