"""
gallery.py: the failure-case gallery. The exams the model got most confidently wrong, with what it looked at.

What it does
    * `pick_failures(task_oof, threshold, k)`: the k worst false negatives (tears the model scored lowest) and the k worst
      false positives (healthy knees it scored highest), among out-of-fold predictions misclassified at `threshold`.
    * `case_figure(models, volumes, task, title)`: one row per plane, showing the key slice, the same slice with its
      Grad-CAM heatmap, and the attention weight of every slice (which slices the model leaned on).
    Each case is explained by the fold model(s) that did NOT train on it, the same ones that made the OOF prediction.
    `models` maps each plane to the model that reads it: the same model for every plane in an ordinary run, or that plane's
    specialist in a combined run (R2). The Reader reads each plane independently, so both give that plane's own attention.

Why look at failures?
    Averages hide *how* a model fails. A gallery shows whether errors are borderline cases, label noise, unusual
    scans, or the model looking at the wrong anatomy, and that is what a doctor will ask about first.

Alternatives
    With a radiologist available, each failure would get an expert second read instead of our short notes.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")   # draw to files only; no window (works on a server and inside scripts)
import matplotlib.pyplot as plt  # noqa: E402

from kneescan import TASKS  # noqa: E402
from kneescan.app.predictor import overlay  # noqa: E402
from kneescan.data.transforms import to_25d_input  # noqa: E402
from kneescan.explain.gradcam import gradcam  # noqa: E402
from kneescan.models.reader import KneeReader  # noqa: E402

HEATMAP_ALPHA = 0.45


def pick_failures(task_oof: pd.DataFrame, threshold: float, k: int) -> pd.DataFrame:
    """Rows of `task_oof` (case_id, fold, label, prob) for the k worst FNs and FPs, with a 'kind' column."""
    misses = task_oof[(task_oof["label"] == 1) & (task_oof["prob"] < threshold)].nsmallest(k, "prob")
    alarms = task_oof[(task_oof["label"] == 0) & (task_oof["prob"] >= threshold)].nlargest(k, "prob")
    return pd.concat([misses.assign(kind="false negative"), alarms.assign(kind="false positive")], ignore_index=True)


@torch.no_grad()
def attention_for(models: dict[str, KneeReader], planes: dict[str, torch.Tensor], task: str) -> dict[str, np.ndarray]:
    """plane -> (S,) attention weights of one finding, each from the model that reads that plane."""
    weights = {}
    for plane, slices in planes.items():
        models[plane].eval()
        out = models[plane]({plane: slices})
        weights[plane] = out.attention[plane][TASKS.index(task)].float().cpu().numpy()
    return weights


def case_figure(models: dict[str, KneeReader], volumes: dict[str, np.ndarray], task: str, title: str) -> plt.Figure:
    """Key slice, heatmap and slice-attention bar chart for each plane of one exam."""
    device = next(iter(models.values())).head_bias.device
    planes = {p: to_25d_input(torch.from_numpy(v).to(device)) for p, v in volumes.items() if p in models}
    attention = attention_for(models, planes, task)
    fig, axes = plt.subplots(len(planes), 3, figsize=(11, 3.4 * len(planes)), squeeze=False)
    for row, (plane, weights) in enumerate(attention.items()):
        index = int(np.argmax(weights))
        heat = gradcam(models[plane], planes[plane], plane, task, [index])[0]
        axes[row, 0].imshow(volumes[plane][index], cmap="gray")
        axes[row, 0].set_title(f"{plane}: key slice {index + 1}/{len(weights)}")
        axes[row, 1].imshow(overlay(volumes[plane][index], heat, HEATMAP_ALPHA))
        axes[row, 1].set_title("Grad-CAM (where it looked)")
        axes[row, 2].bar(np.arange(1, len(weights) + 1), weights, color="tab:blue")
        axes[row, 2].set_title("attention per slice")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
