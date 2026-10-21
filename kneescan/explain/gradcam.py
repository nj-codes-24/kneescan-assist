"""
gradcam.py: Grad-CAM heatmaps, i.e. *where* inside a slice the Reader looked for a finding.

What it does
    `gradcam(model, slices, plane, task, slice_indices)`:
      1. runs one plane through the Reader, keeping the backbone's last feature map (S, D, h, w)
      2. takes the gradient of that plane's logit for `task` with respect to the feature map
      3. weights each of the D feature channels by its average gradient ("how much does this channel push the
         logit up?"), sums the channels, and keeps only the positive part (ReLU)
      4. upsamples the h × w map (8 × 8 for ResNet-18 at 256 px) to the slice size and scales each map to 0..1
    Only the requested slices are returned (usually the key slices), which keeps it fast enough for a CPU.
    `gradcam_per_task(model, slices, plane, key_slices)`: the same maps for several findings from ONE forward pass. The
    gradient only flows back through the small head, so the backbone runs once, without recording gradients, instead of
    once per finding with them (faster and far less memory on a CPU server, same numbers).

    `randomisation_similarity(...)`: the Adebayo et al. (2018) sanity check (EVAL_PROTOCOL R13). The maps of a trained
    model are compared with those of an untrained one. If they look alike (Spearman >= 0.5), the heatmap shows
    image edges, not what the model learned, and the app must say so.

Why Grad-CAM?
    It's the standard, well-understood method for CNNs, needs no retraining, and is one backward pass.

Alternatives
    Grad-CAM is coarse (8 × 8 blocks here). Occlusion maps or Grad-CAM++ are sharper, but slower or less standard.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from kneescan import TASKS
from kneescan.models.reader import KneeReader


def gradcam(model: KneeReader, slices: torch.Tensor, plane: str, task: str,
            slice_indices: list[int]) -> np.ndarray:
    """(len(slice_indices), H, W) heatmaps in 0..1 for one plane's 2.5D input `slices` (S, 3, H, W)."""
    return gradcam_per_task(model, slices, plane, {task: slice_indices})[task]


def gradcam_per_task(model: KneeReader, slices: torch.Tensor, plane: str,
                     slice_indices: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """{task: (len(indices), H, W) heatmaps in 0..1} for every task in `slice_indices`, from one forward pass."""
    model.eval()                                                   # no dropout, BatchNorm in inference mode
    maps = {}
    with torch.no_grad():
        _, feature_map = model.slice_features(slices)              # the backbone: no gradient bookkeeping needed
    feature_map = feature_map.detach().requires_grad_(True)
    with torch.enable_grad():
        logits, _ = model.head_from_map(plane, feature_map)        # only the small head records gradients
        for i, (task, indices) in enumerate(slice_indices.items()):
            keep_graph = i < len(slice_indices) - 1                # reuse the forward pass for the next task
            grads = torch.autograd.grad(logits[TASKS.index(task)], feature_map, retain_graph=keep_graph)[0]
            maps[task] = _cam(grads, feature_map.detach(), indices, slices.shape[-2:])
    return maps


def _cam(grads: torch.Tensor, feature_map: torch.Tensor, slice_indices: list[int], size: torch.Size) -> np.ndarray:
    channel_weights = grads.mean(dim=(2, 3), keepdim=True)         # (S, D, 1, 1): importance of each channel
    cam = F.relu((channel_weights * feature_map).sum(dim=1))       # (S, h, w): positive evidence only
    cam = cam[slice_indices].unsqueeze(1).float()
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False).squeeze(1)
    return normalise_maps(cam.cpu().numpy())


def normalise_maps(maps: np.ndarray) -> np.ndarray:
    """Scale each map to 0..1 on its own; an all-zero map stays zero (the model saw no positive evidence)."""
    out = np.zeros_like(maps, dtype=np.float32)
    for i, m in enumerate(maps):
        peak = m.max()
        if peak > 0:
            out[i] = m / peak
    return out


def randomisation_similarity(trained: KneeReader, untrained: KneeReader, slices: torch.Tensor,
                             plane: str, task: str, slice_indices: list[int]) -> float:
    """Mean Spearman correlation between trained and untrained heatmaps (R13: must be < 0.5)."""
    maps_trained = gradcam(trained, slices, plane, task, slice_indices)
    maps_random = gradcam(untrained, slices, plane, task, slice_indices)
    scores = []
    for a, b in zip(maps_trained, maps_random):
        rho = spearmanr(a.ravel(), b.ravel()).statistic
        scores.append(0.0 if np.isnan(rho) else float(rho))   # a flat map has no ranking: count it as unrelated
    return float(np.mean(scores))
