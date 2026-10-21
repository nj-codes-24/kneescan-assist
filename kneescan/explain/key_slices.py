"""
key_slices.py: which slices did the Reader rely on for each finding?

What it does
    `key_slices(output, k)` reads the attention weights the Reader already computed, and returns for every
    (finding, plane) the k slices with the highest weight, with those weights. The weights of one plane sum to 1,
    so 0.40 means "40% of this plane's summary came from this slice".

Why attention weights and not something fancier?
    They are part of the model's actual computation: the plane summary *is* the weighted sum of slices.
    So they are a faithful answer to "which slices", at no extra cost. They don't say *where* in the slice
    the model looked. That is Grad-CAM's job (gradcam.py).

Alternatives
    For models without attention (e.g. B0's max pooling), you'd use occlusion: hide one slice at a time and
    measure how much the prediction drops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kneescan import TASKS
from kneescan.models.reader import ReaderOutput


@dataclass(frozen=True)
class KeySlice:
    task: str
    plane: str
    slice_index: int
    weight: float     # share of the plane summary that came from this slice (0..1)


def key_slices(output: ReaderOutput, k: int = 3) -> list[KeySlice]:
    """Top-k slices per (finding, plane), highest weight first."""
    result = []
    for plane, weights in output.attention.items():             # weights: (n_tasks, S)
        top = torch.topk(weights.detach().float(), k=min(k, weights.shape[1]), dim=1)
        for t, task in enumerate(TASKS):
            for weight, index in zip(top.values[t].tolist(), top.indices[t].tolist()):
                result.append(KeySlice(task, plane, int(index), float(weight)))
    return result


def slice_importance(output: ReaderOutput, task: str) -> dict[str, list[float]]:
    """Every slice's attention weight for one finding, per plane (for a bar chart under the viewer)."""
    t = TASKS.index(task)
    return {plane: weights[t].detach().float().tolist() for plane, weights in output.attention.items()}
