"""
baseline_mrnet.py: "B0", our reproduction of the MRNet model (Bien et al., PLOS Med 2018).

What it does
    For ONE plane and ONE task:
        slices (S, 3, H, W) -> AlexNet conv layers (ImageNet weights) -> (S, 256, h, w)
        -> global average pool per slice -> (S, 256)
        -> max over slices                -> (256,)
        -> linear layer                   -> 1 logit (sigmoid gives the probability)
    The paper trains 9 of these (3 planes x 3 tasks) and combines the 3 planes of each task
    with a logistic regression (see kneescan/eval/fusion.py).

Why reproduce it exactly?
    It is the published "beat this" number. Running it on *our* folds gives a fair,
    same-data comparison (EVAL_PROTOCOL §3).

Alternatives
    For the real product model, see the Reader (DESIGN.md §1). It uses a stronger backbone,
    attention pooling (explainable key slices), one model for all tasks, and robustness to
    missing planes.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MRNetBaseline(nn.Module):
    """One plane, one task. Input: (S, 3, H, W) float, ImageNet-normalised. Output: scalar logit."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = alexnet(weights=weights).features  # conv layers only, 256 output channels
        self.classifier = nn.Linear(256, 1)

    def slice_features(self, x: torch.Tensor) -> torch.Tensor:
        """(S, 3, H, W) -> (S, 256): one feature vector per slice."""
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.slice_features(x).max(dim=0).values  # strongest evidence across slices
        return self.classifier(pooled).squeeze(-1)
