"""
pretrained.py: load RadImageNet weights (Mei et al., Radiology: AI 2022) into the timm ResNet-50 the Reader uses (idea I4, D-017).

What it does
    The authors' PyTorch file (`ResNet50.pt`, from github.com/BMEII-AI/RadImageNet; licence: docs/DATA.md §5b)
    is a plain state dict of torchvision's ResNet-50 *without* its classifier, wrapped in an nn.Sequential named "backbone":
        backbone.0 = conv1, backbone.1 = bn1, backbone.4–7 = layer1–layer4   (2 = ReLU and 3 = max-pool have no weights)
    `radimagenet_to_timm` renames those keys to timm's names. The DenseNet-121 file (`DenseNet121.pt`) wraps torchvision's
    `features` block as "backbone.0", so there the only change is the prefix "backbone.0." → "features.". `load_backbone_weights` loads them with `weights_only=True`, which
    refuses any code hidden in the pickle, and `strict=True`, so a missing or extra weight is an error rather than a silent gap.

Why RadImageNet?
    It was pretrained on 1.35 million CT, MRI and ultrasound images (including musculoskeletal MRI) instead of photographs, so its
    early features may suit MRI better than ImageNet's.

Alternatives
    With an MRI dataset of your own, self-supervised pretraining on it would match the domain even better, at much higher cost.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

SEQUENTIAL_TO_TIMM = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def radimagenet_to_timm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """'backbone.4.0.conv1.weight' -> 'layer1.0.conv1.weight', and so on."""
    renamed = {}
    for key, value in state.items():
        prefix, index, rest = key.split(".", 2) if key.count(".") >= 2 else (*key.split(".", 1), "")
        if prefix != "backbone" or index not in SEQUENTIAL_TO_TIMM:
            raise ValueError(f"unexpected key in RadImageNet weights: {key}")
        renamed[SEQUENTIAL_TO_TIMM[index] + ("." + rest if rest else "")] = value
    return renamed


def densenet_to_timm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """'backbone.0.denseblock1....' -> 'features.denseblock1....'."""
    if not all(k.startswith("backbone.0.") for k in state):
        raise ValueError("unexpected key layout in the RadImageNet DenseNet weights")
    return {"features." + k[len("backbone.0."):]: v for k, v in state.items()}


def load_backbone_weights(backbone: nn.Module, path: str | Path) -> None:
    """Replace a freshly built timm backbone's weights with RadImageNet's (all of them, or fail)."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    is_densenet = any(".denseblock" in k for k in state)
    backbone.load_state_dict(densenet_to_timm(state) if is_densenet else radimagenet_to_timm(state), strict=True)
