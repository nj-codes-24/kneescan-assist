"""
transforms.py: turn a uint8 MRI series into network input, plus training augmentation.

What it does
    * `random_affine(vol, cfg, gen)`: one random rotation, shift and flip, applied
      identically to every slice of a series (EVAL_PROTOCOL §3: rotation +-25 degrees,
      shift +-25 px, horizontal flip with p = 0.5).
    * `to_network_input(vol)`: uint8 (S, H, W) -> float (S, 3, H, W) scaled to [0, 1] and
      normalised with ImageNet mean/std. The grey slice is repeated into 3 channels so
      ImageNet-pretrained weights apply.

Why on the GPU, on whole volumes?
    The same transform has to hit every slice (the knee doesn't move between slices), and
    doing it on the GPU keeps the data loader light.

Alternatives
    The Reader uses 2.5D input instead: the neighbouring slices go into the 3 channels
    (DESIGN.md §1).
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2.functional as TF

from kneescan.models.baseline_mrnet import IMAGENET_MEAN, IMAGENET_STD


def random_affine(vol: torch.Tensor, rotate_deg: float, shift_px: float, flip_p: float,
                  gen: torch.Generator) -> torch.Tensor:
    """Same random rotation/shift/flip for every slice. vol: (S, H, W), any dtype."""
    def uniform(a: float) -> float:
        return float((torch.rand(1, generator=gen) * 2 - 1) * a)

    angle, dx, dy = uniform(rotate_deg), uniform(shift_px), uniform(shift_px)
    flip = bool(torch.rand(1, generator=gen) < flip_p)
    out = TF.affine(vol.unsqueeze(1), angle=angle, translate=[round(dx), round(dy)], scale=1.0,
                    shear=[0.0, 0.0], interpolation=TF.InterpolationMode.BILINEAR).squeeze(1)
    return out.flip(-1) if flip else out


def random_gamma(vol: torch.Tensor, strength: float, gen: torch.Generator) -> torch.Tensor:
    """Random contrast change for a whole series: (v/255) ** gamma * 255, gamma = exp(U(-strength, strength)).

    Different scanners and settings render the same tissue brighter or darker. This teaches the model not to rely on exact
    intensities. gamma < 1 brightens dark tissue, gamma > 1 darkens it.
    """
    gamma = float(torch.exp((torch.rand(1, generator=gen) * 2 - 1) * strength))
    return (vol.float().clamp(0, 255) / 255.0) ** gamma * 255.0


def random_slice_dropout(x: torch.Tensor, p: float, gen: torch.Generator, keep_at_least: int = 3) -> torch.Tensor:
    """Hide each slice with probability p (training only), so the model can't depend on any single slice."""
    keep = torch.rand(x.shape[0], generator=gen) >= p
    if int(keep.sum()) < keep_at_least:
        keep[torch.randperm(x.shape[0], generator=gen)[:keep_at_least]] = True
    return x[keep.to(x.device)]


def to_25d_input(vol: torch.Tensor) -> torch.Tensor:
    """uint8/float (S, H, W) in 0..255 -> float (S, 3, H, W): channels = slices [i-1, i, i+1].

    The first and last slices reuse themselves as the missing neighbour (edge replication),
    so every slice gets a full 3-channel input. Normalised with ImageNet mean/std.
    """
    x = vol.float().div(255.0)
    previous = torch.cat([x[:1], x[:-1]], dim=0)
    following = torch.cat([x[1:], x[-1:]], dim=0)
    stacked = torch.stack([previous, x, following], dim=1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (stacked - mean) / std


def to_network_input(vol: torch.Tensor) -> torch.Tensor:
    """uint8/float (S, H, W) in 0..255 -> float (S, 3, H, W), ImageNet-normalised."""
    x = vol.float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
