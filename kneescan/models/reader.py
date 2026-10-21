"""
reader.py: "The Reader", the main model of KneeScan Assist (DESIGN.md §1).

What it does
    For one exam with up to three series (sagittal, coronal, axial):
        each slice, stacked with its two neighbours (2.5D)  -> (S, 3, H, W)
        -> one shared ImageNet CNN (ResNet-18 by default)   -> (S, D) slice features
        -> + a learned "which plane is this" vector
        -> gated attention, one head per finding            -> (3, S) slice weights per plane
        -> weighted sum of slice features                   -> (3, D) per plane
        -> one linear score per (plane, finding)            -> plane logits (3 planes x 3 findings)
        -> learned weighted average over the planes present -> 3 logits (abnormal, ACL, meniscus)

    Whiteboard version: "read every slice, decide which slices matter for each finding,
    summarise each plane, then let the planes vote."

Why this design?
    * Attention instead of B0's max pooling: the weights *are* the key-slice explanation, and
      every slice contributes some gradient, which trains more smoothly.
    * One shared backbone for all planes and findings: about 9x fewer weights than B0's 9 CNNs,
      which matters on 1,130 exams.
    * Plane dropout + averaging only over the planes present: the model still works when a
      series is missing (KneeMRI has sagittal only).

Alternatives
    A 3D CNN if slices were thin and the dataset large. Feature concatenation across planes
    (instead of voting) if every exam always had all three planes.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

from kneescan import PLANES, TASKS


class GatedAttention(nn.Module):
    """Gated attention pooling (Ilse et al. 2018): one softmax over slices for each finding."""

    def __init__(self, in_dim: int, att_dim: int, n_heads: int) -> None:
        super().__init__()
        self.content = nn.Linear(in_dim, att_dim)   # "what is in this slice"
        self.gate = nn.Linear(in_dim, att_dim)      # "how much of it to let through"
        self.score = nn.Linear(att_dim, n_heads)    # one relevance score per finding

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(S, D) slice features -> (S, n_heads) weights; each column sums to 1 over the slices."""
        hidden = torch.tanh(self.content(features)) * torch.sigmoid(self.gate(features))
        return torch.softmax(self.score(hidden), dim=0)


def max_pool_slices(features: torch.Tensor, n_tasks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """B0-style max pooling: each feature keeps its strongest value across slices.

    Returns the (n_tasks, D) summary (the same for every finding) and (S, n_tasks) slice weights for explanation:
    the share of features whose maximum came from that slice ("how much of the summary this slice supplied").
    """
    best, best_slice = features.max(dim=0)                                      # (D,), (D,)
    share = torch.bincount(best_slice, minlength=features.shape[0]).float() / features.shape[1]
    return best.unsqueeze(0).expand(n_tasks, -1), share.unsqueeze(1).expand(-1, n_tasks)


@dataclass
class ReaderOutput:
    logits: torch.Tensor                      # (n_tasks,) final logits, in TASKS order
    plane_logits: dict[str, torch.Tensor]     # plane -> (n_tasks,), only for planes used
    attention: dict[str, torch.Tensor]        # plane -> (n_tasks, S) slice weights, for key slices
    planes_used: list[str]


class KneeReader(nn.Module):
    """Input: dict plane -> (S, 3, H, W) normalised 2.5D slices (or None if missing)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, att_dim: int = 128,
                 dropout: float = 0.25, plane_dropout: float = 0.2, freeze_batchnorm: bool = False,
                 pooling: str = "attention") -> None:
        super().__init__()
        if pooling not in ("attention", "max"):
            raise ValueError("pooling must be 'attention' or 'max'")
        self.freeze_batchnorm = freeze_batchnorm
        self.pooling = pooling
        # global_pool="" keeps the spatial feature map, which Grad-CAM needs later
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        dim = self.backbone.num_features
        n_planes, n_tasks = len(PLANES), len(TASKS)
        self.plane_embedding = nn.Parameter(torch.zeros(n_planes, dim))   # starts neutral, learns plane identity
        # Pooled CNN features are tiny (mean ~0.25), so the head could barely move the logits; LayerNorm puts every
        # feature on unit scale. Without it the Reader failed the overfit check (scripts/sanity_overfit.py, D-007).
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.ModuleList([GatedAttention(dim, att_dim, n_tasks) for _ in PLANES])
        self.head_weight = nn.Parameter(torch.randn(n_planes, n_tasks, dim) * 0.01)
        self.head_bias = nn.Parameter(torch.zeros(n_planes, n_tasks))
        self.fusion_logits = nn.Parameter(torch.zeros(n_tasks, n_planes))  # equal plane votes at the start
        self.dropout = nn.Dropout(dropout)
        self.plane_dropout = plane_dropout
        self.inference_batch: int | None = None   # app only: run the backbone on this many slices at a time (less memory)

    def train(self, mode: bool = True) -> "KneeReader":
        """Like nn.Module.train, but can keep the backbone's BatchNorm layers in inference mode.

        A training "batch" here is the slices of ONE knee. BatchNorm would normalise every exam by its own statistics,
        while evaluation uses long-run averages, so the model learns on one input distribution and is tested on another
        (D-010). Frozen BatchNorm uses the pretrained ImageNet statistics in both, and its scale/shift still learn.
        """
        super().train(mode)
        if mode and self.freeze_batchnorm:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def slice_features(self, slices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(S, 3, H, W) -> pooled (S, D) features and the (S, D, h, w) feature map.

        Without gradients and with `inference_batch` set, the slices go through the backbone a few at a time. Every slice
        is processed on its own (BatchNorm is in inference mode), so the result is the same; the peak memory isn't.
        On a 2-core server: 3.5 -> 2.25 GiB for a 3-plane study with 8 slices at a time (decision D-026).
        """
        n = self.inference_batch
        if n and not torch.is_grad_enabled() and slices.shape[0] > n:
            feature_map = torch.cat([self.backbone(slices[i:i + n]) for i in range(0, slices.shape[0], n)])
        else:
            feature_map = self.backbone(slices)
        return feature_map.mean(dim=(2, 3)), feature_map

    def choose_planes(self, available: list[str]) -> list[str]:
        """During training drop each plane with probability p, but always keep at least one."""
        if not self.training or self.plane_dropout <= 0 or len(available) == 1:
            return available
        kept = [p for p in available if float(torch.rand(1)) >= self.plane_dropout]
        if kept:
            return kept
        return [available[int(torch.randint(len(available), (1,)))]]

    def read_plane(self, plane: str, slices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One series -> (n_tasks,) plane logits and (n_tasks, S) attention weights."""
        logits, weights, _ = self.read_plane_with_map(plane, slices)
        return logits, weights

    def read_plane_with_map(self, plane: str, slices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like read_plane, but also returns the backbone's (S, D, h, w) feature map (Grad-CAM needs it)."""
        _, feature_map = self.slice_features(slices)
        logits, weights = self.head_from_map(plane, feature_map)
        return logits, weights, feature_map

    def head_from_map(self, plane: str, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Everything after the backbone: (S, D, h, w) feature map -> (n_tasks,) logits and (n_tasks, S) attention.
        Grad-CAM calls this on its own, so the backbone can run without recording gradients."""
        p = PLANES.index(plane)
        pooled = feature_map.mean(dim=(2, 3))
        features = self.norm(pooled.float() + self.plane_embedding[p])
        if self.pooling == "max":
            summary, weights = max_pool_slices(features, len(TASKS))
        else:
            weights = self.attention[p](features)                   # (S, T)
            summary = weights.t() @ features                        # (T, D): one summary per finding
        summary = self.dropout(summary)
        logits = (summary * self.head_weight[p]).sum(dim=-1) + self.head_bias[p]
        return logits, weights.t()

    def fuse(self, plane_logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted average of plane logits; the weights are renormalised over the planes present."""
        idx = [PLANES.index(p) for p in plane_logits]
        votes = torch.softmax(self.fusion_logits[:, idx], dim=1)     # (T, planes present)
        stacked = torch.stack(list(plane_logits.values()), dim=1)    # (T, planes present)
        return (votes * stacked).sum(dim=1)

    def forward(self, planes: dict[str, torch.Tensor | None]) -> ReaderOutput:
        available = [p for p in PLANES if planes.get(p) is not None]
        if not available:
            raise ValueError("an exam needs at least one plane")
        used = self.choose_planes(available)
        plane_logits, attention = {}, {}
        for plane in used:
            plane_logits[plane], attention[plane] = self.read_plane(plane, planes[plane])
        return ReaderOutput(self.fuse(plane_logits), plane_logits, attention, used)
