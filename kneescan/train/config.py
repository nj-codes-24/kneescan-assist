"""
config.py: experiment settings as a typed, hashable object.

What it does
    Loads a YAML file (e.g. configs/b0_alexnet.yaml) into a `B0Config` dataclass. A YAML key
    that doesn't exist in the dataclass is an error, so typos fail loudly. `config_hash()`
    gives a short fingerprint that goes into every log line, which shows which runs used
    identical settings.

Why a dataclass, not a plain dict?
    Your editor can autocomplete the field names, the type hints document what each value
    means, and defaults live in exactly one place.

Alternatives
    Bigger projects often use Hydra or OmegaConf, which support config composition and
    command-line overrides. That is overkill for a handful of experiments.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class B0Config:
    """Settings for the Bien et al. (2018) baseline. Defaults follow EVAL_PROTOCOL.md §3."""

    name: str = "b0_alexnet"
    backbone: str = "alexnet"
    pretrained: bool = True            # ImageNet weights (torchvision AlexNet_Weights.IMAGENET1K_V1)
    epochs: int = 50
    lr: float = 1e-5
    weight_decay: float = 0.01         # Adam's (coupled) L2 penalty, as in the public MRNet code
    scheduler_factor: float = 0.3      # ReduceLROnPlateau on inner-tuning loss
    scheduler_patience: int = 5
    rotate_deg: float = 25.0           # augmentation: rotation in [-25, 25] degrees
    shift_px: float = 25.0             # augmentation: shift in [-25, 25] pixels
    flip_p: float = 0.5                # augmentation: horizontal flip probability
    amp: bool = True                   # mixed precision on GPU (ignored on CPU)
    deterministic: bool = True
    num_workers: int = 4               # DataLoader worker processes (0 = load in main process)
    seed: int = 42


@dataclass(frozen=True)
class ReaderConfig:
    """Settings for the Reader (DESIGN.md §1). Every experiment is one YAML file that changes a few of these."""

    name: str = "reader_r18"
    backbone: str = "resnet18"         # any timm CNN name; ablations try bigger ones
    backbone_weights: str = ""         # optional file of non-timm pretrained weights, e.g. RadImageNet ResNet-50 (D-017)
    pretrained: bool = True            # ImageNet weights from timm
    att_dim: int = 128                 # width of the gated-attention hidden layer
    dropout: float = 0.25              # on each per-finding plane summary, before the linear score
    plane_dropout: float = 0.2         # chance of hiding each plane during training (robust to missing series)
    freeze_batchnorm: bool = False     # keep backbone BatchNorm in inference mode while training (D-010)
    plane_loss_weight: float = 0.0     # >0: each plane's own diagnosis is also trained ("deep supervision", D-012)
    pooling: str = "attention"         # "attention" (gated, per finding) or "max" (B0-style); a diagnostic switch (D-013)
    planes: tuple[str, ...] = ("sagittal", "coronal", "axial")   # train on a subset of planes (single-plane diagnostics)
    # Anti-overfitting options (idea I3, D-017). All off by default, so older configs keep their meaning.
    intensity_jitter: float = 0.0      # random gamma in [exp(-x), exp(+x)] per series (contrast changes between scanners)
    slice_dropout: float = 0.0         # hide each slice with this probability while training (keeps at least 3)
    frozen_prefixes: tuple[str, ...] = ()   # backbone parameter-name prefixes that don't train, e.g. [conv1, bn1, layer1, layer2]
    loss_tasks: tuple[str, ...] = ("abnormal", "acl", "meniscus")   # findings that train the model and pick its epoch (I5, D-018)
    epochs: int = 20
    lr: float = 1e-4                   # AdamW learning rate of the pretrained backbone, cosine-decayed to 0 over `epochs`
    head_lr: float = 1e-3              # everything after the backbone starts from scratch, so it learns 10x faster (D-007)
    weight_decay: float = 1e-4
    accumulate_exams: int = 8          # gradient accumulation: one optimiser step every 8 exams (batch = 1 exam)
    grad_clip: float = 0.0             # >0: cap the gradient norm per optimiser step (stabilises RadImageNet training, D-020)
    rotate_deg: float = 25.0           # same augmentation as B0, so the comparison isolates the model
    shift_px: float = 25.0
    flip_p: float = 0.5
    amp: bool = True
    deterministic: bool = True
    num_workers: int = 0               # Windows worker start-up costs more than loading 3 small .npy files
    seed: int = 42


def _load_dataclass(path: str | Path, cls: type):
    """Read a YAML file into the dataclass `cls`. Unknown keys raise ValueError."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return cls(**raw)


def load_b0_config(path: str | Path) -> B0Config:
    """Read a YAML file into a B0Config. Unknown keys raise ValueError."""
    return _load_dataclass(path, B0Config)


def load_reader_config(path: str | Path) -> ReaderConfig:
    """Read a YAML file into a ReaderConfig. Unknown keys raise ValueError."""
    return _load_dataclass(path, ReaderConfig)


def config_hash(cfg: B0Config | ReaderConfig) -> str:
    """Short, stable fingerprint of all settings (same settings -> same hash)."""
    canonical = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
