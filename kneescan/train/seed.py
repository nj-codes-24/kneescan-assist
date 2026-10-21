"""
seed.py: make runs repeatable and resumable.

What it does
    * `seed_everything(seed)` seeds Python, NumPy and PyTorch (CPU + CUDA) in one call.
    * `capture_rng_state()` / `restore_rng_state()` save and reload every random-number
      generator, so a run resumed from a checkpoint continues with exactly the random
      numbers it would have drawn without the interruption.

Why
    Two runs with the same config should give the same numbers, and an interrupted
    overnight run should not silently become a different experiment after `--resume`.

Alternatives
    `deterministic=True` asks cuDNN for deterministic kernels. That is slightly slower and
    not every GPU operation has a deterministic version. For pure speed, pass False and
    accept tiny run-to-run differences (about the 3rd decimal of AUC).
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all random-number generators used in this project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def capture_rng_state(extra_generators: dict[str, torch.Generator] | None = None) -> dict[str, Any]:
    """Return a dict with the state of every RNG (put it inside a checkpoint)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generators": {},
    }
    for name, gen in (extra_generators or {}).items():
        state["generators"][name] = gen.get_state()
    return state


def restore_rng_state(state: dict[str, Any], extra_generators: dict[str, torch.Generator] | None = None) -> None:
    """Undo `capture_rng_state`: put every RNG back where it was."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(..., map_location="cuda") moves every tensor in a checkpoint to the GPU, these RNG states included,
    # but PyTorch only accepts them as CPU byte tensors. Without .cpu(), resuming on a GPU crashes (found 2026-10-20).
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    for name, gen in (extra_generators or {}).items():
        if name in state.get("generators", {}):
            gen.set_state(state["generators"][name].cpu())
