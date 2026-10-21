"""
synthetic.py: a tiny fake MRNet dataset (random pixels, the exact MRNet v1.0 folder layout and label files).

What it does
    `make_fake_mrnet(base, n_train, n_valid, hw)` writes <base>/MRNet-v1.0/ with train/valid series and label CSVs.
    Labels follow MRNet's logic (a tear implies "abnormal") but are random, so no model can learn real anatomy from it.

Why
    The tests and scripts/smoke_test.py need data that has the right *shape* and runs in seconds, on any machine,
    without the licensed dataset.

Alternatives
    To test whether a model *learns*, you'd plant a visible pattern in the positive cases. Here we only test plumbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kneescan import PLANES


def make_fake_mrnet(base: Path, n_train: int = 40, n_valid: int = 10, hw: int = 64,
                    seed: int = 0, header: bool = False) -> Path:
    """Write a fake MRNet-v1.0 folder under `base` and return its root."""
    rng = np.random.default_rng(seed)
    root = base / "MRNet-v1.0"
    ids = {"train": [f"{i:04d}" for i in range(n_train)],
           "valid": [f"{i:04d}" for i in range(n_train, n_train + n_valid)]}
    for split, case_ids in ids.items():
        abnormal = rng.random(len(case_ids)) < 0.8
        acl = abnormal & (rng.random(len(case_ids)) < 0.35)
        meniscus = abnormal & (rng.random(len(case_ids)) < 0.45)
        for task, lab in (("abnormal", abnormal), ("acl", acl), ("meniscus", meniscus)):
            lines = (["case,label"] if header else []) + [f"{c},{int(v)}" for c, v in zip(case_ids, lab)]
            (root).mkdir(parents=True, exist_ok=True)
            (root / f"{split}-{task}.csv").write_text("\n".join(lines) + "\n")
        for plane in PLANES:
            d = root / split / plane
            d.mkdir(parents=True, exist_ok=True)
            for c in case_ids:
                s = int(rng.integers(6, 12))
                np.save(d / f"{c}.npy", rng.integers(0, 256, size=(s, hw, hw), dtype=np.uint8))
    return root
