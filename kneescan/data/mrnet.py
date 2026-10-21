"""
mrnet.py: find, read and validate the Stanford MRNet v1.0 dataset.

What it does
    * `find_mrnet_root(base)` finds the folder that holds `train/`, `valid/` and the label
      CSVs, wherever the download put it (e.g. data/mrnet/raw/<table>/MRNet-v1.0/).
    * `read_labels(root, split)` merges the three label files into one table:
      case_id | abnormal | acl | meniscus.
    * `load_volume(root, split, plane, case_id)` returns one series as a uint8 array of shape
      (slices, 256, 256).
    * `MRNetPlaneDataset` is a PyTorch Dataset that returns one exam's series for one plane.
    * `validate_mrnet(root)` checks that every expected file exists and has a sane shape.

Expected layout (MRNet v1.0 as released by Stanford)
    <root>/train/{sagittal,coronal,axial}/0000.npy ... 1129.npy
    <root>/valid/{sagittal,coronal,axial}/1130.npy ... 1249.npy
    <root>/train-abnormal.csv, train-acl.csv, train-meniscus.csv   (rows: "0000,1", no header)
    <root>/valid-abnormal.csv, valid-acl.csv, valid-meniscus.csv

Why `.npy` with allow_pickle=False?
    A `.npy` file is a plain array. Refusing pickles means a tampered file cannot run code.

Alternatives
    If loading is the training bottleneck (check GPU utilisation), cache all volumes in RAM
    (about 6 GB for MRNet; pc_c has 64 GB) or convert them to one memory-mapped file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kneescan import PLANES, TASKS

SPLITS: tuple[str, ...] = ("train", "valid")


def find_mrnet_root(base: str | Path, max_depth: int = 4) -> Path:
    """Return the folder under `base` that contains 'train-acl.csv' and a 'train' folder."""
    base = Path(base)
    candidates = [base] + [p for p in base.rglob("train-acl.csv") if len(p.relative_to(base).parts) <= max_depth]
    for c in candidates:
        root = c if c.is_dir() else c.parent
        if (root / "train-acl.csv").exists() and (root / "train").is_dir():
            return root
    raise FileNotFoundError(f"No MRNet root (train-acl.csv + train/) found under {base}")


def _read_one_label_file(path: Path, task: str) -> pd.DataFrame:
    """Read one '<split>-<task>.csv'. Handles files with or without a header row."""
    df = pd.read_csv(path, header=None, dtype=str)
    if not df.iloc[0, 0].strip().isdigit():  # first row is a header -> drop it
        df = df.iloc[1:]
    df = df.iloc[:, :2]
    df.columns = ["case_id", task]
    df["case_id"] = df["case_id"].str.strip().str.zfill(4)
    df[task] = df[task].astype(int)
    if not df[task].isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0/1")
    return df


def read_labels(root: str | Path, split: str) -> pd.DataFrame:
    """One row per exam: case_id (4-digit string), abnormal, acl, meniscus (0/1 ints)."""
    root = Path(root)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    merged: pd.DataFrame | None = None
    for task in TASKS:
        df = _read_one_label_file(root / f"{split}-{task}.csv", task)
        merged = df if merged is None else merged.merge(df, on="case_id", how="outer", validate="one_to_one")
    assert merged is not None
    if merged[list(TASKS)].isna().any().any():
        raise ValueError(f"{split}: some exams are missing a label in one of the CSVs")
    return merged.sort_values("case_id").reset_index(drop=True)


def volume_path(root: str | Path, split: str, plane: str, case_id: str) -> Path:
    return Path(root) / split / plane / f"{case_id}.npy"


def load_volume(root: str | Path, split: str, plane: str, case_id: str) -> np.ndarray:
    """Load one series: uint8 array (slices, H, W). Memory-mapped, never unpickled."""
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {PLANES}")
    vol = np.load(volume_path(root, split, plane, case_id), allow_pickle=False, mmap_mode="r")
    if vol.ndim != 3:
        raise ValueError(f"{split}/{plane}/{case_id}: expected 3-D array, got shape {vol.shape}")
    return vol


class MRNetPlaneDataset(Dataset):
    """
    One item = one exam, one plane.

    Returns a dict with:
        "volume": torch.uint8 tensor (slices, H, W). Model-specific transforms happen on the GPU.
        "labels": torch.float32 tensor (3,) in TASKS order
        "case_id": str
    """

    def __init__(self, root: str | Path, split: str, plane: str, labels: pd.DataFrame,
                 case_ids: list[str] | None = None) -> None:
        self.root, self.split, self.plane = Path(root), split, plane
        table = labels.set_index("case_id")
        self.case_ids = list(case_ids) if case_ids is not None else list(table.index)
        self.labels = table.loc[self.case_ids, list(TASKS)].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, i: int) -> dict:
        vol = np.array(load_volume(self.root, self.split, self.plane, self.case_ids[i]))  # copy out of the memory map
        return {"volume": torch.from_numpy(vol), "labels": torch.from_numpy(self.labels[i]),
                "case_id": self.case_ids[i]}


class MRNetExamDataset(Dataset):
    """
    One item = one exam, all three planes (used by the Reader, which reads the planes together).

    Returns a dict with:
        "volumes": dict plane -> torch.uint8 tensor (slices, H, W)
        "labels": torch.float32 tensor (3,) in TASKS order
        "case_id": str
    """

    def __init__(self, root: str | Path, split: str, labels: pd.DataFrame,
                 case_ids: list[str] | None = None, planes: tuple[str, ...] = PLANES) -> None:
        self.root, self.split, self.planes = Path(root), split, tuple(planes)
        table = labels.set_index("case_id")
        self.case_ids = list(case_ids) if case_ids is not None else list(table.index)
        self.labels = table.loc[self.case_ids, list(TASKS)].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, i: int) -> dict:
        case_id = self.case_ids[i]
        volumes = {plane: torch.from_numpy(np.array(load_volume(self.root, self.split, plane, case_id)))
                   for plane in self.planes}
        return {"volumes": volumes, "labels": torch.from_numpy(self.labels[i]), "case_id": case_id}


@dataclass
class ValidationReport:
    n_exams: dict[str, int]
    missing_files: list[str]
    bad_shapes: list[str]
    slice_range: dict[str, tuple[int, int]]

    @property
    def ok(self) -> bool:
        return not self.missing_files and not self.bad_shapes


def validate_mrnet(root: str | Path, expected_hw: tuple[int, int] = (256, 256)) -> ValidationReport:
    """Check every exam has all 3 planes, each a uint8 (S, 256, 256) array. Labels are not inspected beyond counts."""
    root = Path(root)
    n_exams, missing, bad, ranges = {}, [], [], {}
    for split in SPLITS:
        labels = read_labels(root, split)
        n_exams[split] = len(labels)
        for plane in PLANES:
            counts = []
            for cid in labels["case_id"]:
                p = volume_path(root, split, plane, cid)
                if not p.exists():
                    missing.append(str(p.relative_to(root)))
                    continue
                vol = np.load(p, allow_pickle=False, mmap_mode="r")
                if vol.ndim != 3 or vol.shape[1:] != expected_hw or vol.dtype != np.uint8:
                    bad.append(f"{p.relative_to(root)} shape={vol.shape} dtype={vol.dtype}")
                else:
                    counts.append(vol.shape[0])
            if counts:
                ranges[f"{split}/{plane}"] = (min(counts), max(counts))
    return ValidationReport(n_exams, missing, bad, ranges)
