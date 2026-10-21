"""
kneemri.py: safely read the KneeMRI dataset (Štajduhar et al. 2017), our external ACL test set.

What it does
    * `safe_load_pck(path)` reads one `.pck` volume with a *restricted unpickler*. It
      rebuilds numpy arrays and refuses anything else.
    * `read_metadata(path)` reads metadata.csv and adds the binary labels we pre-registered
      (EVAL_PROTOCOL R9): `acl_tear` = partial or complete, `acl_complete` = complete only.
    * `preprocess_volume(vol)` applies the fixed preprocessing (EVAL_PROTOCOL R10): clip to the
      [0.5, 99.5] intensity percentiles, scale to uint8 0..255, resize in-plane to 256 x 256.
    * `KneeMRIDataset` returns preprocessed sagittal volumes in MRNet's format.

Why a restricted unpickler?
    Python's `pickle.load` will happily run any code embedded in a file. KneeMRI ships its
    volumes as pickles, and a downloaded file is untrusted. We inspected `example.pck`
    without executing it: it only needs four "globals" (numpy's array rebuild function,
    `numpy.ndarray`, `numpy.dtype` and `_codecs.encode` for Python-2-era byte strings).
    Anything else raises `UnsafePickleError`.

Alternatives
    Convert the volumes once to `.npy` (safe format) and load those instead. Under the
    licence (CC BY-NC-ND) such converted copies must stay private (data/ is git-ignored).

REMINDER: this is the *external test set*. Until the one-time final run, we may only load it
and check shapes and counts (EVAL_PROTOCOL R1). No model may run on it before then.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

TARGET_HW = 256
CLIP_PERCENTILES = (0.5, 99.5)


class UnsafePickleError(pickle.UnpicklingError):
    """Raised when a .pck file asks for anything other than a plain numpy array."""


def _numpy_reconstruct():
    """numpy's array-rebuild function (moved from numpy.core to numpy._core in numpy 2)."""
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import _reconstruct
    return _reconstruct


def _codecs_encode():
    import _codecs
    return _codecs.encode


_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct,
    ("numpy", "ndarray"): lambda: np.ndarray,
    ("numpy", "dtype"): lambda: np.dtype,
    ("_codecs", "encode"): _codecs_encode,
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        factory = _ALLOWED_GLOBALS.get((module, name))
        if factory is None:
            raise UnsafePickleError(f"refusing to load '{module}.{name}' from a pickle")
        return factory()


def safe_load_pck(path: str | Path) -> np.ndarray:
    """Load one KneeMRI volume (z, y, x) without ever executing code from the file."""
    data = Path(path).read_bytes()
    obj = _NumpyOnlyUnpickler(io.BytesIO(data), encoding="latin1").load()
    if not isinstance(obj, np.ndarray) or obj.ndim != 3:
        raise UnsafePickleError(f"{path}: expected a 3-D numpy array, got {type(obj).__name__}")
    return obj


def index_volumes(volumes_dir: str | Path) -> dict[str, Path]:
    """Map 'examId-seriesNo.pck' -> full path. The archive spreads files over sub-folders."""
    index: dict[str, Path] = {}
    for p in Path(volumes_dir).rglob("*.pck"):
        if p.name in index:
            raise ValueError(f"duplicate volume file name {p.name}")
        index[p.name] = p
    return index


def read_metadata(path: str | Path) -> pd.DataFrame:
    """metadata.csv plus our two pre-registered binary labels (EVAL_PROTOCOL R9)."""
    meta = pd.read_csv(path)
    expected = {"examId", "seriesNo", "aclDiagnosis", "kneeLR", "roiX", "roiY", "roiZ",
                "roiHeight", "roiWidth", "roiDepth", "volumeFilename"}
    missing = expected - set(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns {sorted(missing)}")
    if not meta["aclDiagnosis"].isin([0, 1, 2]).all():
        raise ValueError("aclDiagnosis must be 0 (healthy), 1 (partial) or 2 (complete)")
    meta["acl_tear"] = (meta["aclDiagnosis"] > 0).astype(int)           # primary (R9)
    meta["acl_complete"] = (meta["aclDiagnosis"] == 2).astype(int)      # secondary (R9)
    return meta


def preprocess_volume(vol: np.ndarray, target_hw: int = TARGET_HW) -> np.ndarray:
    """Fixed preprocessing (R10): percentile clip -> uint8 -> bilinear resize to 256 x 256."""
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, CLIP_PERCENTILES)
    if hi <= lo:  # a constant volume; avoid division by zero
        scaled = np.zeros_like(v)
    else:
        scaled = (np.clip(v, lo, hi) - lo) / (hi - lo) * 255.0
    t = torch.from_numpy(scaled).unsqueeze(1)  # (z, 1, y, x): resize each slice
    if t.shape[-2:] != (target_hw, target_hw):
        t = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return t.squeeze(1).round().clamp(0, 255).to(torch.uint8).numpy()


class KneeMRIDataset(Dataset):
    """
    One item = one KneeMRI volume, preprocessed to look like an MRNet sagittal series.

    Returns {"volume": uint8 (S, 256, 256), "acl_tear": float, "acl_complete": float,
             "exam_id": int, "filename": str}.
    """

    def __init__(self, volumes_dir: str | Path, metadata: pd.DataFrame) -> None:
        self.meta = metadata.reset_index(drop=True)
        self.paths = index_volumes(volumes_dir)
        missing = set(self.meta["volumeFilename"]) - set(self.paths)
        if missing:
            raise FileNotFoundError(f"{len(missing)} volumes listed in metadata are missing, e.g. {sorted(missing)[:3]}")

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int) -> dict:
        row = self.meta.iloc[i]
        vol = preprocess_volume(safe_load_pck(self.paths[row["volumeFilename"]]))
        return {"volume": torch.from_numpy(vol), "acl_tear": float(row["acl_tear"]),
                "acl_complete": float(row["acl_complete"]), "exam_id": int(row["examId"]),
                "filename": str(row["volumeFilename"])}
