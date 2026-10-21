"""
dicom_io.py: turn an uploaded knee MRI study (a .zip of DICOM files, or MRNet-style .npy files) into the
three uint8 volumes the Reader expects. Patient data is never looked at, shown or kept.

What it does
    * `read_dicom_zip(path)`: reads every DICOM in the zip *in memory* (nothing is extracted to disk) and keeps
      only four things per file: pixels, SeriesInstanceUID, ImageOrientationPatient, ImagePositionPatient.
      pydicom parses the file, but names, IDs, dates and every other tag are never accessed, and each dataset
      object is dropped right after.
    * `plane_from_orientation(iop)`: the slice normal (row × column direction) points mostly along x → sagittal,
      y → coronal, z → axial.
    * Slices are sorted by their physical position along the normal (file order is often random), contrast-stretched
      to the 0.5–99.5 percentile and resized to 256 × 256, the same scaling as KneeMRI (EVAL_PROTOCOL R10).
    * `read_npy_files(paths)`: MRNet format, one (S, 256, 256) uint8 file per plane, plane taken from the file name.

Why so strict about tags?
    The demo must never display or store patient identifiers (docs/DESIGN.md §4). Not reading them at all is the
    simplest way to guarantee that.

Alternatives
    A clinical tool would read SeriesDescription to tell T1 from T2/PD and ask the user to confirm the sequence.
    MRNet's exact sequences can't be matched reliably here, so the app shows that as a limitation.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from kneescan import PLANES

TARGET_SIZE = 256
CLIP_PERCENTILES = (0.5, 99.5)
MAX_FILES = 5000                     # a knee study has a few hundred images; more suggests the wrong upload
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3  # refuse "zip bombs" before reading them


def plane_from_orientation(iop: list[float]) -> str:
    """ImageOrientationPatient (row xyz, column xyz) -> 'sagittal' / 'coronal' / 'axial'."""
    normal = np.cross(np.array(iop[:3], dtype=float), np.array(iop[3:], dtype=float))
    return PLANES[int(np.argmax(np.abs(normal)))]   # PLANES order is (x, y, z) = (sagittal, coronal, axial)


def _read_minimal(raw: bytes) -> dict | None:
    """Pixels + 3 geometry tags of one DICOM file, or None if it isn't an image. No other tag is touched."""
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        return {"series": str(ds.SeriesInstanceUID), "iop": [float(v) for v in ds.ImageOrientationPatient],
                "ipp": [float(v) for v in ds.ImagePositionPatient], "pixels": ds.pixel_array.astype(np.float32)}
    except (AttributeError, KeyError, ValueError, TypeError):
        return None   # reports, localisers without geometry, non-image files


def _zip_members(path: Path) -> list[bytes]:
    with zipfile.ZipFile(path) as zf:
        files = [i for i in zf.infolist() if not i.is_dir()]
        if len(files) > MAX_FILES or sum(i.file_size for i in files) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("this zip is too large to be one knee MRI study")
        return [zf.read(i) for i in files]


def _stack_series(slices: list[dict]) -> np.ndarray:
    """Sort one series' slices by physical position, then stack them.

    The sort key is the patient coordinate the slices step along (x for sagittal, y for coronal, z for axial),
    ascending. Sorting along the slice normal instead would flip the order on scanners whose normal points the
    other way. The patient axis gives the same order for every scanner.
    """
    normal = np.cross(np.array(slices[0]["iop"][:3]), np.array(slices[0]["iop"][3:]))
    axis = int(np.argmax(np.abs(normal)))
    ordered = sorted(slices, key=lambda s: s["ipp"][axis])
    shapes = {s["pixels"].shape for s in ordered}
    if len(shapes) != 1:
        raise ValueError("slices of one series have different sizes")
    return np.stack([s["pixels"] for s in ordered])


def to_uint8_256(volume: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch to 0..255, then bilinear resize of every slice to 256 × 256."""
    lo, hi = np.percentile(volume, CLIP_PERCENTILES)
    scaled = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    resized = F.interpolate(torch.from_numpy(scaled).float().unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE),
                            mode="bilinear", align_corners=False).squeeze(1)
    return resized.round().clamp(0, 255).to(torch.uint8).numpy()


def read_dicom_zip(path: str | Path) -> dict[str, np.ndarray]:
    """plane -> uint8 (S, 256, 256). When a plane has several series, the one with the most slices is used."""
    series: dict[str, list[dict]] = {}
    for raw in _zip_members(Path(path)):
        info = _read_minimal(raw)
        if info is not None:
            series.setdefault(info["series"], []).append(info)
    volumes: dict[str, np.ndarray] = {}
    for slices in sorted(series.values(), key=len, reverse=True):   # biggest series first
        plane = plane_from_orientation(slices[0]["iop"])
        if plane not in volumes and len(slices) >= 3:                 # 1–2 images are localisers, not a series
            volumes[plane] = to_uint8_256(_stack_series(slices))
    if not volumes:
        raise ValueError("no usable MRI series found in the zip")
    return volumes


def read_npy_files(paths: list[str | Path]) -> dict[str, np.ndarray]:
    """MRNet-style files; the plane comes from the file name (e.g. 'sagittal_0001.npy')."""
    volumes = {}
    for path in paths:
        plane = next((p for p in PLANES if p in Path(path).name.lower()), None)
        if plane is None:
            raise ValueError(f"can't tell the plane of {Path(path).name}: put sagittal/coronal/axial in the name")
        vol = np.load(path, allow_pickle=False)     # never unpickle an uploaded file
        if vol.ndim != 3:
            raise ValueError(f"{Path(path).name}: expected (slices, H, W)")
        already_mrnet_format = vol.shape[1:] == (TARGET_SIZE, TARGET_SIZE) and vol.dtype == np.uint8
        volumes[plane] = vol if already_mrnet_format else to_uint8_256(vol.astype(np.float32))
    return volumes
