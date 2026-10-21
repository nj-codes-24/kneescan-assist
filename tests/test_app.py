"""
test_app.py: the Clinic App's building blocks. DICOM reading (plane detection, physical slice order, no patient
data in the output), confidence voting, heatmap overlays, and an end-to-end prediction with tiny fake models.

Why synthetic DICOMs?
    They let us control the geometry exactly (shuffled file order, known planes) and plant a fake patient name to
    prove it never reaches the output. No real patient file is needed or stored.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from kneescan import TASKS
from kneescan.app.dicom_io import plane_from_orientation, read_dicom_zip, read_npy_files, to_uint8_256
from kneescan.app.predictor import Predictor, confidence_from_votes, overlay
from kneescan.models.reader import KneeReader

ORIENTATION = {"sagittal": [0, 1, 0, 0, 0, -1], "coronal": [1, 0, 0, 0, 0, -1], "axial": [1, 0, 0, 0, 1, 0]}
NORMAL_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def make_dicom(plane: str, series_uid: str, position: float, value: int) -> bytes:
    """One 32 × 32 MR slice whose pixels all equal `value`, at `position` mm along the plane's normal."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName, ds.PatientID = "Doe^Jane", "SECRET-123"     # planted: must never come out
    ds.SOPClassUID, ds.SOPInstanceUID, ds.SeriesInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID, series_uid
    ds.ImageOrientationPatient = ORIENTATION[plane]
    ipp = [0.0, 0.0, 0.0]
    ipp[NORMAL_AXIS[plane]] = position
    ds.ImagePositionPatient = ipp
    ds.Rows = ds.Columns = 32
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelData = np.full((32, 32), value, dtype=np.uint16).tobytes()
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, ds, enforce_file_format=True)
    return buffer.getvalue()


def make_study_zip(path: Path) -> Path:
    """Sagittal series of 5 slices written in shuffled order (value = 100 * physical position) + a coronal series."""
    with zipfile.ZipFile(path, "w") as zf:
        sag_uid = generate_uid()
        for n, pos in enumerate([3, 0, 4, 1, 2]):
            zf.writestr(f"IMG{n:03d}", make_dicom("sagittal", sag_uid, float(pos), 100 * pos + 50))
        cor_uid = generate_uid()
        for pos in range(4):
            zf.writestr(f"COR/{pos}.dcm", make_dicom("coronal", cor_uid, float(pos), 500))
    return path


def test_plane_from_orientation() -> None:
    for plane, iop in ORIENTATION.items():
        assert plane_from_orientation(iop) == plane


def test_dicom_zip_is_sorted_by_position_and_has_no_patient_data(tmp_path: Path) -> None:
    volumes = read_dicom_zip(make_study_zip(tmp_path / "study.zip"))
    assert set(volumes) == {"sagittal", "coronal"}
    sag = volumes["sagittal"]
    assert sag.shape == (5, 256, 256) and sag.dtype == np.uint8
    assert list(sag[:, 0, 0]) == sorted(sag[:, 0, 0])            # brightness rises with physical position
    assert all(isinstance(v, np.ndarray) for v in volumes.values())  # only pixel arrays come out, no datasets


def test_zip_with_too_many_files_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kneescan.app.dicom_io.MAX_FILES", 3)
    with pytest.raises(ValueError, match="too large"):
        read_dicom_zip(make_study_zip(tmp_path / "study.zip"))


def test_npy_upload_needs_plane_in_name(tmp_path: Path) -> None:
    np.save(tmp_path / "axial_0001.npy", np.zeros((4, 256, 256), dtype=np.uint8))
    np.save(tmp_path / "scan.npy", np.zeros((4, 256, 256), dtype=np.uint8))
    assert set(read_npy_files([tmp_path / "axial_0001.npy"])) == {"axial"}
    with pytest.raises(ValueError, match="plane"):
        read_npy_files([tmp_path / "scan.npy"])


def test_uint8_conversion_range() -> None:
    out = to_uint8_256(np.random.default_rng(0).normal(500, 100, (3, 64, 64)))
    # the stretch uses the full 0..255 range; resizing smooths the extreme pixels a little, so not exactly 0 and 255
    assert out.shape == (3, 256, 256) and out.dtype == np.uint8 and out.min() < 30 and out.max() > 225


def test_confidence_from_votes() -> None:
    assert [confidence_from_votes(v, 5) for v in (0, 1, 2, 3, 4, 5)] == ["High", "Medium", "Low", "Low", "Medium", "High"]


def test_overlay_keeps_grey_where_no_evidence() -> None:
    image = overlay(np.full((4, 4), 128, np.uint8), np.zeros((4, 4), np.float32), alpha=0.5)
    assert image.shape == (4, 4, 3) and (image == 128).all()


def write_fake_run(run_dir: Path, n_folds: int = 2, planes: str = "") -> None:
    """A run folder like a finished CV run, with tiny untrained models (optionally a single-plane specialist)."""
    run_dir.mkdir(parents=True)
    plane_line = f"planes: [{planes}]\n" if planes else ""
    (run_dir / "config.yaml").write_text("name: fake\nbackbone: resnet18\npretrained: false\natt_dim: 16\n" + plane_line)
    for fold in range(n_folds):
        (run_dir / f"fold{fold}").mkdir()
        torch.manual_seed(fold)
        model = KneeReader("resnet18", pretrained=False, att_dim=16)
        torch.save(model.state_dict(), run_dir / f"fold{fold}" / "best.pt")


def test_predictor_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fake"
    write_fake_run(run_dir)
    predictor = Predictor(run_dir, device="cpu")
    volumes = {"sagittal": np.random.default_rng(0).integers(0, 256, (5, 64, 64), dtype=np.uint8)}
    results = predictor.predict(volumes)
    assert [r.task for r in results] == list(TASKS)
    for r in results:
        assert 0.0 <= r.probability <= 1.0 and len(r.fold_probabilities) == 2
        assert r.cv_auc == "n/a" and r.threshold == 0.5            # run not scored yet
        heatmap = r.heatmaps[0]
        assert heatmap.plane == "sagittal" and heatmap.image.shape == (64, 64, 3)
        assert 0 <= heatmap.slice_index < heatmap.n_slices and "sagittal slice" in heatmap.caption


def test_combined_run_uses_only_the_specialists_whose_plane_is_present(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_fake_run(runs / "sag", planes="sagittal")
    write_fake_run(runs / "ax", planes="axial")
    (runs / "r2").mkdir()
    (runs / "r2" / "components.yaml").write_text("rule: mean\ncomponents: [runs/sag, runs/ax]\n")
    predictor = Predictor(runs / "r2", device="cpu")
    assert [m.plane for m in predictor.members] == ["sagittal", "axial"]
    volumes = {"sagittal": np.random.default_rng(0).integers(0, 256, (5, 64, 64), dtype=np.uint8)}
    results = predictor.predict(volumes)            # axial missing: only the sagittal specialist votes
    assert len(results[0].fold_probabilities) == 2 and [h.plane for h in results[0].heatmaps] == ["sagittal"]
