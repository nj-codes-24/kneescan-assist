"""Tests for kneescan.data.kneemri: the restricted unpickler, labels and preprocessing."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from kneescan.data.kneemri import (KneeMRIDataset, UnsafePickleError, preprocess_volume,
                                   read_metadata, safe_load_pck)
from tests.conftest import REPO_ROOT

RAW = REPO_ROOT / "data" / "kneemri" / "raw"


class _Evil:
    """A pickle that would run a shell command if loaded with plain pickle.load."""

    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_refuses_code_execution(tmp_path):
    path = tmp_path / "evil.pck"
    path.write_bytes(pickle.dumps(_Evil(), protocol=2))
    with pytest.raises(UnsafePickleError):
        safe_load_pck(path)


def test_refuses_non_array(tmp_path):
    path = tmp_path / "list.pck"
    path.write_bytes(pickle.dumps([1, 2, 3], protocol=2))
    with pytest.raises(UnsafePickleError):
        safe_load_pck(path)


@pytest.mark.parametrize("protocol", [2, 4])
def test_accepts_numpy_volume(tmp_path, protocol):
    vol = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7)
    path = tmp_path / "ok.pck"
    path.write_bytes(pickle.dumps(vol, protocol=protocol))
    assert np.array_equal(safe_load_pck(path), vol)


def test_metadata_binarisation(tmp_path):
    rows = [dict(examId=1, seriesNo=1, aclDiagnosis=d, kneeLR=0, roiX=0, roiY=0, roiZ=0,
                 roiHeight=1, roiWidth=1, roiDepth=1, volumeFilename=f"1-{d}.pck") for d in (0, 1, 2)]
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)
    meta = read_metadata(tmp_path / "metadata.csv")
    assert meta["acl_tear"].tolist() == [0, 1, 1]       # partial counts as a tear (R9)
    assert meta["acl_complete"].tolist() == [0, 0, 1]


def test_preprocess_shape_range():
    rng = np.random.default_rng(0)
    out = preprocess_volume(rng.integers(0, 4096, size=(4, 320, 300)).astype(np.uint16))
    assert out.shape == (4, 256, 256) and out.dtype == np.uint8
    # without resizing, clipping + scaling must use the full 0..255 range
    same_size = preprocess_volume(rng.integers(0, 4096, size=(2, 256, 256)).astype(np.uint16))
    assert same_size.min() == 0 and same_size.max() == 255


def test_preprocess_constant_volume():
    assert preprocess_volume(np.full((2, 10, 10), 7, dtype=np.uint16)).max() == 0


def test_dataset_reports_missing_files(tmp_path):
    meta = pd.DataFrame([dict(examId=1, volumeFilename="nope.pck", acl_tear=0, acl_complete=0)])
    with pytest.raises(FileNotFoundError):
        KneeMRIDataset(tmp_path, meta)


@pytest.mark.data
def test_bundled_example_loads():
    """example.pck is the publisher's sample file; loading it is an integrity check (R1)."""
    if not (RAW / "example.pck").exists():
        pytest.skip("KneeMRI not downloaded")
    vol = safe_load_pck(RAW / "example.pck")
    assert vol.ndim == 3
    meta = read_metadata(RAW / "metadata.csv")
    assert len(meta) == 917 and meta["examId"].nunique() == 909
    assert meta["aclDiagnosis"].value_counts().to_dict() == {0: 690, 1: 172, 2: 55}
