"""Tests for kneescan.data.mrnet: finding the root, reading labels, loading volumes, validating."""

from __future__ import annotations

import numpy as np
import pytest

from kneescan import TASKS
from kneescan.data.mrnet import (MRNetPlaneDataset, find_mrnet_root, load_volume, read_labels,
                                 validate_mrnet)
from tests.conftest import REPO_ROOT, make_fake_mrnet


def test_find_root_when_nested(tmp_path):
    root = make_fake_mrnet(tmp_path / "raw" / "some_table")
    assert find_mrnet_root(tmp_path) == root


def test_read_labels_merges_three_tasks(fake_mrnet):
    labels = read_labels(fake_mrnet, "train")
    assert list(labels.columns) == ["case_id", *TASKS]
    assert len(labels) == 40
    assert labels["case_id"].iloc[0] == "0000"
    # by construction a tear implies abnormal
    assert ((labels["acl"] == 0) | (labels["abnormal"] == 1)).all()


def test_read_labels_tolerates_header(tmp_path):
    root = make_fake_mrnet(tmp_path, header=True)
    assert len(read_labels(root, "valid")) == 10


def test_load_volume_shape_and_dtype(fake_mrnet):
    vol = load_volume(fake_mrnet, "train", "sagittal", "0003")
    assert vol.ndim == 3 and vol.shape[1:] == (64, 64) and vol.dtype == np.uint8


def test_load_volume_refuses_pickles(fake_mrnet):
    path = fake_mrnet / "train" / "axial" / "0001.npy"
    np.save(path, np.array([{"not": "an image"}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        load_volume(fake_mrnet, "train", "axial", "0001")


def test_dataset_item(fake_mrnet):
    labels = read_labels(fake_mrnet, "train")
    ds = MRNetPlaneDataset(fake_mrnet, "train", "coronal", labels, case_ids=["0005", "0002"])
    item = ds[1]
    assert item["case_id"] == "0002"
    assert item["volume"].ndim == 3
    assert item["labels"].shape == (3,)
    expected = labels.set_index("case_id").loc["0002", list(TASKS)].to_numpy(dtype=np.float32)
    assert np.array_equal(item["labels"].numpy(), expected)


def test_validate_flags_missing_and_bad(fake_mrnet):
    assert validate_mrnet(fake_mrnet, expected_hw=(64, 64)).ok
    (fake_mrnet / "valid" / "axial" / "0041.npy").unlink()
    np.save(fake_mrnet / "train" / "sagittal" / "0000.npy", np.zeros((5, 32, 32), dtype=np.uint8))
    report = validate_mrnet(fake_mrnet, expected_hw=(64, 64))
    assert not report.ok
    assert len(report.missing_files) == 1 and len(report.bad_shapes) == 1


@pytest.mark.data
def test_real_mrnet_if_present():
    """Runs only once the real MRNet is on disk: counts must match the published 1,130 / 120."""
    try:
        root = find_mrnet_root(REPO_ROOT / "data" / "mrnet")
    except FileNotFoundError:
        pytest.skip("MRNet not downloaded yet")
    assert len(read_labels(root, "train")) == 1130
    assert len(read_labels(root, "valid")) == 120
