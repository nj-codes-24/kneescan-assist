"""Tests for the pure helpers in scripts/fetch_mrnet.py (no network)."""

from __future__ import annotations

import importlib.util

import pandas as pd

from tests.conftest import REPO_ROOT

_spec = importlib.util.spec_from_file_location("fetch_mrnet", REPO_ROOT / "scripts" / "fetch_mrnet.py")
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def test_repair_restores_first_row_eaten_as_header():
    # what Redivis shows for train-acl: header "_0000", "_0" and 1,129 remaining rows
    raw = pd.DataFrame({"_0000": ["0001", "0002"], "_0": [1, 0]})
    fixed = fetch.repair_label_table(raw)
    assert fixed["case_id"].tolist() == ["0000", "0001", "0002"]
    assert fixed["label"].tolist() == [0, 1, 0]


def test_repair_leaves_proper_header_alone():
    raw = pd.DataFrame({"case": ["0000", "0001"], "label": [1, 0]})
    assert len(fetch.repair_label_table(raw)) == 2


def test_plane_and_case():
    assert fetch.plane_and_case("train/axial/0123.npy") == ("axial", "0123")
    assert fetch.plane_and_case("Sagittal/7.npy") == ("sagittal", "0007")
    assert fetch.plane_and_case("0123.npy") is None
