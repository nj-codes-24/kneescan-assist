"""
conftest.py: shared test fixtures.

`fake_mrnet` builds a tiny dataset in a temp folder with the *exact* MRNet v1.0 layout
(random pixels, small images), so every loader and training test runs in seconds without
the real data. Real-data checks are marked `@pytest.mark.data` and skip when the data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kneescan.data.synthetic import make_fake_mrnet  # noqa: F401  (re-exported for tests)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _redirect_lab_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from writing into the real project diary logs/pc_c.jsonl."""
    monkeypatch.setenv("KNEESCAN_LOG_DIR", str(tmp_path / "lab_logs"))


@pytest.fixture
def fake_mrnet(tmp_path: Path) -> Path:
    return make_fake_mrnet(tmp_path)
