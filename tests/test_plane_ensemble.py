"""
test_plane_ensemble.py: R2's equal logit vote is computed exactly, and refuses components that don't line up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kneescan.eval.plane_ensemble import average_logits, combine_runs, read_components


def component(probs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"fold": 0, "subset": "oof", "case_id": ["0001", "0002"], "task": "acl",
                         "label": [1, 0], "prob": probs})


def test_mean_logit_of_two_components() -> None:
    combined = average_logits([component([0.9, 0.2]), component([0.5, 0.2])])
    # logit(0.9) = 2.197, logit(0.5) = 0 -> mean 1.099 -> prob 0.75; identical inputs stay identical
    assert np.allclose(combined["prob"], [0.75, 0.2], atol=1e-3)
    assert list(combined["label"]) == [1, 0]


def test_mismatched_components_are_refused() -> None:
    with pytest.raises(ValueError, match="same exams"):
        average_logits([component([0.9, 0.2]), component([0.5, 0.2]).iloc[:1]])


def test_combine_writes_a_normal_looking_run(tmp_path: Path) -> None:
    for name in ("sag", "cor"):
        (tmp_path / name / "fold0").mkdir(parents=True)
        component([0.7, 0.3]).to_csv(tmp_path / name / "fold0" / "preds.csv", index=False)
    combine_runs([tmp_path / "sag", tmp_path / "cor"], tmp_path / "r2")
    assert (tmp_path / "r2" / "fold0" / "preds.csv").exists()
    assert read_components(tmp_path / "r2") == [tmp_path / "sag", tmp_path / "cor"]
    assert read_components(tmp_path / "sag") is None


def test_per_finding_combination_uses_only_that_findings_runs(tmp_path: Path) -> None:
    from kneescan.eval.plane_ensemble import combine_task_runs
    for name, prob in (("acl_spec", 0.9), ("other_spec", 0.1)):
        (tmp_path / name / "fold0").mkdir(parents=True)
        df = pd.concat([component([prob, prob]).assign(task=t) for t in ("abnormal", "acl", "meniscus")])
        df.to_csv(tmp_path / name / "fold0" / "preds.csv", index=False)
    out = combine_task_runs({"acl": [tmp_path / "acl_spec"], "abnormal": [tmp_path / "other_spec"],
                             "meniscus": [tmp_path / "other_spec"]}, tmp_path / "r5")
    assert np.allclose(out.query("task == 'acl'")["prob"], 0.9) and np.allclose(out.query("task == 'meniscus'")["prob"], 0.1)
