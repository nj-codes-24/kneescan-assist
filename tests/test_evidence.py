"""
test_evidence.py: every headline number in the README matches the ORIGINAL output files in evidence/, and those files
are the ones that were produced (fingerprints), in the right order (frozen before the one-time test).

Why: writing numbers in a README is easy. These checks make a typo, a rounded-up claim or an edited report fail the tests,
so a reader can trust the README as much as the files the scripts wrote (scripts/collect_evidence.py copies them unchanged).
"""

from __future__ import annotations

import hashlib
import json

import yaml

from tests.conftest import REPO_ROOT

EVIDENCE = REPO_ROOT / "evidence"
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
TASKS = ("abnormal", "acl", "meniscus")


def load(rel: str) -> dict:
    return json.loads((EVIDENCE / rel).read_text(encoding="utf-8"))


def signed(x: float) -> str:
    """+0.012 / −0.032, written the way the README writes them (a real minus sign)."""
    return f"{'+' if x >= 0 else '−'}{abs(x):.3f}"


def readme_line(starts_with: str) -> str:
    lines = [line for line in README.splitlines() if line.startswith(starts_with)]
    assert len(lines) == 1, f"expected one README line starting with {starts_with!r}, found {len(lines)}"
    return lines[0]


def test_files_match_their_fingerprints() -> None:
    lines = (EVIDENCE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) > 100
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((EVIDENCE / rel).read_bytes()).hexdigest() == digest, rel


def test_final_test_table_matches_final_json() -> None:
    final = load("final_test/final.json")["mrnet_valid"]
    rows = {"baseline": "| B0: our reproduction", "reader": "| **KneeScan Reader**", "hybrid": "| Hybrid: Reader + B0 (mean logit)"}
    for model, start in rows.items():
        line, auc, lo, hi = readme_line(start), *final[model]["macro_auc"]
        assert f"{auc:.3f}" in line and f"({lo:.3f}–{hi:.3f})" in line, model
        cells = [c.strip(" *") for c in line.split("|")[3:6]]
        assert cells == [f"{final[model]['per_task'][t]['auc']:.3f}" for t in TASKS], model


def test_paired_gains_and_external_test_match_final_json() -> None:
    report = load("final_test/final.json")
    gain, lo, hi = report["mrnet_valid"]["reader_minus_baseline"]["macro"]
    assert f"Reader {signed(gain)} (95% CI {signed(lo)} to {signed(hi)})" in README
    gain, lo, hi = report["mrnet_valid"]["hybrid_minus_baseline"]["macro"]
    assert f"Hybrid {signed(gain)} ({signed(lo)} to {signed(hi)})" in README
    knee = report["kneemri"]
    assert (f"AUC {knee['baseline']['acl_auc'][0]:.3f} (B0), {knee['reader']['acl_auc'][0]:.3f} (Reader), "
            f"{knee['hybrid']['acl_auc'][0]:.3f} (Hybrid)") in README
    assert f"({knee['n_volumes']} volumes" in README


def test_cross_validation_table_matches_cv_reports() -> None:
    rows = {"b0": "| B0 |", "reader_r18_fbn": "| R1b: one Reader", "reader_r2": "| R2: three plane specialists",
            "reader_r2_seeds": "| Reader (R2, 2 seeds)", "hybrid_r2s_b0": "| Hybrid: Reader + B0 |"}
    for run, start in rows.items():
        report, line = load(f"cv/{run}.json"), readme_line(start)
        lo, hi = report["macro_auc_ci"]
        assert f"{report['macro_auc']:.3f} ({lo:.3f}–{hi:.3f})" in line, run
        assert [c.strip() for c in line.split("|")[3:6]] == [f"{report['per_task'][t]['auc']:.3f}" for t in TASKS], run
    hybrid = load("cv/hybrid_r2s_b0.json")["comparison"]
    assert (f"In CV the hybrid beat B0 ({signed(hybrid['gain'])}, CI {signed(hybrid['gain_ci'][0])} to "
            f"{signed(hybrid['gain_ci'][1])})") in README


def test_model_was_frozen_from_cv_before_the_one_time_test() -> None:
    frozen = yaml.safe_load((REPO_ROOT / "docs" / "frozen.yaml").read_text(encoding="utf-8"))
    final = load("final_test/final.json")
    assert frozen["frozen_at"] < final["run_at"]                        # ISO timestamps compare as text
    assert final["reader_run"] == frozen["models"]["reader"]["run"] and final["rerun_reason"] is None
    lock = (EVIDENCE / "final_test" / "LOCK.txt").read_text(encoding="utf-8")
    assert "reason: first run" in lock and lock.count("run started") == 1   # the test ran exactly once
    for model, run in (("reader", "reader_r2_seeds"), ("baseline", "b0")):
        cv = load(f"cv/{run}.json")
        for t in TASKS:                                                 # operating thresholds come from CV, not the test
            assert abs(frozen["models"][model]["thresholds_t_hs"][t] - cv["per_task"][t]["t_hs"]) < 1e-12


def test_evidence_names_no_individual_exam() -> None:
    for path in EVIDENCE.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").lower()
            assert not any(m in text for m in ("case_id", "exam_id", "volumefilename", ".pck")), path
