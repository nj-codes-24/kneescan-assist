"""
test_audit.py: the repository audit really catches what it promises to (planted labels, tokens, UIDs, e-mails),
and lets ordinary code and the label-free split files through.
"""

from __future__ import annotations

import importlib.util

from tests.conftest import REPO_ROOT

_spec = importlib.util.spec_from_file_location("audit_repo", REPO_ROOT / "scripts" / "audit_repo.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_label_csv_is_caught_but_split_file_passes() -> None:
    assert audit.text_problems("x/labels.csv", "case_id,acl\n0000,1\n")
    assert audit.text_problems("splits/mrnet_train_5fold.csv", "case_id,fold\n0000,3\n") == []


def test_secrets_uids_and_emails_are_caught() -> None:
    fake_token = "ghp_" + "a1B2" * 9          # planted values are built at runtime, so this file passes the audit itself
    assert any("GitHub token" in p for p in audit.text_problems("a.py", f"t = '{fake_token}'"))
    fake_uid = ".".join(["1", "2", "840", "113619", "2", "55", "3", "604688", "1234567"])
    fake_email = "jane.doe" + "@" + "hospital.org"
    assert any("UID" in p for p in audit.text_problems("a.md", f"series {fake_uid}"))
    assert any("e-mail" in p for p in audit.text_problems("a.md", f"write to {fake_email}"))
    assert audit.text_problems("a.md", "Co-Authored-By: Jo <123+jo@users.noreply.github.com>") == []


def test_binary_types_are_refused() -> None:
    assert audit.suffix_of("data/x.npy") not in audit.TEXT_SUFFIXES
    assert audit.suffix_of(".gitignore") in audit.TEXT_SUFFIXES


def test_figures_only_in_docs_images_and_only_when_described(tmp_path, monkeypatch) -> None:
    (tmp_path / "docs" / "images").mkdir(parents=True)
    (tmp_path / "docs" / "images" / "chart.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "docs" / "images" / "scan.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "docs" / "images" / "SOURCES.md").write_text("| chart.png | a chart |\n", encoding="utf-8")
    (tmp_path / "elsewhere.png").write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit, "FIGURE_LIST", tmp_path / "docs" / "images" / "SOURCES.md")
    assert audit.file_problems("docs/images/chart.png") == []
    assert "not described" in audit.file_problems("docs/images/scan.png")[0]
    assert "could hold data" in audit.file_problems("elsewhere.png")[0]
