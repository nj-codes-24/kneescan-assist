"""
audit_repo.py: check that nothing data-derived, licensed or secret would reach GitHub. Read-only: it changes nothing.

What it does
    1. Asks git which files a commit could contain: tracked files plus untracked files that .gitignore does not exclude
       (`git ls-files --cached --others --exclude-standard`, a read-only command).
    2. Checks every one of them. Any problem makes the audit FAIL (exit code 1):
       * file type: only code and text (.py .ps1 .css .md .yaml .toml .txt .csv .json .jsonl .gitignore). Images, arrays, weights,
         archives and DICOMs are refused outright, with one exception: README figures (.png .svg) in docs/images/, each
         listed in docs/images/SOURCES.md with what it shows. They are charts and app screenshots, never a scan (D-025).
       * size: over 600 KB is suspicious for hand-written code or docs.
       * CSV content: a column named like a label (abnormal / acl / meniscus / label / aclDiagnosis) means labels,
         which the MRNet and KneeMRI licences don't allow us to redistribute. Split files may hold case IDs + folds only.
       * evidence/ (the original reports behind the numbers) must be aggregate only: no case IDs, exam IDs or file names.
       * DICOM identifiers (long dotted UIDs), credential patterns (GitHub, Hugging Face, Kaggle, Redivis, AWS, private
         keys, passwords) and e-mail addresses.
    3. Prints a JSON summary.

Adapted from the Kaggle project's `tools/publish_github.py`. Dropped: its whitelist export and git push (commits are made by
hand here) and its report-text check (this project has no report text).

Usage
    .venv\\Scripts\\python.exe scripts\\audit_repo.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 600 * 1024
FIGURE_DIR = "docs/images/"
FIGURE_SUFFIXES = {".png", ".svg"}
FIGURE_LIST = REPO_ROOT / "docs" / "images" / "SOURCES.md"
TEXT_SUFFIXES = {".py", ".ps1", ".css", ".md", ".yaml", ".yml", ".toml", ".txt", ".csv", ".json", ".jsonl", ".gitignore", ".gitattributes", ".cfg", ".ini"}
LABEL_COLUMNS = {"abnormal", "acl", "meniscus", "label", "labels", "acldiagnosis"}
UID_RE = re.compile(r"\b[0-2](?:\.\d+){5,}\.\d{6,}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
ALLOWED_EMAIL = re.compile(r"(?:users\.noreply\.github\.com|@example\.(?:com|org))$", re.I)
SECRETS = {
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "Kaggle key": re.compile(r"""["']?key["']?\s*[:=]\s*["'][0-9a-f]{32}["']""", re.I),
    "Kaggle token": re.compile(r"\bKGAT_[A-Za-z0-9]{20,}"),
    "Redivis / API token": re.compile(r"\b(?:REDIVIS_API_TOKEN|api_token)\s*[:=]\s*['\"][^'\"\s]{16,}['\"]", re.I),
    "AWS key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "API secret key (sk-...)": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{30,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "password assignment": re.compile(r"""\b(?:password|passwd|secret)\b\s*[:=]\s*["'][^"'\s]{8,}["']""", re.I),
}


def candidate_files() -> list[str]:
    """Every path a `git add .` could commit (read-only git query)."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return sorted(line for line in out.splitlines() if line.strip())


def suffix_of(path: str) -> str:
    name = Path(path).name
    return name if name.startswith(".git") else Path(path).suffix.lower()


def csv_label_columns(text: str) -> set[str]:
    header = text.splitlines()[0] if text else ""
    return {c.strip().lower() for c in header.split(",")} & LABEL_COLUMNS


EVIDENCE_DIR = "evidence/"
PER_EXAM_MARKERS = ("case_id", "exam_id", "volumefilename", ".pck")


def text_problems(path: str, text: str) -> list[str]:
    problems = []
    if path.startswith(EVIDENCE_DIR) and any(m in text.lower() for m in PER_EXAM_MARKERS):
        problems.append(f"{path}: evidence must be aggregate only, but it names individual exams")
    if path.endswith(".csv") and csv_label_columns(text):
        problems.append(f"{path}: has label column(s) {sorted(csv_label_columns(text))}; labels may not be redistributed")
    if UID_RE.search(text):
        problems.append(f"{path}: contains a DICOM-style UID")
    for name, pattern in SECRETS.items():
        if pattern.search(text):
            problems.append(f"{path}: looks like a {name}")
    emails = {m for m in EMAIL_RE.findall(text) if not ALLOWED_EMAIL.search(m)}
    if emails:
        problems.append(f"{path}: e-mail address(es) {[e[:3] + '***' for e in sorted(emails)]}")
    return problems


def file_problems(path: str) -> list[str]:
    full = REPO_ROOT / path
    if not full.exists():          # tracked but deleted in the working tree: nothing to commit
        return []
    is_figure = path.startswith(FIGURE_DIR) and suffix_of(path) in FIGURE_SUFFIXES
    if suffix_of(path) not in TEXT_SUFFIXES and not is_figure:
        return [f"{path}: file type '{suffix_of(path)}' could hold data, images or weights"]
    if full.stat().st_size > MAX_BYTES:
        return [f"{path}: {full.stat().st_size // 1024} KB is over the {MAX_BYTES // 1024} KB limit"]
    if is_figure:
        listed = FIGURE_LIST.exists() and Path(path).name in FIGURE_LIST.read_text(encoding="utf-8")
        if not listed:
            return [f"{path}: README figure not described in docs/images/SOURCES.md"]
        if suffix_of(path) == ".png":
            return []                       # binary; SVGs are text and get the content checks below
    try:
        text = full.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}: not UTF-8 text"]
    return text_problems(path, text)


def main() -> int:
    files = candidate_files()
    problems = [p for f in files for p in file_problems(f)]
    by_folder: dict[str, int] = {}
    for f in files:
        top = f.split("/")[0] if "/" in f else "(root)"
        by_folder[top] = by_folder.get(top, 0) + 1
    print(json.dumps({"files_checked": len(files), "by_top_level": dict(sorted(by_folder.items())),
                      "problems": problems}, indent=1))
    print("AUDIT FAILED" if problems else "AUDIT PASSED: nothing data-derived, licensed or secret would be committed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
