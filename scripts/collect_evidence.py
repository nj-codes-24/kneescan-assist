"""
collect_evidence.py: copy the ORIGINAL outputs behind every reported number into evidence/, so readers can check them.

What it does
    Copies, unchanged, into evidence/ (committed to git):
      evidence/final_test/   final.json + final.md written by scripts/final_eval.py during the one-time test, and LOCK.txt
                             (the lock file: when that run started and finished)
      evidence/cv/           every cross-validation report (results/cv/*.json and *.md): AUCs with CIs, calibration, thresholds
      evidence/training_logs/<run>/fold<k>/   each training run's per-epoch log (metrics.csv: losses, tuning AUC, seconds per
                             epoch) and done.json (best epoch, config fingerprint, finish time)
    Then writes evidence/SHA256SUMS.txt (a fingerprint of every file) and refuses to copy anything per-exam.

What stays private, and why
    Per-exam predictions (runs/*/fold*/preds.csv) and anything with case IDs: together with public IDs they would reveal the
    dataset labels, which the MRNet and KneeMRI licences don't let us redistribute. Weights stay private (D-025, D-027); their
    SHA-256s are already public in docs/frozen.yaml. Images aren't evidence of numbers and may show scans.

Alternatives
    With redistributable data you'd publish the per-exam predictions too, so anyone could recompute every AUC from scratch.

Usage (from the repo root)
    .venv\\Scripts\\python.exe scripts\\collect_evidence.py
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "evidence"
PER_EXAM_MARKERS = ("case_id", "exam_id", "volumefilename", ".pck")   # never in a published file
TRAINING_LOG_NAMES = ("metrics.csv", "done.json")


def has_per_exam_data(path: Path) -> bool:
    text = path.read_text(encoding="utf-8").lower()
    return any(marker in text for marker in PER_EXAM_MARKERS)


def sources(repo_root: Path) -> dict[Path, Path]:
    """{source file: destination relative to evidence/} for everything that gets published."""
    pairs = {repo_root / "results" / "final" / name: Path("final_test") / name for name in ("final.json", "final.md")}
    pairs[repo_root / "results" / "final" / "LOCK"] = Path("final_test") / "LOCK.txt"
    for report in sorted((repo_root / "results" / "cv").glob("*")):
        if report.suffix in (".json", ".md"):
            pairs[report] = Path("cv") / report.name
    for log in sorted((repo_root / "runs").rglob("*")):
        if log.name in TRAINING_LOG_NAMES:
            pairs[log] = Path("training_logs") / log.relative_to(repo_root / "runs")
    return pairs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(repo_root: Path = REPO_ROOT, dest: Path = EVIDENCE) -> list[str]:
    pairs = sources(repo_root)
    leaking = [str(src) for src in pairs if has_per_exam_data(src)]
    if leaking:
        raise ValueError(f"refusing to publish per-exam data: {leaking[:3]}")
    for sub in ("final_test", "cv", "training_logs"):
        shutil.rmtree(dest / sub, ignore_errors=True)            # only our own generated sub-folders
    for src, rel in pairs.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / rel)
    lines = [f"{sha256(dest / rel)}  {rel.as_posix()}" for rel in sorted(pairs.values())]
    (dest / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return lines


def main() -> None:
    lines = collect()
    print(f"evidence/: {len(lines)} files copied unchanged; fingerprints in evidence/SHA256SUMS.txt")


if __name__ == "__main__":
    main()
