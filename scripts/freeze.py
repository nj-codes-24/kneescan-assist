"""
freeze.py: record EXACTLY what goes into the one-time final test, before it runs (EVAL_PROTOCOL R14).

What it does
    Writes docs/frozen.yaml (read by scripts/final_eval.py) and docs/FROZEN.md (for people) with:
      * the two models compared (R8): the baseline B0 and the chosen Reader run (an ordinary run or a combined one such as R2);
      * the SHA-256 of every weight file they use. final_eval.py refuses to run if any file has changed;
      * each model's frozen operating threshold (T_hs, 90% sensitivity on out-of-fold data) and Platt calibrator per finding,
        copied from results/cv/<run>.json;
      * each model's CV macro AUC, the date, and the git commit (read-only `git rev-parse`).
    It refuses to overwrite an existing freeze once results/final/LOCK exists (the test has run).

Usage
    .venv\\Scripts\\python.exe scripts\\freeze.py --reader runs\\reader_r2 --baseline runs\\b0
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from kneescan import TASKS
from kneescan.eval.plane_ensemble import read_components

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_YAML = REPO_ROOT / "docs" / "frozen.yaml"
FROZEN_MD = REPO_ROOT / "docs" / "FROZEN.md"
LOCK = REPO_ROOT / "results" / "final" / "LOCK"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_files(run_dir: Path) -> list[Path]:
    """Every checkpoint a run's predictions use (B0: 45 task CNNs; Reader: fold models of the run or of its components)."""
    if (run_dir / "fold0" / "sagittal" / "best_acl.pt").exists():           # B0 layout
        return sorted(run_dir.glob("fold*/*/best_*.pt"))
    components = read_components(run_dir) or [run_dir]
    return sorted(p for c in components for p in c.glob("fold*/best.pt"))


def model_entry(run_dir: Path) -> dict:
    report = json.loads((REPO_ROOT / "results" / "cv" / f"{run_dir.name}.json").read_text(encoding="utf-8"))
    files = weight_files(run_dir)
    return {"run": str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            "cv_macro_auc": round(report["macro_auc"], 4),
            "cv_macro_auc_ci": [round(v, 4) for v in report["macro_auc_ci"]],
            "thresholds_t_hs": {t: float(report["per_task"][t]["t_hs"]) for t in TASKS},
            "platt": {t: [float(report["calibration"][t]["platt_a"]), float(report["calibration"][t]["platt_b"])] for t in TASKS},
            "weights": {str(f.relative_to(REPO_ROOT)).replace("\\", "/"): sha256(f) for f in files}}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip() + (" (+ uncommitted changes)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def markdown(frozen: dict) -> str:
    lines = ["# FROZEN: what the one-time final test evaluates (EVAL_PROTOCOL R14)", "",
             f"Frozen {frozen['frozen_at']} on pc_c, git {frozen['git']}. Machine-readable copy: `docs/frozen.yaml`.", ""]
    for role, entry in frozen["models"].items():
        lines += [f"## {role}: `{entry['run']}`", "",
                  f"- CV macro AUC {entry['cv_macro_auc']} (95% CI {entry['cv_macro_auc_ci'][0]}–{entry['cv_macro_auc_ci'][1]})",
                  (f"- {len(entry['weights'])} weight files, SHA-256 listed in frozen.yaml" if entry["weights"]
                   else "- no weights of its own: the mean logit of the baseline and the reader above"),
                  "- Operating thresholds T_hs (90% sensitivity on OOF): "
                  + ", ".join(f"{t} {v:.3f}" for t, v in entry["thresholds_t_hs"].items()), ""]
    lines += ["The test sets (MRNet `valid`, KneeMRI) had not been used by any model before this freeze.",
              "\"uncommitted changes\": the code was committed after the freeze, so the weight hashes above, not a",
              "git commit, are the binding record of what was tested."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=REPO_ROOT / "runs" / "b0")
    parser.add_argument("--hybrid", type=Path, default=None,
                        help="optional pre-registered extra variant (R8): the OOF run made by make_hybrid.py from --reader and --baseline")
    args = parser.parse_args()
    if LOCK.exists():
        raise SystemExit("the final test has already run (results/final/LOCK); the freeze can't change any more")
    frozen = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"), "git": git_commit(),
              "models": {"baseline": model_entry(args.baseline.resolve()), "reader": model_entry(args.reader.resolve())}}
    if args.hybrid is not None:      # no weights of its own: it is the mean logit of the two frozen models above
        frozen["models"]["hybrid"] = model_entry(args.hybrid.resolve())
    FROZEN_YAML.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    FROZEN_MD.write_text(markdown(frozen), encoding="utf-8")
    print(markdown(frozen))


if __name__ == "__main__":
    main()
