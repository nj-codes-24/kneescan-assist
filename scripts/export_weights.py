"""
export_weights.py: copy ONLY what the demo app needs out of runs/ into one small folder, checked against the frozen fingerprints.

What it does
    Reads the final Reader from docs/frozen.yaml and writes export/weights/ (git-ignored) with the repo's own layout:
      runs/reader_r2_seeds/components.yaml    which plane specialists make up the final model (paths rewritten with "/")
      runs/<specialist>/config.yaml           how to rebuild each network
      runs/<specialist>/fold<k>/best.pt       the 30 tested weight files (6 specialists x 5 folds)
      results/cv/reader_r2_seeds.json         thresholds, Platt calibration and CV AUCs the app shows
      MANIFEST.yaml                           SHA-256 of every file in the folder
    Every best.pt must match the SHA-256 recorded in docs/frozen.yaml before the final test, or the export stops.
    Resume checkpoints (last.pt), TensorBoard logs and per-exam predictions stay behind: the app doesn't need them.

Why a separate folder?
    It's what goes to a private Hugging Face model repo (and your backup drive) when this PC is handed back. The fingerprint
    check proves the copy is exactly the model that was tested, not a later or half-trained one.

Alternatives
    For a model you may redistribute openly, you'd also convert to safetensors and publish a model card. These weights are
    trained on MRNet, so they stay private (docs/decisions.md D-025).

Usage (from the repo root)
    .venv\\Scripts\\python.exe scripts\\export_weights.py              # -> export\\weights
    .venv\\Scripts\\python.exe scripts\\export_weights.py --dest E:\\backup\\kneescan-weights
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kneescan.eval.plane_ensemble import COMPONENTS_FILE, read_components  # noqa: E402

FROZEN_YAML = REPO_ROOT / "docs" / "frozen.yaml"
README_TEXT = """# KneeScan Assist: trained weights (private)

The final Reader of KneeScan Assist: 6 plane specialists x 5 cross-validation folds, frozen on 2026-10-20 before the
one-time final test. Code: https://github.com/nj-codes-24/kneescan-assist

Trained on MRNet (Stanford AIMI) under its Research Use Agreement: personal, non-commercial research only.
Keep this repository PRIVATE and don't redistribute the files. Research prototype, not a medical device.
Every file's SHA-256 is in MANIFEST.yaml; the weight files match docs/frozen.yaml in the code repository.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_reader(frozen_yaml: Path = FROZEN_YAML) -> tuple[str, dict[str, str]]:
    """(run folder of the final Reader, {relative weight path: SHA-256}) as frozen before the final test."""
    reader = yaml.safe_load(frozen_yaml.read_text(encoding="utf-8"))["models"]["reader"]
    return reader["run"], reader["weights"]


def files_to_copy(repo_root: Path, run: str) -> list[str]:
    """Relative paths (with "/") of everything the app loads for `run`, except the rewritten components.yaml."""
    run_dir = repo_root / run
    components = read_components(run_dir) or [run_dir]
    files = []
    for component in components:
        rel = component.relative_to(repo_root).as_posix()
        files.append(f"{rel}/config.yaml")
        files += [f"{rel}/{p.parent.name}/best.pt" for p in sorted(component.glob("fold*/best.pt"))]
    files.append(f"results/cv/{run_dir.name}.json")
    return files


def check_against_frozen(repo_root: Path, files: list[str], frozen: dict[str, str]) -> None:
    """Every frozen weight file is exported, unchanged. Raises ValueError otherwise."""
    weights = [f for f in files if f.endswith(".pt")]
    if sorted(weights) != sorted(frozen):
        raise ValueError(f"weights on disk ({len(weights)}) don't match the frozen list ({len(frozen)})")
    changed = [f for f in weights if sha256(repo_root / f) != frozen[f]]
    if changed:
        raise ValueError(f"{len(changed)} weight file(s) changed since the freeze, e.g. {changed[0]}")


def export(repo_root: Path, dest: Path, frozen_yaml: Path) -> dict[str, str]:
    """Copy, rewrite components.yaml, write README and MANIFEST. Returns the manifest {path: SHA-256}."""
    run, frozen = frozen_reader(frozen_yaml)
    files = files_to_copy(repo_root, run)
    check_against_frozen(repo_root, files, frozen)
    for rel in files:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / rel, dest / rel)
    components = read_components(repo_root / run)
    if components is not None:
        names = [c.relative_to(repo_root).as_posix() for c in components]
        source = yaml.safe_load((repo_root / run / COMPONENTS_FILE).read_text(encoding="utf-8"))
        (dest / run).mkdir(parents=True, exist_ok=True)
        (dest / run / COMPONENTS_FILE).write_text(yaml.safe_dump({**source, "components": names}, sort_keys=False),
                                                  encoding="utf-8")
    (dest / "README.md").write_text(README_TEXT, encoding="utf-8")
    manifest = {p.relative_to(dest).as_posix(): sha256(p) for p in sorted(dest.rglob("*"))
                if p.is_file() and p.name != "MANIFEST.yaml"}
    (dest / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "export" / "weights")
    args = parser.parse_args()
    if args.dest.exists() and any(args.dest.iterdir()):
        sys.exit(f"{args.dest} is not empty; choose another --dest or empty it first")
    manifest = export(REPO_ROOT, args.dest, FROZEN_YAML)
    size_mb = sum((args.dest / f).stat().st_size for f in manifest) / 1e6
    print(f"exported {len(manifest)} files ({size_mb:.0f} MB) to {args.dest}; all weights match docs/frozen.yaml")


if __name__ == "__main__":
    main()
