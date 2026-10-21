"""
pipeline.py: run one stage of the project with one command. prepare → train → evaluate → explain (→ final, once).

What each stage runs (settings in configs/pipeline.yaml):
    prepare   fetch MRNet (your Redivis login), make the frozen CV splits, verify and extract KneeMRI
    train     B0 (5 folds × 3 planes) and every plane specialist of the final model (5 folds each), then combine them
    evaluate  cross-validation report of B0, the final model and the hybrid (final model + B0), each compared with B0
    explain   failure-case gallery and the heatmap randomisation check for the final model
    final     freeze the models, then the ONE-TIME test on MRNet valid + KneeMRI (refuses to run twice)
Every step is an existing script, run in order. A failing step stops the stage. Training steps resume from their checkpoints.

Why a thin wrapper and not a workflow tool (Make, Snakemake)?
    Five stages of a few commands each don't need a new tool to learn. The script reads top to bottom.

Usage
    .venv\\Scripts\\python.exe scripts\\pipeline.py --stage prepare
    .venv\\Scripts\\python.exe scripts\\pipeline.py --stage train
    .venv\\Scripts\\python.exe scripts\\pipeline.py --stage evaluate
    .venv\\Scripts\\python.exe scripts\\pipeline.py --stage explain
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(args: list[str], isolated: bool = False) -> None:
    """Run one project script; stop the stage if it fails."""
    command = [PYTHON] + (["-I"] if isolated else []) + args
    print(">>", " ".join(args), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def component_names(settings: dict) -> list[str]:
    return [yaml.safe_load((REPO_ROOT / c).read_text(encoding="utf-8"))["name"] for c in settings["final_model"]["components"]]


def stage_steps(stage: str, settings: dict) -> list[tuple[list[str], bool]]:
    final = settings["final_model"]["name"]
    hybrid = settings["hybrid_name"]
    runs = [f"runs/{n}" for n in component_names(settings)]
    steps = {
        "prepare": [(["scripts/fetch_mrnet.py"], False)]
                   + ([] if (REPO_ROOT / "splits" / "mrnet_train_5fold.csv").exists()     # splits are frozen once made
                      else [(["scripts/make_splits.py"], False)])
                   + [(["scripts/prepare_kneemri.py"]
                       + (["--skip-extract"] if (REPO_ROOT / "data" / "kneemri" / "volumes").exists() else []), True)],
        "train": [(["scripts/train_b0_cv.py", "--config", settings["baseline_config"]], False)]
                 + [(["scripts/train_reader_cv.py", "--config", c], False) for c in settings["final_model"]["components"]]
                 + [(["scripts/combine_planes.py", "--out", f"runs/{final}"] + runs, False)],
        "evaluate": [(["scripts/score_cv.py", "--runs", "runs/b0", "--change", "B0 (pipeline)"], False),
                     (["scripts/score_cv.py", "--runs", f"runs/{final}", "--compare", "runs/b0", "--change", f"{final} (pipeline)"], False),
                     (["scripts/make_hybrid.py", "--reader", f"runs/{final}", "--baseline", "runs/b0", "--out", f"runs/{hybrid}"], False),
                     (["scripts/score_cv.py", "--runs", f"runs/{hybrid}", "--compare", "runs/b0", "--change", f"{hybrid} (pipeline)"], False)],
        "explain": [(["scripts/explain_cases.py", "--runs", f"runs/{final}", "--k", str(settings["explain_cases_per_finding"])], False)],
        "final": [(["scripts/freeze.py", "--reader", f"runs/{final}", "--baseline", "runs/b0", "--hybrid", f"runs/{hybrid}"], False),
                  (["scripts/final_eval.py"], True)],
    }
    return steps[stage]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "train", "evaluate", "explain", "final"])
    args = parser.parse_args()
    settings = yaml.safe_load((REPO_ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    for script_args, isolated in stage_steps(args.stage, settings):
        run(script_args, isolated)
    print(f"stage '{args.stage}' finished")


if __name__ == "__main__":
    main()
