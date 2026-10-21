"""
explain_cases.py: the "explain" stage. It builds the failure-case gallery and runs the heatmap sanity check
for one finished Reader run.

What it does
    1. For each finding: the k most confident false negatives and false positives among out-of-fold predictions
       (threshold = the run's T_hs from results/cv/<run>.json, else 0.5). Each one gets a figure, drawn with the
       fold model that never trained on it.
    2. Writes outputs/gallery/<run>/gallery.md, a table linking every figure, ready for a short note per case.
    3. EVAL_PROTOCOL R13: for 3 positive exams per finding, compares the heatmaps of the trained model with those of a
       randomly initialised one (Spearman). It must be < 0.5, or the heatmaps don't reflect learning.
       The result goes to results/cv/<run>_r13.json.
    Images stay under outputs/, which is git-ignored, because MRNet images may not be redistributed. Only MRNet `train`
    exams are used, never `valid` (the test set).

Usage
    .venv\\Scripts\\python.exe scripts\\explain_cases.py --runs runs\\reader_r18_fbn --k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kneescan import PLANES, TASKS
from kneescan.app.predictor import load_cv_summary
from kneescan.data.mrnet import find_mrnet_root, load_volume
from kneescan.data.transforms import to_25d_input
from kneescan.eval.oof import load_reader_oof
from kneescan.explain.gallery import case_figure, pick_failures
from kneescan.eval.plane_ensemble import read_components
from kneescan.explain.gradcam import randomisation_similarity
from kneescan.models.reader import KneeReader
from kneescan.train.config import load_reader_config
from kneescan.train.reader_trainer import build_reader

REPO_ROOT = Path(__file__).resolve().parents[1]
R13_CASES_PER_TASK = 3
R13_LIMIT = 0.5


def fold_model(run_dir: Path, fold: int, device: torch.device) -> KneeReader:
    model = build_reader(load_reader_config(run_dir / "config.yaml"), pretrained=False)
    model.load_state_dict(torch.load(run_dir / f"fold{fold}" / "best.pt", map_location=device))
    return model.to(device).eval()


def fold_models_by_plane(run_dir: Path, fold: int, device: torch.device) -> dict[str, KneeReader]:
    """plane -> the fold model that reads it (one shared model, or each plane's specialist for a combined run)."""
    components = read_components(run_dir)
    if components is None:
        shared = fold_model(run_dir, fold, device)
        return {p: shared for p in PLANES}
    by_plane = {}
    for component in components:
        for plane in load_reader_config(component / "config.yaml").planes:
            by_plane[plane] = fold_model(component, fold, device)
    return by_plane


def untrained_model(run_dir: Path, device: torch.device) -> KneeReader:
    """A randomly initialised Reader with the same settings, for the randomisation check."""
    components = read_components(run_dir)
    config_dir = run_dir if components is None else components[0]
    torch.manual_seed(0)
    return build_reader(load_reader_config(config_dir / "config.yaml"), pretrained=False).to(device).eval()


def exam_volumes(root: Path, case_id: str) -> dict[str, np.ndarray]:
    return {p: np.array(load_volume(root, "train", p, case_id)) for p in PLANES}


def write_gallery(run_dir: Path, root: Path, out: Path, k: int, device: torch.device) -> list[str]:
    """Draw every selected failure; returns the markdown table rows."""
    oof, summary = load_reader_oof(run_dir), load_cv_summary(run_dir)
    models = {f: fold_models_by_plane(run_dir, f, device) for f in sorted(oof["fold"].unique())}
    rows = []
    for task in TASKS:
        threshold = summary[task]["threshold"]
        for _, case in pick_failures(oof[oof["task"] == task], threshold, k).iterrows():
            name = f"{task}_{case['kind'].replace(' ', '_')}_{case['case_id']}.png"
            title = f"{task}, {case['kind']}: exam {case['case_id']}, p = {case['prob']:.2f} (threshold {threshold:.2f})"
            fig = case_figure(models[int(case["fold"])], exam_volumes(root, case["case_id"]), task, title)
            fig.savefig(out / name, dpi=90)
            fig.clf()
            rows.append(f"| {case['case_id']} | {task} | {case['kind']} | {case['prob']:.2f} | ![]({name}) | _note_ |")
    return rows


def randomisation_check(run_dir: Path, root: Path, device: torch.device) -> dict:
    """R13 on a few OOF-positive exams per finding: mean Spearman(trained map, random-weights map)."""
    oof = load_reader_oof(run_dir)
    untrained = untrained_model(run_dir, device)
    result = {}
    for task in TASKS:
        positives = oof[(oof["task"] == task) & (oof["label"] == 1)].head(R13_CASES_PER_TASK)
        scores = []
        for _, case in positives.iterrows():
            model = fold_models_by_plane(run_dir, int(case["fold"]), device)["sagittal"]
            slices = to_25d_input(torch.from_numpy(exam_volumes(root, case["case_id"])["sagittal"]).to(device))
            scores.append(randomisation_similarity(model, untrained, slices, "sagittal", task, list(range(len(slices)))))
        result[task] = {"mean_spearman": float(np.mean(scores)), "passes": bool(np.mean(scores) < R13_LIMIT)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "mrnet")
    parser.add_argument("--k", type=int, default=3, help="false negatives and false positives per finding")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_mrnet_root(args.data)
    out = REPO_ROOT / "outputs" / "gallery" / args.runs.name
    out.mkdir(parents=True, exist_ok=True)
    rows = write_gallery(args.runs, root, out, args.k, device)
    header = ["# Failure-case gallery: " + args.runs.name, "",
              "Out-of-fold predictions on MRNet train, each explained by the fold model that never saw the exam.", "",
              "| Exam | Finding | Kind | Probability | Figure | Note |", "|---|---|---|---|---|---|"]
    (out / "gallery.md").write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    r13 = randomisation_check(args.runs, root, device)
    results = REPO_ROOT / "results" / "cv"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{args.runs.name}_r13.json").write_text(json.dumps(r13, indent=2), encoding="utf-8")
    print(f"gallery: {out / 'gallery.md'} ({len(rows)} cases)\nR13: {r13}")


if __name__ == "__main__":
    main()
