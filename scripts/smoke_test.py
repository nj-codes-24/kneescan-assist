"""
smoke_test.py: run the WHOLE pipeline end to end on a tiny synthetic dataset, in a few minutes, on CPU or GPU.

Stages (the same library code the real scripts use):
    1. prepare  : write a fake MRNet (80 exams, 64 × 64 px) and make the 5 CV folds + inner tuning split
    2. train    : train the Reader for 1 epoch on 2 of the folds (checkpoint, resume logic and preds.csv included)
    3. evaluate : macro AUC with CI, per-finding metrics, calibration, and a paired comparison of the run with itself
    4. explain  : pick failure cases, draw one gallery figure, run the heatmap randomisation check
    5. app      : load the trained fold models in the app's Predictor and analyse one exam
Everything is written to a temporary folder that is deleted at the end. Real runs, results and logs are untouched.

Numbers printed here are meaningless (random pixels, random labels). The point is that nothing crashes.

Usage
    .venv\\Scripts\\python.exe scripts\\smoke_test.py
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from kneescan import PLANES, TASKS  # noqa: E402
from kneescan.app.predictor import Predictor  # noqa: E402
from kneescan.data.mrnet import read_labels  # noqa: E402
from kneescan.data.splits import make_cv_folds, make_inner_tune, split_for  # noqa: E402
from kneescan.data.synthetic import make_fake_mrnet  # noqa: E402
from kneescan.data.transforms import to_25d_input  # noqa: E402
from kneescan.eval.cv_report import compare, score  # noqa: E402
from kneescan.eval.oof import load_reader_oof  # noqa: E402
from kneescan.explain.gallery import case_figure, pick_failures  # noqa: E402
from kneescan.explain.gradcam import randomisation_similarity  # noqa: E402
from kneescan.train.config import ReaderConfig  # noqa: E402
from kneescan.train.reader_trainer import ReaderFoldJob, build_reader  # noqa: E402

SMOKE_FOLDS = (0, 1)


def stage(name: str, started: float) -> None:
    print(f"[{time.time() - started:6.1f} s] {name}", flush=True)


def tiny_config() -> ReaderConfig:
    return dataclasses.replace(ReaderConfig(), name="smoke", pretrained=False, att_dim=16, epochs=1,
                               accumulate_exams=4, num_workers=0)


def train_folds(cfg: ReaderConfig, root: Path, run_dir: Path) -> None:
    labels = read_labels(root, "train")
    folds = make_cv_folds(labels)
    inner = make_inner_tune(labels, folds)
    (run_dir).mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg)), encoding="utf-8")
    for fold in SMOKE_FOLDS:
        ReaderFoldJob(cfg, root, labels, split_for(fold, folds, inner), fold, run_dir / f"fold{fold}").run()


def explain(cfg: ReaderConfig, root: Path, run_dir: Path, out: Path) -> None:
    oof = load_reader_oof(run_dir)
    failures = pick_failures(oof[oof["task"] == "acl"], threshold=0.5, k=1)
    case = failures.iloc[0] if len(failures) else oof.iloc[0]
    model = build_reader(cfg, pretrained=False)
    model.load_state_dict(torch.load(run_dir / f"fold{int(case['fold'])}" / "best.pt", map_location="cpu"))
    volumes = {p: np.load(root / "train" / p / f"{case['case_id']}.npy") for p in PLANES}
    case_figure({p: model for p in PLANES}, volumes, "acl", "smoke test").savefig(out / "smoke_case.png", dpi=60)
    untrained = build_reader(cfg, pretrained=False).eval()
    slices = to_25d_input(torch.from_numpy(volumes["sagittal"]))
    print("      R13 similarity (meaningless on random data):",
          round(randomisation_similarity(model, untrained, slices, "sagittal", "acl", [0, 1]), 3))


def main() -> int:
    started = time.time()
    os.environ["KNEESCAN_LOG_DIR"] = tempfile.mkdtemp()      # keep the real project diary clean
    work = Path(tempfile.mkdtemp(prefix="kneescan_smoke_"))
    try:
        root = make_fake_mrnet(work / "data", n_train=80, n_valid=4, hw=64)   # 80: the smallest size the inner tuning split accepts
        stage("1 prepare: fake MRNet + folds", started)
        cfg, run_dir = tiny_config(), work / "runs" / "smoke"
        train_folds(cfg, root, run_dir)
        stage(f"2 train: Reader, folds {SMOKE_FOLDS}, 1 epoch each", started)
        report = score(run_dir)
        same = compare(run_dir, run_dir)
        print(f"      macro AUC {report['macro_auc']:.3f}; gain vs itself {same['gain']:+.3f} (must be +0.000)")
        stage("3 evaluate: macro AUC, per-finding metrics, calibration, paired comparison", started)
        explain(cfg, root, run_dir, work)
        stage("4 explain: failure pick, gallery figure, randomisation check", started)
        results = Predictor(run_dir, device="cpu").predict({"sagittal": np.load(root / "train" / "sagittal" / "0000.npy")})
        print("      app:", {r.task: (round(r.probability, 2), r.confidence) for r in results})
        assert [r.task for r in results] == list(TASKS)
        stage("5 app: Predictor on one exam", started)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
