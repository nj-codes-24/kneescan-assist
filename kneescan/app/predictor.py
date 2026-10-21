"""
predictor.py: everything the app shows for one exam. Probabilities, a confidence level, key slices and
heatmaps, from the 5 cross-validation models of one run used as an ensemble.

What it does
    * Loads the run's *members*. An ordinary run has one member (5 fold models reading all planes). A combined run
      such as R2 (`components.yaml`, kneescan/eval/plane_ensemble.py) has one member per plane specialist, each with
      5 fold models that read only their own plane. Only the specialists whose plane is in the upload vote.
    * Fold k's answer = mean logit of the members' fold-k models (exactly how the run was scored in CV).
    * Loads the run's CV results (`results/cv/<run>.json`).
    * Probability = mean of the 5 fold answers' probabilities, then Platt-calibrated with the calibrator fitted on the run's
      out-of-fold predictions (kneescan/eval/calibration.py), so "0.70" means about 70%.
    * Confidence = how many of the 5 models land on the same side of the operating threshold:
      5/5 agree → "High", 4/5 → "Medium", otherwise "Low". The threshold is the one that reached 90%
      sensitivity on out-of-fold data (EVAL_PROTOCOL §5, T_hs), never chosen on test data.
    * Key slice per plane = highest attention weight, averaged over the 5 models.
    * Heatmap = Grad-CAM on that slice, averaged over the 5 models, drawn over the image.

Why "agreement of 5 models" as confidence?
    It's easy to explain ("five independently trained readers voted 5 to 0"), and it rises exactly where the
    data is ambiguous. A single model's probability can't tell you that on its own.

Alternatives
    Conformal prediction gives confidence sets with a guaranteed error rate, but needs a calibration set
    and is harder to explain to a clinician in one sentence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import torch

from kneescan import TASKS
from kneescan.data.transforms import to_25d_input
from kneescan.eval.calibration import apply_platt
from kneescan.eval.plane_ensemble import read_components
from kneescan.explain.gradcam import gradcam_per_task
from kneescan.models.reader import KneeReader
from kneescan.train.config import load_reader_config
from kneescan.train.reader_trainer import build_reader

DEFAULT_THRESHOLD = 0.5   # used only until the run has been scored (then T_hs from out-of-fold data)


@dataclass
class Heatmap:
    """One explanation image: the key slice of one plane for one finding, with Grad-CAM drawn over it."""
    plane: str
    slice_index: int          # 0-based position in the series
    n_slices: int
    attention: float          # share of the plane's summary that came from this slice (0..1)
    image: np.ndarray         # RGB uint8 overlay

    @property
    def caption(self) -> str:
        return f"{self.plane} slice {self.slice_index + 1}/{self.n_slices} (attention {self.attention:.2f})"


@dataclass
class FindingResult:
    task: str
    probability: float
    fold_probabilities: list[float]
    threshold: float
    votes_positive: int
    confidence: str
    cv_auc: str                                   # e.g. "0.912 (0.89–0.93)" from cross-validation, or "n/a"
    heatmaps: list[Heatmap] = field(default_factory=list)   # one per plane present


def confidence_from_votes(votes_positive: int, n_models: int) -> str:
    """'High' if all models agree, 'Medium' if all but one do, otherwise 'Low'."""
    agreeing = max(votes_positive, n_models - votes_positive)
    if agreeing == n_models:
        return "High"
    return "Medium" if agreeing == n_models - 1 else "Low"


def overlay(slice_uint8: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    """Grey slice + 'jet' coloured heatmap -> RGB uint8. Areas with no evidence stay plain grey."""
    grey = np.repeat(slice_uint8[..., None].astype(np.float32) / 255.0, 3, axis=2)
    colour = matplotlib.colormaps["jet"](heat)[..., :3]
    weight = alpha * heat[..., None]               # colour only where the heatmap is strong
    return ((1 - weight) * grey * 255 + weight * colour * 255).round().astype(np.uint8)


def load_fold_models(run_dir: Path, device: torch.device, slice_batch: int | None = None) -> list[KneeReader]:
    cfg = load_reader_config(run_dir / "config.yaml")
    models = []
    for checkpoint in sorted(run_dir.glob("fold*/best.pt")):
        model = build_reader(cfg, pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.inference_batch = slice_batch
        models.append(model.to(device).eval())
    if not models:
        raise FileNotFoundError(f"no fold*/best.pt under {run_dir}")
    return models


@dataclass
class Member:
    plane: str | None              # the one plane this member reads, or None for a model that reads every plane
    models: list[KneeReader]       # one per CV fold, in fold order

    def inputs(self, planes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """What this member sees of the exam, or None if its plane is missing (then it doesn't vote)."""
        if self.plane is None:
            return planes
        return {self.plane: planes[self.plane]} if self.plane in planes else None


def load_members(run_dir: Path, device: torch.device, slice_batch: int | None = None) -> list[Member]:
    components = read_components(run_dir)
    if components is None:
        return [Member(None, load_fold_models(run_dir, device, slice_batch))]
    members = []
    for component in components:
        planes = load_reader_config(component / "config.yaml").planes
        members.append(Member(planes[0] if len(planes) == 1 else None, load_fold_models(component, device, slice_batch)))
    if len({len(m.models) for m in members}) != 1:
        raise ValueError("all members of a combined run need the same number of fold models")
    return members


def load_cv_summary(run_dir: Path) -> dict[str, dict]:
    """Per finding: threshold (T_hs), CV AUC text and Platt calibrator. Before scoring: 0.5, 'n/a', no calibration."""
    path = run_dir.parents[1] / "results" / "cv" / f"{run_dir.name}.json"
    if not path.exists():
        return {t: {"threshold": DEFAULT_THRESHOLD, "cv_auc": "n/a", "platt": None} for t in TASKS}
    report = json.loads(path.read_text(encoding="utf-8"))
    per_task, calibration = report["per_task"], report["calibration"]
    return {t: {"threshold": per_task[t]["t_hs"],
                "cv_auc": f"{per_task[t]['auc']:.3f} ({per_task[t]['auc_ci'][0]:.3f}–{per_task[t]['auc_ci'][1]:.3f})",
                "platt": (calibration[t]["platt_a"], calibration[t]["platt_b"])}
            for t in TASKS}


class Predictor:
    def __init__(self, run_dir: str | Path, heatmap_alpha: float = 0.45, device: str | None = None,
                 slice_batch: int | None = None) -> None:
        has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0     # a hidden GPU can pass the first check
        self.device = torch.device(device or ("cuda" if has_gpu else "cpu"))
        self.run_dir = Path(run_dir)
        self.members = load_members(self.run_dir, self.device, slice_batch)
        self.n_folds = len(self.members[0].models)
        self.summary = load_cv_summary(self.run_dir)
        self.alpha = heatmap_alpha

    def models_for(self, plane: str) -> list[KneeReader]:
        """Every fold model that reads `plane` (used for that plane's heatmaps)."""
        return [m for member in self.members if member.plane in (None, plane) for m in member.models]

    @torch.no_grad()
    def _forward_all(self, planes: dict[str, torch.Tensor]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """(n_folds, n_tasks) probabilities and plane -> (n_tasks, S) attention averaged over the models."""
        active = [(member, member.inputs(planes)) for member in self.members if member.inputs(planes) is not None]
        fold_probs, attention_parts = [], {p: [] for p in planes}
        for fold in range(self.n_folds):
            outputs = [member.models[fold](inputs) for member, inputs in active]
            mean_logits = torch.stack([o.logits.float() for o in outputs]).mean(dim=0)
            fold_probs.append(torch.sigmoid(mean_logits).cpu().numpy())
            for o in outputs:
                for plane, weights in o.attention.items():
                    attention_parts[plane].append(weights.float().cpu().numpy())
        attention = {p: np.mean(parts, axis=0) for p, parts in attention_parts.items() if parts}
        return np.stack(fold_probs), attention

    def _heatmaps(self, volumes: dict[str, np.ndarray], planes: dict[str, torch.Tensor],
                  attention: dict[str, np.ndarray]) -> dict[str, list[Heatmap]]:
        """Per task, one overlay per plane on the slice with the highest attention for that task (one backbone pass per model)."""
        images: dict[str, list[Heatmap]] = {task: [] for task in TASKS}
        for plane, weights in attention.items():
            key = {task: [int(np.argmax(weights[t]))] for t, task in enumerate(TASKS)}
            per_model = [gradcam_per_task(m, planes[plane], plane, key) for m in self.models_for(plane)]
            for t, task in enumerate(TASKS):
                index = key[task][0]
                heat = np.mean([maps[task][0] for maps in per_model], axis=0)
                images[task].append(Heatmap(plane, index, weights.shape[1], float(weights[t, index]),
                                            overlay(volumes[plane][index], heat, self.alpha)))
        return images

    def _calibrated(self, task: str, probability: float) -> float:
        platt = self.summary[task]["platt"]
        return probability if platt is None else float(apply_platt(np.array([probability]), *platt)[0])

    def fold_probabilities(self, volumes: dict[str, np.ndarray]) -> np.ndarray:
        """(n_folds, n_tasks) raw (uncalibrated) probabilities; used by the one-time final evaluation."""
        planes = {p: to_25d_input(torch.from_numpy(v).to(self.device)) for p, v in volumes.items()}
        return self._forward_all(planes)[0]

    def predict(self, volumes: dict[str, np.ndarray], with_heatmaps: bool = True) -> list[FindingResult]:
        """volumes: plane -> uint8 (S, 256, 256), one to three planes."""
        planes = {p: to_25d_input(torch.from_numpy(v).to(self.device)) for p, v in volumes.items()}
        probs, attention = self._forward_all(planes)
        heatmaps = self._heatmaps(volumes, planes, attention) if with_heatmaps else {task: [] for task in TASKS}
        results = []
        for t, task in enumerate(TASKS):
            threshold = self.summary[task]["threshold"]           # on the raw scale, like the fold probabilities
            votes = int((probs[:, t] >= threshold).sum())
            results.append(FindingResult(
                task=task,
                probability=self._calibrated(task, float(probs[:, t].mean())),
                fold_probabilities=probs[:, t].round(3).tolist(),
                threshold=self._calibrated(task, threshold),     # shown on the same scale as the probability
                votes_positive=votes,
                confidence=confidence_from_votes(votes, self.n_folds),
                cv_auc=self.summary[task]["cv_auc"],
                heatmaps=heatmaps[task]))
        return results
