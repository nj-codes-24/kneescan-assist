"""
b0_trainer.py: train the B0 baseline for one (fold, plane) job. It checkpoints every epoch,
can be resumed, and can pause on schedule.

What it does
    One *job* = one CV fold x one plane. It trains the three task models of that plane
    (abnormal, ACL, meniscus), each with its own weights, optimiser and LR schedule, as in
    Bien et al. To save time, all three see the same augmented copy of each exam: the
    volume is loaded and augmented once and used three times. That is a speed choice, not
    a modelling change, and it is logged.

    Per epoch:
      1. train on the fold's training cases (shuffled with seed + epoch, so the order is
         reproducible after a resume)
      2. evaluate on the inner tuning cases (loss + AUC per task)
      3. step each task's ReduceLROnPlateau on its tuning loss
      4. save best_<task>.pt when that task's tuning loss improves (= epoch selection, R4)
      5. save last.pt with EVERYTHING needed to continue: weights, optimisers, schedulers,
         AMP scalers, RNG states, history. It is written atomically, so a crash mid-save
         can't corrupt it.
      6. if --stop-at is near (the next epoch wouldn't finish in time): stop cleanly
    After the last epoch, `predict_job` writes probabilities for the train, tune and
    OOF cases. It never touches MRNet `valid` (the final test set).

Why per-epoch checkpoints?
    The GPU was shared (only free at night after the first few days), so every run must
    survive being stopped and continued without changing its result.

Alternatives
    PyTorch Lightning gives checkpoint/resume for free, but hides the loop. This is a
    beginner-readable project, so the loop stays explicit.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from kneescan import PLANES, TASKS
from kneescan.data.mrnet import MRNetPlaneDataset
from kneescan.data.splits import FoldSplit
from kneescan.data.transforms import random_affine, to_network_input
from kneescan.eval.metrics import auc
from kneescan.models.baseline_mrnet import MRNetBaseline
from kneescan.train.config import B0Config, config_hash
from kneescan.train.run_logger import RunLogger, append_lab_log
from kneescan.train.seed import capture_rng_state, restore_rng_state, seed_everything

DONE_FILE = "done.json"
LAST_CKPT = "last.pt"


# ---------------------------------------------------------------- scheduling helpers
def compute_deadline(stop_at: str | None, now: dt.datetime) -> dt.datetime | None:
    """'08:00' -> the next 08:00 after `now` (today or tomorrow). None means no deadline."""
    if not stop_at:
        return None
    hour, minute = (int(v) for v in stop_at.split(":"))
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return deadline if deadline > now else deadline + dt.timedelta(days=1)


def should_pause(now: dt.datetime, deadline: dt.datetime | None, last_epoch_seconds: float) -> bool:
    """Pause if another epoch (plus a 10% margin) would run past the deadline."""
    if deadline is None:
        return False
    return now + dt.timedelta(seconds=1.1 * last_epoch_seconds) > deadline


# ---------------------------------------------------------------- loss
def class_weights(labels: np.ndarray) -> torch.Tensor:
    """(3, 2) tensor of [w_negative, w_positive] per task; rarer class gets the larger weight."""
    n = len(labels)
    pos = labels.sum(axis=0).clip(min=1)
    neg = (n - labels.sum(axis=0)).clip(min=1)
    return torch.tensor(np.stack([n / (2 * neg), n / (2 * pos)], axis=1), dtype=torch.float32)


def weighted_bce(logit: torch.Tensor, target: torch.Tensor, w_neg_pos: torch.Tensor) -> torch.Tensor:
    """BCE scaled inversely to class prevalence (Bien et al. 2018)."""
    weight = w_neg_pos[1] if float(target) > 0.5 else w_neg_pos[0]
    return weight * F.binary_cross_entropy_with_logits(logit, target)


class EpochOrderSampler(Sampler[int]):
    """Yields indices in an order we set before each epoch (works with persistent workers)."""

    def __init__(self, n: int) -> None:
        self.order = list(range(n))

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


# ---------------------------------------------------------------- the job
class PlaneJob:
    """Trains abnormal/ACL/meniscus B0 models for one (fold, plane)."""

    def __init__(self, cfg: B0Config, data_root: Path, labels: pd.DataFrame, split: FoldSplit,
                 fold: int, plane: str, run_dir: Path, device: str | None = None) -> None:
        self.cfg, self.fold, self.plane = cfg, fold, plane
        self.run_dir = Path(run_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.seed = cfg.seed + 1000 * fold + PLANES.index(plane)
        seed_everything(self.seed, cfg.deterministic)

        self.train_ds = MRNetPlaneDataset(data_root, "train", plane, labels, split.train)
        self.tune_ds = MRNetPlaneDataset(data_root, "train", plane, labels, split.tune)
        self.oof_ds = MRNetPlaneDataset(data_root, "train", plane, labels, split.oof)
        self.weights = class_weights(self.train_ds.labels).to(self.device)

        self.models = {t: MRNetBaseline(cfg.pretrained).to(self.device) for t in TASKS}
        self.opts = {t: torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                     for t, m in self.models.items()}
        self.scheds = {t: torch.optim.lr_scheduler.ReduceLROnPlateau(
            o, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
            for t, o in self.opts.items()}
        self.scalers = {t: torch.amp.GradScaler("cuda", enabled=self.use_amp) for t in TASKS}
        self.aug_gen = torch.Generator().manual_seed(self.seed)

        self.next_epoch = 0
        self.best = {t: {"tune_loss": float("inf"), "epoch": -1, "tune_auc": None} for t in TASKS}
        self.last_epoch_seconds = 0.0
        self.sampler = EpochOrderSampler(len(self.train_ds))
        self.logger = RunLogger(self.run_dir)

    # ---------------- checkpointing
    def _state(self) -> dict:
        return {
            "next_epoch": self.next_epoch, "best": self.best, "last_epoch_seconds": self.last_epoch_seconds,
            "models": {t: m.state_dict() for t, m in self.models.items()},
            "opts": {t: o.state_dict() for t, o in self.opts.items()},
            "scheds": {t: s.state_dict() for t, s in self.scheds.items()},
            "scalers": {t: s.state_dict() for t, s in self.scalers.items()},
            "rng": capture_rng_state({"aug": self.aug_gen}),
            "config": dataclasses.asdict(self.cfg), "config_hash": config_hash(self.cfg),
        }

    def save_last(self) -> None:
        tmp = self.run_dir / (LAST_CKPT + ".tmp")
        torch.save(self._state(), tmp)
        os.replace(tmp, self.run_dir / LAST_CKPT)  # atomic: either old or new file, never half

    def load_last(self) -> None:
        state = torch.load(self.run_dir / LAST_CKPT, map_location=self.device, weights_only=False)
        if state["config_hash"] != config_hash(self.cfg):
            raise RuntimeError("checkpoint was made with a different config; refusing to resume")
        for t in TASKS:
            self.models[t].load_state_dict(state["models"][t])
            self.opts[t].load_state_dict(state["opts"][t])
            self.scheds[t].load_state_dict(state["scheds"][t])
            self.scalers[t].load_state_dict(state["scalers"][t])
        self.next_epoch, self.best = state["next_epoch"], state["best"]
        self.last_epoch_seconds = state["last_epoch_seconds"]
        restore_rng_state(state["rng"], {"aug": self.aug_gen})
        self.logger.drop_rows_after(self.next_epoch - 1)

    # ---------------- one epoch
    def _loader(self, ds, sampler=None) -> DataLoader:
        return DataLoader(ds, batch_size=None, sampler=sampler, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=self.device.type == "cuda",
                          persistent_workers=False)

    def _prepare(self, item: dict, augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
        vol = item["volume"].to(self.device, non_blocking=True)
        if augment:
            vol = random_affine(vol.float(), self.cfg.rotate_deg, self.cfg.shift_px, self.cfg.flip_p, self.aug_gen)
        return to_network_input(vol), item["labels"].to(self.device)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        for m in self.models.values():
            m.train()
        self.sampler.order = np.random.default_rng(self.seed + epoch).permutation(len(self.train_ds)).tolist()
        totals = {t: 0.0 for t in TASKS}
        for item in self._loader(self.train_ds, self.sampler):
            x, y = self._prepare(item, augment=True)
            for i, task in enumerate(TASKS):
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    logit = self.models[task](x)
                loss = weighted_bce(logit.float(), y[i], self.weights[i])
                self.opts[task].zero_grad(set_to_none=True)
                self.scalers[task].scale(loss).backward()
                self.scalers[task].step(self.opts[task])
                self.scalers[task].update()
                totals[task] += float(loss.detach())
        return {t: v / len(self.train_ds) for t, v in totals.items()}

    @torch.no_grad()
    def predict(self, ds: MRNetPlaneDataset, models: dict | None = None) -> pd.DataFrame:
        """Probabilities and weighted losses for every case in `ds` (no augmentation)."""
        models = models or self.models
        for m in models.values():
            m.eval()
        rows = []
        for item in self._loader(ds):
            x, y = self._prepare(item, augment=False)
            for i, task in enumerate(TASKS):
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    logit = models[task](x).float()
                rows.append({"case_id": item["case_id"], "task": task, "label": int(y[i]),
                             "prob": float(torch.sigmoid(logit)),
                             "loss": float(weighted_bce(logit, y[i], self.weights[i]))})
        return pd.DataFrame(rows)

    def evaluate_tune(self) -> dict[str, dict[str, float | None]]:
        df = self.predict(self.tune_ds)
        out = {}
        for task, g in df.groupby("task"):
            two_classes = g["label"].nunique() == 2
            out[task] = {"loss": float(g["loss"].mean()), "auc": auc(g["label"], g["prob"]) if two_classes else None}
        return out

    # ---------------- the main loop
    def run(self, resume: bool = True, stop_at: str | None = None,
            max_epochs_this_call: int | None = None) -> str:
        """Train until all epochs are done ('done') or the schedule says stop ('paused')."""
        if (self.run_dir / DONE_FILE).exists():
            return "done"
        if (self.run_dir / LAST_CKPT).exists():
            if not resume:
                raise RuntimeError(f"{self.run_dir} has a checkpoint; pass resume=True or delete the folder")
            self.load_last()
        deadline = compute_deadline(stop_at, dt.datetime.now())
        append_lab_log("b0_job_start", fold=self.fold, plane=self.plane, start_epoch=self.next_epoch,
                       config_hash=config_hash(self.cfg), device=str(self.device), run_dir=str(self.run_dir))
        epochs_this_call = 0
        while self.next_epoch < self.cfg.epochs:
            epoch, t0 = self.next_epoch, time.time()
            train_loss = self.train_one_epoch(epoch)
            tune = self.evaluate_tune()
            self.last_epoch_seconds = time.time() - t0
            for task in TASKS:
                self.scheds[task].step(tune[task]["loss"])
                is_best = tune[task]["loss"] < self.best[task]["tune_loss"]
                if is_best:
                    self.best[task] = {"tune_loss": tune[task]["loss"], "epoch": epoch, "tune_auc": tune[task]["auc"]}
                    torch.save(self.models[task].state_dict(), self.run_dir / f"best_{task}.pt")
                self.logger.log_epoch({"epoch": epoch, "task": task, "train_loss": train_loss[task],
                                       "tune_loss": tune[task]["loss"], "tune_auc": tune[task]["auc"],
                                       "lr": self.opts[task].param_groups[0]["lr"],
                                       "seconds": round(self.last_epoch_seconds, 1), "is_best": is_best})
            self.next_epoch = epoch + 1
            self.save_last()
            epochs_this_call += 1
            if self.next_epoch >= self.cfg.epochs:
                break
            if max_epochs_this_call is not None and epochs_this_call >= max_epochs_this_call:
                return "paused"
            if should_pause(dt.datetime.now(), deadline, self.last_epoch_seconds):
                append_lab_log("b0_job_paused", fold=self.fold, plane=self.plane, next_epoch=self.next_epoch)
                return "paused"
        self._finish()
        return "done"

    def _finish(self) -> None:
        """Predict train/tune/OOF with the best checkpoint of each task, then mark the job done."""
        best_models = {}
        for task in TASKS:
            m = MRNetBaseline(pretrained=False).to(self.device)
            m.load_state_dict(torch.load(self.run_dir / f"best_{task}.pt", map_location=self.device))
            best_models[task] = m
        parts = []
        for subset, ds in (("train", self.train_ds), ("tune", self.tune_ds), ("oof", self.oof_ds)):
            df = self.predict(ds, best_models)
            df.insert(0, "subset", subset)
            parts.append(df)
        preds = pd.concat(parts, ignore_index=True)
        preds.insert(0, "plane", self.plane)
        preds.insert(0, "fold", self.fold)
        preds.drop(columns="loss").to_csv(self.run_dir / "preds.csv", index=False)
        summary = {"fold": self.fold, "plane": self.plane, "best": self.best,
                   "epochs": self.cfg.epochs, "config_hash": config_hash(self.cfg),
                   "finished": dt.datetime.now().isoformat(timespec="seconds"),
                   "note": "3 task models share one augmented stream per exam (speed choice)"}
        (self.run_dir / DONE_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.logger.close()
        append_lab_log("b0_job_done", fold=self.fold, plane=self.plane, best=self.best)
