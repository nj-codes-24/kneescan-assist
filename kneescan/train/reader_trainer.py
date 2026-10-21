"""
reader_trainer.py: train the Reader for one CV fold. It checkpoints every epoch, can be resumed,
and can pause on schedule (same guarantees as b0_trainer.py).

What it does
    One *job* = one CV fold. One model learns all three findings from all three planes.
    Per epoch:
      1. train on the fold's training exams (order = seed + epoch, so a resume is exact).
         Batch = 1 exam (exams have different slice counts), with gradients accumulated
         over `accumulate_exams` exams before each optimiser step.
      2. evaluate on the inner tuning exams (loss + AUC per finding)
      3. keep best.pt when the mean tuning loss improves (epoch selection, EVAL_PROTOCOL R4/R7)
      4. save last.pt atomically with weights, optimiser, scheduler, AMP scaler, RNG states
      5. stop cleanly if --stop-at is near
    At the end it writes preds.csv for train / tune / OOF exams with best.pt. MRNet `valid` is never touched.

Why BCE with pos_weight?
    ACL tears are ~23% of exams. pos_weight = negatives / positives makes a missed tear cost as
    much in total as all the false alarms, so the model doesn't just learn "no tear".

Alternatives
    With equal-size inputs you'd batch several exams. Here, padding to the longest series
    would waste memory, and accumulation gives the same averaged gradient.
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
from torch.utils.data import DataLoader

from kneescan import TASKS
from kneescan.data.mrnet import MRNetExamDataset
from kneescan.data.splits import FoldSplit
from kneescan.data.transforms import random_affine, random_gamma, random_slice_dropout, to_25d_input
from kneescan.eval.metrics import auc
from kneescan.models.pretrained import load_backbone_weights
from kneescan.models.reader import KneeReader
from kneescan.train.b0_trainer import EpochOrderSampler, compute_deadline, should_pause
from kneescan.train.config import ReaderConfig, config_hash
from kneescan.train.run_logger import RunLogger, append_lab_log
from kneescan.train.seed import capture_rng_state, restore_rng_state, seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
DONE_FILE = "done.json"
LAST_CKPT = "last.pt"
BEST_CKPT = "best.pt"


def build_reader(cfg: ReaderConfig, pretrained: bool | None = None) -> KneeReader:
    """The Reader as described by a config (pretrained=False when weights come from a checkpoint).

    With `backbone_weights` set, the backbone starts from that file (e.g. RadImageNet) instead of timm's ImageNet weights.
    """
    use_pretrained = cfg.pretrained if pretrained is None else pretrained
    from_file = use_pretrained and bool(cfg.backbone_weights)
    model = KneeReader(cfg.backbone, use_pretrained and not from_file, cfg.att_dim, cfg.dropout,
                       cfg.plane_dropout, cfg.freeze_batchnorm, cfg.pooling)
    if from_file:
        load_backbone_weights(model.backbone, REPO_ROOT / cfg.backbone_weights)
    return model


def parameter_groups(model: KneeReader, cfg: ReaderConfig) -> list[dict]:
    """Two learning rates: small for the pretrained backbone, larger for the new layers on top of it.

    The backbone already knows useful image features and should only be nudged. The attention, norm and
    heads start from random numbers and need bigger steps (fine-tuning practice; also the old K recipe).
    """
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    head = [p for p in model.parameters() if id(p) not in backbone_ids]
    return [{"params": backbone, "lr": cfg.lr}, {"params": head, "lr": cfg.head_lr}]


def freeze_backbone_prefixes(model: KneeReader, prefixes: tuple[str, ...]) -> None:
    """Stop early backbone layers from training (their ImageNet edge/texture detectors are kept as they are)."""
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(False)


def positive_weights(labels: np.ndarray) -> torch.Tensor:
    """(n_tasks,) negatives / positives per finding, computed on the training exams only."""
    pos = labels.sum(axis=0).clip(min=1)
    neg = (len(labels) - labels.sum(axis=0)).clip(min=1)
    return torch.tensor(neg / pos, dtype=torch.float32)


class ReaderFoldJob:
    """Trains one Reader on one CV fold."""

    def __init__(self, cfg: ReaderConfig, data_root: Path, labels: pd.DataFrame, split: FoldSplit,
                 fold: int, run_dir: Path, device: str | None = None) -> None:
        self.cfg, self.fold, self.run_dir = cfg, fold, Path(run_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.seed = cfg.seed + 1000 * fold
        seed_everything(self.seed, cfg.deterministic)

        self.train_ds = MRNetExamDataset(data_root, "train", labels, split.train, cfg.planes)
        self.tune_ds = MRNetExamDataset(data_root, "train", labels, split.tune, cfg.planes)
        self.oof_ds = MRNetExamDataset(data_root, "train", labels, split.oof, cfg.planes)
        self.pos_weight = positive_weights(self.train_ds.labels).to(self.device)
        # 1 for each finding that trains this model; all 1s by default. A per-finding specialist (I5) has a single 1.
        self.task_mask = torch.tensor([float(t in cfg.loss_tasks) for t in TASKS], device=self.device)

        self.model = build_reader(cfg).to(self.device)
        freeze_backbone_prefixes(self.model, cfg.frozen_prefixes)
        self.opt = torch.optim.AdamW(parameter_groups(self.model, cfg), weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.epochs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.aug_gen = torch.Generator().manual_seed(self.seed)

        self.next_epoch = 0
        self.best = {"tune_loss": float("inf"), "epoch": -1, "tune_auc": None}
        self.last_epoch_seconds = 0.0
        self.sampler = EpochOrderSampler(len(self.train_ds))
        self.logger = RunLogger(self.run_dir)

    # ---------------- checkpointing
    def _state(self) -> dict:
        return {"next_epoch": self.next_epoch, "best": self.best, "last_epoch_seconds": self.last_epoch_seconds,
                "model": self.model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(), "rng": capture_rng_state({"aug": self.aug_gen}),
                "config": dataclasses.asdict(self.cfg), "config_hash": config_hash(self.cfg)}

    def save_last(self) -> None:
        tmp = self.run_dir / (LAST_CKPT + ".tmp")
        torch.save(self._state(), tmp)
        os.replace(tmp, self.run_dir / LAST_CKPT)  # atomic: either the old or the new file, never half

    def load_last(self) -> None:
        state = torch.load(self.run_dir / LAST_CKPT, map_location=self.device, weights_only=False)
        if state["config_hash"] != config_hash(self.cfg):
            raise RuntimeError("checkpoint was made with a different config; refusing to resume")
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
        self.scaler.load_state_dict(state["scaler"])
        self.next_epoch, self.best = state["next_epoch"], state["best"]
        self.last_epoch_seconds = state["last_epoch_seconds"]
        restore_rng_state(state["rng"], {"aug": self.aug_gen})
        self.logger.drop_rows_after(self.next_epoch - 1)

    # ---------------- one exam
    def _loader(self, ds: MRNetExamDataset, sampler=None) -> DataLoader:
        return DataLoader(ds, batch_size=None, sampler=sampler, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=self.device.type == "cuda")

    def _prepare(self, item: dict, augment: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        planes = {}
        for plane, vol in item["volumes"].items():
            vol = vol.to(self.device, non_blocking=True).float()
            if augment:
                vol = random_affine(vol, self.cfg.rotate_deg, self.cfg.shift_px, self.cfg.flip_p, self.aug_gen)
                if self.cfg.intensity_jitter > 0:
                    vol = random_gamma(vol, self.cfg.intensity_jitter, self.aug_gen)
            slices = to_25d_input(vol)
            if augment and self.cfg.slice_dropout > 0:      # after 2.5D stacking, so each kept slice keeps its true neighbours
                slices = random_slice_dropout(slices, self.cfg.slice_dropout, self.aug_gen)
            planes[plane] = slices
        return planes, item["labels"].to(self.device)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """(n_tasks,) BCE per finding, with pos_weight against class imbalance."""
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight, reduction="none")

    def _training_loss(self, planes: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(scalar loss to minimise, (n_tasks,) loss of the combined answer for logging).

        With plane_loss_weight > 0, every plane must also diagnose on its own ("deep supervision", D-012). Otherwise
        a weak plane can free-ride on the others inside the vote.
        """
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            out = self.model(planes)
        fused = self._loss(out.logits.float(), labels)
        total = (fused * self.task_mask).sum() / self.task_mask.sum()   # only the findings this model specialises in (I5)
        if self.cfg.plane_loss_weight > 0:
            per_plane = [(self._loss(lg.float(), labels) * self.task_mask).sum() / self.task_mask.sum()
                         for lg in out.plane_logits.values()]
            total = total + self.cfg.plane_loss_weight * torch.stack(per_plane).mean()
        return total, fused

    # ---------------- one epoch
    def train_one_epoch(self, epoch: int) -> np.ndarray:
        """Returns the mean training loss per finding."""
        self.model.train()
        self.sampler.order = np.random.default_rng(self.seed + epoch).permutation(len(self.train_ds)).tolist()
        totals = np.zeros(len(TASKS))
        self.opt.zero_grad(set_to_none=True)
        for step, item in enumerate(self._loader(self.train_ds, self.sampler), start=1):
            planes, labels = self._prepare(item, augment=True)
            total, per_task = self._training_loss(planes, labels)
            self.scaler.scale(total / self.cfg.accumulate_exams).backward()
            totals += per_task.detach().cpu().numpy()
            if step % self.cfg.accumulate_exams == 0 or step == len(self.train_ds):
                if self.cfg.grad_clip > 0:                      # unscale first, so the cap applies to the true gradient size
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
        return totals / len(self.train_ds)

    @torch.no_grad()
    def predict(self, ds: MRNetExamDataset, model: KneeReader | None = None) -> pd.DataFrame:
        """Probabilities and losses for every exam in `ds` (no augmentation, all planes)."""
        model = model or self.model
        model.eval()
        rows = []
        for item in self._loader(ds):
            planes, labels = self._prepare(item, augment=False)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = model(planes).logits.float()
            losses = self._loss(logits, labels)
            for i, task in enumerate(TASKS):
                rows.append({"case_id": item["case_id"], "task": task, "label": int(labels[i]),
                             "prob": float(torch.sigmoid(logits[i])), "loss": float(losses[i])})
        return pd.DataFrame(rows)

    def evaluate_tune(self) -> dict[str, dict[str, float | None]]:
        df = self.predict(self.tune_ds)
        out = {}
        for task, g in df.groupby("task"):
            two_classes = g["label"].nunique() == 2
            out[task] = {"loss": float(g["loss"].mean()), "auc": auc(g["label"], g["prob"]) if two_classes else None}
        return out

    # ---------------- the main loop
    def _log_epoch(self, epoch: int, train_loss: np.ndarray, tune: dict, is_best: bool) -> None:
        for i, task in enumerate(TASKS):
            self.logger.log_epoch({"epoch": epoch, "task": task, "train_loss": float(train_loss[i]),
                                   "tune_loss": tune[task]["loss"], "tune_auc": tune[task]["auc"],
                                   "lr": self.opt.param_groups[0]["lr"],
                                   "seconds": round(self.last_epoch_seconds, 1), "is_best": is_best})

    def _end_epoch(self, epoch: int, train_loss: np.ndarray, tune: dict) -> None:
        """Select the epoch on mean tuning loss (over the findings it trains on), log, step the LR schedule and save last.pt."""
        mean_tune_loss = float(np.mean([tune[t]["loss"] for t in self.cfg.loss_tasks]))
        is_best = mean_tune_loss < self.best["tune_loss"]
        if is_best:
            self.best = {"tune_loss": mean_tune_loss, "epoch": epoch,
                         "tune_auc": {t: tune[t]["auc"] for t in TASKS}}
            torch.save(self.model.state_dict(), self.run_dir / BEST_CKPT)
        self._log_epoch(epoch, train_loss, tune, is_best)
        self.sched.step()
        self.next_epoch = epoch + 1
        self.save_last()

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
        append_lab_log("reader_job_start", fold=self.fold, start_epoch=self.next_epoch,
                       config=self.cfg.name, config_hash=config_hash(self.cfg), device=str(self.device))
        epochs_this_call = 0
        while self.next_epoch < self.cfg.epochs:
            epoch, t0 = self.next_epoch, time.time()
            train_loss = self.train_one_epoch(epoch)
            tune = self.evaluate_tune()
            self.last_epoch_seconds = time.time() - t0
            self._end_epoch(epoch, train_loss, tune)
            epochs_this_call += 1
            if self.next_epoch >= self.cfg.epochs:
                break
            if max_epochs_this_call is not None and epochs_this_call >= max_epochs_this_call:
                return "paused"
            if should_pause(dt.datetime.now(), deadline, self.last_epoch_seconds):
                append_lab_log("reader_job_paused", fold=self.fold, next_epoch=self.next_epoch)
                return "paused"
        self._finish()
        return "done"

    def _finish(self) -> None:
        """Predict train/tune/OOF with best.pt, write preds.csv and mark the job done."""
        best_model = build_reader(self.cfg, pretrained=False).to(self.device)
        best_model.load_state_dict(torch.load(self.run_dir / BEST_CKPT, map_location=self.device))
        parts = []
        for subset, ds in (("train", self.train_ds), ("tune", self.tune_ds), ("oof", self.oof_ds)):
            df = self.predict(ds, best_model)
            df.insert(0, "subset", subset)
            parts.append(df)
        preds = pd.concat(parts, ignore_index=True)
        preds.insert(0, "fold", self.fold)
        preds.drop(columns="loss").to_csv(self.run_dir / "preds.csv", index=False)
        summary = {"fold": self.fold, "best": self.best, "epochs": self.cfg.epochs, "config": self.cfg.name,
                   "config_hash": config_hash(self.cfg), "finished": dt.datetime.now().isoformat(timespec="seconds")}
        (self.run_dir / DONE_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.logger.close()
        append_lab_log("reader_job_done", fold=self.fold, config=self.cfg.name, best=self.best)
