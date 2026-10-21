"""
run_logger.py: write down everything a training run does.

What it does
    * `RunLogger` writes one CSV row per (epoch, task) to `<run_dir>/metrics.csv` and the
      same numbers to TensorBoard (`<run_dir>/tb/`). On resume it keeps appending to the
      same files.
    * `append_lab_log()` adds one JSON line per event (run started / paused / finished) to
      `logs/pc_c.jsonl`, the project-wide diary. Each line carries the machine name, the
      git commit and the config hash.

Why both CSV and TensorBoard?
    CSV is easy to read with pandas and to diff. TensorBoard shows live curves while a run
    is going (`tensorboard --logdir runs`).

Alternatives
    MLflow or Weights & Biases add a web dashboard and artifact tracking. W&B sends data to
    a third-party server, which we avoid for anything derived from licensed medical data.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from kneescan import MACHINE_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    """Current git commit (with a '+dirty' suffix when there are uncommitted changes)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_lab_log(event: str, log_dir: Path | None = None, **fields: Any) -> None:
    """Append one event to logs/<machine>.jsonl (the project diary)."""
    log_dir = log_dir or Path(os.environ.get("KNEESCAN_LOG_DIR", REPO_ROOT / "logs"))  # tests redirect this
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "machine": MACHINE_NAME,
        "git": git_commit(),
        "event": event,
        **fields,
    }
    with open(log_dir / f"{MACHINE_NAME}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


class RunLogger:
    """Per-run metrics: CSV (always) + TensorBoard (if installed)."""

    FIELDS = ["epoch", "task", "train_loss", "tune_loss", "tune_auc", "lr", "seconds", "is_best"]

    def __init__(self, run_dir: Path, use_tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:  # TensorBoard is optional; CSV is the source of truth
                self._writer = None

    def drop_rows_after(self, epoch: int) -> None:
        """On resume, remove rows from epochs that were logged but never checkpointed."""
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= epoch]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def log_epoch(self, row: dict[str, Any]) -> None:
        """Write one (epoch, task) row."""
        new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in self.FIELDS})
        if self._writer is not None:
            task, epoch = row["task"], int(row["epoch"])
            for key in ("train_loss", "tune_loss", "tune_auc", "lr"):
                if row.get(key) is not None:
                    self._writer.add_scalar(f"{task}/{key}", float(row[key]), epoch)
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
