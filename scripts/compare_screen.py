"""
compare_screen.py: compare a screen (a run trained on one fold) with a reference run on that same fold's out-of-fold exams.

What it does
    Prints, for both runs: macro AUC with a 95% bootstrap CI and the AUC per finding. Then the paired macro-AUC gain (new − reference)
    on the same exams, and the paired AUC gain for one chosen finding (e.g. ACL for an ACL-only specialist).
    A screen only points the way (226 exams → wide CIs). Ideas that look promising get a full 5-fold run (docs/decisions.md D-017).

Usage
    .venv\\Scripts\\python.exe scripts\\compare_screen.py --new runs\\reader_cor_wd --ref runs\\reader_cor_att --fold 1
    .venv\\Scripts\\python.exe scripts\\compare_screen.py --new runs\\reader_cor_acl --ref runs\\reader_cor_att --fold 1 --finding acl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from kneescan import TASKS
from kneescan.eval.macro import macro_auc_ci, paired_macro_gain, wide_oof
from kneescan.eval.metrics import auc, paired_bootstrap_delta
from kneescan.eval.oof import load_oof


def fold_table(run: Path, fold: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    oof = load_oof(run)
    return wide_oof(oof[oof["fold"] == fold])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--finding", choices=TASKS, default=None, help="also report the paired gain for this finding")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # the report uses – and −; the Windows console defaults to cp1252
    y, p_new, ids_new = fold_table(args.new, args.fold)
    y_ref, p_ref, ids_ref = fold_table(args.ref, args.fold)
    if ids_new != ids_ref or not np.array_equal(y, y_ref):
        raise SystemExit("the two runs don't cover the same exams on this fold")
    for name, p in ((args.ref.name, p_ref), (args.new.name, p_new)):
        m, lo, hi = macro_auc_ci(y, p)
        per = ", ".join(f"{t} {auc(y[:, k], p[:, k]):.3f}" for k, t in enumerate(TASKS))
        print(f"{name:22s} macro {m:.3f} ({lo:.3f}–{hi:.3f})   {per}")
    g, lo, hi = paired_macro_gain(y, p_new, p_ref)
    print(f"paired macro gain {args.new.name} − {args.ref.name}: {g:+.4f} ({lo:+.4f} to {hi:+.4f}), {len(ids_new)} exams")
    if args.finding:
        k = TASKS.index(args.finding)
        d, dlo, dhi = paired_bootstrap_delta(y[:, k], p_new[:, k], p_ref[:, k], auc)
        print(f"paired {args.finding} AUC gain: {d:+.4f} ({dlo:+.4f} to {dhi:+.4f})")


if __name__ == "__main__":
    main()
