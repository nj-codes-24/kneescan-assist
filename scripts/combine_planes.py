"""
combine_planes.py: build the R2 run (three plane specialists, equal logit vote) from finished single-plane runs.

Usage
    .venv\\Scripts\\python.exe scripts\\combine_planes.py --out runs\\reader_r2 runs\\reader_sag_att runs\\reader_cor_att runs\\reader_ax_att
Then score it like any other run:
    .venv\\Scripts\\python.exe scripts\\score_cv.py --runs runs\\reader_r2 --compare runs\\b0 --change "R2: three plane specialists"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kneescan.eval.plane_ensemble import combine_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("components", type=Path, nargs="+", help="finished single-plane run folders")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combined = combine_runs(args.components, args.out)
    print(f"{args.out}: {combined['case_id'].nunique()} exams, {combined['fold'].nunique()} folds, "
          f"{len(args.components)} components")


if __name__ == "__main__":
    main()
