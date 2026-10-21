"""
fetch_mrnet.py: download MRNet v1.0 from Stanford AIMI's Redivis page and rebuild the
original folder layout.

What it does
    1. Connects to Redivis with *your* account. On first use the Redivis client prints a
       login link; open it and approve. (A token in REDIVIS_API_TOKEN also works.) This
       script never reads, prints or stores credentials.
    2. Labels: Redivis stores the six label CSVs as *tables*. The original CSVs had no header
       row, so Redivis turned the first exam into column names (e.g. "_0000", "_0") and
       each table has 1,129 rows instead of 1,130. We put that first row back.
    3. Images: downloads the `train` and `valid` file tables (3,390 + 360 .npy files), then
       checks every file's size and MD5 against the Redivis file index.
    4. Writes the layout the loader expects:
           data/mrnet/MRNet-v1.0/{train,valid}/{sagittal,coronal,axial}/NNNN.npy
           data/mrnet/MRNet-v1.0/{train,valid}-{abnormal,acl,meniscus}.csv

Why a script
    It is resumable (existing files are skipped), verifiable (MD5) and repeatable.

Alternatives
    If Redivis refuses export for your access level, download from the web page instead and
    put the files in data/mrnet/ by hand. The loader finds them wherever they are.

Usage (from the repo root; run it in a terminal you can see, because of the login link)
    .venv\\Scripts\\python.exe scripts\\fetch_mrnet.py --inspect     # look only, download nothing
    .venv\\Scripts\\python.exe scripts\\fetch_mrnet.py               # download everything
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "mrnet"
DATASET_REF = "aimi.mrnet_knee_mri_s:4a2c:v1_0"   # found via the Redivis API on 2026-10-18
PLANES = ("sagittal", "coronal", "axial")
TASKS = ("abnormal", "acl", "meniscus")
SPLITS = ("train", "valid")


# ---------------------------------------------------------------- labels
def repair_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the 'first row became the header' problem. Returns columns case_id, label."""
    if len(df.columns) != 2:
        raise ValueError(f"expected 2 columns, got {list(df.columns)}")
    id_col, label_col = df.columns
    first_id, first_label = str(id_col).lstrip("_"), str(label_col).lstrip("_")
    rows = pd.DataFrame({"case_id": df[id_col].astype(str).str.zfill(4),
                         "label": df[label_col].astype(int)})
    if first_id.isdigit() and first_label in ("0", "1"):
        first = pd.DataFrame({"case_id": [first_id.zfill(4)], "label": [int(first_label)]})
        rows = pd.concat([first, rows], ignore_index=True)
    return rows.sort_values("case_id").reset_index(drop=True)


def fetch_labels(dataset, out_root: Path) -> dict[str, int]:
    counts = {}
    for split in SPLITS:
        for task in TASKS:
            table = dataset.table(f"{split}-{task}")
            raw = table.to_pandas_dataframe(progress=False, dtype_backend="numpy_nullable")
            fixed = repair_label_table(raw)
            fixed.to_csv(out_root / f"{split}-{task}.csv", header=False, index=False)
            counts[f"{split}-{task}"] = len(fixed)
            print(f"labels {split}-{task}: {len(raw)} rows on Redivis -> {len(fixed)} after repair")
    return counts


# ---------------------------------------------------------------- files
def plane_and_case(file_name: str) -> tuple[str, str] | None:
    """'.../axial/0123.npy' -> ('axial', '0123'); None if the plane can't be read from the path."""
    parts = PurePosixPath(file_name.replace("\\", "/")).parts
    plane = next((p for p in parts if p.lower() in PLANES), None)
    if plane is None or not parts[-1].endswith(".npy"):
        return None
    return plane.lower(), parts[-1][:-4].zfill(4)


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


def file_index(dataset, split: str) -> pd.DataFrame:
    df = dataset.table(split).to_pandas_dataframe(progress=False, dtype_backend="numpy_nullable")
    return df[["file_id", "file_name", "size", "md5_hash"]]


def inspect(dataset) -> None:
    for split in SPLITS:
        idx = file_index(dataset, split)
        parsed = idx["file_name"].map(plane_and_case)
        print(f"{split}: {len(idx)} files, {idx['size'].sum() / 1e9:.2f} GB, "
              f"plane readable for {parsed.notna().sum()}/{len(idx)}")
        print("  examples:", idx["file_name"].head(5).tolist())
    for task in TASKS:
        cols = list(dataset.table(f"train-{task}").to_pandas_dataframe(max_results=1, progress=False).columns)
        print(f"train-{task} columns: {cols}")


def fetch_files(dataset, split: str, out_root: Path, staging: Path, parallel: int) -> dict[str, int]:
    """Download one split's files, verify MD5, and move them into <split>/<plane>/NNNN.npy."""
    idx = file_index(dataset, split)
    parsed = idx["file_name"].map(plane_and_case)
    if parsed.isna().any():
        bad = idx.loc[parsed.isna(), "file_name"].head(5).tolist()
        raise SystemExit(f"{split}: can't tell the plane from file names like {bad}. "
                         "Run --inspect and check the output.")
    stage = staging / split
    stage.mkdir(parents=True, exist_ok=True)
    print(f"{split}: downloading {len(idx)} files ({idx['size'].sum() / 1e9:.2f} GB) -> {stage}")
    dataset.table(split).to_directory().download(str(stage), overwrite=False, progress=True,
                                                 max_parallelization=parallel)
    moved = bad_md5 = 0
    for (_, row), (plane, case) in zip(idx.iterrows(), parsed):
        src = stage / row["file_name"]
        if not src.exists():
            hits = list(stage.rglob(PurePosixPath(row["file_name"]).name))
            src = next((h for h in hits if plane in h.as_posix().lower()), None) or src
        dst = out_root / split / plane / f"{case}.npy"
        if dst.exists() and dst.stat().st_size == row["size"]:
            moved += 1
            continue
        if not src.exists():
            raise SystemExit(f"missing after download: {row['file_name']}")
        if md5_b64(src) != row["md5_hash"]:
            bad_md5 += 1
            print(f"MD5 MISMATCH {row['file_name']}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        moved += 1
    if bad_md5:
        raise SystemExit(f"{split}: {bad_md5} files failed MD5; re-run to retry them")
    return {"files": moved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=DATASET_REF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inspect", action="store_true", help="show file names / columns, download nothing")
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    import redivis
    dataset = redivis.dataset(args.ref)
    if args.inspect:
        inspect(dataset)
        return

    out_root = args.out / "MRNet-v1.0"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"reference": args.ref, "labels": fetch_labels(dataset, out_root)}
    for split in SPLITS:
        summary[split] = fetch_files(dataset, split, out_root, args.out / "_staging", args.parallel)
    summary["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (args.out / "download_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("done. Next: scripts/check_mrnet.py validates shapes; then scripts/make_splits.py.")


if __name__ == "__main__":
    main()
