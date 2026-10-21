"""
prepare_kneemri.py: verify, extract and integrity-check the KneeMRI download.

What it does
    1. Checks the MD5 of each downloaded file against the values published on Zenodo.
    2. Lists the .7z members and refuses absolute paths or '..' (an archive must not write
       outside its folder), then extracts into data/kneemri/volumes/ with py7zr.
    3. Loads every volume with the restricted unpickler and records only integrity facts
       (shape, dtype, whether the ROI box fits inside the volume) in
       data/kneemri/manifest.csv.

What it deliberately does NOT do (EVAL_PROTOCOL R1)
    It runs no model and computes no image statistics: KneeMRI is the external test set.

Usage (from the repo root; -I = isolated mode, so nothing in the data folder can be imported)
    .venv\\Scripts\\python.exe -I scripts\\prepare_kneemri.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Published on https://zenodo.org/records/14789903 (checked 2026-10-18)
ZENODO_MD5 = {
    "volumetric_data.7z": "669d47b0f7771304bf2dcd97b152c9e6",
    "metadata.csv": "2b574f92a7865ff23d28f1056e33c20b",
    "READ.ME": "5d8d41509f9f8d551668a9a993a906c9",
    "example.py": "badf0dc1e49ce561cc9565271b351a57",
    "example.pck": "2432007600f0f312ce8513fdeeba0061",
}


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_checksums(raw_dir: Path) -> None:
    for name, expected in ZENODO_MD5.items():
        got = md5_of(raw_dir / name)
        status = "OK" if got == expected else "MISMATCH"
        print(f"{status:8} {name}  {got}")
        if got != expected:
            raise SystemExit(f"{name}: checksum mismatch (incomplete or corrupted download?)")


def is_safe_member(name: str) -> bool:
    """True if an archive member stays inside the extraction folder."""
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and ":" not in name


def extract(archive: Path, out_dir: Path) -> None:
    import py7zr
    with py7zr.SevenZipFile(archive, mode="r") as z:
        names = z.getnames()
        unsafe = [n for n in names if not is_safe_member(n)]
        if unsafe:
            raise SystemExit(f"refusing to extract: unsafe paths in archive, e.g. {unsafe[:3]}")
        print(f"extracting {len(names)} members -> {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        z.extractall(path=out_dir)


def integrity_manifest(volumes_dir: Path, meta_path: Path) -> pd.DataFrame:
    from kneescan.data.kneemri import index_volumes, read_metadata, safe_load_pck

    meta = read_metadata(meta_path)
    paths = index_volumes(volumes_dir)
    rows = []
    for _, r in meta.iterrows():
        path = paths.get(r["volumeFilename"])
        if path is None:
            rows.append({"volumeFilename": r["volumeFilename"], "found": False})
            continue
        vol = safe_load_pck(path)
        z, y, x = vol.shape
        roi_inside = (r["roiZ"] + r["roiDepth"] <= z and r["roiY"] + r["roiHeight"] <= y
                      and r["roiX"] + r["roiWidth"] <= x)
        rows.append({"volumeFilename": r["volumeFilename"], "found": True, "z": z, "y": y, "x": x,
                     "dtype": str(vol.dtype), "roi_inside": bool(roi_inside)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "kneemri" / "raw")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "kneemri" / "volumes")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    verify_checksums(args.raw)
    if not args.skip_extract:
        extract(args.raw / "volumetric_data.7z", args.out)
    manifest = integrity_manifest(args.out, args.raw / "metadata.csv")
    manifest.to_csv(args.out.parent / "manifest.csv", index=False)
    found = manifest["found"].sum()
    print(f"volumes found: {found}/{len(manifest)}")
    if found:
        ok = manifest[manifest["found"]]
        print(f"dtypes: {ok['dtype'].value_counts().to_dict()}")
        print(f"z range {ok['z'].min()}-{ok['z'].max()}, y range {ok['y'].min()}-{ok['y'].max()}, "
              f"x range {ok['x'].min()}-{ok['x'].max()}")
        print(f"ROI inside volume: {ok['roi_inside'].sum()}/{len(ok)}")


if __name__ == "__main__":
    main()
