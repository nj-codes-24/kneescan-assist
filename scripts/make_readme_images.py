"""
make_readme_images.py: draw the README's figures. A results chart and two app screenshots, none of which shows a scan.

What it does
    * docs/images/final_results.svg: the one-time final test as a bar chart with 95% CIs (results/final/final.json):
      macro AUC on MRNet `valid` and ACL AUC on the external KneeMRI set, for B0, the Reader and the hybrid.
    * docs/images/app_results.png: the results page for the hosted demo's example study (KneeMRI exam in configs/deploy.yaml),
      rendered from app/views.py WITHOUT the "where it looked" section, so no scan appears (D-025).
    * docs/images/app_landing.png (only with --landing-url): the landing page of a running copy of the app.
    Screenshots are taken with Microsoft Edge in headless mode (installed with Windows).

Why no scans?
    MRNet images may not be published (MRNet RUA clause 4), and KneeMRI images with our heatmaps would be adapted material
    (CC BY-NC-ND). The live demo shows the heatmaps; the repository doesn't.

Alternatives
    With an openly licensed knee MRI you could add a heatmap figure. Re-run this script after any change to the results or the page.

Usage (from the repo root; -I because it opens a KneeMRI .pck file; needs the weights and data on this computer)
    .venv\\Scripts\\python.exe -I scripts\\make_readme_images.py --landing-url http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.views import Wording, finding_row, limits_html, note_html, verdict_html  # noqa: E402
from kneescan.app.dicom_io import to_uint8_256  # noqa: E402
from kneescan.app.predictor import Predictor  # noqa: E402
from kneescan.data.kneemri import index_volumes, safe_load_pck  # noqa: E402

IMAGES = REPO_ROOT / "docs" / "images"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
MODELS = (("baseline", "B0 (MRNet paper\nreproduction)", "#86868b"), ("reader", "KneeScan Reader", "#0071e3"),
          ("hybrid", "Hybrid\n(Reader + B0)", "#5ac8fa"))


def results_chart(final_json: Path, out: Path) -> None:
    """Two panels of AUC bars with 95% CI whiskers, numbers written on the bars."""
    report = json.loads(final_json.read_text(encoding="utf-8"))
    panels = (("MRNet test (120 exams)\nmacro AUC", report["mrnet_valid"], "macro_auc"),
              ("KneeMRI, other hospital (917 scans)\nACL AUC", report["kneemri"], "acl_auc"))
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, section, key) in zip(axes, panels):
        for i, (name, label, colour) in enumerate(MODELS):
            auc, lo, hi = section[name][key]
            ax.bar(i, auc, color=colour, width=0.62)
            ax.errorbar(i, auc, yerr=[[auc - lo], [hi - auc]], color="#1d1d1f", capsize=4, linewidth=1)
            ax.text(i, 0.515, f"{auc:.3f}", ha="center", va="bottom", color="white", fontweight="bold")
        ax.set_xticks(range(len(MODELS)), [m[1] for m in MODELS], fontsize=8.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0.5, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUC (0.5 = chance, 1.0 = perfect)")
    fig.text(0.5, -0.02, "Bars: AUC. Whiskers: 95% bootstrap CI. Neither the Reader nor the hybrid differs reliably from B0 "
             "on either test set (paired bootstrap).",
             ha="center", fontsize=8.5, color="#6e6e73")
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def results_page_html(exam_file: str, note: str, wording: Wording) -> str:
    """The results page for the example study, minus the heatmaps, as a standalone HTML page."""
    volume = safe_load_pck(index_volumes(REPO_ROOT / "data" / "kneemri" / "volumes")[exam_file])
    volumes = {"sagittal": to_uint8_256(volume.astype("float32"))}
    results = Predictor(REPO_ROOT / "runs" / "reader_r2_seeds").predict(volumes, with_heatmaps=False)
    css = (REPO_ROOT / "app" / "theme.css").read_text(encoding="utf-8")
    body = (f'<div class="ks-results">{verdict_html(results, wording)}{note_html(note)}'
            f'<p class="ks-series">Sagittal {volumes["sagittal"].shape[0]} slices</p>'
            f'<div class="ks-rows">{"".join(finding_row(r, wording) for r in results)}</div>{limits_html()}</div>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head>'
            f'<body><div class="gradio-container">{body}</div></body></html>')


def screenshot(url: str, out: Path, height: int) -> None:
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=10000",
                    f"--window-size=1280,{height}", f"--screenshot={out}", url], check=True, capture_output=True, timeout=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--landing-url", help="a running copy of the app, e.g. http://127.0.0.1:7860")
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    results_chart(REPO_ROOT / "results" / "final" / "final.json", IMAGES / "final_results.svg")
    space = yaml.safe_load((REPO_ROOT / "configs" / "deploy.yaml").read_text(encoding="utf-8"))
    app = yaml.safe_load((REPO_ROOT / "configs" / "app.yaml").read_text(encoding="utf-8"))
    page = results_page_html(space["kneemri_sample"], space["sample_note"], Wording(app["likely_from"], app["unlikely_below"]))
    with tempfile.TemporaryDirectory() as tmp:
        html_file = Path(tmp) / "results.html"
        html_file.write_text(page, encoding="utf-8")
        screenshot(html_file.as_uri(), IMAGES / "app_results.png", 1180)
    if args.landing_url:
        screenshot(args.landing_url, IMAGES / "app_landing.png", 960)
    print("written:", ", ".join(sorted(p.name for p in IMAGES.iterdir() if p.suffix in {".png", ".svg"})))


if __name__ == "__main__":
    main()
