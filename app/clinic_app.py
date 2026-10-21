"""
clinic_app.py: the KneeScan Assist demo. Two screens: a landing stage with one job (add a knee MRI study), then a results page that
opens with the answer in plain words, followed by where the model looked.

What it does
    1. Landing: drop a .zip of DICOM files or 1–3 MRNet-style .npy files (kneescan/app/dicom_io.py), or try the local sample study.
    2. The uploaded files are deleted as soon as they have been read. Patient tags are never accessed.
    3. The 5-model ensemble predicts with explanations (kneescan/app/predictor.py).
    4. The landing stage hides and the results page shows (app/views.py, styled by app/theme.css). "New analysis" goes back.
    The weights come from `run_dir` on this computer, or, when `weights_repo` is set (the hosted demo), from that private
    Hugging Face model repo, downloaded once at start-up with the HF_TOKEN secret (scripts/export_weights.py makes its files).
    scripts/build_deploy.py packs this same file into a container with the hosted settings (configs/deploy.yaml).

Why Gradio?
    Upload, buttons and the page switch are a few lines of Python, with no separate web front-end (D-008). The look is our own:
    HTML from small functions plus one stylesheet (D-023, D-024).

Alternatives
    A real clinical integration would plug into a PACS viewer (e.g. via DICOMweb / OHIF) instead of a web form.

Usage (from the repo root)
    .venv\\Scripts\\python.exe app\\clinic_app.py            # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # no usage data leaves this PC (must be set before import)

import gradio as gr  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))   # lets `python app/clinic_app.py` import the kneescan and app packages

from app.views import Wording, brand_html, landing_html, landing_note_html, results_html  # noqa: E402
from kneescan.app.dicom_io import read_dicom_zip, read_npy_files  # noqa: E402
from kneescan.app.predictor import Predictor  # noqa: E402

SYSTEM_FONTS = ("-apple-system", "BlinkMacSystemFont", "Segoe UI Variable Display", "Segoe UI", "system-ui", "sans-serif")


def load_settings() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "app.yaml").read_text(encoding="utf-8"))


def resolve_run_dir(settings: dict) -> Path:
    """The final model's run folder: local, or inside a download of the private weights repo (the token is never logged)."""
    if not settings.get("weights_repo"):
        return REPO_ROOT / settings["run_dir"]
    from huggingface_hub import snapshot_download
    local = snapshot_download(settings["weights_repo"], token=os.environ.get("HF_TOKEN"))
    return Path(local) / settings["run_dir"]


def read_upload(paths: list[str]) -> dict:
    """Read the uploaded files into volumes, then delete them (they may contain patient data)."""
    try:
        if len(paths) == 1 and paths[0].lower().endswith(".zip"):
            return read_dicom_zip(paths[0])
        return read_npy_files(paths)
    finally:
        for path in paths:
            Path(path).unlink(missing_ok=True)


def show_results(predictor: Predictor, volumes: dict, started: float, wording: Wording, note: str = "") -> tuple:
    """Predict, then switch screens: hide the landing stage, show the results page."""
    page = results_html(predictor.predict(volumes), volumes, time.time() - started, wording, note)
    return gr.update(visible=False), gr.update(visible=True), page


def make_analyse(predictor: Predictor, wording: Wording):
    def analyse(files: list[str] | None) -> tuple:
        if not files:
            raise gr.Error("Add a .zip of DICOM files or 1–3 .npy files first.")
        started = time.time()
        try:
            volumes = read_upload([str(f) for f in files])
        except ValueError as problem:                       # e.g. no usable series, wrong file name
            raise gr.Error(str(problem)) from problem
        return show_results(predictor, volumes, started, wording)
    return analyse


def make_sample(predictor: Predictor, wording: Wording, sample_dir: Path, note: str = ""):
    """The example study (never deleted: it is a kept copy, not an upload), with a note on where it comes from.
    Analysed once, then served from memory: on a CPU-only server the first analysis takes several seconds. The lock makes
    a visitor who clicks during that first analysis wait for it instead of starting a second one."""
    cached: list[tuple] = []
    lock = threading.Lock()
    def sample() -> tuple:
        with lock:
            if not cached:
                started = time.time()
                cached.append(show_results(predictor, read_npy_files(sorted(sample_dir.glob("*.npy"))), started, wording, note))
        return cached[0]
    return sample


def back_to_start() -> tuple:
    return gr.update(visible=True), gr.update(visible=False), "", None


def build_theme() -> gr.themes.Base:
    """Gradio's own widgets (upload box, buttons) in the same quiet style as our HTML."""
    return gr.themes.Base(primary_hue="blue", neutral_hue="gray", radius_size="lg", font=SYSTEM_FONTS).set(
        body_background_fill="#fbfbfd", body_background_fill_dark="#000000",
        block_background_fill="#ffffff", block_background_fill_dark="#161617",
        block_border_width="0px", block_shadow="none",
        button_primary_background_fill="#0071e3", button_primary_background_fill_hover="#0077ed",
        button_primary_text_color="#ffffff")


def build_ui(predictor: Predictor, settings: dict) -> gr.Blocks:
    wording = Wording(settings["likely_from"], settings["unlikely_below"])
    sample_dir = REPO_ROOT / settings.get("sample_dir", "")
    has_sample = bool(settings.get("sample_dir")) and any(sample_dir.glob("*.npy"))
    show_sample = make_sample(predictor, wording, sample_dir, settings.get("sample_note", ""))
    if has_sample and settings.get("hosted"):
        # analyse the example in the background, so the server opens its port at once (a host restarts slow starters)
        threading.Thread(target=show_sample, daemon=True).start()
    with gr.Blocks(title="KneeScan", delete_cache=(settings["cache_seconds"], settings["cache_seconds"])) as ui:
        with gr.Column(elem_classes="ks-stage") as landing:                       # screen 1: one job
            gr.HTML(landing_html())
            with gr.Column(elem_classes="ks-glass"):
                files = gr.File(file_count="multiple", file_types=[".zip", ".npy"], type="filepath", show_label=False)
                with gr.Row(elem_classes="ks-actions"):
                    analyse = gr.Button("Analyse study", variant="primary", elem_classes="ks-primary", scale=2)
                    sample = gr.Button("Try a sample study", elem_classes="ks-secondary", scale=1, visible=has_sample)
            gr.HTML(landing_note_html(settings.get("hosted", False)))
        with gr.Column(visible=False, elem_classes="ks-results-page") as results_page:   # screen 2: the answer
            with gr.Row(elem_classes="ks-topbar"):
                gr.HTML(brand_html())
                again = gr.Button("New analysis", elem_classes="ks-secondary", size="sm")
            results = gr.HTML()
        analyse.click(make_analyse(predictor, wording), inputs=files, outputs=[landing, results_page, results])
        sample.click(show_sample, outputs=[landing, results_page, results])
        again.click(back_to_start, outputs=[landing, results_page, results, files])
    return ui


def main() -> None:
    settings = load_settings()
    predictor = Predictor(resolve_run_dir(settings), settings["heatmap_alpha"], slice_batch=settings.get("slice_batch"))
    css = (REPO_ROOT / "app" / "theme.css").read_text(encoding="utf-8")
    build_ui(predictor, settings).launch(server_name=settings["host"], server_port=settings["port"], theme=build_theme(), css=css)


if __name__ == "__main__":
    main()
