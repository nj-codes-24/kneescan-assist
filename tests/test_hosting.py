"""
test_hosting.py: the pieces behind the hosted demo do what they promise.

Why: the weights export is the only copy of the model that survives this PC, so it must refuse changed files and keep
the layout the app loads. The deploy settings decide what the public sees (D-025), and the sample is served from memory.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from kneescan import TASKS
from kneescan.app.predictor import Predictor
from kneescan.eval.plane_ensemble import read_components
from tests.conftest import REPO_ROOT
from tests.test_app import write_fake_run


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_weights = load_script("export_weights")
build_deploy = load_script("build_deploy")


def fake_frozen_repo(root: Path) -> Path:
    """runs/sag + runs/ax specialists, a combined run written on Windows, its CV summary and a frozen.yaml."""
    runs = root / "runs"
    write_fake_run(runs / "sag", planes="sagittal")
    write_fake_run(runs / "ax", planes="axial")
    (runs / "r2").mkdir()
    (runs / "r2" / "components.yaml").write_text("rule: mean\ncomponents: ['runs\\sag', 'runs\\ax']\n")
    (root / "results" / "cv").mkdir(parents=True)
    summary = {"per_task": {t: {"t_hs": 0.3, "auc": 0.8, "auc_ci": [0.7, 0.9]} for t in TASKS},
               "calibration": {t: {"platt_a": 1.0, "platt_b": 0.0} for t in TASKS}}
    (root / "results" / "cv" / "r2.json").write_text(json.dumps(summary))
    weights = {p.relative_to(root).as_posix(): export_weights.sha256(p) for p in sorted(runs.glob("*/fold*/best.pt"))}
    frozen = root / "frozen.yaml"
    frozen.write_text(yaml.safe_dump({"models": {"reader": {"run": "runs/r2", "weights": weights}}}))
    return frozen


def test_windows_component_paths_load_anywhere(tmp_path: Path) -> None:
    (tmp_path / "r2").mkdir()
    (tmp_path / "r2" / "components.yaml").write_text("components: ['runs\\sag']\n")
    assert read_components(tmp_path / "r2") == [tmp_path / "sag"]


def test_export_copies_only_what_the_app_needs_and_still_loads(tmp_path: Path) -> None:
    frozen = fake_frozen_repo(tmp_path / "repo")
    dest = tmp_path / "export"
    manifest = export_weights.export(tmp_path / "repo", dest, frozen)
    assert "runs/sag/fold0/best.pt" in manifest and "results/cv/r2.json" in manifest and "README.md" in manifest
    assert not list(dest.rglob("last.pt"))
    assert yaml.safe_load((dest / "runs" / "r2" / "components.yaml").read_text())["components"] == ["runs/sag", "runs/ax"]
    predictor = Predictor(dest / "runs" / "r2", device="cpu")
    assert [m.plane for m in predictor.members] == ["sagittal", "axial"]


def test_export_refuses_weights_changed_after_the_freeze(tmp_path: Path) -> None:
    frozen = fake_frozen_repo(tmp_path / "repo")
    with (tmp_path / "repo" / "runs" / "ax" / "fold1" / "best.pt").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="changed since the freeze"):
        export_weights.export(tmp_path / "repo", tmp_path / "export", frozen)


def test_deploy_settings_are_hosted_and_never_use_mrnet_samples() -> None:
    settings, build_only = build_deploy.deploy_settings(REPO_ROOT / "configs" / "app.yaml", REPO_ROOT / "configs" / "deploy.yaml")
    assert settings["hosted"] is True and settings["host"] == "0.0.0.0" and settings["weights_repo"]
    assert settings["slice_batch"] == 8                                 # fits the 4 GiB server (D-026)
    assert "demo_inputs" not in settings["sample_dir"] and build_only["kneemri_sample"].endswith(".pck")
    assert "complete ACL tear" in settings["sample_note"] and "AUC 0.70" in settings["sample_note"]
    assert not build_deploy.DEPLOY_ONLY_KEYS & set(settings)            # build-only keys don't reach the app


def test_container_runs_as_a_normal_user_with_cpu_pytorch() -> None:
    assert "USER kneescan" in build_deploy.DOCKERFILE and 'CMD ["python", "app/clinic_app.py"]' in build_deploy.DOCKERFILE
    assert "whl/cpu" in build_deploy.REQUIREMENTS and "gradio=={gradio_version}" in build_deploy.REQUIREMENTS
    assert "not a medical device" in build_deploy.README and "AIMI" in build_deploy.README


def test_sample_is_analysed_once_then_served_from_memory(tmp_path: Path) -> None:
    import app.clinic_app as clinic

    class CountingPredictor:
        calls = 0

        def predict(self, volumes):
            CountingPredictor.calls += 1
            return []

    np.save(tmp_path / "sagittal_x.npy", np.zeros((3, 256, 256), np.uint8))
    original = clinic.results_html
    clinic.results_html = lambda *args: "page"
    try:
        sample = clinic.make_sample(CountingPredictor(), clinic.Wording(), tmp_path, "note")
        assert sample()[2] == "page" and sample()[2] == "page"
    finally:
        clinic.results_html = original
    assert CountingPredictor.calls == 1


def test_local_settings_load_weights_from_this_computer() -> None:
    import app.clinic_app as clinic
    assert clinic.resolve_run_dir({"run_dir": "runs/x", "weights_repo": ""}) == REPO_ROOT / "runs" / "x"
