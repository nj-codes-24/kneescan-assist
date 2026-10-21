"""
test_explain.py: key slices and Grad-CAM give well-formed answers.

Why: the app shows these to doctors. Wrong shapes, NaNs or maps outside 0..1 would make a misleading picture.
The *meaning* of the maps is checked on real data by the randomisation test (EVAL_PROTOCOL R13), not here.
"""

from __future__ import annotations

import numpy as np
import torch

from kneescan import TASKS
from kneescan.data.transforms import to_25d_input
from kneescan.explain.gradcam import gradcam, gradcam_per_task, normalise_maps, randomisation_similarity
from kneescan.explain.key_slices import key_slices, slice_importance
from kneescan.models.reader import KneeReader


def tiny_model(seed: int) -> KneeReader:
    torch.manual_seed(seed)
    return KneeReader("resnet18", pretrained=False, att_dim=16, dropout=0.0, plane_dropout=0.0).eval()


def exam() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {p: to_25d_input(torch.randint(0, 256, (s, 64, 64), generator=gen, dtype=torch.uint8))
            for p, s in (("sagittal", 6), ("coronal", 5), ("axial", 4))}


def test_key_slices_are_sorted_and_cover_every_finding_and_plane() -> None:
    out = tiny_model(0)(exam())
    keys = key_slices(out, k=2)
    assert len(keys) == len(TASKS) * 3 * 2
    acl_sag = [k for k in keys if k.task == "acl" and k.plane == "sagittal"]
    assert acl_sag[0].weight >= acl_sag[1].weight
    assert abs(sum(slice_importance(out, "acl")["axial"]) - 1.0) < 1e-5


def test_gradcam_shape_and_range() -> None:
    maps = gradcam(tiny_model(0), exam()["coronal"], "coronal", "meniscus", [0, 3])
    assert maps.shape == (2, 64, 64)
    assert np.isfinite(maps).all() and maps.min() >= 0.0 and maps.max() <= 1.0


def test_one_pass_for_all_findings_matches_one_pass_per_finding() -> None:
    model, slices = tiny_model(0), exam()["sagittal"]
    together = gradcam_per_task(model, slices, "sagittal", {"abnormal": [1], "acl": [4], "meniscus": [0, 2]})
    for task, indices in (("abnormal", [1]), ("acl", [4]), ("meniscus", [0, 2])):
        assert np.allclose(together[task], gradcam(model, slices, "sagittal", task, indices), atol=1e-6)


def test_backbone_in_small_batches_gives_the_same_features() -> None:
    model, slices = tiny_model(0), exam()["sagittal"]               # 6 slices
    with torch.no_grad():
        whole = model.slice_features(slices)[1]
        model.inference_batch = 4
        batched = model.slice_features(slices)[1]
    assert torch.allclose(whole, batched, atol=1e-6)


def test_normalise_keeps_empty_map_empty() -> None:
    maps = normalise_maps(np.stack([np.zeros((2, 2)), np.full((2, 2), 4.0)]))
    assert maps[0].max() == 0.0 and maps[1].max() == 1.0


def test_randomisation_similarity_of_a_model_with_itself_is_high() -> None:
    model = tiny_model(0)
    slices = exam()["sagittal"]
    assert randomisation_similarity(model, model, slices, "sagittal", "acl", [0, 1, 2]) > 0.99
