"""
test_reader.py: the Reader model, its 2.5D input, the macro-AUC scorer, and a one-epoch training smoke test.

Why these tests?
    * Shapes and attention sums catch wiring mistakes before a 5-hour run does.
    * Missing planes must work, because KneeMRI and real uploads often have only one series.
    * The macro-AUC helpers decide what goes on the resume, so they get exact, hand-checkable cases.
All tests use random tiny images and an untrained backbone, so they run on CPU in seconds.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kneescan import PLANES, TASKS
from kneescan.data.mrnet import MRNetExamDataset, read_labels
from kneescan.data.splits import make_cv_folds, make_inner_tune, split_for
from kneescan.data.transforms import to_25d_input
from kneescan.eval.macro import macro_auc, macro_auc_ci, paired_macro_gain, wide_oof
from kneescan.models.reader import GatedAttention, KneeReader
from kneescan.train.config import ReaderConfig
from kneescan.train.reader_trainer import ReaderFoldJob, positive_weights
from tests.conftest import make_fake_mrnet


def tiny_reader(plane_dropout: float = 0.0) -> KneeReader:
    return KneeReader("resnet18", pretrained=False, att_dim=16, dropout=0.0, plane_dropout=plane_dropout)


def random_exam(slices: dict[str, int], hw: int = 64) -> dict[str, torch.Tensor]:
    return {p: to_25d_input(torch.randint(0, 256, (s, hw, hw), dtype=torch.uint8)) for p, s in slices.items()}


def test_25d_input_puts_neighbours_in_channels() -> None:
    vol = torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 2, 2) * 100  # slice i is filled with 100*i
    x = to_25d_input(vol)
    assert x.shape == (3, 3, 2, 2)
    raw = x * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    # middle slice: channels are slices 0, 1, 2; first slice repeats itself as its missing "previous" neighbour
    assert torch.allclose(raw[1, :, 0, 0] * 255, torch.tensor([0.0, 100.0, 200.0]), atol=1e-3)
    assert torch.allclose(raw[0, :, 0, 0] * 255, torch.tensor([0.0, 0.0, 100.0]), atol=1e-3)


def test_gated_attention_weights_sum_to_one_per_finding() -> None:
    weights = GatedAttention(in_dim=8, att_dim=4, n_heads=3)(torch.randn(10, 8))
    assert weights.shape == (10, 3)
    assert torch.allclose(weights.sum(dim=0), torch.ones(3), atol=1e-5)


def test_reader_forward_all_planes() -> None:
    model = tiny_reader().eval()
    out = model(random_exam({"sagittal": 7, "coronal": 5, "axial": 6}))
    assert out.logits.shape == (len(TASKS),)
    assert out.planes_used == list(PLANES)
    assert out.attention["coronal"].shape == (len(TASKS), 5)
    assert torch.isfinite(out.logits).all()


def test_reader_works_with_one_plane_only() -> None:
    model = tiny_reader().eval()
    out = model({"sagittal": random_exam({"sagittal": 4})["sagittal"], "coronal": None, "axial": None})
    assert out.planes_used == ["sagittal"]
    # with one plane, the vote is that plane's own logits
    assert torch.allclose(out.logits, out.plane_logits["sagittal"], atol=1e-5)


def test_plane_dropout_always_keeps_one_plane() -> None:
    model = tiny_reader(plane_dropout=0.99).train()
    for _ in range(20):
        assert len(model.choose_planes(list(PLANES))) >= 1


def test_positive_weights_balance_classes() -> None:
    labels = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    assert torch.allclose(positive_weights(labels), torch.tensor([1 / 3, 3.0, 3.0]))


def long_table(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    rows = [{"case_id": f"{i:04d}", "task": t, "label": int(labels[i, k]), "prob": float(probs[i, k])}
            for i in range(len(labels)) for k, t in enumerate(TASKS)]
    return pd.DataFrame(rows)


def test_macro_auc_is_mean_of_task_aucs() -> None:
    labels = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]])
    probs = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.2, 0.1, 0.4], [0.8, 0.3, 0.6]])  # tasks 1 and 3 perfect
    # task 2: positives score 0.8 and 0.1, negatives 0.2 and 0.3 -> 2 of 4 pairs ranked right = 0.5
    assert macro_auc(labels, probs) == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    y, p, ids = wide_oof(long_table(labels, probs))
    assert ids == ["0000", "0001", "0002", "0003"] and np.array_equal(y, labels) and np.allclose(p, probs)


def test_macro_ci_and_paired_gain() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=(200, 3))
    weak = np.clip(labels * 0.2 + rng.random((200, 3)) * 0.8, 0, 1)
    strong = np.clip(labels * 0.6 + rng.random((200, 3)) * 0.4, 0, 1)
    point, lo, hi = macro_auc_ci(labels, weak, n_boot=200)
    assert lo <= point <= hi
    gain, g_lo, g_hi = paired_macro_gain(labels, strong, weak, n_boot=200)
    assert gain > 0 and g_lo > 0      # clearly better model -> CI excludes 0 -> "kept"
    same, s_lo, s_hi = paired_macro_gain(labels, weak, weak, n_boot=50)
    assert same == s_lo == s_hi == 0.0


def test_reader_job_one_epoch_and_resume(tmp_path: Path) -> None:
    root = make_fake_mrnet(tmp_path / "data", n_train=80, n_valid=4, hw=64)  # 80 exams: enough for an inner tune split
    labels = read_labels(root, "train")
    folds = make_cv_folds(labels)
    split = split_for(0, folds, make_inner_tune(labels, folds))
    cfg = dataclasses.replace(ReaderConfig(), pretrained=False, att_dim=16, epochs=2, accumulate_exams=4,
                              amp=False, num_workers=0)
    run_dir = tmp_path / "fold0"
    job = ReaderFoldJob(cfg, root, labels, split, 0, run_dir, device="cpu")
    assert job.run(max_epochs_this_call=1) == "paused"
    resumed = ReaderFoldJob(cfg, root, labels, split, 0, run_dir, device="cpu")
    assert resumed.run() == "done"
    preds = pd.read_csv(run_dir / "preds.csv", dtype={"case_id": str})
    assert set(preds["subset"]) == {"train", "tune", "oof"}
    assert len(preds[preds["subset"] == "oof"]) == len(split.oof) * len(TASKS)
    assert preds["prob"].between(0, 1).all()


def test_frozen_batchnorm_stays_in_inference_mode_while_training() -> None:
    model = KneeReader("resnet18", pretrained=False, att_dim=16, freeze_batchnorm=True).train()
    batchnorms = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert batchnorms and all(not m.training for m in batchnorms)
    assert model.training and model.attention.training          # everything else still trains
    before = batchnorms[0].running_mean.clone()
    model(random_exam({"sagittal": 4}))
    assert torch.equal(before, batchnorms[0].running_mean)       # running statistics untouched


def test_deep_supervision_adds_the_plane_losses(tmp_path: Path) -> None:
    root = make_fake_mrnet(tmp_path / "data", n_train=80, n_valid=4, hw=64)
    labels = read_labels(root, "train")
    folds = make_cv_folds(labels)
    split = split_for(0, folds, make_inner_tune(labels, folds))
    totals = []
    for weight in (0.0, 1.0):
        cfg = dataclasses.replace(ReaderConfig(), pretrained=False, att_dim=16, amp=False, plane_dropout=0.0,
                                  dropout=0.0, plane_loss_weight=weight)
        torch.manual_seed(0)
        job = ReaderFoldJob(cfg, root, labels, split, 0, tmp_path / f"w{weight}", device="cpu")
        planes, y = job._prepare(job.train_ds[0], augment=False)
        total, fused = job._training_loss(planes, y)
        assert torch.isclose(fused.mean(), total) if weight == 0 else total > fused.mean()
        totals.append(float(total))
    assert totals[1] > totals[0]


def test_max_pooling_summary_and_slice_shares() -> None:
    from kneescan.models.reader import max_pool_slices
    features = torch.tensor([[1.0, 5.0, 0.0, 2.0], [3.0, 1.0, 0.0, 9.0]])    # 2 slices, 4 features
    summary, weights = max_pool_slices(features, n_tasks=3)
    assert torch.equal(summary[0], torch.tensor([3.0, 5.0, 0.0, 9.0])) and summary.shape == (3, 4)
    assert torch.allclose(weights[:, 0], torch.tensor([0.5, 0.5]))            # each slice supplied 2 of 4 maxima
    model = KneeReader("resnet18", pretrained=False, att_dim=16, pooling="max").eval()
    out = model(random_exam({"sagittal": 5, "axial": 3}))
    assert out.logits.shape == (3,) and torch.allclose(out.attention["axial"].sum(dim=1), torch.ones(3))


def test_exam_dataset_can_load_a_subset_of_planes(fake_mrnet: Path) -> None:
    labels = read_labels(fake_mrnet, "train")
    item = MRNetExamDataset(fake_mrnet, "train", labels, planes=("coronal",))[0]
    assert list(item["volumes"]) == ["coronal"]


def test_anti_overfitting_options() -> None:
    from kneescan.data.transforms import random_gamma, random_slice_dropout
    from kneescan.train.reader_trainer import freeze_backbone_prefixes
    gen = torch.Generator().manual_seed(0)
    vol = torch.full((4, 8, 8), 64.0)
    out = random_gamma(vol, 0.3, gen)
    assert out.shape == vol.shape and 0 <= float(out.min()) and float(out.max()) <= 255   # same shape, still in range
    assert random_gamma(vol, 0.0, gen).allclose(vol)                                       # strength 0 = unchanged
    x = torch.randn(10, 3, 8, 8)
    assert random_slice_dropout(x, 0.99, gen).shape[0] == 3                                # never fewer than 3 slices
    assert random_slice_dropout(x, 0.0, gen).shape[0] == 10
    model = KneeReader("resnet18", pretrained=False, att_dim=16)
    freeze_backbone_prefixes(model, ("conv1", "bn1", "layer1"))
    trainable = {n for n, p in model.backbone.named_parameters() if p.requires_grad}
    assert not any(n.startswith("layer1") for n in trainable) and any(n.startswith("layer4") for n in trainable)


def test_single_finding_specialist_only_trains_on_its_finding(tmp_path: Path) -> None:
    root = make_fake_mrnet(tmp_path / "data", n_train=80, n_valid=4, hw=64)
    labels = read_labels(root, "train")
    folds = make_cv_folds(labels)
    split = split_for(0, folds, make_inner_tune(labels, folds))
    cfg = dataclasses.replace(ReaderConfig(), pretrained=False, att_dim=16, amp=False, plane_dropout=0.0, dropout=0.0,
                              loss_tasks=("acl",))
    job = ReaderFoldJob(cfg, root, labels, split, 0, tmp_path / "acl_only", device="cpu")
    planes, y = job._prepare(job.train_ds[0], augment=False)
    total, per_task = job._training_loss(planes, y)
    assert torch.isclose(total, per_task[TASKS.index("acl")])     # the other findings contribute nothing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_rng_state_restores_after_being_mapped_to_the_gpu() -> None:
    from kneescan.train.seed import capture_rng_state, restore_rng_state
    gen = torch.Generator().manual_seed(1)
    state = capture_rng_state({"aug": gen})
    expected = torch.rand(3)
    # what torch.load(map_location="cuda") does to a checkpoint: tensors land on the GPU
    state["torch_cpu"] = state["torch_cpu"].cuda()
    state["generators"]["aug"] = state["generators"]["aug"].cuda()
    restore_rng_state(state, {"aug": gen})
    assert torch.equal(torch.rand(3), expected)
