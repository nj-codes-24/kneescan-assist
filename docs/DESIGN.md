# DESIGN.md: KneeScan Assist architecture (one page)

> **Research prototype for decision support. Not a medical device. Not for clinical use.**
> It reads a knee MRI exam (sagittal, coronal and axial series) and outputs three
> probabilities, *abnormal*, *ACL tear* and *meniscus tear*, with an explanation for each.

```
 DICOM / .npy ─► [1 The Reader] ─► 3 probabilities ─► [4 The Clinic App] ─► viewer + PDF
                     │  attention weights, feature maps        ▲
                     └──────────► [2 The Explainer] ───────────┘ key slices + heatmaps
 [3 The Honest Judge] scores everything above under docs/EVAL_PROTOCOL.md
```

## 1. The Reader (the model)

Per plane `p ∈ {sag, cor, ax}`, a volume has `S` slices (17 to 61 in MRNet) of 256 × 256.

1. **2.5D slice input.** Each slice is stacked with its two neighbours, giving 3 channels
   `[i−1, i, i+1]`. This gives the 2D CNN some 3D context. It also fits ImageNet-pretrained
   RGB weights directly. *Choose differently:* a full 3D CNN if we had much more data,
   because 3D models overfit on 1,130 exams.
2. **Backbone.** One shared ImageNet-pretrained 2D CNN (default **ResNet-18** from timm;
   EfficientNet-B0 and ConvNeXt-Tiny are CV ablations), plus a learned plane embedding.
   The output is a 512-d feature per slice, plus the last conv feature map, which Grad-CAM
   uses. Sharing the backbone across planes means 3× fewer weights, which matters on a small
   dataset. Separate backbones per plane are an ablation.
3. **Attention pooling over slices.** Gated attention MIL (Ilse et al. 2018): it gives
   `a_{p,t,i}` = softmax over slices of `wᵀ(tanh(V h_i) ⊙ σ(U h_i))`. There is **one attention
   head per (plane, task)**, so each finding gets its own key slices. Plane feature:
   `z_{p,t} = Σ_i a_{p,t,i} h_i`. The paper's baseline uses max pooling instead; ours is
   softer and more readable.
4. **Plane fusion.** Each plane-task embedding goes to a plane logit `ℓ_{p,t}`. The three logits
   are combined by a learned weighted sum, with weights renormalised over the planes that are
   present. **Plane dropout** during training (each plane dropped with p = 0.2, at least one
   kept) makes the model robust to missing series. This is needed for KneeMRI (sagittal only)
   and for real uploads. *Choose differently:* concatenating features before the head is
   stronger when all planes are always present; it is a CV ablation.
5. **Outputs and loss.** Three sigmoid heads, using BCE with a per-task `pos_weight`.
   Platt recalibration is fit on OOF only. If the outputs disagree (for example ACL high but
   abnormal low), they are shown as-is and flagged, never silently "fixed".
6. **Training infrastructure (shared with B0).** Each YAML config is hashed. Seeds are set
   in Python, NumPy and torch (cuDNN deterministic where available). Mixed precision is used.
   **A checkpoint is written every epoch** (model, optimiser, scheduler, AMP scaler, RNG
   states, epoch, best score), and `--resume` continues from `last.pt`. `--stop-at HH:MM`
   finishes the current epoch and exits cleanly, for the 18:00–08:00 GPU window after the
   free period. Logs go to CSV + TensorBoard per run, plus `lab_log.jsonl` (machine `pc_c`,
   git SHA, config hash).

**Baseline B0** (Bien et al. reproduction, EVAL_PROTOCOL §3) uses the same data and training
code, with an AlexNet backbone, max pooling over slices, one CNN per plane × task, and LR fusion.

## 2. The Explainer

- **Key slices:** for each finding, the top-3 slices per plane ranked by `a_{p,t,i}`, shown
  with their attention weight.
- **Heatmaps:** Grad-CAM for the task logit on the backbone's last conv block, computed on
  key slices only (cheap enough on CPU). It is upsampled to 256 × 256 and overlaid.
- **Honesty built in:** the model-randomisation sanity test (EVAL_PROTOCOL R13) runs in
  `tests/`. On KneeMRI we measure the pointing rate against the ACL ROI box (R12). The UI
  says: *"Heatmaps show where the model looked, not proof of a tear. Resolution is coarse
  (8 × 8 for ResNet-18)."* *Choose differently:* Grad-CAM++ or occlusion maps would be
  sharper but slower; occlusion is a later option for the demo.

## 3. The Honest Judge

This part is fully specified in **docs/EVAL_PROTOCOL.md**, fixed before training. In short:

- **Development:** 5-fold CV on MRNet train (patient grouping isn't possible, as noted).
- **Tests:** MRNet valid (120 exams) and KneeMRI (917 exams, ACL), each run **once** via a
  frozen, locked script.
- **Metrics:** AUC with a bootstrap CI, paired ΔAUC vs B0, sensitivity/specificity at 0.5 and
  at an OOF-chosen high-sensitivity threshold, reliability plots, ECE and Brier score.
- **Reported:** everything, including losses.

## 4. The Clinic App

- **Backend: FastAPI** (`app/api.py`).
  - `POST /studies` (zip of DICOMs) → `study_id`
  - `GET /studies/{id}`: the series found and the plane assigned to each
  - `PUT /studies/{id}/series` lets the user correct the plane/sequence mapping
  - `POST /studies/{id}/analyse` → findings
  - `GET /studies/{id}/slices/{plane}/{i}.png?overlay=acl`
  - `GET /studies/{id}/report.pdf`
- **DICOM handling (pydicom):** series are grouped by `SeriesInstanceUID` and sorted by
  `ImagePositionPatient`. The plane is assigned from the `ImageOrientationPatient` normal
  vector (dominant axis x → sagittal, y → coronal, z → axial). The sequence type is guessed
  from `SeriesDescription` and **confirmed by the user**, because the model expects MRNet-like
  contrasts. Pixels are resampled to 256 × 256 with the same intensity scaling as training.
- **Privacy:** only pixel arrays and the geometry tags are kept. Studies stay in a temp
  folder that is deleted at session end or after 1 h. Patient identifiers are never logged or
  put in URLs. The PDF shows "Study A". The app runs locally (127.0.0.1) by default.
- **Viewer:** plain HTML + vanilla JS, served by FastAPI, with no build step. It has a slice
  scroller per plane, a heatmap opacity slider, jump-to-key-slice buttons, and a findings
  panel showing probability, the threshold used, and the model's validated AUC with its CI.
  A permanent banner reads *"Research prototype, not a medical device."*
- **PDF summary (reportlab, pure Python):** findings table, key slices with overlays, model
  version + weights hash, evaluation summary, limitations, and the disclaimer.
  *Choose differently:* WeasyPrint gives nicer HTML→PDF, but needs GTK on Windows.

## 5. Repository layout (planned)

```
kneescan/            Python package
  data/              mrnet.py, kneemri.py (restricted unpickler), dicom_io.py, transforms.py
  models/            backbones.py, attention.py, reader.py, baseline_mrnet.py
  train/             loop.py (checkpoint/resume/stop-at), cv.py, logging.py, seed.py
  eval/              metrics.py (AUC+bootstrap, Wilson, ECE), calibration.py, plots.py
  explain/           gradcam.py, key_slices.py
app/                 api.py, static/ (viewer), report_pdf.py
configs/             base.yaml, b0_alexnet.yaml, reader_resnet18.yaml
scripts/             make_splits.py, train_cv.py, final_eval.py, fetch_kneemri.py
splits/  docs/  tests/  runs/ (git-ignored)  data/ (git-ignored)
```

## 6. Build order and GPU plan

1. Setup, loaders + tests, and splits committed (no GPU needed).
2. **B0 5-fold, 45 small AlexNet runs.** Start the day MRNet arrives, while the GPU is free
   24/7. Estimated one day or less; this will be measured on the first fold.
3. Reader CV and ablations, in the remaining 24/7 window. After that, only overnight with `--stop-at 08:00`.
4. Explainer, then the app (CPU work, daytime).
5. Freeze, final run once, then the README results and model card.
