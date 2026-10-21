# EVAL_PROTOCOL.md: "The Honest Judge"

> Fixed on **2026-10-18, before any training** (pc_c). This file is the contract. Changes after
> training starts go in **§9 Amendments** only (append-only, dated, with a reason), and must be
> made *before* seeing any result they could affect.
>
> **Why so strict?** With only 120 test exams, every peek at the test set is a hidden tuning
> step. Small medical test sets make numbers look better than they are unless the rules are
> written down first. KneeScan Assist is a research prototype, **not a medical device**.

---

## 1. Data roles (never mixed)

| Set | Exams | Role | Touched when |
|---|---|---|---|
| MRNet `train` | 1,130 | Development: 5-fold CV, all tuning and all model choices | Any time |
| MRNet `valid` | 120 | **Final test**. The paper calls it the "tuning set"; for us it is untouched | **Once, at the end** |
| KneeMRI | 917 | **External test**, ACL only, sagittal only | **Once, at the end** |

**R1. Allowed before the end on test sets:** load the files and check counts, shapes, dtypes and
metadata integrity (this is what the loader tests do). **Forbidden:** running any model on
them, viewing label-vs-prediction anything, or using their images or statistics for any
preprocessing choice.

**R2. Nothing is fit on test data.** That includes normalisation statistics, thresholds,
calibrators, logistic-regression fusers, epoch selection and ensembling weights.
Intensity normalisation uses fixed ImageNet mean/std after the per-volume scaling in §6.

## 2. Cross-validation on MRNet `train`

**R3. Folds.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, stratified on the
label combination string `f"{abnormal}{acl}{meniscus}"`. Folds are saved to
`splits/mrnet_train_5fold.csv` (columns `case_id, fold`, **no labels**) and committed
**before the first training run**. Every model uses the same folds.

**R4. Inner tuning split (for epoch selection).** Inside each outer training split (4 folds),
10% is held out (stratified as above, `random_state=42+fold`) as the *inner tuning set*.
Checkpoint/epoch selection uses only this set. The outer held-out fold is never used for
selection, so out-of-fold (OOF) predictions stay clean. Saved to `splits/mrnet_inner_tune.csv`.

**R5. Known limitation.** MRNet does not release patient IDs, so folds cannot be grouped by
patient. About 42 train exams are repeat exams of a patient already in `train`. CV may be slightly
optimistic. The final test does not have this problem: the official splits are patient-disjoint.

**R6. CV score = pooled OOF.** For each task we pool the 1,130 OOF predictions and compute one
AUC, with a bootstrap CI as in §5. We also report the per-fold mean ± SD.
**Model-selection metric:** the mean of the three pooled-OOF task AUCs.
Every configuration we try is logged (`runs/` + `lab_log.jsonl`), including the failed ones.
The winner's CV score is optimistic because it was selected, and that is exactly why the final
test exists.

## 3. Baseline "B0": reproduction of Bien et al. (2018)

This is the "beat this" number. It follows the paper as closely as the paper allows:

- **Per-plane, per-task CNN** (3 planes × 3 tasks = 9 CNNs per fold, 45 in total).
  Each slice goes through an ImageNet-pretrained **AlexNet** feature extractor, then global
  average pooling (256-d), then **max-pool across slices**, then a linear layer and sigmoid.
- **Loss:** BCE weighted inversely to class prevalence (computed on the training split).
- **Augmentation:** rotation U(−25°, 25°), shift U(−25, 25) px, horizontal flip p = 0.5.
- **Optimiser:** the paper does not state it. We use Adam, lr 1e-5, weight decay 0.01,
  ReduceLROnPlateau (factor 0.3, patience 5), 50 epochs, batch = 1 exam. These are the values
  common in public reimplementations; logged as a stated assumption.
- **Epoch selection:** lowest loss on the inner tuning set (R4). The paper's equivalent was its
  tuning set.
- **Plane fusion:** one logistic regression per task on the 3 plane probabilities. As in the
  paper, it is **fit on the training split's own predictions**. (Variant B0-oof, reported
  alongside: an LR fit on inner-tuning predictions, to show how much the faithful version
  over-trusts training predictions.)
- **Final-test prediction:** the mean of the 5 fold pipelines' probabilities (each = 3 CNNs + LR).
- **External test (KneeMRI):** the mean of the 5 fold **sagittal ACL CNN** outputs, with no LR.
  This matches the paper's external experiment.
- **Published target (Bien et al. 2018, their hidden test set, threshold 0.5):**

  | Task | AUC (95% CI) | Sensitivity | Specificity |
  |---|---|---|---|
  | Abnormal | 0.937 (0.895–0.980) | 0.879 | 0.714 |
  | ACL | 0.965 (0.938–0.993) | 0.759 | 0.968 |
  | Meniscus | 0.847 (0.780–0.914) | not given in text | 0.741 |
  | KneeMRI ACL, no retraining | 0.824 (0.757–0.892) | n/a | n/a |

  **Not directly comparable:** the paper evaluated on the *hidden* test set, and evaluated
  KneeMRI on its own 183-exam split, which is not published. We evaluate on the public
  `valid` set and on all 917 KneeMRI exams. The comparison with the paper is descriptive.
  The **statistical comparison is our model vs our B0**, on identical data.

## 4. Our model ("The Reader", see DESIGN.md)

**R7.** Same folds, inner tuning split and selection rule as B0. Final-test prediction is the
mean of the 5 fold models. Any recalibrator (Platt / temperature) or threshold is fit on
**pooled OOF predictions only**.

**R8. Pre-declared final comparisons (only these go to the final test):**
1. B0 (and B0-oof) vs **the single Reader configuration with the best CV score (R6)**.
2. At most **one** extra pre-registered variant, named in §9 before the final run.

## 5. Metrics (computed identically for every model)

- **AUC** (ROC), with a **95% CI from a percentile bootstrap**: 2,000 exam-level resamples,
  seed 20261018.
- **Paired comparison Reader − B0:** ΔAUC with a paired-bootstrap 95% CI (same resampled exams
  for both models). We report the CI. Expect wide intervals with n = 120.
- **Sensitivity and specificity, with Wilson 95% CIs, at two fixed thresholds:**
  - **T₀.₅ = 0.5**: the paper's threshold, for comparability.
  - **T_hs (high-sensitivity)**: the threshold that achieves sensitivity ≥ 0.90 on pooled OOF,
    frozen in `docs/FROZEN.md` before the final run. This is the screening use case.
  - Thresholds are never re-chosen on test data.
- **Calibration:**
  - A reliability diagram with 10 equal-width bins and a bin-count histogram, plus the same
    diagram with 10 equal-mass bins.
  - **ECE** (10 equal-width bins, count-weighted) with a bootstrap CI, plus the **Brier score**.
  - Both **raw** and **OOF-recalibrated** probabilities are reported. The app uses the
    recalibrated ones.
- Also reported: ROC curves, confusion matrices at both thresholds, and positive/negative
  counts per task.

## 6. External test: KneeMRI (ACL only)

- **R9. Labels:** positive = partially injured **or** completely ruptured (as in Bien et al.).
  Secondary analysis: complete rupture vs healthy, with partials excluded.
- **R9b. Unit of analysis:** one prediction per volume (917), as in the paper. The metadata
  shows 917 volumes from 909 `examId`s (7 exams have 2–3 series), so the KneeMRI bootstrap
  **resamples whole exams** (cluster bootstrap by `examId`). Note: `aclDiagnosis` in
  `metadata.csv` is coded 0/1/2, while READ.ME numbers the classes 1/2/3.
- **R10. Preprocessing, fixed now without looking at the images:** take each volume as given
  in the `.pck` (axis order checked against the bundled example, read-only); clip
  per-volume intensities to the [0.5, 99.5] percentile; scale to [0, 255] uint8; resize
  in-plane to 256 × 256 (bilinear); then apply the same pipeline as MRNet. We use the
  **whole volume** and ignore the ROI box for prediction.
- **R11. Missing planes:** only sagittal exists. B0 uses its sagittal ACL CNN. The Reader runs
  with coronal and axial **masked**; it is trained for this with plane dropout (DESIGN.md).
- **R12. Explanation check (secondary, descriptive):** on ACL-positive volumes, the
  *pointing rate* is the share of volumes where (a) the top-attention slice lies in the
  ROI's slice range, and (b) the Grad-CAM maximum on that slice lies inside the ROI box.
  We report it with a Wilson CI. It is not used for any decision.

## 7. Explanation sanity check (on CV data, any time)

- **R13. Model-randomisation test (Adebayo et al., 2018):** Grad-CAM maps and attention
  rankings must change substantially when the trained weights are replaced by random ones
  (Spearman rank correlation of the maps < 0.5). If they don't, the explanation is not
  showing what the model learned, and the app must say so.

## 8. The one-time final run

- **R14. Freeze first:** before running, `docs/FROZEN.md` records the git commit, configs,
  SHA-256 of every weight file, T_hs values, calibrator parameters, and the exact list of
  models (R8). This is committed.
- **R15. One script, one run:** `scripts/final_eval.py` checks that FROZEN.md matches the
  files on disk, runs both test sets, writes `results/final/` and a `LOCK` file, and
  **refuses to run again** unless given `--rerun-reason "<text>"`. Every rerun is logged and
  reported next to the first result, never instead of it.
- **R16. Everything gets reported:** all R8 models on all metrics, including results that are
  worse than the paper or worse than B0.

## 9. Amendments (append-only)

### A1. 2026-10-20, written BEFORE the final run. Which models go to the final test (R8), and five deviations
**Models (R8):**
1. **B0** (`runs/b0`, §3), as specified. Only the faithful variant is tested; B0-oof stays a CV-only number to keep the test small.
2. **The Reader configuration with the best CV score:** `runs/reader_r2_seeds`. That's three plane specialists (sagittal, coronal, axial), each
   ResNet-18 + gated attention, 2 seeds per plane, combined by an equal vote (mean logit). CV macro AUC 0.854 (0.839–0.868).
3. **The one extra pre-registered variant:** the hybrid `runs/hybrid_r2s_b0`, the mean logit of (2) and (1). CV macro AUC 0.864 (0.849–0.877),
   paired gain over B0 +0.009 (+0.006 to +0.013).

**Deviations from the text above, all decided on CV data only and logged in docs/decisions.md:**
- **R3 / R5:** unchanged (folds and inner tuning split as written; no patient grouping possible).
- **§4 "single Reader configuration":** the best Reader is an ensemble of single-plane Readers (D-014: joint training made planes interfere),
  with 2 seeds (L3). Each component follows R4/R7 (epoch chosen on the inner tuning set).
- **R11 (missing planes):** the Reader handles KneeMRI's sagittal-only exams by letting only its sagittal specialists vote, not by plane
  dropout (single-plane specialists don't need it).
- **§5 thresholds:** T_hs (≥ 90% sensitivity on pooled OOF) and the Platt calibrators are taken from `results/cv/<run>.json` for each of the
  three models and frozen in `docs/frozen.yaml`. The hybrid has its own (computed on its own OOF predictions).
- **§6 KneeMRI:** sensitivity/specificity at T_hs are not reported there. The thresholds were set on 3-plane MRNet exams, and the Reader votes
  with one plane only on KneeMRI, so they aren't meaningful. AUC (primary, exam-clustered CI) and complete-vs-healthy AUC (secondary) are
  reported for all three models.
- **R12 (pointing rate vs the KneeMRI ACL box):** deferred to after the final run, as a descriptive explainability check. It doesn't affect any choice.

**What happens to these numbers:** reported once, next to each other, whatever they are (R16). No model, threshold or calibrator changes after this.
