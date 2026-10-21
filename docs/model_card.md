# Model card: KneeScan Assist

> **Research and education prototype. NOT a medical device.** Not for diagnosis, treatment or any clinical decision.

## Model
- **What it does:** from a knee MRI exam (sagittal, coronal, axial series), it estimates the probability of **any abnormality**, an **ACL
  tear** and a **meniscus tear**, with a confidence level and the key slices and heatmaps behind each estimate.
- **Architecture:** three plane specialists (the "Reader": ImageNet ResNet-18 on 2.5D slices, gated attention over slices, one score per
  finding). They are combined by an equal vote (mean of logits). Each specialist is an ensemble of 5 cross-validation models.
  Two seeds per plane (6 specialists × 5 folds = 30 networks). Details: [DESIGN.md](DESIGN.md) and [decisions.md](decisions.md) (D-014, D-022).
- **Outputs:** Platt-calibrated probabilities (calibrator fitted on out-of-fold predictions). The operating threshold per finding is the one that
  reached 90% sensitivity out-of-fold. Confidence = agreement of the 5 fold models (5/5 High, 4/5 Medium, else Low).
- **Version / files:** the exact weights are listed with SHA-256 in `docs/FROZEN.md` (frozen 2026-10-20, before the final test).

## Intended use
- **For:** teaching, research, and demonstrating to clinicians what an explainable knee-MRI triage aid could look like.
- **Not for:** patient care, screening programmes, or any decision about a real patient. It hasn't been validated prospectively or by a regulator.
- **Users:** students, researchers, clinicians *evaluating the idea*, on public or de-identified cases only.

## Training data
- **MRNet v1.0** (Stanford University Medical Center; Bien et al., PLOS Medicine 2018): 1,130 `train` exams, 3 series each, labels from
  clinical reports (abnormal 80.6%, ACL 23.3%, meniscus 37.1% over the full dataset). Stanford Research Use Agreement. See [DATA.md](DATA.md).
- ImageNet-pretrained ResNet-18 backbone weights (timm). RadImageNet was evaluated but not used (decisions.md D-022).

## Evaluation
- **Model selection:** 5-fold cross-validation on MRNet `train` (pooled out-of-fold predictions, paired bootstrap comparisons with the
  B0 baseline, a reproduction of Bien et al. 2018). Fixed in [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) before training.
- **Metrics:** per-finding AUC and macro AUC (mean of the three) with bootstrap 95% CIs; sensitivity and specificity at 0.5 and at the
  90%-sensitivity threshold (Wilson CIs); calibration (ECE, Brier), raw and calibrated.
- **CV results (out-of-fold, 1,130 exams):** B0 macro AUC 0.855 (0.839–0.869); final Reader 0.854 (0.839–0.868), tied with B0 (paired −0.001,
  −0.010 to +0.009). Full table: [experiments.md](experiments.md).
- **Final test, used once (2026-10-20):**
  - MRNet `valid` (120 exams): Reader macro AUC **0.895** (0.848–0.931); abnormal 0.929, ACL 0.942, meniscus 0.814. B0 0.904 (0.868–0.935).
    Paired Reader − B0 −0.008 (−0.032 to +0.012): a tie.
  - KneeMRI (Rijeka, 917 sagittal volumes, external, ACL only): Reader ACL AUC **0.699** (0.659–0.737), B0 0.702. A large domain-shift drop.
  - Operating point: the threshold for ≥ 90% sensitivity on OOF data, frozen before the test (FROZEN.md). Test sensitivity/specificity are in
    `results/final/final.json` (kept local).

## Explanations
- Key slices = the attention weights the model actually uses. Heatmaps = Grad-CAM on the key slice (coarse, 8 × 8 blocks).
- Randomisation check (Adebayo et al. 2018): heatmaps of trained vs random weights correlate 0.26 / 0.18 / −0.11
  (abnormal / ACL / meniscus), all below the 0.5 limit, so they reflect learning.
- Failure analysis: [failure_cases.md](failure_cases.md). Patterns: joint fluid as a shortcut, image-quality problems
  (including a noise-only slice), and off-target heatmaps.

## Limitations and risks
- **One training site:** one US hospital and one vendor (GE), specific sequences. Performance on other scanners, sequences, populations
  or children is unknown and may fail silently. KneeMRI tests one such shift (Siemens, PD-weighted, sagittal only).
- **Labels from reports:** some "errors" are likely label noise. No second expert read.
- **No patient IDs in MRNet:** CV folds can't be grouped by patient (~42 repeat exams). The official test split is patient-disjoint.
- **Small test set:** 120 exams give wide CIs (about ±0.03–0.05 AUC).
- **Shortcuts:** the model may rely on correlated signs (e.g. joint fluid with ACL tears). The heatmaps make this visible, they don't prevent it.
- **Automation bias:** a confident output can mislead. That's why the app shows model agreement, validated AUCs and the disclaimer next to every result.

## Ethics, privacy, licences
- Uses only datasets licensed for academic research. No data, labels, images or weights are redistributed (the repo audit enforces this).
- The demo reads only pixels and geometry from DICOM files, never patient tags. Uploads are deleted after reading. Locally it runs on
  127.0.0.1 only; the hosted demo (docs/HOSTING.md) asks for anonymised scans and keeps the weights in a private repository.
- Cite MRNet (Bien et al. 2018) and KneeMRI (Štajduhar et al. 2017) when reporting results.
