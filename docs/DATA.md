# DATA.md: datasets, licences, and what we may do with them

> **KneeScan Assist is a research prototype, not a medical device.** Every dataset
> below is used for non-commercial academic research only.

**Hard rule:** this project uses **no RSNA Kaggle competition data and no weights trained on it**
(this includes our own "K" models and the nartaa and heliosli weights). Allowed inputs are the
datasets listed here plus ImageNet-pretrained backbones (see §5).

Last checked: 2026-10-18 (pc_c). Licences change, so re-check the source page before any publication.

---

## Summary

| Dataset | Role in this project | Licence | Access | Size | Status |
|---|---|---|---|---|---|
| **MRNet v1.0** (Stanford) | Development (5-fold CV on train) + **final test** (official validation, 120 exams) | Stanford Research Use Agreement (non-commercial, no redistribution; terms summarised in decision D-025) | Registration via Stanford AIMI | ~5–6 GB | Access granted 2026-10-19. Goes in `data/mrnet` |
| **KneeMRI** (Rijeka, Zenodo) | **External test**, ACL only, used once | CC BY-NC-ND 4.0 | Open, no signup | 3.7 GB archive | Licence accepted 2026-10-18. Downloaded |
| fastMRI + fastMRI+ | Optional (localisation check for the Explainer) | NYU data sharing agreement + MIT-licensed annotations repo | Signup at NYU | large | Not planned yet |
| OAI | Optional (more external testing) | NIMH Data Archive data use certification | NDA account | very large | Not planned yet |

---

## 1. MRNet v1.0 (Stanford ML Group)

- **Source:** https://stanfordmlgroup.github.io/competitions/mrnet/ → download via Stanford AIMI
  (https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c).
- **Paper:** Bien N, Rajpurkar P, et al. *Deep-learning-assisted diagnosis for knee magnetic
  resonance imaging: Development and retrospective validation of MRNet.* PLOS Medicine 15(11):
  e1002699, 2018. https://doi.org/10.1371/journal.pmed.1002699 (the article itself is CC BY 4.0).
- **Licence:** a Stanford *Research Use Agreement*, accepted at registration.
  **Verbatim text: TODO.** The agreement is shown on the registration page, which I can't open.
  Once you register, please paste the text, or save it as `docs/licences/MRNet_RUA.txt`, and I
  will quote the operative clauses here.
  Stanford's standard AIMI template for its other datasets (e.g. INSPECT, EHRSHOT) grants use
  "for personal, non-commercial research purposes only" and forbids redistribution, sale and
  re-identification. Some versions also forbid "derivative works" of the dataset. The MRNet
  agreement, read on 2026-10-21 (archived copy of the original MRNet page, 2020), says the same: decision D-025.
- **What we may do under the RUA:**
  - Allowed: Train and evaluate models for academic research. Present the results, the aggregate
    metrics and a small number of figures in an academic talk, with citation.
  - Not allowed: Redistribute images, preprocessed arrays, or the label CSVs. `data/` is git-ignored, and
    split files in the repo hold **case IDs + fold numbers only, never labels**.
  - Not allowed: Commercial use of any kind, and any attempt to re-identify patients.
  - Not allowed: showing MRNet slices publicly (clause 4). Open question: whether trained weights count as a
    "derivative work" (clause 6); they stay private (D-025).
- **Redivis release (checked 2026-10-18):** owner AIMI, reference
  `aimi.mrnet_knee_mri_s:4a2c:v1_0`, DOI 10.71718/rcbp-8c35. There are 8 tables: `train`
  (3,390 files) and `valid` (360 files) file tables, plus 6 label tables. **Known defect:**
  the label tables have 1,129 / 119 rows, because the original header-less CSVs lost their
  first exam to the column names (`_0000`, `_0`). `scripts/fetch_mrnet.py` repairs this, and
  `tests/test_fetch_mrnet.py` checks the repair. Downloads use your own Redivis login
  (device-code flow, approved in your browser). Credentials never pass through the
  project code or logs.
- **Size / format:** 1,370 exams (1,312 patients) from Stanford University Medical Center.
  The Redivis page says 56.6% of exams were acquired at 3.0 T, the rest at 1.5 T (GE scanners).
  Each exam has three series: sagittal T2 (fat-sat), coronal T1, axial PD (fat-sat). Each series
  is a `.npy` array of shape `(slices, 256, 256)`, uint8, with a variable slice count.
  `.npy` files can be loaded with `allow_pickle=False` (safe).
- **Splits as released:**
  - `train`: 1,130 exams / 1,088 patients. We use it for **development** (5-fold CV).
  - `valid`: 120 exams / 111 patients. This is the paper's *"tuning"* set and our **final test**.
  - hidden test: 120 exams / 113 patients. Never released; this is what the paper reports on.
  - Per the website, "all exams from a patient are in the same split" across released splits.
    **Patient IDs are not released, so our CV folds inside `train` cannot be grouped by patient.**
    About 42 train exams are repeat exams of the same patient (1,130 − 1,088), so CV estimates
    may be slightly optimistic. EVAL_PROTOCOL.md records this.
- **Labels:** three binary labels per exam, extracted from clinical reports: `abnormal`
  (80.6% overall), `acl` tear (23.3%), `meniscus` tear (37.1%). By construction, an ACL or
  meniscus tear implies abnormal.

## 2. KneeMRI (Štajduhar et al., Clinical Hospital Centre Rijeka)

- **Source:** https://zenodo.org/records/14789903 (Zenodo mirror, created 2025-02-02, of the
  original 2017 release).
- **Paper:** Štajduhar I, Mamula M, Miletić D, Ünal G. *Semi-automated detection of anterior
  cruciate ligament injury from MRI.* Computer Methods and Programs in Biomedicine 140:151–164,
  2017. https://doi.org/10.1016/j.cmpb.2016.12.006
- **Licence (quoted from the record):** "This dataset is licensed under a Creative Commons
  Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)."
  The record asks users to acknowledge the source (Clinical Hospital Centre Rijeka, Croatia)
  and cite the paper. **Accepted on 2026-10-18.**
- **What we may do under CC BY-NC-ND 4.0:**
  - Allowed: Download, store and use it privately for non-commercial research, including running a
    model on it. CC 4.0 lets you *produce* adapted material for yourself; it forbids *sharing* it.
  - Allowed: Report aggregate results (AUC etc.) with attribution and citation.
  - Not allowed: **Share** any adapted material: converted volumes, crops, resampled arrays, or images
    with heatmap overlays. To be safe, any public KneeMRI figure is an unmodified slice with
    attribution. Heatmap demos use MRNet or our own sample cases (see the open question above).
    Update 2026-10-21 (D-025): the public demo's example is one KneeMRI exam shared **unchanged** (format change only)
    with credits. The heatmap over it is drawn live in the demo, never published as a file; an accepted grey area.
  - Not allowed: Commercial use.
  - Not allowed: Re-uploading the dataset anywhere. Point people to the Zenodo link instead.
- **Size / format:** 917 volumes, 12-bit grayscale, sagittal **PD-weighted fat-suppressed**
  only, Siemens Avanto 1.5 T, exams from 2006–2014.
  `volumetric_data.7z` (3.7 GB; MD5 `669d47b0f7771304bf2dcd97b152c9e6`), plus `metadata.csv`,
  `READ.ME`, `example.py` and `example.pck`.
  - Caution: **Volumes are Python pickles (`.pck`).** Unpickling an untrusted file can run arbitrary
    code. Our loader uses a *restricted unpickler* that only allows the numpy classes needed to
    rebuild an array, and refuses everything else. We never run the bundled `example.py`.
- **Labels:** per-volume ACL diagnosis (`aclDiagnosis` column): 0 = healthy (690),
  1 = partially injured (172), 2 = completely ruptured (55). There are 917 volumes from 909
  exams (7 exams have more than one series). Other columns: `examId, seriesNo, kneeLR, roiX,
  roiY, roiZ, roiHeight, roiWidth, roiDepth, volumeFilename`. Volumes are indexed `[z, y, x]`. There is also a manually drawn rectangular ROI (bounding box)
  around the ACL region.
  **Our binarisation (matches Bien et al. 2018): positive = partial or complete (1 or 2).**
- **Known domain shift vs MRNet:** sagittal PD-FS instead of sagittal T2-FS, a different
  scanner and field strength, a different country, and no coronal or axial series.
  This makes it a real external test.

## 3. fastMRI (NYU) + fastMRI+ annotations (optional, not planned yet)

- **Sources:** https://fastmri.med.nyu.edu/ (raw/DICOM data, requires a signed data sharing
  agreement) and https://github.com/microsoft/fastmri-plus (bounding-box pathology annotations
  for the knee set. The repo shows an MIT licence, but it is unclear whether that covers the
  annotation files).
- The README quotes the data terms: "You will be able to download and use the data for academic
  purposes after signing the data sharing agreement."
- **Possible use:** checking whether Grad-CAM points at the annotated lesion (localisation).
  Mostly coronal knee series, so this would test the coronal branch.
- **Before using:** read the NYU agreement and the fastMRI+ licence, and quote them here before
  accepting.

## 4. Osteoarthritis Initiative (OAI) (optional, not planned yet)

- **Source:** NIMH Data Archive, https://nda.nih.gov/oai. Requires an NDA account and a data
  use certification.
- **Possible use:** a second external test for meniscus findings (semi-quantitative MOAKS
  readings exist for a subset). The cohort is older, mostly osteoarthritis patients, a
  different population.
- **Before using:** read and quote the NDA data use certification before accepting.

## 5. Pretrained weights

- ImageNet-pretrained CNN backbones from `torchvision` / `timm`. The code is BSD/Apache-licensed.
  ImageNet itself is non-commercial research only, which fits this project. We record the exact
  weight name and hash in each run's log.
- **Never:** any checkpoint trained on RSNA Kaggle competition data (K models, nartaa, heliosli).

## 5b. RadImageNet pretrained weights (idea I4, added 2026-10-20)

- **Source:** https://github.com/BMEII-AI/RadImageNet, PyTorch models via the authors' Google Drive link in the README
  (file id 1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR, a zip with DenseNet121 / InceptionV3 / ResNet50). Only `ResNet50.pt` is used.
  SHA-256 `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`. Stored in `data/pretrained/` (git-ignored).
- **Paper:** Mei X, et al. *RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning.*
  Radiology: Artificial Intelligence 4(5): e210315, 2022. https://doi.org/10.1148/ryai.210315
- **Licence:** the repository shows an MIT licence. The README states no separate licence for the weights, and other listings disagree
  (a third-party Hugging Face copy says MIT, the RSNA Atlas says CC BY 4.0). **I went with the MIT reading on 2026-10-20**. Weights are never redistributed by this project. Cite the paper if any result uses them.
- **Safety:** the file is a pickle. It's loaded only with `torch.load(..., weights_only=True)` (tensors only, no code), and the
  zip member paths were checked before extraction.

## 6. Storage and handling rules

- All data lives under `data/` (git-ignored): `data/mrnet/`, `data/kneemri/raw/`, `data/kneemri/volumes/`.
- Downloaded archives are untrusted. They are extracted into their own folder, and our Python
  reads them with `-I` (isolated mode). We never execute files that ship inside a dataset.
- The repo commits only code, docs, configs, and split files that hold case IDs + folds.
- Uploaded DICOMs in the Clinic App are processed in memory or a temp folder and deleted at the
  end of the session. No patient identifiers are written to logs (see DESIGN.md §4).
