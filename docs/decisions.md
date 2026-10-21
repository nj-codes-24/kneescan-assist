# Decision log

Every important choice, with the reason and the evidence. Newest at the bottom.
Experiment IDs refer to `experiments/runs.csv`. Earlier design choices (2026-10-18) are in `docs/DESIGN.md`,
`docs/DATA.md` and `docs/EVAL_PROTOCOL.md`. This log starts on 2026-10-19 and doesn't repeat them.

---

## D-001: This folder (`D:\knee_mri_tool`) is the one project
Decided: All work continues here. The draft folder `D:\knee_kaggle_comp\knee-kaggle` (made on 2026-10-19 before this decision) is
abandoned; nothing in it is needed.
Why: This folder, started the day before, already holds the design,
the evaluation protocol fixed before training, tested loaders, splits and the B0 trainer (40 tests pass).
Rejected alternatives: Rebuilding from scratch in the new folder. That would duplicate a day of tested work.
Evidence: `pytest`: 40 passed, 1 skipped (2026-10-19).

## D-002: Public data only, and no link to the Kaggle competition
Decided: Train and evaluate on public datasets only: MRNet v1.0 (development + final test) and KneeMRI (external test).
No RSNA competition data, and no weights trained on it.
Why: I wanted a project of my own, independent of the competition. Public data also lets the code, results and demo be
shared without the competition's no-redistribution licence (RSNA MIRA RUA §4).
Rejected alternatives: Training on the RSNA data (allowed for research) and scoring with a Kaggle late submission. That gives a number
comparable with the Kaggle leaderboard, but the data can never be shown, so I chose a clean, separate project.
Evidence: `docs/DATA.md`.

## D-003: Kaggle-style metrics, reported only against our own baseline
Decided: The headline metric is **macro AUC**, the unweighted mean of the per-finding AUCs (same definition as the Kaggle competition),
here over MRNet's 3 findings. We also report per-finding AUC, sensitivity/specificity at fixed thresholds, calibration (ECE, Brier), and
bootstrap 95% CIs on everything (EVAL_PROTOCOL §5).
Why: A familiar, quantifiable metric. **Caveat:** a macro AUC on MRNet (3 findings, Stanford exams) can't
be compared with the Kaggle score (12 findings, other hospitals). The improvement we claim is always *our model vs our B0 baseline on
identical exams* (paired ΔAUC with CI), plus a descriptive comparison with the published MRNet paper.
Rejected alternatives: Claiming "Kaggle 0.941 → MRNet X". Different data and different labels make that claim meaningless.
Evidence: EVAL_PROTOCOL §3 (published targets), §5 (metrics).

## D-004: How we avoid the "gold-58" mistake
Decided: Pick models only on pooled out-of-fold CV over the 1,130 MRNet `train` exams. The 120-exam `valid` set and KneeMRI are each
used **once**, at the end (EVAL_PROTOCOL R1, R14–R16).
Why: In the Kaggle work, a 58-study expert set ranked models the wrong way. Its CI (about ±0.035) was wider than the differences being
compared, and some models had been selected on it. MRNet `valid` has the same weakness (120 exams, CI about ±0.03–0.05). Choosing on
it would repeat the mistake. Pooled OOF on 1,130 exams is ~10× larger, and every exam is scored by a model that never trained on it.
Rejected alternatives: Choosing on `valid`, which is what many MRNet reimplementations do. That makes the reported test score optimistic.
Evidence: the Kaggle project's final report (§3.2, not public); EVAL_PROTOCOL R5–R8.

## D-005: "Patient-level CV" on MRNet is only approximate. A known, logged limitation
Decided: Keep EVAL_PROTOCOL R3/R5. CV folds are stratified by label combination but can't be grouped by patient, because MRNet doesn't
release patient IDs (1,130 exams from 1,088 patients, so ≤42 repeat exams). The official train/valid/test splits *are* patient-disjoint,
so the final test is clean.
Why: The protocol asks for "no patient in more than one split". For the final test this holds. Inside CV it can't be checked, and saying so
is more honest than pretending.
Rejected alternatives: Guessing patient identity from image similarity. Too error-prone to verify, and it would add machinery to explain.
Evidence: MRNet site (1,130 exams / 1,088 patients); EVAL_PROTOCOL R5.

## D-006: Windows background jobs instead of tmux
Decided: Long runs start as background processes that log to a file. They write a checkpoint every epoch and use `--resume` / `--stop-at`.
Why: pc_c runs Windows with no tmux. Resumability comes from the checkpoints (DESIGN §1.6), not the terminal.
Rejected alternatives: Installing WSL or tmux, which is a system-wide change.

## D-007: The Reader gets a LayerNorm and a 10× learning rate for its new layers, found by an overfit check
Decided: (1) Apply `LayerNorm` to slice features before attention. (2) Use two learning rates: 1e-4 for the pretrained backbone and
1e-3 (`head_lr`) for everything new (norm, attention, heads, plane fusion). (3) Run `scripts/sanity_overfit.py` (memorise 32 exams)
before every new architecture gets a full CV run.
Why: The first one-epoch smoke test on real data barely learned. Abnormal tuning AUC was 0.38, and the losses sat at the
"predict 50/50" level. The overfit check confirmed it: over 8 epochs on 32 exams, the losses didn't move (abnormal 0.303 → 0.303).
A single exam *could* be fitted, so the wiring was fine. The cause was scale: pooled ResNet features are tiny (mean 0.25,
std 0.19), and with `pos_weight` balancing, only *separating* exams lowers the loss. With both fixes the same check reaches
training AUC 0.93 / 1.00 / 0.97 (abnormal / ACL / meniscus), and every loss falls (ACL 3.28 → 0.77). LayerNorm alone wasn't
enough (losses still flat).
Rejected alternatives: Raising one global LR, which would also shake the pretrained backbone harder. Removing pos_weight would
make the imbalance problem come back.
Evidence: `runs/sanity_reader_r18/metrics.csv` (three runs on 2026-10-19: no fix, LayerNorm only, both); old K recipe
`lr_bb=1e-4, lr_head=1e-3` (`knee-kaggle-old/src/knee/train.py:32`).

## D-008: The demo is a local Gradio app. Confidence = agreement of the 5 fold models
Decided: (1) The Clinic App is `app/clinic_app.py` (Gradio, ~100 lines). It replaces DESIGN §4's FastAPI + hand-written viewer +
PDF report. (2) The app runs the 5 cross-validation models as an ensemble. Probability = their mean. Confidence = how many land
on the same side of the out-of-fold T_hs threshold (5/5 High, 4/5 Medium, else Low). (3) Key slice per plane = highest mean
attention, and its heatmap = Grad-CAM averaged over the 5 models. (4) The app listens on 127.0.0.1 only and deletes uploads right
after reading them. DICOM tags other than pixels, SeriesInstanceUID, orientation and position are never accessed. Gradio analytics are off.
Why: I wanted a simple web UI and code I can explain line by line. Gradio needs no front-end code. Ensemble
agreement is a confidence measure that fits in one sentence for a clinician, and it uses models we train anyway.
Rejected alternatives: FastAPI + JS viewer + PDF (several hundred lines, three technologies). Using the probability alone as
"confidence" (it conflates "likely" with "sure"). Conformal prediction (sound, but needs another held-out set and is harder to explain).
Evidence: `tests/test_app.py` (8 passed), including a planted fake patient name that never reaches the output.

## D-009: DICOM slices are ordered along the patient axis, not the slice normal
Decided: Uploaded series are sorted by ascending patient coordinate (x for sagittal, y for coronal, z for axial).
Why: The first version sorted along the computed slice normal. A test with a sagittal orientation whose normal points to −x
showed that this reverses the stack on some scanners. File order is no help either: in the Kaggle work it was uncorrelated
with position (recap). The patient axis gives the same order on every scanner.
Evidence: `tests/test_app.py::test_dicom_zip_is_sorted_by_position_and_has_no_patient_data` (failed before the fix, passes after).

## D-010: Freeze the backbone's BatchNorm while training the Reader
Decided: `freeze_batchnorm: true`. The backbone's BatchNorm layers stay in inference mode (pretrained ImageNet statistics) during
training. Their scale and shift still learn. R1 (`reader_r18`) was stopped after fold 0 and is replaced by R1b (`reader_r18_fbn`),
identical except for this switch.
Why: A training batch is the ~20–60 slices of *one* knee. With normal BatchNorm, each exam is normalised by its own statistics
during training but by long-run averages at evaluation, so the model is tested on inputs it never trained on. Evidence on R1
fold 0, same model and same 91 inner-tuning exams: evaluation-mode BatchNorm gave AUC 0.895 / 0.754 / 0.760 (abnormal / ACL /
meniscus), per-exam statistics 0.938 / 0.805 / 0.869. R1's best epoch was also epoch 2 of 20, and its ACL tuning AUC was far below
B0's sagittal network alone (0.92). B0 doesn't have the problem: AlexNet has no BatchNorm.
Rejected alternatives: (a) Using per-exam statistics at inference too: it works, but a prediction would then depend on how many
slices a series has, which is hard to explain. (b) Switching to a BatchNorm-free backbone (ConvNeXt, LayerNorm) as the fix. That
stays as a *separate* ablation, so its effect isn't mixed up with fixing this defect.
Evidence: R1 fold 0, `runs/reader_r18/fold0/` (stopped 2026-10-19 09:45); the diagnostic is described above; `sanity_overfit.py`
on R1b passes; `tests/test_reader.py::test_frozen_batchnorm_stays_in_inference_mode_while_training`.

## D-011: Fix the Reader's training recipe before trying bigger backbones (amends the pre-registered ladder)
Decided: Postpone L1 (ConvNeXt). Screen two recipe changes on fold 1 first (S1: backbone lr 1e-5; S2: 40 epochs). See
`docs/experiments.md` Amendment 1.
Why: An early read-out on the 452 exams both models had finished showed the Reader reliably *behind* B0 (paired macro-AUC gain
−0.025, 95% CI −0.042 to −0.007), mostly on ACL. A bigger backbone trained with the same recipe would cost ~1 GPU-day and
probably inherit the problem. This change is logged *before* the screens run, so the screens can't be chosen to fit their result.
Rejected alternatives: (a) Continue the ladder as written. Pre-registration matters, but not more than not wasting a day on a known
weak recipe, and the amendment is recorded openly. (b) Tune many hyperparameters at once. Then we couldn't say which change helped.
Evidence: preview computed with `kneescan/eval/macro.py` on R1b folds 0–1 vs B0 folds 0–1 (fused, paper-faithful), 2026-10-19 15:35.

## D-012: Deep supervision. Each plane must also diagnose on its own
Decided: Screen S3 adds a per-plane loss: loss = BCE(combined answer) + `plane_loss_weight` × mean over planes of BCE(that plane's answer).
If it beats R1b, it becomes the base recipe for the full 5-fold run and the ladder.
Why: Per-plane analysis of R1b fold 1 showed the coronal branch almost useless for ACL (AUC 0.609, against 0.837 for B0's coronal
network), while the learned vote still gave it a third of the say. Training only the combined answer lets a weak plane free-ride on the
others. B0 avoids this by construction: every plane network is graded alone. A per-plane loss restores that pressure while keeping
one shared model and the attention explanations. It should also help sagittal-only exams (KneeMRI).
Rejected alternatives: (a) Separate networks per plane (B0 style): 3× the weights and three explanations to reconcile. (b) Fitting the
plane vote by logistic regression afterwards: it would down-weight coronal, but not make coronal any better. (c) S1, the lower learning
rate: +0.005, CI −0.019 to +0.030, no clear effect. (d) Later epochs: ACL doesn't improve (0.805 vs 0.811).
Evidence: `docs/experiments.md` Amendment 1 table and diagnosis; `tests/test_reader.py::test_deep_supervision_adds_the_plane_losses`.
Note: adding the `plane_loss_weight` field changes every Reader config hash, so a run started before this change (S2) can't be
resumed after a restart. It would have to start again.

## D-013: Stop guessing. Isolate why the Reader's per-plane branches are weaker than B0's networks
Decided: Two single-plane diagnostics on fold 1 (coronal only; attention vs max pooling) instead of more recipe tweaks. Two switchable
options were added for this (`planes`, `pooling`), both defaulting to the current behaviour.
Why: Three plausible recipe fixes changed fold-1 macro AUC by at most +0.005 (S1 lower LR, S3 deep supervision; S2 longer training
only overfits more). Every result points to the same gap: each Reader plane branch is weaker than B0's per-plane network, most of all
coronal (ACL 0.61–0.70 vs B0 0.80 on fold 1). A controlled single-plane comparison isolates where the gap is, at a third of the cost of a
full Reader screen.
Rejected alternatives: More hyperparameter screens (weight decay, dropout, augmentation strength). Without knowing where the gap is,
that's a random search on a 226-exam read-out.
Evidence: docs/experiments.md Amendment 1 (S1, S3, S2 curve, per-plane AUCs).

## D-014: R2 = one Reader per plane ("plane specialists"), combined by an equal logit vote
Decided: Train the Reader separately on sagittal, coronal and axial (`configs/reader_{sag,cor,ax}_att.yaml`, all 5 folds). Combine them
by averaging their logits (`kneescan/eval/plane_ensemble.py`, `scripts/combine_planes.py`). In the app, only the specialists whose plane
is in the upload vote.
Why: The coronal-only diagnostic (fold 1) scored abnormal / ACL / meniscus 0.908 / 0.841 / 0.840, above B0's coronal network
(0.901 / 0.801 / 0.790) and far above the coronal branch of the jointly trained Reader (0.848 / 0.609 / 0.726). So the per-plane design
works, and joint training is what hurt (planes interfering or free-riding; deep supervision, S3, didn't fix it). Attention beat B0-style max
pooling on the same plane (ACL 0.841 vs 0.760), so the explainable pooling stays.
Equal vote, not fitted weights: nothing to overfit, nothing to leak, and it handles missing planes naturally.
Rejected alternatives: (a) Keep fixing joint training (S1–S3 gave ≤ +0.005). (b) Fitted stacking of the planes (B0's logistic regression):
more to explain and to overfit on 1,130 exams; it can be a later ladder step if R2 leaves room. (c) Max pooling: worse on the diagnostic.
Cost: three single-plane runs ≈ the GPU time of one 3-plane run. Three explanations (one per plane) are what the app shows anyway.
Evidence: docs/experiments.md (coronal-only diagnostics table); tests/test_plane_ensemble.py; tests/test_app.py (combined run, missing plane).

## D-015: No slice-position (depth) input. The failure-gallery hunch failed the all-exam check
Decided: Don't add a depth embedding to the Reader on the basis of the failure gallery.
Why: 5 of the 12 gallery failures had a key slice at the edge of a series, which suggested the model is fooled by edge slices.
`scripts/key_slice_position.py` checked this on all 1,130 out-of-fold exams × 3 planes × 3 findings: the key slice is in the outer 15% of
its series for **18.1%** of wrong predictions and **19.6%** of right ones, so there's no difference overall. On the side view, edge key
slices are common even when correct (31–45%), which fits anatomy (meniscal horns are seen on the side slices). Only abnormal/coronal
differs (25.7% wrong vs 12.8% right).
Rejected alternatives: Adding the depth embedding anyway "because K had one". That would be a change without evidence, and the
protocol asks every trick to earn its place.
Evidence: `results/cv/reader_r2_key_slice_position.json`; docs/failure_cases.md.

## D-016: Lift the 3-miss stop rule. Remaining known ideas are tried cheapest-first
Decided: Try, in order: I1 weighted plane vote, I2 R2+B0 hybrid, I3 anti-overfitting screens, I4 medical-image pretraining (if the
licence allows), I5 per-view-per-finding specialists. Every idea still has to pass the paired-CI rule on all 1,130 OOF exams.
Why: I wanted to beat the baseline, and not everything known had been tried yet.
Risk, stated openly: every extra idea tried on the same CV exams raises the chance of a lucky CV win ("garden of forking paths"). The
protection is the untouched 120-exam test, used once, plus logging *every* idea here, including failures. With 120 test exams, a gain of
~0.01 is unlikely to be statistically clear on the test, so any claim leans on the CV paired result and the test agreeing in direction.
Evidence: docs/experiments.md "Round 2".

## D-017: Idea I3 (anti-overfitting) and I4 (RadImageNet) are screened with one change each, on the coronal specialist, fold 1
Decided: Four I3 screens (weight decay 1e-2, contrast/gamma jitter 0.3, frozen early layers conv1–layer2, slice dropout 0.2) and two I4
screens (ResNet-50 with ImageNet weights as the control; ResNet-50 with RadImageNet weights), all compared with `reader_cor_att`
fold 1 (macro 0.863). A screen whose point gain is ≥ +0.01 moves on to all 3 planes × 5 folds and the full paired rule.
Why: Overfitting was the one problem we measured repeatedly (D-011, S2). These are the standard, explainable remedies. RadImageNet tests
whether medical rather than photographic pretraining helps. The ResNet-50 ImageNet control stops "bigger network" from being mistaken for
"medical pretraining". A 226-exam screen can't prove small gains, so it only filters which ideas earn a full run.
Rejected alternatives: Running every idea on all 5 folds × 3 planes (about 15× the GPU time per idea).
Evidence: configs/reader_cor_{wd,gamma,frz,sdrop,r50,r50rad}.yaml; the RadImageNet licence check (docs/DATA.md §5b).

## D-018: Idea I5. Per-finding specialists, screened on ACL first
Decided: Added `loss_tasks` (the findings that train a model and choose its epoch) and `combine_task_runs` (per finding, average only the runs
trained for it). First screens: ACL-only coronal and sagittal specialists on fold 1, compared with the all-findings specialists' fold-1 ACL AUC.
Why: B0 still leads on ACL (CV 0.859 vs R2 0.846), and B0's design is one network per plane *and* finding, with its epoch picked per finding.
R2's specialists share one network and one epoch across three findings. ACL is where that should matter most (few slices, 23% prevalence).
Rejected alternatives: Training all 9 plane × finding specialists on 5 folds straight away (3× R2's GPU time) before a screen shows a gain.
Evidence: configs/reader_{cor,sag}_acl.yaml; tests/test_reader.py::test_single_finding_specialist_only_trains_on_its_finding;
tests/test_plane_ensemble.py::test_per_finding_combination_uses_only_that_findings_runs.

## D-019: Two infrastructure lessons. Never oversubscribe GPU memory; GPU resume was broken (fixed)
Decided: (1) Keep the total GPU memory of concurrent jobs under ~14 of 16 GB. Run ResNet-50 jobs one at a time next to at most one small
job. (2) `restore_rng_state` now moves saved RNG states to the CPU before restoring them.
Why: (1) With 5 jobs at 15.3 GB, the Windows driver silently spilled GPU memory into system RAM. One ResNet-50 epoch took 20,144 s
(5.6 h) instead of minutes, and the RadImageNet screen made no progress in 7 hours. No error, just a ~70× slowdown. (2) Stopping that
screen exposed a bug: `torch.load(map_location="cuda")` puts the saved CPU RNG state on the GPU, and restoring it crashed
("RNG state must be a torch.ByteTensor"). So every GPU resume, B0's included, would have failed. The tests resumed on the CPU only.
Evidence: runs/reader_cor_r50/fold1/metrics.csv (epoch 0: 20144.5 s); logs/reader_cor_r50_fold1.err.log (the traceback);
tests/test_reader.py::test_rng_state_restores_after_being_mapped_to_the_gpu (new, runs when a GPU is present).

## D-020: RadImageNet is retried on DenseNet-121 with gradient clipping, not on ResNet-50
Decided: Drop the ResNet-50 route. Screen DenseNet-121 (the only RadImageNet model near ResNet-18's size) with RadImageNet weights vs its own
ImageNet weights, both with `grad_clip: 1.0` (new option: cap the gradient norm per optimiser step, after unscaling).
Why: (1) ResNet-50 with ImageNet weights lost clearly to ResNet-18 on the same fold (−0.048, CI −0.075 to −0.022), so even a RadImageNet gain
on ResNet-50 would very likely stay below the current model. (2) RadImageNet ResNet-50 diverged: in a 40-step trace the scaled gradient norm
repeatedly hit inf and the loss jumped to 9.9; the run crashed on NaN predictions after its first epoch. Clipping is the standard stabiliser.
It goes on *both* DenseNet screens, so pretraining stays the only difference between them.
Rejected alternatives: Lowering only the RadImageNet learning rate (it would differ from its control). Retrying ResNet-50 (the size handicap).
Evidence: runs/reader_cor_r50/fold1; logs/reader_cor_r50rad_fold1.err.log ("Input contains NaN"); configs/reader_cor_d121{,rad}.yaml.

## D-021: RadImageNet runs in full precision (fp32), and so does its control
Decided: Both DenseNet-121 screens use `amp: false`.
Why: Diagnosis (2026-10-20 20:10). RadImageNet DenseNet-121's backbone outputs reach 3,198 in fp32, and inside the network they exceed fp16's
range (max 65,504), so in mixed precision the output is inf. The gradient scaler then skips every update: the logits were identical before
and after 5 optimiser steps. NaN reached the first evaluation. That's why gradient clipping (D-020) didn't help: it was never a divergence
problem. ImageNet weights stay well inside fp16's range (ResNet-50: max 26). The control also runs in fp32, so precision isn't a second difference.
Cost: ~2× slower and 14.4 GB of GPU memory for one job, so the two screens run strictly one after the other.
Evidence: diagnostic trace (fp16 logits non-finite, backbone max 3198.1, NaN weights 0, logits unchanged after 5 steps).

## D-022: RadImageNet dropped. Model search closed. Final candidates: R2 with 2 seeds, plus the R2+B0 hybrid
Decided: (1) Stop the RadImageNet DenseNet screen and its control. (2) The final test compares, per EVAL_PROTOCOL R8: B0 vs the best Reader
configuration by CV, **R2 with 2 seeds per plane specialist** (`runs/reader_r2_seeds`, CV 0.854), plus ONE extra pre-registered variant, **the
hybrid R2 (2 seeds) + B0** (`runs/hybrid_r2s_b0`, CV 0.864, +0.009 vs B0, CI +0.006 to +0.013). Both are named in EVAL_PROTOCOL §9 before the run.
Why: (1) Fed one MRNet exam (CPU check), the RadImageNet DenseNet-121 weights gave backbone outputs with max 1,737 for our ImageNet-normalised
input, 0.1 for [0,1] or [−1,1] inputs, and 205,333 for 0–255. No standard scaling gives a working pretrained network (ImageNet DenseNet: max 59),
and after 6 fp32 epochs ACL was at chance. Likely a problem with the published PyTorch conversion. Fixing a third party's weights is out of scope.
(2) Round 2 has no idea left with evidence of a real Reader-only gain, so the search stops (the results have plateaued). The 2-seed
model is the best Reader by CV (kept over R2: +0.004, CI +0.000 to +0.007). The weighted vote on top (0.8546) wasn't kept, so the simpler equal vote stays.
Rejected alternatives: A third seed (expected +0.001–0.002, and it can't turn a tie with B0 into a win; ~4 GPU-hours). Digging into the
RadImageNet conversion.

## D-023: A custom, minimal design for the demo (2026-10-20)
Decided: Keep Gradio, but render the page ourselves. `app/views.py` (pure functions returning HTML) and `app/theme.css` (one stylesheet with
light and dark colour tokens) give a hero header, a slim always-visible notice, one upload card, a result card per finding (large calibrated
percentage, a meter with the screening threshold marked, a "Look closely" / "Not flagged" pill, model agreement, validated AUC), and a
"Where it looked" section whose segmented control shows one heatmap strip at a time. The segmented control is pure CSS (radio buttons + labels),
with no JavaScript. Heatmaps are embedded as PNG data URIs, so nothing loads from the internet. The predictor now returns structured
`Heatmap` objects (plane, slice, attention, image) instead of captions.
Why: I wanted a modern, minimal, premium look in the spirit of Apple's design principles: clarity (one hierarchy, the result as the hero),
deference (quiet colours, white space), depth (soft cards). A results table plus a flat gallery was hard to read.
Rejected alternatives: A separate React/Svelte front end (much more code to explain). Many small Gradio components (layout control too limited).
Evidence: tests/test_app_views.py (6 tests: values, flag rule, plane order, no web links, escaping, disclaimer); checked in the live app on
2026-10-20 in light, dark and phone widths, with a real file upload and the segmented control working.

## D-024: Redesign v2. Two screens, the answer in words first, the reading guide built in (2026-10-20)
Decided: (1) **Landing stage**: one job only. A gradient background, a one-sentence promise naming the three findings, one frosted upload
card ("Analyse study" + "Try a sample study"), and one line on privacy and the disclaimer. (2) After analysis **the page switches** (Gradio
column visibility): a verdict headline from the abnormality finding ("Signs of an abnormal knee."), a subline for ACL and meniscus, then one
row per finding with a plain word (**Likely ≥ 70% / Possible / Unlikely < 30%**, cut-offs in configs/app.yaml), the percentage, a meter with
the screening line, and one built-in explanation sentence. Then "Where it looked" and a short "Know the limits" paragraph. **No dropdowns.**
"New analysis" goes back. (3) The flag sentence never contradicts the word: an "Unlikely" finding above the cautious screening line reads
"Unlikely, but above the cautious screening line (7%) that is set to catch 9 in 10 cases".
Why: The answer should be readable at first glance, with a simple page, clear communication, and the guidance inside the page rather than in
dropdowns. The first build showed "Unlikely" next to "worth a closer look" for a 16% ACL probability above a 7% screening line. Both were
true, but together they confused, so the wording was fixed and a test now guards it.
Words, not just numbers: the bands only change wording, never a model output, a threshold or a flag.
Rejected alternatives: Colouring by the screening flag alone (it flags many healthy knees by design, so everything would look alarming);
a separate "how to read" section (too much to read).
Evidence: tests/test_app_views.py (8 tests); checked live on 2026-10-20 at desktop and phone widths, in light and dark mode, with the sample
study, the heatmap switcher and "New analysis".
Note: "Try a sample study" uses the local MRNet copy in data/demo_inputs (git-ignored). Whether MRNet images may be shown to people outside
the project depends on the MRNet licence.
Update 2026-10-21: settled by D-025. The MRNet sample stays on this PC; the public demo uses a KneeMRI example.

## D-025: A public demo within the data licences (2026-10-21)
Decided: Aim for a "live demo" link and accept the grey areas below. **Allowed and done:** public code, docs and all
metrics; README figures that show no scan (docs/images/SOURCES.md). **Grey area, accepted:** (1) a public Hugging Face Space
running the app, (2) with the weights in a **private** HF model repo the Space reads with a token (nobody can download them), and
(3) an example study taken from KneeMRI, not MRNet. Exam 510057-5 (complete ACL tear) is copied **unchanged** (a format change from
.pck to .npy only), with credits; the app preprocesses it at run time, and the heatmap drawn over it is shown only live.
**Not done:** no MRNet scan anywhere public (not the sample button, screenshots or videos), no downloadable weights, nothing commercial.
Why: The MRNet Research Use Agreement (checked 2026-10-21 against the copy on the original MRNet page, archived 13 Mar 2020; the Redivis
version may differ) allows personal, non-commercial research use, forbids publishing any part of the dataset without written permission
(clause 4) and forbids derivative works of it (clause 6) without saying whether a trained model is one. I chose not to ask
Stanford for permission. KneeMRI's CC BY-NC-ND 4.0 allows sharing a verbatim copy with credits; a format change isn't an adaptation
(CC 4.0 §2(a)(4)).
How the example was chosen: the final model was run on the first 40 complete tears and 15 healthy knees in KneeMRI's metadata order,
and the case with the clearest correct answer was kept (ACL 80%, 5 of 5 models). This is a hand-picked example, not a typical one,
and the note under the verdict says so, with the external AUC (0.70). Picking it touched none of the locked final-test numbers.
Speed: on 2 CPU cores (a free Space), Grad-CAM now reuses one backbone pass for all three findings. Same probabilities and identical
heatmaps, about twice as fast: 17 s for the sagittal-only example (was 31 s) and 54 s for a 3-plane study (was 107 s). The example is
analysed once at start-up and then served from memory.
Update: the host changed from a Hugging Face Space to Azure Container Apps (D-026); the rules above are unchanged.
Rejected alternatives: Vercel (serverless functions are capped at about 250 MB; PyTorch plus 1.4 GB of weights and a long-running
Gradio server don't fit). The MRNet sample in a public demo (clause 4). Public weights (the riskiest grey item).
Evidence: tests/test_hosting.py (export refuses changed weights, layout loads, Space settings, sample cache), the Grad-CAM equivalence
test in tests/test_explain.py, two view tests; the hosted app was run locally on CPU only with the exported weights on 2026-10-21
(exported weights give identical outputs to runs/; the sample file equals the original volume).

## D-026: Host the demo on Azure Container Apps (Azure for Students), and make it fit 4 GiB (2026-10-21)
Decided: The demo runs as a Docker container on **Azure Container Apps**, consumption plan, **2 vCPU / 4 GiB**, scale to zero,
at most 1 replica (Gradio needs every request of a visitor on the same server), under an **Azure for Students** subscription.
The weights stay in the private Hugging Face model repo (D-025) and are downloaded at start-up with a read-only token stored as an
Azure secret. `scripts/build_deploy.py` makes the container folder (Dockerfile, CPU-only PyTorch, the code, the KneeMRI example).
To fit the server: (1) Grad-CAM runs the backbone once **without recording gradients** and turns gradients on only for the small
head, which is all the heatmap needs; (2) in the app the backbone reads **8 slices at a time** (`slice_batch: 8`; every slice is
processed independently, BatchNorm in inference mode); (3) the example is analysed in a background thread after the port opens.
Why: Hugging Face now requires a paid PRO plan for any Gradio Space, even on the free CPU (HTTP 402 on 2026-10-21). The goal was
no payment and no card. Modal's own billing docs say a payment method must be on file; Google Cloud Run and Oracle also need a card.
Azure for Students gives $100 for 12 months without a card (school e-mail), and Container Apps has a monthly free allowance
(180,000 vCPU-s, 360,000 GiB-s): about 25 hours awake per month at 2 vCPU / 4 GiB before the credit is touched. When the credit ends,
the resources are switched off unless you choose pay-as-you-go. A consumption-only environment allows at most 2 vCPU / 4 GiB.
Measured (2026-10-21, a 3-plane MRNet study, 2 CPU threads, this PC): the committed code before D-025 took 117 s with a 4.09 GiB peak;
with (1) and the shared pass from D-025, 55 s and 3.52 GiB (too close to 4 GiB); with (2) as well, 66 s and 2.25 GiB. Probabilities
and heatmaps are bit-identical in all three. Disabling oneDNN (4.2 GiB) or its cache (no change) didn't help; 16-slice batches gave
2.75 GiB. In the built container on Linux, limited to 2 CPUs and 4 GiB (Docker, weights mounted locally): port open after 20 s,
peak 2.0 GiB, a 3-plane upload in 48.5 s, the example in about 5 s; no restart, not out of memory.
Rejected alternatives: Hugging Face PRO (paid). Modal, Cloud Run, Oracle Always Free (card). Render/Koyeb free tiers (about 512 MB).
Streamlit Community Cloud (about 2.7 GB cap and a full rewrite of the interface). Weights inside the image (the image would then hold
MRNet-derived weights; the private HF repo keeps one controlled copy).
Evidence: tests/test_hosting.py (deploy settings, container recipe), tests/test_explain.py (8-slice batches give the same features;
one pass for all findings equals one pass per finding), the measurements above (one-off measurement scripts; the numbers are recorded here).

## D-027: No live demo; the repository is the deliverable (2026-10-21)
Decided: Drop the hosted demo. Polish the GitHub repository instead; I keep the weights privately (Google Drive, sharing
restricted; the private Hugging Face repo optionally as a second copy). The container build (D-026) stays in the repo, tested but not
deployed, with docs/HOSTING.md marked "not deployed". The README shows screenshots and the local demo instead of a link.
Why: Hugging Face charges for Gradio Spaces; the free hosts that fit 2 GB of memory need a card, the Azure for Students sign-up page
was unavailable, and Render's free tier (512 MB RAM, 0.1 CPU) can't load the weights (the app peaks at 2.0 GiB).
Shrinking the model to fit would mean showing a different model from the one the reported numbers describe.
Kept: everything that's true without a server: the memory and speed work (bit-identical outputs), the container test, the figures.

## D-028: Publish the evidence, and test the README against it (2026-10-21)
Decided: Commit the original outputs behind every number to `evidence/` (copied unchanged by `scripts/collect_evidence.py`, with
SHA-256 fingerprints): the one-time final test (final.json, final.md, the lock file), every CV report, and the per-epoch training log
and done.json of every run (149 files, 0.8 MB). `tests/test_evidence.py` checks every number in the README tables and the resume
bullets against them, that the model was frozen before the test (frozen_at 21:00:57 < run_at 21:01:17) with thresholds from CV, and
that the test ran once. GitHub Actions runs all tests, the smoke test and the audit at every push (badge in the README).
Why: Anyone can type numbers into a README. Original files plus an automatic check make them verifiable.
Safe to publish: every published file is aggregate (checked: no case IDs, exam IDs or file names; the audit now fails on any).
Per-exam predictions (preds.csv) stay private: with public exam IDs they would reveal the licensed labels.
Checked: a deliberately wrong README number (0.899 instead of 0.895) fails the test. A fresh copy of only the committable files,
without GPU, data, weights or results: 108 passed, 3 skipped (GPU-only), smoke test passed, audit passed.
