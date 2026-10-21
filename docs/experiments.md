# Experiments: every change, scored and kept or dropped

How to read it:
- **Score** = pooled out-of-fold **macro AUC** over abnormal / ACL / meniscus, on the 1,130 MRNet `train` exams (5-fold CV).
- **CI** = paired bootstrap 95% CI of the gain over the previous kept model, on the same exams.
- **Kept?** = yes only if the CI excludes 0.
- **Stop rule:** stop after 3 consecutive experiments that gain less than their CI.

Raw numbers are in `experiments/runs.csv`. The final test numbers go in the README, once (EVAL_PROTOCOL §8).

| ID | Change | Why | Score (95% CI) | Gain vs previous (95% CI) | Kept? |
|---|---|---|---|---|---|
| B0 | Bien et al. 2018 reproduction (AlexNet, max-pool, 9 CNNs + LR) | The "beat this" baseline | **0.855** (0.839–0.869); abnormal 0.896, ACL 0.859, meniscus 0.809 | — | baseline |
| R1 | Reader: ResNet-18, 2.5D, gated attention, plane voting | Stronger, explainable, one model | **stopped after fold 0**: BatchNorm defect (D-010). Fold-0 tuning AUC 0.895 / 0.754 / 0.760 | — | no (defect) |
| R1b | R1 + frozen BatchNorm | Fix the train/eval mismatch of D-010 | 0.833 (0.816–0.849); abnormal 0.889, ACL 0.822, meniscus 0.787 | vs B0: **−0.022 (−0.034 to −0.011)** | no (reliably worse than B0) |

## Pre-registered ladder (written 2026-10-19 10:00, before R1b or B0 had a CV score)

Each step changes **one** thing relative to the last *kept* model. It is scored with `scripts/score_cv.py --compare <last kept run>`
and kept only if the paired 95% CI of the macro-AUC gain is entirely above 0. The ladder stops after 3 consecutive steps that aren't
kept, or when the list runs out. The final model is the simplest one whose macro AUC lies within the best model's CI.

| Step | Change (one at a time) | Why it might help | Cost |
|---|---|---|---|
| L1 | Backbone ResNet-18 → ConvNeXt-Tiny (ImageNet-22k) | stronger features; LayerNorm, so no BatchNorm issue; the Kaggle K model used it | ~3–4× GPU time |
| L2 | 20 → 35 epochs (same cosine schedule) | R1's best epoch came early; with frozen BatchNorm, longer training may now pay off | +75% time |
| L3 | Ensemble of 3 seeds per fold (mean probability) | averages out training noise; cheap at inference | 3× training |
| L4 | Test-time augmentation: mean over the original + left-right flipped volume | flips were used in training; cheap at inference | 2× inference only |
| L5 | Plane dropout 0.2 → 0 | checks whether robustness to missing planes costs accuracy when all planes exist | same |

L4 and L5 are "advanced tricks" under the protocol's rule. They stay only if their gain beats the CI.

### Amendment 1 (2026-10-19 15:40): fix the recipe before the ladder
**Observed, before any official score:** on the 452 exams of folds 0–1 (the only folds both models had finished), R1b scored macro AUC
0.840 (0.814–0.866) vs B0 0.865 (0.843–0.888). Paired gain −0.025 (−0.042 to −0.007): reliably *worse*, mostly in ACL (0.836 vs 0.879).
R1b's training loss keeps falling (0.78 → 0.40) while tuning loss plateaus near epoch 10. ACL is still rising at epoch 20.
**Change to the plan:** L1 (ConvNeXt, ~1 GPU-day) is postponed. Running it on a recipe that loses to B0 would waste that day.
First, two one-change screens on **fold 1** (the weakest fold) decide which recipe gets the full 5-fold run:

| Screen | Change vs R1b | Hypothesis | Read-out (fold-1 OOF, 226 exams) |
|---|---|---|---|
| S1 | backbone lr 1e-4 → 1e-5 | 1e-4 overwrites the pretrained features on ~800 exams; B0 uses 1e-5 | macro AUC + paired gain vs R1b fold 1 and vs B0 fold 1 |
| S2 | 20 → 40 epochs | ACL is under-trained | same |

A screen is only a direction finder (226 exams → wide CI). The winner then runs all 5 folds and is judged by the usual full-CV paired rule.
The fold-1 run of a screen is reused as fold 1 of that full run, so no GPU time is wasted. After this, the ladder L1–L5 resumes on top of the best recipe.

**Screen results (fold 1, 226 OOF exams; direction finders, not ladder decisions):**

| Model | Macro AUC (95% CI) | Abnormal | ACL | Meniscus | Paired gain vs R1b fold 1 |
|---|---|---|---|---|---|
| B0 (fold 1) | 0.853 (0.817–0.886) | 0.916 | 0.849 | 0.793 | — |
| R1b (fold 1) | 0.837 (0.797–0.871) | 0.904 | 0.811 | 0.795 | — |
| S1 backbone lr 1e-5 | 0.842 (0.802–0.876) | 0.900 | 0.818 | 0.807 | +0.005 (−0.019 to +0.030): no clear effect |
| R1b's last epoch instead of the selected one (diagnostic) | — | 0.918 | 0.805 | 0.802 | ACL doesn't improve later, so epoch selection isn't the cause |
| S2 40 epochs | 0.826 (0.787–0.861) | 0.906 | 0.789 | 0.782 | −0.011 (−0.029 to +0.006): no gain; overfits (train loss 0.55 → 0.25, tuning loss 0.63 → 0.78) |
| S3 deep supervision (each plane also trained alone) | 0.836 (0.798–0.870) | 0.906 | 0.815 | 0.787 | −0.001 (−0.017 to +0.016): no effect |

**Diagnosis that motivated S3** (R1b fold 1, per-plane logits): ACL AUC sagittal 0.782, **coronal 0.609**, axial 0.805, combined 0.811.
The learned plane vote stayed near equal (0.29–0.37), so the weak coronal plane drags every answer down. B0's planes, trained
separately, score coronal ACL 0.837 (folds 0–1) and gain a lot from combining (0.879).

**S3 per-plane (fold 1):** ACL sagittal 0.758 (R1b 0.782), coronal **0.698** (R1b 0.609), axial 0.811 (R1b 0.805). Deep supervision lifts
the weak plane but costs the others, so the total doesn't move.

**Next diagnostic (D-013):** the Reader trained on the coronal series only, with attention (`reader_cor_att`) and with B0-style max pooling
(`reader_cor_max`), on fold 1. Reference: B0's coronal networks on fold 1, abnormal 0.901 / ACL 0.801 / meniscus 0.790. If the
coronal-only Reader matches B0, the problem is training planes together. If it stays far below, it's inside the per-plane pipeline,
and attention vs max pooling is the first suspect. These are diagnostics, not candidate models.

**Coronal-only diagnostics (fold 1, D-013) settled it:**

| Fold 1 | Abnormal | ACL | Meniscus |
|---|---|---|---|
| B0 coronal network | 0.901 | 0.801 | 0.790 |
| **Reader coronal-only, attention** | **0.908** | **0.841** | **0.840** |
| Reader coronal-only, max pooling | 0.884 | 0.760 | 0.796 |
| coronal branch *inside* the 3-plane R1b | 0.848 | 0.609 | 0.726 |

The Reader's per-plane design beats B0's per-plane network, and attention beats max pooling. What hurt was training the three planes
*together* in one shared network. → **R2 = three single-plane Readers (plane specialists), equal logit vote** (D-014), running on all
5 folds since 20:40. It will be scored against B0 and R1b with the usual full-CV paired rule.

| R2 | Three plane specialists (Reader per plane, attention), mean of logits | Joint training made planes interfere (D-013) | **0.850** (0.835–0.865); abnormal 0.896, ACL 0.846, meniscus 0.809 | vs R1b: **+0.017 (+0.008 to +0.027)**; vs B0: −0.005 (−0.015 to +0.005), tied | **yes** over R1b; tied with B0 |

R2's single specialists alone (pooled OOF, 1,130 exams): sagittal 0.801, coronal 0.832, axial 0.824 macro AUC. Their vote (0.850) beats each one.

## Ladder on top of R2 (the pre-registered L1–L5, applied to the plane specialists)

| Step | Change vs R2 | Result | Paired gain (95% CI) | Kept? |
|---|---|---|---|---|
| L1 screen | coronal specialist ResNet-18 → ConvNeXt-T (fold 1, 226 exams) | macro 0.866 vs 0.863; abnormal 0.917 vs 0.908, ACL 0.838 vs 0.841, meniscus 0.843 vs 0.840 | +0.003 (−0.019 to +0.024) | **no** (miss 1 of 3). ~3.5× GPU cost, training loss 0.07 = heavy memorising |
| L2 | 20 → 35 epochs | **skipped**: screen S2 already showed that longer training only overfits on this data (train loss 0.55 → 0.25, tuning loss up; fold-1 −0.011) | — | not run |
| L3 | Seeds per specialist, mean logit. **Amended 09:30: 2 seeds (42, 43) first** | 2 seeds: macro **0.854** (0.839–0.868); abnormal 0.900, ACL 0.852, meniscus 0.810 (seed 43 alone 0.847) | vs R2 **+0.0037 (+0.0001 to +0.0075)**; vs B0 −0.001 (−0.010 to +0.009) | **yes** over R2 (barely); still tied with B0 |
| L4 | mirror TTA (original + left-right flip, mean logit) | macro 0.850 (0.834–0.865); abnormal 0.897, ACL 0.844, meniscus 0.810 | 0.000 (−0.002 to +0.002) | **no** (miss 2 of 3). Dropped: no gain for 2× inference cost |
| L5 | plane dropout 0.2 → 0 | **not applicable** to single-plane specialists (no plane to drop) | — | not run |

## Round 2 of ideas (2026-10-20: the remaining known ideas; the 3-miss stop rule is lifted, D-016)

| Idea | Change | Macro AUC (95% CI) | Gain vs R2 | Gain vs B0 | Kept? |
|---|---|---|---|---|---|
| I1 | Weighted plane vote per finding: logistic regression on the 3 specialists' logits, **cross-fitted** (fold k's weights learned on the other folds' OOF) | 0.853 (0.838–0.867); abnormal 0.893, ACL 0.851, meniscus 0.814 | +0.003 (−0.002 to +0.008) | −0.002 (−0.012 to +0.009) | no (too small) |
| I2 | **R2 + B0 hybrid**: mean of R2's and B0's logits (equal weights, nothing fitted) | **0.863 (0.849–0.877)**; abnormal 0.903, ACL 0.866, meniscus 0.821 | **+0.013 (+0.007 to +0.020)** | **+0.0085 (+0.005 to +0.012)** | **yes: first reliable gain over B0** |

I2 caveat: the hybrid contains B0. The honest claim is "adding the attention Reader to the published baseline improves it", not "our
model alone beats the baseline". R2 and B0 logits correlate 0.89 / 0.82 / 0.82 (abnormal / ACL / meniscus), so they make partly different
mistakes, which is why averaging helps. Ideas I3–I5 aim at a Reader that beats B0 alone.

**I3 screens (coronal specialist, fold 1, 226 exams, vs `reader_cor_att` macro 0.863)**, via `scripts/compare_screen.py`:

| Screen | Macro AUC (95% CI) | Abnormal / ACL / meniscus | Paired macro gain | Verdict |
|---|---|---|---|---|
| weight decay 1e-4 → 1e-2 | 0.849 (0.814–0.883) | 0.888 / 0.834 / 0.827 | −0.014 (−0.029 to +0.001) | dropped |
| contrast (gamma) jitter 0.3 | 0.809 (0.766–0.847) | 0.885 / 0.773 / 0.767 | **−0.054 (−0.085 to −0.027)**, reliably worse | dropped. In MRI, signal brightness (e.g. bright fluid) *is* diagnostic information, so jittering it removes signal |
| frozen early layers (conv1–layer2) | 0.827 (0.785–0.864) | 0.882 / 0.789 / 0.809 | **−0.036 (−0.065 to −0.011)**, reliably worse | dropped. Adapting the early layers to MRI matters |
| slice dropout 0.2 | 0.856 (0.816–0.891) | 0.915 / 0.804 / 0.848 | −0.007 (−0.034 to +0.020) | dropped (ACL −0.037) |

All four standard anti-overfitting remedies failed on this data (I3 closed).

**More round-2 results (2026-10-20 evening):**

| Idea | Result | Gain | Verdict |
|---|---|---|---|
| I5 screen: coronal specialist trained on ACL only (fold 1) | ACL AUC 0.838 vs 0.841 for the all-findings specialist | −0.003 (−0.048 to +0.042) | no gain on coronal (sagittal ACL screen running) |
| 2 seeds + cross-fitted weighted vote | macro 0.8546 (0.839–0.869) | vs 2 seeds +0.0006 (−0.003 to +0.004); vs B0 −0.0003 (−0.010 to +0.009) | the two small gains don't stack |
| I4 ResNet-50 screens | first attempt invalid: GPU memory oversubscribed (see D-019); rerunning one at a time | — | pending |

**Per-plane diagnosis (all 1,130 OOF exams, macro AUC per plane):** B0's per-plane networks vs our specialists (1 seed / 2 seeds):
sagittal 0.812 vs 0.801 / 0.809; coronal 0.829 vs 0.832 / 0.836; axial 0.834 vs 0.824 / 0.837.
→ Plane by plane, the Reader already matches B0. Both designs hit the same ceiling on 1,130 exams, but with partly different errors
(logit correlation 0.82–0.89), which is why the R2+B0 hybrid (I2) is the only reliable gain so far. Beating B0 with a Reader alone needs a
better *per-plane* model, which is what I4 (RadImageNet) tests.
| I5 screen: sagittal specialist trained on ACL only (fold 1) | ACL AUC 0.753 vs 0.772 for the all-findings specialist | −0.020 (−0.059 to +0.018) | worse. Learning all 3 findings together helps ACL; I5 closed |
| Formal scoring | R2 with 2 seeds: 0.854 (0.839–0.868), +0.004 vs R2 (kept). Hybrid R2(2 seeds)+B0: **0.864 (0.849–0.877), +0.009 vs B0 (+0.006 to +0.013)** | | `runs/reader_r2_seeds`, `runs/hybrid_r2s_b0` |
| I4 screen: coronal ResNet-50, ImageNet weights (fold 1) | macro 0.815 (0.776–0.853); abnormal 0.889, ACL 0.763, meniscus 0.793 | **−0.048 (−0.075 to −0.022)** vs ResNet-18 | bigger network clearly worse (overfits on 1,130 exams) |
| I4 screen: coronal ResNet-50, RadImageNet weights | **training diverged** in epoch 1 (NaN predictions): gradients repeatedly overflowed and the loss spiked (D-020) | — | invalid. ResNet-50 is the wrong size anyway |
| I4 second try: DenseNet-121 (ResNet-18-sized), RadImageNet vs ImageNet weights, both with gradient clipping 1.0 | running (RadImageNet first, then the ImageNet control) | pending | pending |
| I4 DenseNet-121 RadImageNet (fp32, clip 1.0, fold 1) | **stopped after 6 epochs**: tuning AUC abnormal 0.69, ACL 0.47–0.53 (chance), meniscus 0.63 (ResNet-18: ~0.90 / 0.80 / 0.80 by then) | — | dropped. The weights don't behave like a working pretrained network under any standard input scaling (D-022); the ImageNet control was not needed |

**Round 2 closed (2026-10-20 ~21:00).** Of the 5 ideas, only I2 (hybrid with B0) reliably beats B0. Seeds (L3) is kept over R2 but ties B0.
The model search stops here: no remaining idea has evidence of a real gain, and each extra try on the same CV exams raises the risk of a
lucky win (D-016).

## ONE-TIME FINAL TEST (2026-10-20 21:01–21:14, `scripts/final_eval.py`, frozen models in docs/FROZEN.md, EVAL_PROTOCOL §9 A1)

**MRNet `valid` (120 exams, never used before):**

| Model | Macro AUC (95% CI) | Abnormal | ACL | Meniscus |
|---|---|---|---|---|
| B0 (reproduction of Bien et al. 2018) | **0.904** (0.868–0.935) | 0.934 (0.879–0.976) | 0.934 (0.890–0.971) | 0.843 (0.766–0.910) |
| Reader `reader_r2_seeds` | 0.895 (0.848–0.931) | 0.929 (0.854–0.978) | **0.942** (0.901–0.975) | 0.814 (0.737–0.885) |
| Hybrid `hybrid_r2s_b0` | **0.905** (0.863–0.938) | 0.934 (0.866–0.980) | 0.941 (0.898–0.974) | 0.840 (0.766–0.906) |

Paired macro gain vs B0: Reader **−0.008 (−0.032 to +0.012)**, Hybrid **+0.001 (−0.011 to +0.011)**. Both ties.

**KneeMRI external test** (917 volumes, 909 exams, 227 ACL tears; sagittal only; exam-clustered CIs): ACL AUC B0 0.702 (0.661–0.741), Reader 0.699
(0.659–0.737), Hybrid 0.706 (0.666–0.744). Complete tear vs healthy: 0.749 / 0.755 / 0.759. Paired vs B0: Reader −0.003 (−0.021 to +0.018),
Hybrid +0.004 (−0.006 to +0.015).

**Reading, honestly:**
- The CV gain of the hybrid (+0.009, CI +0.006 to +0.013) **did not replicate** on the 120-exam test (+0.001). As warned in D-016, a 120-exam test
  can't confirm a gain of that size, and some of the CV gain may be selection optimism from trying many ideas on the same exams.
- Our B0 reproduction lands close to the paper's own hidden-test numbers (abnormal 0.934 vs 0.937, ACL 0.934 vs 0.965, meniscus 0.843 vs 0.847),
  which supports the reproduction being faithful. The two tests are different 120-exam sets, so this is descriptive only.
- All models drop to ACL AUC ≈ 0.70 on KneeMRI (Siemens 1.5 T, PD-weighted, Croatia): a large domain shift. Bien et al. reported 0.824 on
  their own 183-exam KneeMRI split (EVAL_PROTOCOL §3); a different subset, so not directly comparable.
