# hybrid_r2s_b0: pooled OOF on MRNet train (1130 exams, 5 folds)

**Macro AUC 0.864** (95% CI 0.849–0.877)

Gain vs b0: +0.009 (95% CI +0.006 to +0.013) -> kept: yes

| Finding | AUC (95% CI) | Sens / Spec @0.5 | Sens / Spec @T_hs | ECE raw → calibrated | Brier raw → calibrated |
|---|---|---|---|---|---|
| abnormal | 0.904 (0.883–0.923) | 0.90 / 0.69 | 0.90 / 0.71 | 0.035 → 0.022 | 0.098 → 0.096 |
| acl | 0.867 (0.837–0.894) | 0.62 / 0.92 | 0.90 / 0.57 | 0.036 → 0.014 | 0.099 → 0.096 |
| meniscus | 0.821 (0.796–0.844) | 0.68 / 0.80 | 0.90 / 0.57 | 0.046 → 0.034 | 0.166 → 0.164 |

T_hs = the threshold reaching 90% sensitivity on these OOF predictions. Calibrated values are cross-fitted (each fold's calibrator was fitted on the other folds).
