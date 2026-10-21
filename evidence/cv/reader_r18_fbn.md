# reader_r18_fbn: pooled OOF on MRNet train (1130 exams, 5 folds)

**Macro AUC 0.833** (95% CI 0.816–0.849)

Gain vs b0: -0.022 (95% CI -0.034 to -0.011) -> kept: no

| Finding | AUC (95% CI) | Sens / Spec @0.5 | Sens / Spec @T_hs | ECE raw → calibrated | Brier raw → calibrated |
|---|---|---|---|---|---|
| abnormal | 0.889 (0.866–0.911) | 0.85 / 0.79 | 0.90 / 0.61 | 0.115 → 0.022 | 0.121 → 0.101 |
| acl | 0.822 (0.790–0.855) | 0.68 / 0.82 | 0.90 / 0.46 | 0.155 → 0.024 | 0.147 → 0.111 |
| meniscus | 0.787 (0.760–0.812) | 0.76 / 0.69 | 0.90 / 0.50 | 0.111 → 0.038 | 0.191 → 0.179 |

T_hs = the threshold reaching 90% sensitivity on these OOF predictions. Calibrated values are cross-fitted (each fold's calibrator was fitted on the other folds).
