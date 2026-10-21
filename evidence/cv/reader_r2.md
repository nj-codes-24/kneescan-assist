# reader_r2: pooled OOF on MRNet train (1130 exams, 5 folds)

**Macro AUC 0.850** (95% CI 0.835–0.865)

Gain vs b0: -0.005 (95% CI -0.015 to +0.005) -> kept: no

| Finding | AUC (95% CI) | Sens / Spec @0.5 | Sens / Spec @T_hs | ECE raw → calibrated | Brier raw → calibrated |
|---|---|---|---|---|---|
| abnormal | 0.896 (0.875–0.915) | 0.83 / 0.82 | 0.90 / 0.63 | 0.128 → 0.027 | 0.122 → 0.100 |
| acl | 0.846 (0.816–0.874) | 0.70 / 0.83 | 0.90 / 0.52 | 0.158 → 0.026 | 0.137 → 0.107 |
| meniscus | 0.809 (0.784–0.833) | 0.77 / 0.71 | 0.90 / 0.55 | 0.099 → 0.045 | 0.178 → 0.170 |

T_hs = the threshold reaching 90% sensitivity on these OOF predictions. Calibrated values are cross-fitted (each fold's calibrator was fitted on the other folds).
