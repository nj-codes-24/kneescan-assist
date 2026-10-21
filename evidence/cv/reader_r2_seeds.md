# reader_r2_seeds: pooled OOF on MRNet train (1130 exams, 5 folds)

**Macro AUC 0.854** (95% CI 0.839–0.868)

Gain vs reader_r2: +0.004 (95% CI +0.000 to +0.007) -> kept: yes

| Finding | AUC (95% CI) | Sens / Spec @0.5 | Sens / Spec @T_hs | ECE raw → calibrated | Brier raw → calibrated |
|---|---|---|---|---|---|
| abnormal | 0.900 (0.879–0.919) | 0.83 / 0.82 | 0.90 / 0.62 | 0.122 → 0.024 | 0.118 → 0.099 |
| acl | 0.852 (0.823–0.880) | 0.73 / 0.83 | 0.90 / 0.57 | 0.171 → 0.022 | 0.139 → 0.103 |
| meniscus | 0.810 (0.785–0.833) | 0.77 / 0.70 | 0.90 / 0.56 | 0.099 → 0.035 | 0.178 → 0.169 |

T_hs = the threshold reaching 90% sensitivity on these OOF predictions. Calibrated values are cross-fitted (each fold's calibrator was fitted on the other folds).
