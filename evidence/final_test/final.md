# Final test (2026-10-20T21:01:17): B0 vs Reader `runs/reader_r2_seeds`

## MRNet valid (120 exams, never used before)

| Model | Macro AUC (95% CI) | abnormal AUC (95% CI) | acl AUC (95% CI) | meniscus AUC (95% CI) |
|---|---|---|---|---|
| baseline | 0.904 (0.868–0.935) | 0.934 (0.879–0.976) | 0.934 (0.890–0.971) | 0.843 (0.766–0.910) |
| reader | 0.895 (0.848–0.931) | 0.929 (0.854–0.978) | 0.942 (0.901–0.975) | 0.814 (0.737–0.885) |
| hybrid | 0.905 (0.863–0.938) | 0.934 (0.866–0.980) | 0.941 (0.898–0.974) | 0.840 (0.766–0.906) |

Paired macro-AUC gain Reader − B0: **-0.008** (95% CI -0.032 to +0.012)
Paired macro-AUC gain Hybrid − B0: **+0.001** (95% CI -0.011 to +0.011)

## KneeMRI external test (917 volumes, 909 exams, 227 ACL tears; sagittal only)

| Model | ACL AUC (95% CI, exam-clustered) | Complete tear vs healthy AUC |
|---|---|---|
| baseline | 0.702 (0.661–0.741) | 0.749 |
| reader | 0.699 (0.659–0.737) | 0.755 |
| hybrid | 0.706 (0.666–0.744) | 0.759 |

Paired ACL AUC gain Reader − B0 on KneeMRI: **-0.003** (95% CI -0.021 to +0.018)
Paired ACL AUC gain Hybrid − B0 on KneeMRI: **+0.004** (95% CI -0.006 to +0.015)
