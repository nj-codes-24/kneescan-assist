"""
KneeScan Assist: a knee-MRI decision-support RESEARCH PROTOTYPE.

NOT A MEDICAL DEVICE. Not for diagnosis or treatment decisions.

Sub-packages
------------
- kneescan.data    : dataset loaders (MRNet, KneeMRI), splits, image transforms
- kneescan.models  : neural networks (B0 = Bien et al. 2018 baseline; later the Reader)
- kneescan.train   : config, seeding, logging, the resumable training loop
- kneescan.eval    : metrics (AUC + bootstrap CIs, sensitivity/specificity, calibration)

See docs/DESIGN.md for the big picture and docs/EVAL_PROTOCOL.md for the rules.
"""

MACHINE_NAME = "pc_c"  # written into every log line, so logs from different machines never mix

PLANES: tuple[str, ...] = ("sagittal", "coronal", "axial")
TASKS: tuple[str, ...] = ("abnormal", "acl", "meniscus")
