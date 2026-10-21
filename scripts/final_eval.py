"""
final_eval.py: the ONE-TIME final test (EVAL_PROTOCOL §8). B0 vs the frozen Reader on MRNet `valid` (120 exams) and KneeMRI (917 volumes).

What it does
    1. Checks every weight file against the SHA-256 recorded in docs/frozen.yaml (scripts/freeze.py). Any change → stop.
    2. Refuses to run if results/final/LOCK exists, unless --rerun-reason "<text>" is given. Every rerun is logged in the LOCK file
       and reported NEXT TO the first result, never instead of it (R15).
    3. Predicts (kneescan/eval/final_predict.py) and scores (kneescan/eval/final_metrics.py) with the FROZEN thresholds and calibrators:
         MRNet valid: per finding and macro AUC for both models, the paired Reader − B0 gains, sensitivity/specificity, calibration.
         KneeMRI (ACL only, sagittal only): AUC with exam-clustered CIs for both, the paired gain, and the secondary complete-vs-healthy AUC.
    4. Writes results/final/final.json and final.md (git-ignored, because they hold per-test-set details), then the LOCK.

Usage
    .venv\\Scripts\\python.exe -I scripts\\final_eval.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))     # -I (isolated mode, for the KneeMRI pickles) drops the script folder from the path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from kneescan import TASKS  # noqa: E402
from kneescan.data.kneemri import KneeMRIDataset, read_metadata  # noqa: E402
from kneescan.data.mrnet import find_mrnet_root  # noqa: E402
from kneescan.eval import final_predict as fp  # noqa: E402
from kneescan.eval.calibration import logit  # noqa: E402
from kneescan.eval.final_metrics import paired_report, task_report  # noqa: E402
from kneescan.eval.macro import macro_auc_ci, paired_macro_gain, wide_oof  # noqa: E402
from kneescan.eval.metrics import auc, bootstrap_ci  # noqa: E402

OUT = REPO_ROOT / "results" / "final"
LOCK = OUT / "LOCK"


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_freeze(frozen: dict) -> None:
    for role, entry in frozen["models"].items():
        for rel, expected in entry["weights"].items():
            if sha256(REPO_ROOT / rel) != expected:
                raise SystemExit(f"{role}: {rel} differs from docs/frozen.yaml. Refusing to run (R14)")


def check_lock(rerun_reason: str | None) -> None:
    if LOCK.exists() and not rerun_reason:
        raise SystemExit("the final test has already run (results/final/LOCK). Use --rerun-reason \"...\" and report both runs")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} run started; reason: {rerun_reason or 'first run'}\n")


def mean_logit(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """The hybrid: equal-weight mean of two models' logits, back to a probability (as in scripts/make_hybrid.py)."""
    return 1.0 / (1.0 + np.exp(-(logit(p_a) + logit(p_b)) / 2))


def mrnet_section(frozen: dict, root: Path, device: torch.device) -> dict:
    b0 = fp.b0_predict_mrnet(REPO_ROOT / frozen["models"]["baseline"]["run"], root, "valid", device)
    reader = fp.reader_predict_mrnet(REPO_ROOT / frozen["models"]["reader"]["run"], root, "valid", str(device))
    section = {"n_exams": int(b0["case_id"].nunique())}
    roles = [("baseline", b0), ("reader", reader)]
    if "hybrid" in frozen["models"]:
        roles.append(("hybrid", reader.assign(prob=mean_logit(reader["prob"].to_numpy(), b0["prob"].to_numpy()))))
    for role, preds in roles:
        entry = frozen["models"][role]
        y, p, _ = wide_oof(preds)
        section[role] = {"macro_auc": list(macro_auc_ci(y, p)),
                         "per_task": {t: task_report(y[:, k], p[:, k], entry["thresholds_t_hs"][t], tuple(entry["platt"][t]))
                                      for k, t in enumerate(TASKS)}}
    y, p_reader, _ = wide_oof(reader)
    _, p_b0, _ = wide_oof(b0)
    section["reader_minus_baseline"] = {"macro": list(paired_macro_gain(y, p_reader, p_b0)),
                                        "per_task": {t: paired_report(y[:, k], p_reader[:, k], p_b0[:, k]) for k, t in enumerate(TASKS)}}
    if "hybrid" in frozen["models"]:
        p_hybrid = mean_logit(p_reader, p_b0)
        section["hybrid_minus_baseline"] = {"macro": list(paired_macro_gain(y, p_hybrid, p_b0)),
                                            "per_task": {t: paired_report(y[:, k], p_hybrid[:, k], p_b0[:, k]) for k, t in enumerate(TASKS)}}
    return section


def kneemri_section(frozen: dict, device: torch.device) -> dict:
    meta = read_metadata(REPO_ROOT / "data" / "kneemri" / "raw" / "metadata.csv")
    dataset = KneeMRIDataset(REPO_ROOT / "data" / "kneemri" / "volumes", meta)
    b0 = fp.b0_predict_kneemri(REPO_ROOT / frozen["models"]["baseline"]["run"], dataset, device)
    reader = fp.reader_predict_kneemri(REPO_ROOT / frozen["models"]["reader"]["run"], dataset, str(device))
    clusters = b0["exam_id"].to_numpy()
    y = b0["acl_tear"].to_numpy()
    section = {"n_volumes": len(b0), "n_exams": int(b0["exam_id"].nunique()), "positives": int(y.sum())}
    roles = [("baseline", b0), ("reader", reader)]
    if "hybrid" in frozen["models"]:
        roles.append(("hybrid", reader.assign(prob=mean_logit(reader["prob"].to_numpy(), b0["prob"].to_numpy()))))
    for role, preds in roles:
        point, lo, hi = bootstrap_ci(y, preds["prob"].to_numpy(), auc, clusters=clusters)
        complete = preds[(preds["acl_complete"] == 1) | (preds["acl_tear"] == 0)]   # R9 secondary: partial tears excluded
        section[role] = {"acl_auc": [point, lo, hi],
                         "acl_complete_vs_healthy_auc": auc(complete["acl_complete"], complete["prob"])}
    section["reader_minus_baseline"] = paired_report(y, reader["prob"].to_numpy(), b0["prob"].to_numpy(), clusters)
    if "hybrid" in frozen["models"]:
        hybrid = mean_logit(reader["prob"].to_numpy(), b0["prob"].to_numpy())
        section["hybrid_minus_baseline"] = paired_report(y, hybrid, b0["prob"].to_numpy(), clusters)
    return section


def markdown(report: dict) -> str:
    m, k = report["mrnet_valid"], report["kneemri"]
    lines = [f"# Final test ({report['run_at']}): B0 vs Reader `{report['reader_run']}`", "",
             f"## MRNet valid ({m['n_exams']} exams, never used before)", "",
             "| Model | Macro AUC (95% CI) | " + " | ".join(f"{t} AUC (95% CI)" for t in TASKS) + " |", "|---|---|" + "---|" * len(TASKS)]
    for role in [r for r in ("baseline", "reader", "hybrid") if r in m]:
        r = m[role]
        cells = [f"{r['per_task'][t]['auc']:.3f} ({r['per_task'][t]['auc_ci'][0]:.3f}–{r['per_task'][t]['auc_ci'][1]:.3f})" for t in TASKS]
        lines.append(f"| {role} | {r['macro_auc'][0]:.3f} ({r['macro_auc'][1]:.3f}–{r['macro_auc'][2]:.3f}) | " + " | ".join(cells) + " |")
    g = m["reader_minus_baseline"]["macro"]
    lines += ["", f"Paired macro-AUC gain Reader − B0: **{g[0]:+.3f}** (95% CI {g[1]:+.3f} to {g[2]:+.3f})"]
    if "hybrid_minus_baseline" in m:
        h = m["hybrid_minus_baseline"]["macro"]
        lines += [f"Paired macro-AUC gain Hybrid − B0: **{h[0]:+.3f}** (95% CI {h[1]:+.3f} to {h[2]:+.3f})"]
    lines += ["",
              f"## KneeMRI external test ({k['n_volumes']} volumes, {k['n_exams']} exams, {k['positives']} ACL tears; sagittal only)", "",
              "| Model | ACL AUC (95% CI, exam-clustered) | Complete tear vs healthy AUC |", "|---|---|---|"]
    for role in [r for r in ("baseline", "reader", "hybrid") if r in k]:
        a = k[role]["acl_auc"]
        lines.append(f"| {role} | {a[0]:.3f} ({a[1]:.3f}–{a[2]:.3f}) | {k[role]['acl_complete_vs_healthy_auc']:.3f} |")
    d = k["reader_minus_baseline"]
    lines += ["", f"Paired ACL AUC gain Reader − B0 on KneeMRI: **{d['delta_auc']:+.3f}** (95% CI {d['delta_ci'][0]:+.3f} to {d['delta_ci'][1]:+.3f})"]
    if "hybrid_minus_baseline" in k:
        h = k["hybrid_minus_baseline"]
        lines += [f"Paired ACL AUC gain Hybrid − B0 on KneeMRI: **{h['delta_auc']:+.3f}** (95% CI {h['delta_ci'][0]:+.3f} to {h['delta_ci'][1]:+.3f})"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    frozen = yaml.safe_load((REPO_ROOT / "docs" / "frozen.yaml").read_text(encoding="utf-8"))
    check_freeze(frozen)
    check_lock(args.rerun_reason)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_mrnet_root(REPO_ROOT / "data" / "mrnet")
    report = {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "reader_run": frozen["models"]["reader"]["run"],
              "rerun_reason": args.rerun_reason, "mrnet_valid": mrnet_section(frozen, root, device),
              "kneemri": kneemri_section(frozen, device)}
    stamp = "" if not args.rerun_reason else "_rerun_" + dt.datetime.now().strftime("%Y%m%d_%H%M")
    (OUT / f"final{stamp}.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (OUT / f"final{stamp}.md").write_text(markdown(report), encoding="utf-8")
    with open(LOCK, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} run finished -> final{stamp}.json\n")
    print(markdown(report))


if __name__ == "__main__":
    main()
