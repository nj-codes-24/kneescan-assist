# evidence: the original outputs behind every number

Everything here was written by the project's own scripts during the work and copied **unchanged** by
`scripts/collect_evidence.py`. `SHA256SUMS.txt` holds a fingerprint of every file. Nothing here names an individual exam.

## What's here

| Folder | What it is | Written by | Backs up |
|---|---|---|---|
| `final_test/final.json`, `final.md` | the one-time final test: AUCs with 95% CIs per model and finding, paired differences, the KneeMRI external test | `scripts/final_eval.py`, 2026-10-20 21:01–21:14 | README *Results* |
| `final_test/LOCK.txt` | the lock file: one start (reason "first run") and one finish. A second run would need `--rerun-reason` and would be listed here | `scripts/final_eval.py` | "the test set was used once" |
| `cv/*.json`, `cv/*.md` | cross-validation reports: pooled out-of-fold AUCs with CIs, calibration, the 90%-sensitivity thresholds, paired comparison with B0 | `scripts/score_cv.py` (B0: `scripts/eval_b0_cv.py`) | README *Cross-validation*, docs/experiments.md |
| `cv/reader_r2_r13.json` | the heatmap randomisation check (Adebayo et al.): trained vs untrained maps | `scripts/explain_cases.py` | docs/failure_cases.md |
| `cv/reader_r2_key_slice_position.json` | where the key slices fall within each series | `scripts/key_slice_position.py` | docs/failure_cases.md |
| `training_logs/<run>/fold<k>/metrics.csv` | every epoch of every training run: train/tune loss, tune AUC per finding, learning rate, seconds | the trainers (`kneescan/train/`) | docs/experiments.md (incl. the failed and dropped runs) |
| `training_logs/<run>/fold<k>/done.json` | best epoch, config fingerprint (`config_hash`), finish time | the trainers | which config produced which run, and when |

## How to check it

- **README numbers = these files.** `tests/test_evidence.py` compares every headline number in README.md
  with `final.json` and the CV reports. It runs on every push (GitHub Actions, see the badge in README.md). Change one digit and
  the test fails.
- **Frozen before tested.** `docs/frozen.yaml` was written at `frozen_at` (21:00:57 on 20 Oct 2026); `final.json` says the test started
  20 seconds later, at `run_at` (21:01:17), with exactly those weights. The frozen operating thresholds equal the CV ones in `cv/*.json`, so they weren't tuned on the test set.
  The test checks both.
- **Which weights.** `docs/frozen.yaml` lists the SHA-256 of every weight file used in the final test. The weights themselves are
  private (they're trained on MRNet, whose licence doesn't allow redistribution).
- **Unchanged since collection.** Recompute the fingerprints and compare with `SHA256SUMS.txt`
  (`tests/test_evidence.py::test_files_match_their_fingerprints` does it).

## What's not here, and why

- **Per-exam predictions** (`runs/*/fold*/preds.csv`): with the public exam IDs they would reveal the dataset labels, which the
  MRNet and KneeMRI licences don't let us redistribute. With your own copy of MRNet you can retrain and compare (README, *How to run*).
- **Weights and scans:** private (decisions D-025 and D-027).
