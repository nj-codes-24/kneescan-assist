# Failure cases of R2 (three plane specialists): what went wrong, and what it suggests

Source: `scripts/explain_cases.py --runs runs/reader_r2 --k 2` (R2 = the seed-42 half of the final model `reader_r2_seeds`). The 2 most confident false negatives and 2 false positives per finding
(12 cases), out-of-fold on MRNet train, each explained by the fold models that never saw the exam. The figures and case IDs stay in
`outputs/gallery/reader_r2/` (git-ignored): MRNet's licence doesn't allow sharing images or labels, and "case X is a missed ACL tear" would be a label.
The notes are a non-radiologist's descriptive reading of the key slices and heatmaps, not diagnoses.

## Heatmap sanity check (EVAL_PROTOCOL R13)
Trained vs randomly initialised heatmaps (3 positive exams per finding, sagittal): mean Spearman **0.26 abnormal, 0.18 ACL, −0.11 meniscus**,
all below the 0.5 limit → the heatmaps reflect what the model learned, not just image edges. (`results/cv/reader_r2_r13.json`)

## The 12 cases, by pattern

| Pattern | Cases | What it looks like |
|---|---|---|
| **Edge slices** | 5 of 12 (ACL miss, meniscus miss, meniscus false alarm, abnormal miss, abnormal false alarm); 2 more have an edge key slice but are filed under image quality / off-target | at least one plane's key slice is among the first/last few of its series, away from the joint centre |
| **Joint fluid as a shortcut** | 3 (ACL false alarm ×2, abnormal false alarm) | heatmaps on large bright fluid collections; fluid often accompanies tears, so the model may over-rely on it |
| **Image quality** | 3 (ACL miss: very noisy series; meniscus false alarm: oblique/distorted acquisition; abnormal miss: a noise-only slice was the coronal key slice) | the model has no defence against corrupted or unusual series |
| **Off-target heatmap** | 2 (ACL false alarm, meniscus miss) | heatmap on skin / posterior soft tissue rather than the joint |
| **Needs expert review** | 2 (meniscus false alarm, abnormal false alarm) | nothing obviously wrong to a non-expert. Possibly label noise (MRNet labels come from clinical reports) or a subtle finding |

## What this suggests (each idea must still pass the paired-CI rule)
1. ~~Tell the model where a slice is.~~ **Checked across all 1,130 exams and NOT supported** (D-015): edge key slices are about as
   common when the model is right (19.6%) as when it's wrong (18.1%). Side-view edge slices are often the *right* place
   (the meniscal horns are seen there). The gallery pattern was a coincidence of 12 hand-picked cases.
2. **An input quality check in the app**: flag series with noise-only slices before predicting.
3. **For doctors**: the app's heatmaps make the fluid shortcut and off-target looks visible. That's exactly why the UI says
   "where the model looked, not proof of a finding".
