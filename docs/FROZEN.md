# FROZEN: what the one-time final test evaluates (EVAL_PROTOCOL R14)

Frozen 2026-10-20T21:00:57 on pc_c, git f2780bc (+ uncommitted changes). Machine-readable copy: `docs/frozen.yaml`.

## baseline: `runs/b0`

- CV macro AUC 0.8549 (95% CI 0.8393–0.8691)
- 45 weight files, SHA-256 listed in frozen.yaml
- Operating thresholds T_hs (90% sensitivity on OOF): abnormal 0.615, acl 0.013, meniscus 0.095

## reader: `runs/reader_r2_seeds`

- CV macro AUC 0.8539 (95% CI 0.8386–0.8681)
- 30 weight files, SHA-256 listed in frozen.yaml
- Operating thresholds T_hs (90% sensitivity on OOF): abnormal 0.338, acl 0.248, meniscus 0.357

## hybrid: `runs/hybrid_r2s_b0`

- CV macro AUC 0.8639 (95% CI 0.8494–0.8773)
- no weights of its own: the mean logit of the baseline and the reader above
- Operating thresholds T_hs (90% sensitivity on OOF): abnormal 0.522, acl 0.064, meniscus 0.197

The test sets (MRNet `valid`, KneeMRI) had not been used by any model before this freeze.
"uncommitted changes": the code was committed after the freeze, so the weight hashes above, not a
git commit, are the binding record of what was tested.
