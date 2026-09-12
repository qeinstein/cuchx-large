# Session 18 — Private-Generalization Audit

Date: 2026-09-12
Status: research complete for this pass; no new candidate promoted or submitted.

## Objective

The target is a stronger model for the hidden/private questions, not a public-score probe. The protected reference is the official 332/342 champion:

`submission_097076_332of342_CHAMPION.csv`

The principal local reference is the grouped, subject-disjoint OOF pipeline at
`research/final_video_20260910/oof_pipeline_base.csv`:

- 3,771 / 4,087 overall (`0.92268`)
- 2,375 / 2,408 action rows
- sequence: 116 / 308
- object: 113 / 133

The composite research audit, which combines the proven structural layers with the sequence/object improvements, reaches 3,834 / 4,087 (`0.93810`). These are OOF measurements, not private-score claims.

## What was tested

### 1. Pairwise sequence decoder

The pairwise precedence model was rerun against the current final-video baseline. With the 500-iteration fit it produced:

- pairwise precedence accuracy: 1,619 / 1,830 (`0.8847`)
- exact sequence accuracy: 185 / 308 (`0.6006`)
- baseline exact sequence: 116 / 308 (`0.3766`)
- net sequence gain: +69 rows

Five test flips were selected only when they were stable in at least four of five subject-disjoint fits and agreed with the full fit. Protected sequence QA IDs were left untouched. The resulting candidate was submitted as Kaggle submission `56192056`; it returned 332 / 342 (`0.97076`), equal to the official champion. This is useful evidence that the sequence branch is strong locally, but it does not establish a private improvement.

Artifacts:

- `champ/seqpair.py`
- `champ/eval_seqpair.py`
- `champ/audit_seqpair_test_stability.py`
- `champ/make_seqpair_stable_candidate.py`
- `research/final_video_20260910/seqpair_test_stability.csv`
- `research/final_video_20260910/submission_seqpair_stable_iter500_v1.csv`

### 2. Positive/unknown action-pool decoder

The old pool assumption treated the union of selected QA answers as a complete action pool. The new `pool_pu.py` decoder treats aligned HARn segment actions as strong positive evidence, selected QA actions as weaker positive evidence, and absence from both channels as only weak negative evidence. It retains the hard category constraints.

The source audit found that aligned HARn segments are high precision but incomplete:

- 272 HAU session keys
- 267 aligned HARn segment sessions
- exact segment-pool = QA-pool in 172 / 267 sessions
- segment pool is a subset of the QA pool in 256 / 267 sessions

That motivated the noisy-channel design, but the OOF result did not justify promotion:

| Evaluation | Result |
| :--- | :--- |
| Full five-fold OOF, 350 iterations | 2,376 / 2,408 action rows versus 2,375 baseline (`+1`) |
| Combination slice | `+2` rows |
| Multi slice | `-1` row |
| Single slice | `0` rows |
| Adjacent-pair/thinned OOF | `-43` rows |
| Test stability | `test_0165`: 3/5 fits; `test_0206`: 2/5 fits; no 4/5 change |

The gated test candidate was built and validated (682 rows, IDs and category constraints intact), but it was not submitted. It changes only `test_0165`, from `CD` to `D`. Its stability is below the pre-registered promotion gate, so submitting it would be a public probe rather than a model improvement.

Artifacts:

- `champ/pool_pu.py`
- `champ/eval_pool_pu.py`
- `champ/make_pool_pu_candidate.py`
- `champ/audit_pool_pu_test_stability.py`
- `research/pool_pu_oof_iter350.csv`
- `research/pool_pu_pair_oof.csv`
- `research/pool_pu_test_stability.csv`
- `research/final_video_20260910/submission_pool_pu_complete3_gate_v1.csv`

Decision: keep the implementation and measurements as a reusable audit, but do not integrate its answers into the champion.

### 3. Direct clip/action head

The direct head attempted to replace structural pool inference with a clip-plus-option classifier using the surviving compact dense presence statistics and visible option metadata. It was evaluated across multi-answer thresholds.

The branch was strongly below the current OOF reference at every threshold. The best tested total delta was approximately `-1,121` rows; the lower threshold was approximately `-1,184`. This is a clear negative control, not a near miss.

Artifacts:

- `champ/clip_action.py`
- `champ/eval_clip_action.py`

Decision: retired; no integration and no test candidate.

### 4. Dense2 segment/MIL/ranking model

Dense2 is the remaining substantive model route. It combines:

- supervised segment cross-entropy where the QA-derived pool gives trustworthy labels;
- clip-level multiple-instance learning over the QA-derived pool;
- a sequence-order ranking loss.

A CPU speed pilot used stride 4 and 64 hidden channels. It covered 797 items and reached approximately 29.3% frame accuracy after two epochs. The run was stopped before a full five-fold evaluation because it was an underfit/throughput pilot, not because the modeling idea had been falsified.

The runner now exposes the research controls `CHAMP_DENSE2_STRIDE`, `CHAMP_DENSE2_CH`, `CHAMP_DENSE2_W_MIL`, and `CHAMP_DENSE2_W_RANK`. The default path remains frame-exact and uses the original model width/loss weights.

Decision: this is the next model to run, but only with a complete five-fold subject-disjoint OOF audit and a pre-registered small grid. A single test inference is not sufficient.

## Storage and reproducibility state

The earlier storage audit identified 144,524 KiB (about 141 MiB / 148 MB) of safe regenerable material. At the time of this update, the two large thermal arrays and `.cache` are already absent; the remaining measured items are:

- `research/thermal_mnv3_clip_features_20260906.npz`: 5,928 KiB
- `champ/torch_cache`: 10,068 KiB
- Python `__pycache__` directories: 1,644 KiB combined at measurement time

The raw `champ/dense_logits_full40.npz` and `champ/dense_logits_screen1_bucket.npz` caches are also absent. Compact out-of-fold statistics under `research/final_video_20260910/dense_oof_f*.npy` remain available for the direct-head audit. Raw-logit branches must therefore regenerate or recover their cache before rerunning.

The large `research/takeover_20260908/kaggle_exact_dataset_tar` path is hardlink-backed to the manual HF dataset; deleting that apparent 1.3G path would recover essentially no physical disk space and is not part of the cleanup recommendation.

## Recommended execution order

1. Recover or regenerate the raw dense-logit cache, preserving the compact OOF statistics.
2. Run Dense2 on all five subject-disjoint folds with a small pre-registered grid over width, temporal stride, MIL weight, and ranking weight.
3. Decode Dense2 outputs through the existing category constraints and compare against both the champion and `oof_composite_v1.csv`.
4. Require a positive OOF delta in the affected category, no material regression in unaffected categories, and at least 4/5 test-fit stability before constructing a candidate.
5. Submit only if the candidate is supported by more than one independent signal. Otherwise retain the champion and record the negative result.

## Reproducibility commands

Historical sequence audit:

```bash
CHAMP_LOGITS=dense_logits_full40.npz CHAMP_SEQPAIR_MAX_ITER=500 \
  python3 -u champ/eval_seqpair.py
```

PU OOF audit after the raw dense cache is restored:

```bash
CHAMP_LOGITS=dense_logits_full40.npz CHAMP_POOL_PU_MAX_ITER=350 \
  python3 -u champ/eval_pool_pu.py
```

Dense2 speed pilot and full audit controls:

```bash
CHAMP_DENSE2_STRIDE=4 CHAMP_DENSE2_CH=64 \
  python3 -u champ/run_dense2.py 2 1
```

The exact commands above are audit commands. They do not submit to Kaggle or modify the protected champion.
