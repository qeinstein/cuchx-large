# EXP-EMO-RELRAW-001 — raw reference-based pair probe

## Question

Can label-free raw temporal motion trajectories decide whether a fixed champion's two
within-session Emotion assignments should be swapped?

This is a deliberately narrow specialist probe, not a replacement for the constrained
session solver.  It uses no IMU, radar, or DTW features.

## Representation and validation contract

For each clip, `emolab/probe_relative_raw.py` computes:

- a root-centred, torso-scale-normalised raw 17-joint skeleton descriptor: joint-wise
  velocity profile, temporal spectrum, pose amplitude, left/right synchrony, and pause
  proxy;
- a native-order DINOv2 Depth-frame trajectory descriptor: delta speed profile,
  spectrum, and fixed label-free projections of local delta direction.

For a pair within a test-visible inferred block, a conservative histogram-gradient head
scores its signed feature difference under the current group assignment and under its
swap.  It is fit only on the other subjects in each fold.  It is evaluated in both the
ordinary pseudo-test and the existing 38% pair-thinned pseudo-test.  The direct
comparison baseline is the immutable `champ/audit_champ.csv` snapshot (734/809 Emotion),
not a later mutable rerun.

The control head has exactly the same candidate-group and slot inputs but all raw motion
differences set to zero.  This prevents a high orientation accuracy from being mistaken
for a physical-motion result when it is actually a protocol prior.

Run:

```sh
./venv/bin/python emolab/probe_relative_raw.py
```

## Findings

| metric | Raw motion + candidate/slot context | Matched context-only control |
| --- | ---: | ---: |
| ordinary held-out pair orientation | 663/707 = **0.9378** | 611/707 = 0.8642 |
| pair-thinned held-out pair orientation | 507/545 = **0.9303** | 467/545 = 0.8569 |

So the raw descriptors do add signal to the *ordinary, true-orientation* pair task.
They do **not** make a safe correction policy on the champion's actual disagreement
subset.  This is the relevant result:

| protocol | swap-evidence gate | pairs | changed answers | W→R | R→W | precision | net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | score ≤ 0.0 | 39 | 78 | 20 | 56 | 0.263 | -36 |
| ordinary | score ≤ -2.0 | 23 | 46 | 10 | 34 | 0.227 | -24 |
| pair-thinned | score ≤ 0.0 | 26 | 52 | 13 | 36 | 0.265 | -23 |
| pair-thinned | score ≤ -2.0 | 11 | 22 | 5 | 16 | 0.238 | -11 |

Every evaluated confidence gate is negative, with fold failures.  The direct
one-transposition override is therefore **killed**.  No test answer was altered, and no
submission file was created.  The scored-only test ledger has 206 valid candidate pair
hypotheses; it must not be promoted because the OOF gate failed.

## Interpretation

The 7–8 point lift over the matched prior-only orientation control is evidence that raw
motion contains additional manner information.  But it does not transfer to the sparse
champion-disagreement slice through this independently trained swap policy.  Likely
reasons include calibration shift from true-label pairs to champion-assignment pairs and
the fact that a majority of residuals are not a single valid group-level transposition.

The next legitimate use of this signal is a learned temporal pair-alignment term that is
combined with the existing joint assignment objective, then audited at the same
decision level—not another threshold sweep or an override from this ledger.

## Artifacts

- `orientation_oracle_oof.csv` and `orientation_prior_control_oof.csv`: held-out
  orientation evidence and nuisance control.
- `orientation_oracle_summary.csv` and `orientation_prior_control_summary.csv`:
  per-fold and triple/pair regime breakdowns.
- `orientation_candidates_oof.csv`: all direct, test-visible candidate swaps before a
  gate is selected.
- `flip_audit_by_threshold.csv`: complete W→R/R→W audit.
- `test_pair_score_ledger.csv`: test scores only; no predictions have been applied.
- `raw_descriptor_coverage.csv`: label-free input coverage.
