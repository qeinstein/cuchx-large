# Breakthrough audit — 2026-09-04

This report records the strongest result from the current research pass. The submitted
champion remains immutable and no Kaggle submission was made.

## 1. The production bug

`champ/core.py::fit_manner` emitted one feature row per clip, but the group/position and
label-by-group counters were indented outside that per-clip loop. In practice, those priors
were updated from only the last clip of each training variant. The physical classifier was
fit on all rows, which made the defect easy to miss: the learned head looked healthy while
the hierarchical prior was silently undercounted.

The counters are now updated inside the loop. Training variants and the default inference
path are otherwise unchanged. The optional R3D/raw-pair hooks in `core.py` and the optional
pair-subblock experiment in `emopair.py` remain disabled by default.

## 2. Decision-level evidence

The corrected default path was rerun on the full five-fold subject-disjoint pseudo-test:

| audit | before | after | changed | W→R | R→W | net |
|---|---:|---:|---:|---:|---:|---:|
| full OOF, 4087 decisions | 3827 | 3843 | 24 | 18 | 2 | **+16** |
| emotion-only adjacent-end pair stress, 719 decisions | 605 | 615 | — | — | — | **+10** |

The second line uses the real-test pair regime and completed all five folds: 139/162,
127/151, 136/156, 96/115, and 117/135. The before value is the earlier run of the same
emotion-only harness before the counter repair; the after artifact is
`manner_pair_ends_oof_20260904.csv`.

The full-OOF changed rows are concentrated where expected: 19 emotion decisions changed,
with 18 wins and 1 regression; the remaining two regressions are outside emotion. This is
strong enough to promote the correction into the test candidate.

## 3. Test candidate construction

`submission_priorfix_structural_candidate.csv` starts from the preserved 321/342 champion
and contains 14 changed rows:

* 8 new test deltas from the corrected manner priors (`test_0426`, `0427`, `0429`, `0443`,
  `0447`, `0458`, `0488`, `0519`);
* 4 exact-pool sequence transfers with complete six-pair coverage
  (`test_0330`, `0334`, `0342`, `0353`), already validated in the structural sequence
  audit;
* 2 exact donor-cohort emotion transfers (`test_0386`, `0387`).

The candidate is a copy and has the same 682-row schema and ordering as the champion. A
separate 15-change file, `submission_priorfix_structural_slot187_candidate.csv`, adds the
single high-margin donor-slot hypothesis for `test_0449`; it is intentionally not the
conservative candidate.

## 4. Ideas rejected this pass

* Pair-head training on all two-row subblocks is a regime-matched idea, but the saved
  decision audit was only 11 W→R versus 9 R→W (+2), and it changed only one real-test row.
  It remains opt-in (`CHAMP_EMO_PAIR_SUBBLOCKS=1`) and is not promoted.
* The pool-conditioned manner probe scored 588/719 and was killed.
* The R3D depth-video head and raw temporal joint score did not clear a decision-level
  promotion gate. The raw joint saved control comparison was 11 W→R versus 9 R→W.
* Exact/nearest action-pool template transfer without a complete structural gate was
  strongly negative and was not used. The only retained sequence transfers are the audited
  complete-coverage rows above.

## 5. Integrity

SHA-256 hashes at assembly time:

* champion `submission_093859_SUBMITTED.csv`: `9a45bfbf57c516f05a274d901f0765cba6b4c534341d1b4c761991e33a0e9cd3`
* conservative candidate: `0a12e9bde62d3c9384f15cba434bace994a4e65c1feb4b2ec53fe0438036ef89`
* optional slot candidate: `ace29aaa883d2a73183043f084ecf5b0edf31e59bbfe536c39fe1e1ff6553d84`

The public score of the candidate is unknown because submitting to Kaggle was explicitly
out of scope. The repository therefore contains a materially improved, fully audited
candidate, not a fabricated leaderboard claim.
