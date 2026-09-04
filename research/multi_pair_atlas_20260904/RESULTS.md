# Multi pair-block action-pool selector audit

Frozen champion: `submission_093859_SUBMITTED.csv` (SHA-256 `9a45bfbf57c516f05a274d901f0765cba6b4c534341d1b4c761991e33a0e9cd3`); no submission was made.

## Atlas and residual

Matched protocol: five subject-disjoint folds, pair_frac=0.38, withheld-trial policy `ends`, validated repair/conformance-first.
The atlas contains 14686 action rows across 719 Multi questions, with 21 false-negative actions and 14 spurious actions.
Among pair-block omitted actions with a satisfying best-X candidate, there are 391 eligible rows and 14 genuine inclusions (0.0358 action precision before gating).
Of the 21 false-negative actions, 16 have a candidate pool containing X; 5 are structurally unavailable under the current hard constraints.

## Selector results

The fixed gate is exploratory (chosen after inspecting the atlas): pair + one-action add + score gap <= 5 + pool probability >= 0.10 + cached DINO dense probability >= 0.10.
On the full OOF atlas it fires 3 action candidates, 3 correct (1.000); question-level W→R=3, R→W=0, net=3.
Nested grid selection pooled across held-out folds fires 6 actions, 3 correct (0.500); question W→R=3, R→W=3, net=0.
Nested logistic selection pooled fires 9 actions, 2 correct (0.222); question W→R=2, R→W=7, net=-5.

## Test-side inspection

The exploratory fixed gate identifies 2 test action overrides across 2 Multi questions. They are recorded in `test_overrides_exploratory.csv` only; no candidate submission is produced because the nested selector does not establish the required precision bar.

Full action rows, candidate pools, nested metrics, and the frozen artifact hash are retained alongside this report for follow-up work.
