# Rank-1 pass — 2026-09-05

The immutable baseline for this pass was `submission_094736_mixed14_SUBMITTED.csv`, public
0.94736 = 324/342.  The user explicitly authorised exactly two submissions; both were made,
in the order documented below, and no other artifact was submitted.

## Exact paired-cohort supervision

The first two real-test HAU cohorts exactly match the ordered visible emotion-option
signatures of training users 20 and 21.  This is the same acquisition pairing visible in
training (users 6↔16, 7↔17, 8↔18, and 9↔19), not a nearest-neighbour guess.

Leave-one-subject-out transfer over the exact training cohort pairs gives:

| category | decisions | changes vs saved OOF champion | W→R | R→W |
|---|---:|---:|---:|---:|
| emotion | 378 | 20 | 20 | 0 |
| sequence | 73 | 10 | 10 | 0 |

The 0.94736 artifact already agrees with nearly every applicable donor decision.  Two new
test corrections remain: `test_0385 B→A` (Carefully) and `test_0397 C→D` (Hastily).

## Exact visible HARn templates

An independent rule keys only on `(source, category, normalized question, unordered option
multiset)`.  Donors exclude the held-out subject; their semantic answers must be unanimous
and are converted back to target letters.  No test label, path identity, or sensor feature is
used.

Subject-disjoint accuracy for unanimous templates is 112/113 object-interaction and 115/118
HARn single.  More importantly, the exact object template was compared with the global-object
prior used by production mechanism W2 on no-sibling test rows:

| minimum donor support | rows | fallback correct | template correct | W→R | R→W |
|---:|---:|---:|---:|---:|---:|
| 1 | 113 | 87 | 112 | 25 | 0 |
| 2 | 89 | 74 | 88 | 14 | 0 |
| 3 | 73 | 66 | 73 | 7 | 0 |

This validates `test_0524 A→B` (a cloth) and `test_0530 B→D` (headphones).  The latter repairs
a known weakness: W2 changed an impossible “towel” answer to the globally frequent “phone,”
while the exact option template occurs twice in training and both answers are headphones.

Two exact-template single disagreements (`test_0478 C→B`, `test_0507 A→C`) are retained only
in the maximal research arm.  Their template rule is strong in aggregate, but the ignored
sensor OOF artifact required for a direct challenger-vs-production W→R/R→W audit is absent on
this machine.

## Candidates

Both start from the byte-preserved 0.94736 artifact:

* `research/submission_rank1_conservative_20260905.csv`: four changes, SHA-256
  `35db99a35a0936878b28d3dde463d4a3867f9a3228478b6331a387973de83acb`.
* `research/submission_rank1_maximal_20260905.csv`: six changes, SHA-256
  `125ca9a4299339dc7bec801d5e0dff9c781892b3fc600d1623f9f99750a8f327`.

The conservative candidate is the strongest evidence-backed artifact.  Its public score is
**0.95614 = 327/342, rank 3**, submission id 56029679: +3 versus the 324/342 baseline.
The byte-identical frozen copy is `submission_095614_327of342_CHAMPION.csv`.

The maximal candidate scored **0.95321 = 326/342**, submission id 56029713: +2 versus the
old champion and -1 versus the conservative candidate. Its preserved copy is
`submission_095321_326of342_maximal_SUBMITTED.csv`.

The score equations determine mechanism-level outcomes without identifying labels.  Across
the conservative arm's four changes, net +3 proves exactly three W->R and zero R->W; the
fourth row may be private or a public W->W.  Across the maximal arm's two added HARn-single
changes, net -1 proves exactly one R->W and zero W->R; the other may be private or a public
W->W.  Therefore exact visible templates transfer for object fallbacks but must not override
the sensor-backed HARn single classifier without a new gate.

At scoring time the leaderboard was: #1 334/342 (0.97660), #2 333/342 (0.97368), and us #3
327/342 (0.95614). The remaining gap to #1 is seven answers.

## Falsification and continued research

The complete Multi candidate-pool atlas was reopened.  An oracle choosing among all 91,287
enumerated satisfying pools gains only 21 exact Multi questions over 719 OOF questions;
the pool is incapable of providing a nine-public-answer breakthrough.

Authorised access to the gated Hugging Face release was established and the 385 MB sensor
archive was downloaded.  From it, `champ/meta.csv` (3,932 rows), `champ/feats.csv`,
`champ/skel_seq.npz` (3,905 units), and `champ/imu_seq.npz` (3,917 units; 3,819 with non-zero
IMU) were reconstructed.  No raw test labels were accessed or created.

The virtual emotion-slot fix was then evaluated in the subject-disjoint pair-ends protocol:
both arms scored 620/719 and the flag changed exactly 0/719 decisions.  It is behaviorally
inert and is falsified, not merely blocked by missing infrastructure.

A dense-logit screening cache was rebuilt over five subject-disjoint folds.  The historical
training code was made memory-safe by computing normalization and class counts from streaming
sufficient statistics, and an opt-in length-bucketed loader/cache name was added.  The
screening cache used one epoch per fold (OOF frame accuracy 45,594/143,598 = 0.3175), so it is
appropriate for mechanism triage only, not candidate promotion.  A paired audit of the
full-session sub-block pool target is retained in
`research/eval_pool_full_subblocks_20260905.py` and its per-fold ledgers.

Reproduction:

```bash
/tmp/cuchx-research-venv/bin/python research/evaluate_user_template_pairs_20260904.py
/tmp/cuchx-research-venv/bin/python research/build_user_template_candidate_20260904.py
/tmp/cuchx-research-venv/bin/python research/exact_visible_template_audit_20260905.py
/tmp/cuchx-research-venv/bin/python research/build_rank1_candidates_20260905.py
```
