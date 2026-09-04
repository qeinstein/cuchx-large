# EXP-EMO-DINO-TRAJ-002 — joint-objective integration probe

`probe_joint_integration.py` asks the follow-on question from `probe_dino_trajectory.py`'s own
README: does the trajectory descriptor help when added as extra columns to the champion's own
pairwise classifier (`champ/emopair.py`, fit end-to-end with its existing physical-difference
and slot-prior evidence), rather than as an independently-trained override policy? This is the
same integration test session 2 ran for DTW features (0.9738 -> 0.9825, +0.9pp, all five folds
improved or flat).

## Retraction

The first run of this script reported champion 0.9738 -> combined 0.9875 (+1.4pp, 4/5 folds
improved). **That result is wrong and is retracted.** The bug: for the "swapped" orientation
row, the trajectory diff features were negated (`-diff` instead of `diff`) in lockstep with the
label being predicted (`y=0` for swap). That sign-flip directly encodes the answer into the
feature — a classic label-leakage artifact — and has nothing to do with the champion's actual
convention, which keeps `bf_i`/`bf_j` and every DTW/PHYS difference column **fixed** regardless
of which manner hypothesis (`ma`, `mb`) is being scored for a given clip pair, letting the
tree-based classifier learn the interaction with `grp_i`/`grp_j` itself.

## Corrected result

With `traj` held fixed per `(clip_i, clip_j)` pair, identical for both the correct and swapped
rows — matching `dtw_of()`'s existing convention exactly — subject-disjoint pairwise
manner-orientation accuracy:

| arm | accuracy | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| champion pairwise features [baseline] | 0.9738 (781/802) | 0.9888 | 0.9379 | 0.9828 | 0.9783 | 0.9851 |
| DINO trajectory descriptor only | 0.0000 (0/802) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| champion + trajectory [combined] | **0.9738** (781/802) | 0.9888 | 0.9379 | 0.9828 | 0.9783 | 0.9851 |

The combined arm is **bit-identical to the baseline in every fold**: the
`HistGradientBoostingClassifier` assigns the 246 trajectory columns zero effective weight. This
is a cleaner, more complete kill than DTW's marginal +0.9pp — the signal genuinely adds nothing
once evaluated correctly, not merely "not enough to promote."

## Combined verdict on temporal DINO trajectory features for manner

Between this probe, `probe_dino_trajectory.py`'s own decision-level audit (independent override
policy: uniformly and badly net-negative, -8 to -127, at every regime/head/threshold), and the
prior `emolab/probe_relative_raw.py` raw skeleton+DINO probe (also net-negative on the champion
disagreement subset despite +7-8pt oracle lift over a matched prior-only control), there is now
a threefold-independent, methodologically distinct confirmation that frame-embedding-derived
temporal motion signal from the existing DINOv2-Depth cache does not improve the champion's
actual manner assignments, whether used as an override or integrated into its own joint
pairwise objective. This is a stronger and more specific claim than "manner has no physical
signal" — the oracle-level lift in `probe_relative_raw.py` shows some manner information is
present — it is specifically that this modality's information is either redundant with what
`champ/emopair.py` already extracts from `PHYS`, or is not resolvable at the residual/hard-case
level where the champion is actually unsure.

## Lesson for future flip-precision claims in this repository

Any pairwise/relative construction that assigns two label roles from one comparison (correct
vs swapped orientation, A-before-B vs B-before-A, etc.) must keep every input feature that
describes the compared *items themselves* fixed across both rows, and let the learned
interaction with the categorical role/hypothesis columns do the discriminating. Deriving or
transforming a feature as a function of the label being predicted for that row — even something
as innocuous-seeming as "negate the difference for the swapped case" — silently leaks the
target. The tell here was that the corrected "trajectory only" arm collapsed to exactly 0.0000
(not e.g. ~0.50): the two rows for every pair became indistinguishable once the leak was
removed, which is itself a useful diagnostic to check for in any future pairwise probe in this
codebase.
