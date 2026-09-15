# The two-clip protocol-slot latent, and a ceiling for the emotion residual — session 5

Champion for this session: the exact **0.93859 = 321/342, rank #5** artifact
(`submission_093859_SUBMITTED.csv`, Kaggle submission id 56007625). Nothing was submitted;
no preserved artifact was modified. Work is in `slotlab/`, plus `champ/clockslot.py` and
`seqlab/run18.py`.

## Verdict up front

**No mechanism found this session clears the shipping bar, and the emotion residual is now
bounded rather than merely unexplained.** Two candidates were built and measured to the
decision level; both land at ~0.60 flip precision against the 0.80 that made S/T/W/X transfer:

| candidate | flips | W→R | R→W | precision | net | per-fold net |
| :--- | :-- | :-- | :-- | :-- | :-- | :--- |
| slot adjacency constraint | 24 | 13 | 8 | **0.619** | +5 | +2, +2, 0, +1, 0 |
| + content-normalised which-end | 27 | 15 | 9 | 0.625 | +6 | +2, +3, **−2**, +3, 0 |
| sequence mech Y, nested weights | 19 | 9 | 6 | **0.600** | +3 | 0, +5, **−1**, **−1** |
| *(oracle slot set, for reference)* | 45 | 33 | 5 | 0.868 | +28 | all positive |

Scaled to the 342 public rows, the two shippable candidates are worth **+0.6 and +0.2
questions**. Combined expected gain is under one question against a spread of roughly ±1.5.
Recommendation: **do not disturb the 321/342 artifact for this.**

The valuable output is the ceiling, below.

## What was established

### 1. The real test never withholds a middle trial (π = 1.000)

`solve_emotion` treats a two-clip block whose emotion-option intersection has size 3 as a
three-slot session with one trial withheld, and maximises over
`present ∈ {(0,1), (0,2), (1,2)}` under a **flat** prior. That prior is misspecified, and
asymmetrically so: two of the three hypotheses put clip 0 in slot 0, which is the measured
+5.6pp over-prediction of the SLOW group.

Estimated the actual policy by maximum likelihood on the 21 pair-block recording gaps against
training-fitted adjacent/skipped log-normal densities (`slotlab/estimate_policy.py`):

* adjacent-trial gaps: median 99.0s (train, n=511); one-trial-skipped: 203.2s (n=259)
* **MLE π = P(adjacent) = 1.0000**, parametric bootstrap 95% CI [0.601, 1.000]
* likelihood-ratio test against the π = 2/3 a flat prior implies: **D = 8.338, p = 0.0039**
* post-repair structure (31 triples + 23 pairs + 5 orphans): same π, **p = 0.0020**
* **control**: the identical estimator returns exactly 1.0000 on the 34 complete test triples,
  where π = 1 is known — so it is not biased high by construction

### 2. Validation protocol had a second hidden parameter

`thin_to_pairs` withheld a *uniformly random* trial. WHICH trial is withheld matters as much
as how often, and 'uniform' is the one policy the real test demonstrably is not in. Added
`policy='ends'` (`champ/pseudotest.py`). Under the matched policy the baseline is **0.7053**
at two-clip emotion, *worse* than uniform's 0.7296 — the flat prior wastes mass on a
hypothesis that never occurs. Every number below uses the matched policy.

### 3. The ceiling: +28 OOF, and 23 of it is unreachable

Arms are cleanly isolated — three-clip emotion is **identical at 0.9219 in every arm**:

| arm | 2-clip emotion (n=190) | OOF total |
| :--- | :-- | :-- |
| baseline flat prior | 0.7053 | 3410 |
| soft clock prior | 0.7053 | 3409 (−1) |
| adjacency-only constraint | 0.7316 | 3415 (**+5**) |
| **oracle slot set** | **0.8526** | 3438 (**+28**) |

So a *perfect* slot-set oracle is worth +28 OOF ≈ **+2.6 public** — that is the ceiling on this
entire latent — and the adjacency fact captures only 5 of the 28.

The remaining 23 need the **identity of the withheld end**, and three things now say that is
out of reach:

* the clock is structurally blind to it — (0,1) and (1,2) are both one trial period apart, and
  the clock model scores **0.5077**, i.e. chance
* the duration ratio is equally blind: consecutive-trial ratios 1.21 and 1.29 against 1.55 for
  a skipped middle, so the ratio identifies adjacency and nothing else
* **the solver already resolves it at ≈0.858** (adjonly 0.7316 / oracle 0.8526). Beating that
  is the actual requirement, and the two models built here score 0.6404 and 0.5904

The which-end probe is worth recording as a *positive* representation result that still failed
to transfer — the seventh of its kind here. Absolute level is the only statistic that can
separate the two adjacent hypotheses, and a session's action content is known (all trials run
the same script; `pool.solve_block` recovers the pool), so content was regressed out and the
residual level classified which end survived: **0.6404** subject-disjoint against 0.5904 for
`slotprior`'s GBM and 0.5000 chance, 5/5 folds above chance, label-shuffle control 0.4992,
top-quartile confidence 0.7692. Integrated into production it moved two-clip emotion by
**0.0000** — because 0.64 is far below the 0.858 the solver already achieves.

### 4. Why the adjacency constraint is only worth +5: the hedge

Removing `(0,2)` forces an all-or-nothing 2-way choice. The `(0,2)` hypothesis was acting as a
**hedge** that banks half credit — it assigns clip 0 the slot-0 manner and clip 1 the slot-2
manner, so whichever end is truly present, exactly one of the two questions is right. Hence
R→W = 8 among 24 flips and precision 0.619 despite the constraint removing a provably
never-true hypothesis. Committing still beats hedging (0.858 × 0.8526 = 0.732 against
0.5 × 0.8526 = 0.426), but the baseline already rarely hedges, so the recoverable margin is small.

### 5. Routes closed cheaply

* **Neighbouring-block gaps cannot identify the withheld trial.** Inferred blocks are
  internally chronological but globally shuffled: of 89 within-block consecutive index pairs
  only 1 goes backwards in time, against 21 of 54 across block boundaries, with jumps of one
  to fifteen days. Between-block gaps read recording-day changes, not withheld trials.
* **No slot structure is left on the table in three-clip blocks.** Per-word slot priors reach
  **0.8314** and the coarse five-group map **0.7132**, both below the champion's **0.8966**.
  The protocol itself is weak — P(SLOW | slot 0) = 0.766, P(FAST | slot 2) = 0.589, slot 1
  nearly uniform, and only 171/265 sessions have three distinct manner groups.
* **`seqlab/run17.py`'s `0 observations` was stale state, not a real negative.** Re-run it
  harvests 385 observations over 304 parent clips at 0.951 child-action accuracy. Counted at
  the right unit — the session block, since the latent order is shared — coverage is 52/104
  training sessions (117 orderable pairs) and 17/39 test sequence questions. But under nested
  subject-disjoint weight selection the gain is +3 at 0.600 precision, not the +6 the
  contaminated grid reported.

### 6. One unvalidatable anomaly, recorded not shipped

Block `[166,167]` has an internal gap of **−100.0s**: its clips are in reverse chronological
order, the only such case in the test set (88/89 within-block pairs are forward). The position
prior is therefore applied backwards for that block's 2 emotion questions. Reordering pair
blocks by the clock rather than by clip index is correct by protocol (trial index equals clock
order for 260/268 training sessions), but it **cannot be validated OOF at all**, because
`make_pseudo` builds pseudo-test blocks in chronological order by construction, so the
anomaly never appears in validation. Flagged as a ~1-row structural candidate with an argument
but no measurement behind it. Not included in any candidate.

## The error budget, verified

Expected public errors from held-out accuracies applied to the real test's category counts:
**19.8 against 21 actual**, so the accuracies are well calibrated and the budget can be
trusted for prioritisation.

| pool | public errors |
| :--- | :-- |
| emotion — 2-clip blocks | 6.8 |
| **sequence** | **4.9** |
| emotion — 3-clip blocks | 3.6 |
| multi (pair 1.9 + triple 1.2) | 3.2 |
| HARn single | 1.1 |
| combination / HAU single / object | 0.7 |

## Sequence: the residual is diffuse, not systematic

48 of 77 residual errors are a **single pairwise inversion** from truth, so the latent order is
nearly right. But the inversions do not concentrate on identifiable action pairs:

* 119 inverted (earlier, later) pairs over **67 distinct pairs**
* **24** distinct pairs are needed to cover half of them; maximum frequency is **3**
* only 5 of 62 unordered pairs invert in both directions — so the direction is mostly
  consistent per pair, but each pair appears 2-3 times, far too sparse to learn from

A per-pair correction is therefore not learnable, and a global precedence prior was already
falsified in session 2. The sequence residual is per-session evidence noise. The only lever is
more order evidence, which is mechanism Y, and that is coverage-limited to +0.2 public.

## Multi: under-prediction is real but not a threshold

A `multi` option is admitted only if **every** action in it is in the recovered pool, so an
under-recovered pool silently drops options. The asymmetry is real and measured — of 32
held-out multi errors, **19 are pure under-prediction against 10 pure over-prediction** (23
missing letters vs 14 extra), concentrated in two-clip blocks (miss 14 vs extra 6; 0.9158
against 0.9733 in triples), exactly what a fixed inclusion threshold does when one clip's
worth of option-repetition evidence is missing.

Adding a per-block-size bias to the pool-membership log-odds (`CHAMP_POOL_BIAS`) nonetheless
makes things **monotonically worse**:

| bias on 2-clip blocks | multi | 2-clip multi | HAU single | OOF total |
| :-- | :-- | :-- | :-- | :-- |
| 0.0 (champion) | 687 | 0.9158 | 717 | 3410 |
| 0.5 | 685 | 0.9000 | 716 | 3407 |
| 1.0 | 685 | 0.9053 | 716 | 3406 |

The reason is structural: `pool._enum_component` already enumerates only pools that *satisfy*
the option constraints, so a bias does not add the one missing action — it selects a different
satisfying pool, and more inclusive pools break the uniqueness that `single` and `combination`
depend on (HAU single 717 → 716). Fixing this would need per-action targeting, not a global
threshold, against a ceiling of +3.2 public.

## What this means for the residual

Combining this session with sessions 3 and 4, the emotion residual is no longer an open
question with unknown headroom:

* **three-clip emotion (≈5 public errors)** — eight independent representation probes have
  failed at the decision level, and the champion already exceeds the best label-structure
  ceiling available. Closed.
* **two-clip emotion (≈6 public errors)** — total available headroom is now *measured* at
  +2.6 public, of which ~0.6 is reachable, because the only informative latent beyond
  adjacency is already resolved at 0.858. Effectively closed.
* **sequence (≈5 public errors)** — mechanism Y is coverage-limited (17/39 questions) and
  honestly worth +0.2 public. 47 of the 80 residual errors are a *single pairwise inversion*
  from truth, which localises the problem but does not supply new evidence to fix it.
* **multi (≈3.2 public errors)** — the under-prediction is diagnosed (19 pure-miss vs 10
  pure-extra, concentrated in pair blocks) but a global inclusion threshold is falsified. Needs
  per-action targeting inside the constraint enumeration, against a +3.2 ceiling. The most
  open of the four, and the best-value next target.
* **HARn single (≈1.1) and the rest (≈0.7)** — too small to carry the target.

Reaching 332/342 requires +11 and would mean clearing essentially all of the above. The
evidence assembled across three sessions says that is not available from these modalities and
this generator structure; the honest reachable range from identified mechanisms is +1 to +2.

## Relevant paths

* `slotlab/recon_slots.py`, `recon_timeline.py` — protocol and clock reconnaissance
* `slotlab/estimate_policy.py`, `policy_summary.json` — the π = 1.000 estimate and its control
* `slotlab/eval_slotmodel.py` — isolated slot-set recovery (clock 0.6038 / GBM 0.6487 3-way)
* `slotlab/whichend_probe.py`, `whichend_summary.json` — the 0.6404 which-end result
* `slotlab/word_slot_ceiling.py` — the 0.8314 word-only triple ceiling
* `slotlab/nested_order_blocks.py` — block-level coverage of the sequence mechanism
* `slotlab/run_arms.sh`, `log_ends_*.txt` — the five production arms
* `champ/clockslot.py` — the generative slot prior (off by default: `CHAMP_W_SLOT=0`)
* `seqlab/run18.py`, `run18_summary.json` — mechanism Y with nested weight selection

## Reproduction note

Three concurrent `eval_pairstress.py` runs exhaust swap on a 17 GB machine (each peaks ~5 GB
as the dense-logit presence-stat cache fills per fold) and thrash to a standstill at 0.1% CPU.
Run the arms sequentially — `slotlab/run_arms.sh` does.
