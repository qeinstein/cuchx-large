# Research ledger

Chronological record of what was tried and what it measured. `README.md` holds the current
architecture and the falsified-directions list; this file is the session-by-session trail.

Same documentation rule: **measured numbers only, labelled by protocol.** No projections.

**Current champion:** `submission_093859_SUBMITTED.csv` — public **0.93859 = 321/342, rank 5**
(Kaggle submission id 56007625). Byte-identical copy of `submission_corrlayer_S_W_T.csv`.
Every experiment is evaluated against this exact file.

---

## Session 1 (2026-08-31 → 09-02) — perceptual era

Multimodal feature caches and per-category specialists. Public 0.17251 → **0.78947**.

Built: `sensor_features_all.npz` (280-dim IMU+skeleton), `visual_features_all.npz`,
`thermal_features_all.npz`, `dinov2_features_all.npz` (384-dim ViT-S/14),
`radar_features_all.npz`, consolidated to a 1720-dim array.

Held-out CV at the end of this era: **75.68%** OOF (v8), emotion **52.29%**.

Two lessons that survived:
* `submission_v4.csv` — a global IMU emotion override, 58 changed rows — **regressed**
  0.78947 → 0.78362. First evidence that aggregate-looking improvements can be net negative.
* Every "championship"/"grandmaster" artifact from this era was a projection, never scored.
  See README §6.1 for the measured refutation (hard upper bound 0.582).

## Session 2 (2026-09-03) — structural breakthrough

The insight that reframed the project: **this is a question-generator problem, not only a
perception problem.** Solve per inferred session block, never per question.

Discoveries: session-triple manner protocol, option-set session leakage, closed-world action
pool, HARn frame-level supervision, one latent action order per session (README §3).

Public **0.78947 → 0.85087 → 0.90643 → 0.92105** (rank 6/166). Plain 5-fold OOF 0.9320.

Also established this session:
* **Pair-thinned validation** — the plain protocol predicted 0.9396 while we scored 0.92105;
  `pair_frac=0.38` gives 0.9208 against an actual 0.92105. Everything is validated this way now.
* **Orphan regime** — `infer_blocks` has `kmin=2` so it cannot emit a one-clip block; four
  real-test blocks are non-conforming, covering 44 questions.
* Residual surface mapped in `FINDINGS_session2_residual_surface.md`.

## Session 3 (2026-09-04) — manner representation search

Systematic hunt for a manner representation to lift the ~0.60 manner-group classifier.
**Five independent kills**, all audited at the decision level (README §6.2).

Key negative: the physical axes a CARE/NERV/NEUT ontology needs — amplitude, pause fraction,
spectral smoothness — carry η² ≈ 0.002–0.028 within-session. Only intensity/duration/energy
separates groups, and it is already in `core.PHYS`.

Retracted in-session: a `+1.4pp` result from `probe_joint_integration.py` was label leakage
(feature sign-flipped in lockstep with the target). See README §6.4.

`FINDINGS_session3_manner_representation.md`.

## Session 4 (2026-09-04) — S/T/W/X shipped; PCRME falsified

Shipped the correction layer that produced the current champion:
**S** (joint sequence order decoding, flip precision 0.853), **T** (block conformance repair,
0.821), **W/X** (object↔single consistency, 1.000). Public **0.92105 → 0.93859**, rank 5.
Both correction layers transferred very close to their subject-disjoint decision-level
estimates — evidence that OOF flip precision predicts public transfer.

Tested phase-conditioned relative manner recognition (PCRME): compare corresponding
HARn-anchored action *phases* across sibling trials rather than whole clips. 983 aligned
session/action tuples, 272/272 sessions, provably unambiguous alignment.

Result: **the first feature source in two sessions to beat the champion's own representation**
(+3.3pp manner-group accuracy, 4/5 folds) — and still net **−2** at the decision level
(11 genuine overrides, precision 0.400). Per-action specialist search: 0/40 clear 0.80.
A frozen NTU60-finetuned MotionBERT probe was also negative.

Two bugs caught and fixed in-session (baseline unfairness; letter-vs-text unit mismatch).

`FINDINGS_session4_phase_conditioned_manner.md`, `phaselab/RESULTS.md`.

## Session 5 (2026-09-04, current) — two-clip protocol-slot latent

Target: 332+/342. Current 321/342, so **+11** required; qualification (+3) is not the goal.

Error budget against the current champion, from pair-thinned held-out accuracy scaled to the
342 public rows (~21 errors remaining):

| pool | held-out acc | est. public errors |
| :--- | :-- | :-- |
| emotion, 2-clip blocks | 0.7296 | ~6 |
| emotion, 3-clip blocks | 0.8966 | ~5 |
| sequence (post-S) | ~0.74 | ~5 |
| multi | 0.9596 | ~3 |
| single / object / combination | — | ~2 |

Opened with the densest pool — emotion in two-clip blocks — because unlike the eight killed
manner probes its failure is **structural, not perceptual**: the candidate manner set is right
(|I|=3 for 20 of 21 test pairs), but *which two of the three protocol slots are present* is an
extra latent that `solve_emotion` maximises over with a flat prior, which measurably
over-predicts the SLOW group by 5.6pp.

Reconnaissance (`slotlab/`), all label-free and test-visible:

* **Trial index is chronological**: 260/268 training sessions (0.9701).
* **Inferred blocks are internally chronological but globally shuffled** — of 89 within-block
  consecutive index pairs only 1 goes backwards in time, against 21 of 54 across block
  boundaries. So neighbouring-block gaps are dominated by user/day changes (jumps of ~1 to
  15 days) and **cannot** identify which trial was withheld. That route is dead.
* **The within-block clock does discriminate.** Training: adjacent-trial gaps median 99.0s
  (p5–p95 63–317), one-trial-skipped gaps median 203.2s (p5–p95 130–557); best balanced
  single-threshold separation 0.834. Test triples confirm the scale in-domain (adjacent
  median 88.4s, 0/66 negative).
* **Applying that likelihood to the 21 real-test pairs: 19/21 favour "adjacent trials" over
  "middle trial withheld."** Under a uniform withholding policy ~7 of 21 would be
  middle-withheld. So the flat prior is badly misspecified, and it is exactly the
  misspecification that inflates SLOW (two of the three slot hypotheses put clip 0 in slot 1).
* **One anomaly:** block `[166,167]` has an internal gap of **−100.0s** — the clips are in
  reverse chronological order, so the position prior is applied backwards for that block
  (2 emotion questions).
* Duration ratio separates adjacent from skipped (1.21/1.29 vs 1.55) but does **not**
  separate "first withheld" from "last withheld"; the physical evidence has to do that.

`slotlab/recon_slots.py`, `slotlab/recon_timeline.py`, `slotlab/log_recon.txt`,
`slotlab/log_timeline.txt`, `slotlab/test_pair_blocks.csv`, `slotlab/test_pair_clock.csv`.

**Outcome: the pool is bounded, and nothing ships.** Full detail in
`FINDINGS_session5_slot_latent.md`.

Production arms (pair-thinned, matched `policy='ends'`; 3-clip emotion identical at 0.9219 in
every arm, so the mechanism is correctly isolated):

| arm | 2-clip emotion (n=190) | OOF total |
| :--- | :-- | :-- |
| baseline flat prior | 0.7053 | 3410 |
| soft clock prior | 0.7053 | 3409 (−1) |
| adjacency-only constraint | 0.7316 | 3415 (+5) |
| **oracle slot set** | **0.8526** | 3438 (**+28**) |

Decision-level audit against the exact champion:

| candidate | flips | W→R | R→W | precision | net |
| :--- | :-- | :-- | :-- | :-- | :-- |
| slot adjacency constraint | 24 | 13 | 8 | **0.619** | +5 |
| + content-normalised which-end | 27 | 15 | 9 | 0.625 | +6 |
| sequence mech Y, nested weights | 19 | 9 | 6 | **0.600** | +3 |
| *(oracle, reference)* | 45 | 33 | 5 | 0.868 | +28 |

Both shippable candidates sit at ~0.60 precision against the 0.80 that made S/T/W/X transfer,
and are worth **+0.6 and +0.2** public questions. Combined expected gain is under one question
against a spread of roughly ±1.5, so the 321/342 artifact was left alone.

The durable results are the ceilings:

* **A perfect slot oracle is worth only +28 OOF ≈ +2.6 public** — the ceiling on this latent.
* **Only 5 of that +28 comes from adjacency; 23 needs which END was withheld, and the solver
  already resolves that at ≈0.858.** Anything new must beat 0.858, not 0.5. The clock is
  structurally blind to it (0.5077) and so is the duration ratio.
* A content-normalised which-end model (regress out E[level | recovered action pool], since a
  session's trials share one script) scored **0.6404** subject-disjoint — a genuine
  representation result, 5/5 folds above chance, clean label-shuffle control — and moved
  production by exactly **0.0000**. Seventh probe with this failure shape.
* `(0,2)` was acting as a **hedge** banking half credit, which is why removing a provably
  never-true hypothesis still only reaches 0.619 precision.
* **No label structure is left in 3-clip blocks**: per-word slot priors reach 0.8314 and the
  coarse 5-group map 0.7132, both below the champion's 0.8966.
* `seqlab/run17.py`'s `0 observations` was stale state, not a real negative — it harvests 385
  observations at 0.951 child-action accuracy, covering 52/104 training sessions and 17/39 test
  sequence questions. But nested weight selection gives +3 at 0.600, not the +6 the
  contaminated grid reported (`seqlab/run18.py`).
* One unvalidatable anomaly recorded but not shipped: block `[166,167]` is the only test block
  whose clips run backwards in time (gap −100.0s), so the position prior is applied backwards
  there. `make_pseudo` builds blocks chronologically, so this can never be validated OOF.

**Error budget, verified against the real test counts: 19.8 expected public errors vs 21
actual.** Composition: emotion 2-clip 6.8, sequence 4.9, emotion 3-clip 3.6, multi 3.2,
HARn single 1.1, rest 0.7.

**All four pools are now bounded:**

* **emotion, 3-clip (3.6)** — closed by eight representation probes plus the label-structure
  ceiling (per-word 0.8314 < champion 0.8966).
* **emotion, 2-clip (6.8)** — closed at a measured +2.6 ceiling, ~0.6 reachable at 0.619
  precision.
* **sequence (4.9)** — 48 of 77 residual errors are a single pairwise inversion, but the
  inversions are **diffuse**: 119 inversions over 67 distinct pairs, 24 pairs needed to cover
  half, max frequency 3. Not learnable per-pair; per-session evidence noise. Mechanism Y is
  coverage-limited to +0.2.
* **multi (3.2)** — under-prediction confirmed (19 pure-miss vs 10 pure-extra; pair blocks
  miss 14 vs extra 6) but a global pool-inclusion bias is **monotonically worse**
  (0 → 3410, 0.5 → 3407, 1.0 → 3406), because `pool._enum_component` already enumerates only
  constraint-satisfying pools, so a bias picks a different satisfying pool rather than adding
  the missing action, and more inclusive pools break the uniqueness `single`/`combination`
  need. Needs per-action targeting. **The most open of the four and the best next target.**

Reaching 332/342 would require clearing essentially all of the above. Across three sessions
the evidence says that is not available from these modalities and this generator structure;
the honest reachable range from identified mechanisms is +1 to +2.

### Multi action-level atlas (2026-09-04)

Built `research/multi_pair_atlas_20260904/` against the byte-frozen champion
`submission_093859_SUBMITTED.csv` (SHA-256
`9a45bfbf57c516f05a274d901f0765cba6b4c534341d1b4c761991e33a0e9cd3`). The five-fold,
subject-disjoint pair-thinned `ends` atlas covers 719 Multi questions and 14,686 candidate
action rows. It records every satisfying full-block candidate pool, candidate rank/score,
pool membership, action-level evidence, modality availability, and sibling/child support.
The independently retained production pool matched the live solver on all 275 OOF blocks.

The champion has 21 false-negative actions versus 14 spurious actions; pair blocks account
for 14 false negatives versus 6 extras. Sixteen of the 21 misses have a satisfying best
pool containing the omitted action; 5 are structurally unavailable under the current hard
constraints. For pair candidates with a satisfying best-X pool, only 14/391 are genuine
inclusions, so global inclusion bias is not a safe repair. Sibling predicted support is
zero for every remaining eligible omission because the pool constraints already force
selected actions; it cannot distinguish these residuals.

The fixed gate discovered after atlas inspection fires 3/3 on full OOF and identifies two
test-side exploratory rows (`test_0206` Stirring and `test_0212` Running). Strict nested
subject-disjoint selection collapses to 3/6 action precision and zero net question gain;
the nested logistic variant is 2/9 with net -5. No override was promoted and no Kaggle
submission was made. Full atlas, candidate pools, selector metrics, and test inspection
are retained in `research/multi_pair_atlas_20260904/RESULTS.md`.
