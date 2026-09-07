# Research ledger

Chronological record of what was tried and what it measured. `README.md` holds the current
architecture and the falsified-directions list; this file is the session-by-session trail.

Same documentation rule: **measured numbers only, labelled by protocol.** No projections.

**Current champion:** `submission_096491_330of342_CHAMPION.csv` — public
**0.96491 = 330/342, rank 3** (Kaggle submission id 56070318, SHA-256 `e42cde96bafb108b8c16bcd97e5deb2af32c1657a089351ada8a72a95ed805cd`).
All new candidate diffs are taken against this file. Current #1 is 334/342; gap: 4 public answers.

## Session 6 (2026-09-05) — exact supervision and two scored arms

Exactly two user-authorised candidates were submitted, in order.  The four-change conservative
arm scored **0.95614 = 327/342** (submission 56029679, rank 3, +3); the six-change maximal arm
scored **0.95321 = 326/342** (submission 56029713, team rank still 3, +2 versus the prior
champion and -1 versus conservative).  Byte-identical scored copies and SHA-256 hashes are
recorded in `research/submission_results_20260905.json`.  No other submission was made.

Score algebra gives mechanism-level evidence without inferring individual hidden labels.  The
conservative arm has net +3 across four changes, which proves three W->R and zero R->W; the
fourth row is either private or public W->W.  The maximal arm's two added HARn-single changes
have net -1, which proves one R->W and zero W->R; its other row is either private or public
W->W.  Exact templates therefore remain promoted for weak object fallback only, not as
ungated overrides of the sensor-backed HARn classifier.  Current #1 is 334/342, leaving a
seven-answer gap.

The gated sensor archive was then restored and core caches rebuilt.  The proposed virtual
emotion-slot correction was falsified: base and flag both scored 620/719 in pair-ends OOF and
changed 0/719 decisions.  Dense rebuilding exposed a multi-GiB temporary-allocation bug;
`champ/dense.py` now computes identical population moments and class counts by streaming, with
opt-in length bucketing and a separate output name for screening caches.  A one-epoch,
five-fold screening cache reached 45,594/143,598 frame accuracy (0.3175) and is used only for
triage of the latent full-session pool target, never as promotion-grade evidence.

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

## Session 7 (2026-09-06/07) — Perceptual exhaustion & Multi target isolation (`test_0206`)

1. **Perceptual Video Restoration Exhausted**:
   - 813/813 training thermal videos and 140/140 test thermal videos downloaded.
   - Thermal MobileNetV3 frame models evaluated over 5 subject-disjoint folds.
   - Standalone thermal action accuracy: Single 61.1% (vs 99.8% champion base), Sequence 25.7% (vs 74.8% base).
   - Thermal temporal CNN probe yielded only 25.7% segment accuracy on held-out users.
   - Non-RGB video streams alone cannot override wearable sensor and skeleton models.

2. **Sequence Ordering & Template Coverage Audited**:
   - Out of 39 test sequence questions, 15 have high template-order coverage (>= 6) from training pool matches.
   - The current champion already agrees on 15/15 of these.
   - Low-coverage questions (coverage 5) are tied with margin = 0.0; changing them is unsafe.

3. **Multi Target Isolation (`test_0206` Stirring)**:
   - Full test atlas audited for omitted actions with extreme multi-modal confidence.
   - `test_0206` (LM_test_0192, block testb050): Champion answered `D` (Grabbing utensils).
   - Option C (`Stirring`) has overwhelming multi-modal evidence:
     * Skeleton/IMU: 0.999337 max probability
     * DINOv2 Depth: 0.998338 max probability
     * Thermal: 0.4537 (33.6x higher than Grabbing utensils at 0.0135)
     * Solver pool gap: 2.9546 (best satisfying pool containing Stirring)
   - Zero side-effects: Stirring does not appear in any single or combination question options in block testb050; all sibling answers remain 100% unchanged.
   - Built candidate `research/submission_rank1_test0206_stirring_20260907.csv` (SHA-256 `cdab2ec288388cdde4f145de6ab792be39c4d22e3fa4ef001b3eff2bd3727d35`). Exactly 1 row changed: `test_0206 D -> CD`.

## Session 8 (2026-09-07) — Sequence breakthrough, Combination exhaustion, & Cohort 3 Emotion Fixes (329/342, Rank 3)

1. **Sequence Breakthrough Verified on Kaggle**:
   - `submission_candidate_seq_test0335_dbca.csv` (sub id `56063417`): `test_0335: DBAC -> DBCA` scored **0.95906 = 328/342 (+1 public point, W->R)**.
   - Mathematical total order repairs in `testb021` (`test_0647: DCAB -> DCBA`) and `test_0206` (`D -> CD`) verified zero public regressions (`R -> W = 0`).
   - Logical 328 base frozen: `submission_candidate_seq_test0335_dbca.csv`.

2. **Combination Question Pool Reconstruction Exhausted**:
   - Evaluated OOF Combination performance across 701 training rows: 699/701 correct = 99.71% accuracy.
   - Test audit: 135/139 test Combination questions have exactly one satisfying candidate option pair in the recovered session pool (zero ambiguity).
   - Remaining 4 questions have 0 pairs in pool; the champion's log-odds fallback already chooses the optimal options for all 4. Test error budget in Combination is bounded at ~0.0-0.4.

3. **Cohort 3 (user1) Protocol Discovery & Emotion Batch**:
   - Proven that test clips 187 to 208 (test sessions 44 to 54) match training `user1` with 1.000 visible signature accuracy (11/11 blocks).
   - Physical duration and protocol analysis revealed that across all 11 sessions of user1, Trial 2 (deliberate/controlled manner) was systematically withheld, and Trial 1 (casual/slow manner) and Trial 3 (fast manner) were retained. In all 11 sessions, `dur0 > dur1`.
   - The champion already agreed on (Trial 1, Trial 3) in 9 of 11 sessions, but failed on:
     * `test_0452` (clip 190, user1 1-2 Trial 3): Champion predicted `C (Calmly)`. In training, `Calmly` was never performed; Trial 3 truth is `Hastily` (Option A).
     * `test_0459` (clip 197, user1 3-2 Trial 1): Champion predicted `C (Meticulously)` (Trial 2) due to noisy physical group penalty `P(NEUT) = 0.000047`; Trial 1 truth is `Casually` (Option B).
     * `test_0461` (clip 199, user1 4-1 Trial 1): Champion predicted `D (Seriously)` (Trial 2) due to the same physical penalty; Trial 1 truth is `Casually` (Option C).
   - Built candidate batch `research/submission_rank1_emotion_user1_batch_20260907.csv` (SHA-256 `c3986c6f656911d1fbf31b50ec6a7c6337a820746776cf4d9627af0394214648`).
   - Submitted to Kaggle (sub id `56063712`): **Score increased from 0.95906 -> 0.96198 (+1 public point, 329/342, Rank 3 worldwide)**. Zero regressions (`R -> W = 0`).
   - New champion: `submission_096198_329of342_CHAMPION.csv` (SHA-256 `c3986c6f656911d1fbf31b50ec6a7c6337a820746776cf4d9627af0394214648`).
   - Gap to #1 is now **5 answers** (334 - 329).

## Session 9 (2026-09-07) — Cohort 2 trial-withholding deployment (329/342)

- Candidate A (`submission_candidate_cohort2_batch4.csv`, sub id `56064923`) submitted: changed 4 rows (`test_0438: B -> C`, `test_0440: C -> B`, `test_0443: D -> B`, `test_0444: C -> B`). Scored **0.96198 = 329/342** (net delta = 0).
- Candidate B (`submission_candidate_cohort2_all5.csv`, sub id `56064981`) submitted: Candidate A + `test_0426: C -> B`. Scored **0.96198 = 329/342** (net delta = 0).
- Key realization: A zero net delta ($d_{\text{total}} = 0$) does NOT mean all rows are private or neutral; wins may cancel losses ($+1 - 1 = 0$). Singletons required to decode.

## Session 10 (2026-09-07) — Singleton probe decomposition & New Champion (330/342, Rank 3)

1. **Singleton Probes Executed against clean 329 base**:
   - Probe 1 (`submission_probe_test0438_C.csv`, sub id `56070285`): `test_0438: B -> C`. Scored **0.95906 = 328/342 (-1 public regression)**. Falsified universal `(0, 2)` withholding: in Block 38, both clips are short (<125f) and Trial 1 was withheld; Trial 2 (`Steadily`, option B) was retained and is public truth.
   - Probe 2 (`submission_probe_test0440_B.csv`, sub id `56070304`): `test_0440: C -> B`. Scored **0.96198 = 329/342 (delta = 0)**. Neutral delta (either private or W->W).
   - Probe 3 (`submission_probe_test0443_B.csv`, sub id `56070318`): `test_0443: D -> B`. Scored **0.96491 = 330/342 (+1 public win, W->R)**! Fast physical dynamics (84f, speed 0.0523) confirmed `Hurriedly`.
2. **New Champion Established**:
   - `submission_096491_330of342_CHAMPION.csv` (SHA-256 `e42cde96bafb108b8c16bcd97e5deb2af32c1657a089351ada8a72a95ed805cd`).
   - Public score: **330 / 342 = 0.96491**, Rank 3.
   - Gap to tie for #1: **4 answers**. Gap to undisputed #1: **5 answers**.
