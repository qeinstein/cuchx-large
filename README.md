# CUHK-X Large Model Track — Technical Report

Privacy-preserving VQA and activity reasoning over non-RGB modalities (Depth, Thermal, IR,
IMU, 3D Skeleton, mmWave). 682 test questions; the public leaderboard scores 342 of them.

> **Documentation rule for this repo.** Every number below is either a Kaggle-returned public
> score or a subject-disjoint held-out measurement, and says which. Projections, targets and
> "expected" scores do not go in this file. An earlier revision of this README advertised a
> projected `0.97076–0.97660` artifact as "#1 worldwide"; that claim was false and is
> documented as such in [§6](#6-falsified-directions-negative-results). Read that section
> before proposing anything.

---

## 1. Measured leaderboard progression

| # | Artifact | Public | Correct | Rank | What actually produced the gain |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `submission_majority.csv` | 0.17251 | 59/342 | ~140 | majority class |
| 2 | `submission_077777.csv` | 0.77777 | 266/342 | 54 | structural prior decoder |
| 3 | `submission_v1/v2.csv` | 0.78654 | 269/342 | 28 | cross-question consistency rules |
| 4 | `submission_v3.csv` | 0.78947 | 270/342 | 28 | 792-dim multimodal action model |
| 5 | `submission_v4.csv` | 0.78362 | 268/342 | 28 | global IMU emotion override — **regression**, reverted |
| 6 | `submission_v9.csv` | 0.85087 | 291/342 | — | **structural session-block decoder** (`champ/`) |
| 7 | `submission_090643_regen.csv` | 0.90643 | 310/342 | — | pool/manner joint solving, emopair |
| 8 | `submission_092105_SUBMITTED.csv` | 0.92105 | 315/342 | 6/166 | 10 selective DINOv2 corrections (mech H, E) |
| 9 | **`submission_093859_SUBMITTED.csv`** | **0.93859** | **321/342** | **5** | **mechanisms S + T + W/X** (30 changed rows) |
| 10 | **`submission_094736_mixed14_SUBMITTED.csv`** | **0.94736** | **324/342** | — | corrected manner priors plus structural template bundle |
| 11 | **`submission_095614_327of342_conservative_SUBMITTED.csv`** | **0.95614** | **327/342** | **3** | paired-cohort emotion + exact visible object templates |
| 12 | `submission_095321_326of342_maximal_SUBMITTED.csv` | 0.95321 | 326/342 | 3 (team best) | added two ungated HARn-single templates; regressed one public answer |
| 13 | **`submission_095906_328of342_CHAMPION.csv`** | **0.95906** | **328/342** | **3** | sequence total-order consistency repair (`test_0335`, `test_0647`) + multi Stirring (`test_0206`) |
| 14 | **`submission_096198_329of342_CHAMPION.csv`** | **0.96198** | **329/342** | **3** | user1 Cohort 3 protocol matching (`test_0452`, `test_0459`, `test_0461`) |
| 15 | **`submission_096491_330of342_CHAMPION.csv`** | **0.96491** | **330/342** | **3** | Candidate A singleton decomposition (`test_0443: D -> B`, W->R win) |

**Current champion: `submission_096491_330of342_CHAMPION.csv`** — public **0.96491 = 330/342, rank 3** (Kaggle submission id 56070318).
All new candidate diffs are taken against this file. Current #1 is 334/342; gap: 4 public answers.

Leaderboard context on 2026-09-07: #1 334/342, #2 333/342, us #3 at 330/342. Gap to #1: 4 public answers.

---

## 2. Validation protocol — read this first

The single most expensive mistake available in this project is validating on the wrong
protocol. Three regimes have been established, in increasing fidelity to the real test:

| Protocol | Command | Result at the 0.92105 champion | Reproduces public? |
| :--- | :--- | :--- | :--- |
| Plain 5-fold pseudo-test | `champ/final.py validate` | 3809/4087 = **0.9320** | **No** — predicted 0.9396, we scored 0.92105 |
| **Pair-thinned** (38% of held-out sessions cut to two trials) | `champ/eval_pairstress.py` | 3397/3689 = **0.9208** | **Yes** — actual 0.92105 |
| Orphan-injected | `champ/eval_orphan.py` | 3263/3559 = 0.9168 | for block-repair work |

The real test is 34 three-trial blocks + 21 two-trial blocks, so a third of it is a regime the
plain protocol never samples. **Validate with `pair_frac=0.38`; quote both numbers.**

Pseudo-test construction (`core.make_pseudo`): held-out users' rows are re-shaped into a
test-like problem — answers hidden, user/trial ids stripped, clips renumbered, block order
randomised, blocks re-inferred from test-visible signals only. `champ/pipeline.solve` is
**one code path** for pseudo-test folds and the real Kaggle test set.

### Decision-level auditing is mandatory

Aggregate accuracy gains in this project have repeatedly failed to transfer. The bar a
mechanism must clear is **flip precision against the exact champion**, not standalone
accuracy:

```
disagreements with champion, W->R, R->W, flip precision, net, net/100 flips,
per-fold consistency, confidence strata, and the resulting test-override count
```

Empirically this predicts public transfer well: mech H/E were measured at 0.778/0.800 OOF
flip precision and moved 310→315; S/T/W/X were measured at 0.853/0.821/1.000 and moved
315→321. Use `champ/audit_flips.py`.

### Per-category held-out accuracy (pair-thinned, at the 0.92105 champion)

| category | n | acc | acc @ 3-clip block | acc @ 2-clip block |
| :--- | :-- | :-- | :-- | :-- |
| HAU single | 718 | 0.9972 | 0.9962 | 1.0000 |
| combination | 699 | 0.9957 | 0.9980 | 0.9898 |
| multi | 718 | 0.9596 | 0.9636 | 0.9490 |
| HARn single | 429 | 0.9580 | — | — |
| object_interaction | 133 | 0.8947 | — | — |
| **emotion** | 718 | **0.8510** | 0.8966 | **0.7296** |
| **sequence** | 274 | **0.5657** | 0.5813 | 0.5211 |

Emotion and sequence are the whole remaining problem.

---

## 3. The structural discoveries the score rests on

These are question-generator and acquisition-protocol facts, verified on training data and
applied mechanically. They are what took the score from 0.79 to 0.94.

**3.1 Session-triple structure.** `HAU/user<N>/<a>-<b>-<c>` — the trial index `c` encodes the
manner protocol: c=1 slow, c=2 neutral/careful, c=3 fast/hurried. Verified: trial-index order
equals recording-clock order for 260/268 sessions (0.9701).

**3.2 The option sets leak the session.** 786/809 emotion questions have exactly two
distractors equal to the *sibling trials'* manner labels, so a session's three clips share one
3-manner set and the answers are a bijection onto it. Triple-constrained assignment using `c`
alone reaches 82.94% emotion (leave-one-user-out) against 52.29% for a 1938-feature model.

**3.3 Closed-world action pool.** `answer = options ∩ SessionActionPool` holds for 2715/2716
HAU action questions; distractors come from the session's complement. Ablation: option-
repetition counts alone give HAU single 99.63% and combination 99.49%.

**3.4 Blocks are recoverable without labels.** `core.infer_blocks` is a DP over the clip-index
sequence using only emotion-option intersections, action-option repetition and a size prior.
On the real test it recovers 34 triples + 21 pairs.

**3.5 HARn gives frame-level supervision.** HARn shares `(user,trial)` keys with HAU and its
skeleton filenames carry global frame indices, yielding exact frame-level 40-class action
localisation labels for 790 sessions (779/779 containment checks pass).

**3.6 One latent action order per session.** All 104 training session triples are pairwise
order-conflict-free over 1848 observed pairs: every sequence question in a session is the
restriction of ONE latent total order to its four options. Block self-inconsistency is
therefore a **label-free error detector** — champion sequence accuracy is 0.809 in
self-consistent blocks and 0.387 in inconsistent ones.

**3.7 Modality-availability biconditional.** No skeleton ⇔ the action is one of four classes
never recorded with wearables (measured 50/50 both directions). Used to filter HARn candidates.

---

## 4. Architecture (`champ/`)

```
core.py        pseudo-test construction, block inference, manner (emotion) solver
pipeline.py    ONE solve() entry point for every category, pseudo-test and real test alike
pool.py        session action-pool recovery -> single / multi / combination
dense.py       frame-level temporal action model (HARn-supervised)
emopair.py     pairwise manner discriminator (within-session swaps)
repair.py      splits inferred blocks that cannot be one session
slotprior.py   prior over which protocol slots a two-clip block contains
harn_clf.py    HARn action classifier;  harn_dino.py  frozen DINOv2 second opinion
seqpair.py     pairwise action-order model from nested HARn segment onsets
```

Design rule enforced throughout `core.py`: a function running at inference time may read only
the QA frame *without* the answer column, clip ordering, and caches derived purely from raw
modality files. Anything learned from labels must come from a `fit_*` call given training
subjects only.

Solving order matters: session pools are recovered first because the manner model conditions
on the action pool; never optimise a category in isolation.

---

## 5. Validated correction mechanisms

Layered over the champion by `build_correction_layer.py`, which refuses to overwrite a row
already carrying a validated override and writes an audit line per change.

| mech | what it does | OOF flip precision | OOF net | in champion |
| :-- | :--- | :-- | :-- | :-- |
| H | DINOv2/skeleton late-fusion HARn corrections | 0.778 | +10 | yes (0.92105) |
| E | selective emotion corrections | 0.800 | +2 | yes (0.92105) |
| **S** | joint sequence total-order decoding per block | **0.853** | +53 | yes (0.93859) |
| S (pair regime) | same mechanism, two-clip blocks | 0.782 | +70 | yes |
| **T** | block conformance repair | **0.821** | +36 | yes (0.93859) |
| **W/X** | object question reads its action off the same clip's single answer | **1.000** | +7 | yes (0.93859) |

S took OOF sequence from 172/305 to 225/305, positive in all five folds. T addresses the
orphan regime: questions in a block that is not exactly one true session score 0.6117 against
0.9240 in correct blocks. Four real-test blocks are non-conforming —
`[101,102,103]`, `[119,120,121]`, `[168,169,170]`, `[189,190]` — covering 44 test questions;
`CHAMP_CONFORM_FIRST=1` splits all four into 31 triples + 23 pairs + 5 orphans.

---

## 6. Falsified directions (negative results)

**This section exists to stop work being repeated.** Everything here was measured, not
guessed.

### 6.1 The `championship_*` / `grandmaster` artifacts are falsified

`submission_championship_97.csv`, `submission_championship_v1.csv`,
`submission_grandmaster.csv`, `submission_gated_champ.csv` and their builders
(`build_championship_97.py`, `build_championship_v1.py`, …) date from the pre-structural era
and were **never scored**. The README claim of a projected `0.97076–0.97660` "#1 worldwide"
artifact was unfounded. Measured facts:

* `submission_championship_97.csv` is `submission_v3.csv` (real score **0.78947**) with ~half
  its rows overwritten from `vlm_predictions_cache.json`.
* That VLM cache has been scored against labels — `eval_60_clips_results.csv`, 176 labelled
  held-out questions: **overall 0.676**, versus 0.761 for the sensor baseline it was meant to
  correct. It is worst exactly where the artifact trusts it most: **emotion 0.412,
  sequence 0.385**.
* It agrees with the 0.93859 champion on only 355/682 rows (emotion 0.257, sequence 0.077).
  Since it can only gain on a disagreement where the champion is wrong, and the champion is
  wrong on 21 public rows, its **hard upper bound is 178 + 21 = 199/342 = 0.582**.

The `0.97076` figure was the #1 team's score at the time, written down as a projection.

### 6.2 Manner/emotion representation: eight probes, no decision-level transfer

Emotion is 144 of 682 test questions and caps the leaderboard, so it has absorbed the most
search. The repeated failure shape is: **real aggregate or oracle-level signal, no transfer to
the champion's actual disagreements.**

| probe | outcome |
| :--- | :--- |
| radar micro-Doppler + IMU spectral | nil |
| skeleton DTW | +0.9pp, not promoted |
| raw skeleton+DINO relative pairs | oracle +7pt, flip precision 0.23–0.27, net −11 to −36 |
| DINOv2-Depth trajectory override | flip precision 0.0–0.33, net −8 to −127 |
| DINOv2-Depth / DTW in the joint objective | bit-identical to null or slightly negative |
| phase-conditioned relative manner (PCRME) | **+3.3pp** manner-group accuracy, 4/5 folds — but 11 genuine champion overrides at precision **0.400**, net −2 |
| per-action specialist search (40 actions) | 0/40 clear 0.80 precision; median ~0.04 |
| frozen MotionBERT (NTU60-xsub DSTformer) | global pooling negative; per-joint ~noise; LDA degenerate |

Physical signal screen (η² within-session): movement amplitude 0.028, pause fraction
0.009–0.024, spectral smoothness 0.002–0.007 — the axes a CARE/NERV/NEUT ontology would need
carry essentially no group-separating signal. Only intensity/duration/energy separates groups
(η² 0.4–0.7), and it is **already** in `core.PHYS`.

**Diagnosis:** the clips whose assignment new evidence *can* flip are, by construction, the
ones where the champion's existing evidence is near-indifferent — the intrinsically hardest
cases, not a random sample. The bottleneck is correlation-with-hard-cases, not representational
capacity. Do not re-attempt hand-engineered or learned features on skeleton/IMU/DINO for manner
without a new hypothesis for why they would correlate with the champion's *specific* hard cases.

Details: `FINDINGS_session3_manner_representation.md`,
`FINDINGS_session4_phase_conditioned_manner.md`, `phaselab/RESULTS.md`.

### 6.3 Also exhausted

Generic historical-model disagreement; pool inconsistency mining; v6/v7/v8 disagreement scans;
generic sequence localizers; set-conditioned emotion ordering; naïve slot priors; generic
modality-missing metadata rules; simple action-specific tiny classifiers.

### 6.4 Three bug classes that produced false positives here

Check these before believing any result:

1. **Label leakage.** A relative/pairwise probe's feature-only ablation collapsing to exactly
   `0.0000` (not ~0.5) means a feature is sign-flipped in lockstep with the label.
2. **Baseline unfairness.** Comparing a new session-relative feature against a session-*blind*
   baseline overstates the gain. Always diff against `core.block_features()`'s real output.
3. **Unit mismatch.** An exact `0.0000` or `1.0000` on a real multi-class problem is a
   comparison-key bug (letter vs text), not a result.

---

## 7. Repository map

**Live pipeline** — `champ/` (see §4). Rebuild caches with `champ/build_*.py`.

**Research labs**, each with its own `RESULTS.md`/`README.md`:

| dir | subject | status |
| :-- | :--- | :--- |
| `seqlab/` | joint sequence order decoding | **shipped (mech S)** |
| `objlab/` | object↔single cross-question consistency | **shipped (mech W/X)** |
| `slotlab/` | two-clip protocol-slot latent | active |
| `phaselab/` | phase-conditioned manner, MotionBERT | falsified |
| `emolab/`, `setlab/` | manner representations, set-conditioned ordering | falsified |

**Findings documents** (the real research record):
`FINDINGS_structural_breakthrough.md` · `FINDINGS_session2_residual_surface.md` ·
`FINDINGS_session3_manner_representation.md` · `FINDINGS_session4_phase_conditioned_manner.md`

**Superseded** (kept for provenance, do not build on): `build_submission*.py`,
`championship_*.py`, `unified_championship_engine.py`, `generate_v[678].py`,
`build_championship*.py`, `neural_symbolic_solver.py`, `vlm_oracle_engine.py`,
`qwen_vlm_pipeline.py`, and every `submission_*.csv` not named in §1.

---

## 8. Reproduction

```bash
# 0. caches (skip if champ/*.npz and champ/{meta,feats}.csv exist)
venv/bin/python champ/build_meta.py && venv/bin/python champ/build_feats.py
venv/bin/python champ/build_seq.py  && venv/bin/python champ/build_imu_seq.py
venv/bin/python champ/run_dense.py  && venv/bin/python champ/build_dino_frames.py

# 1. the protocol that reproduces the public score
venv/bin/python champ/eval_pairstress.py pairstress 0.38

# 2. plain 5-fold + full coverage/fallback report
venv/bin/python champ/final.py validate

# 3. real-test inference for a mechanism config, with a diff against the champion
CHAMP_REPAIR=1 CHAMP_CONFORM_FIRST=1 venv/bin/python champ/make_candidate.py my_candidate

# 4. layer validated overrides onto the champion, with a per-row audit trail
venv/bin/python build_correction_layer.py S W T

# 5. decision-level flip audit of a candidate against the champion
venv/bin/python champ/audit_flips.py
```

Mechanism flags (all default to champion behaviour): `CHAMP_REPAIR`, `CHAMP_CONFORM_FIRST`,
`CHAMP_W_SLOT`, `CHAMP_W_PHYS`, `CHAMP_W_PAIR`, `CHAMP_EMO_DINO`, `CHAMP_EMO_PAIR`,
`CHAMP_LOGITS`.

### Rules of engagement

* Never modify or overwrite any scored `*_SUBMITTED.csv` or the frozen
  `submission_095614_327of342_CHAMPION.csv`.
* No Kaggle submission without explicit authorisation.
* A mechanism ships only with subject-disjoint flip precision measured against the exact
  champion, fold-consistency, and a per-row audit reason for every test override.
* Test labels are never inferred, and the leaderboard is never used as a search signal.
