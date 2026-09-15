# Manner/protocol representation search — session 3

Primary objective this session: find a perceptual representation that distinguishes
Emotion/manner (the "protocol" a fixed action script is performed under) beyond what the
champion's existing manner-group classifier (~0.60 accuracy) captures. Secondary objective:
an honest missing-wearable ablation for the 13 no-LMT HARn test clips.

Everything below is training-only / analysis-only unless stated otherwise. No Kaggle
submission was made. No preserved artifact (`submission_092105_SUBMITTED.csv`,
`submission_090643_regen.csv`, `submission_final_assault.csv`, or the S/T/W/X commit history)
was modified.

**Bottom line: no new perceptual signal was found.** Five independent, methodologically
distinct probes across radar, IMU, skeleton, and DINOv2-Depth all converge on the same
result: manner information beyond the existing intensity/duration axis is not extractable
from these modalities in a way that helps the champion's actual decisions, whether used as an
override policy or integrated into its own joint pairwise objective. The no-LMT ablation is
also killed, on its own pre-declared bar. The session's only positive, validated deliverable
remains the S/T/W/X correction layer inherited from session 2, which is finalized and ready.

## 1. Physical map of the manner ontology (this session's own contribution)

`research/emotion_manner_physics_map_20260904/` (RESULTS.md, `tables/*.csv`) reverse-engineers
the label ontology against every physical feature available, session-normalised (centred and
ranked inside each `(user, aa, bb)` session so subject and script are held fixed). Full
one-way-eta² ranking of 47 candidate features against the 5-group ontology
(`tables/physical_signal_screen_within_session.csv`):

| feature | eta2 (group-separating power) |
|---|---|
| imu_gyr_mean, imu_dacc_mean, imu_acc_energy/std, **duration_s** | 0.68 - 0.71 |
| imu_gyr_energy, dino_path | 0.65 - 0.67 |
| dino_straight, raw_speed_mean, sk_vankle_mean, sk_v_mean | 0.40 - 0.50 |
| ... (intensity/energy/duration features, monotonically decreasing) | |
| **raw_pose_amplitude** (movement amplitude) | **0.028** |
| **raw_longest_low_motion_s**, **raw_low_motion_frac** (pause proxies) | **0.024, 0.009** |
| **raw_hip_efficiency**, **raw_wrist_ankle_ratio** | 0.058, 0.008 |
| **sk_spec_cent**, **dino_spec_cent** (periodicity/smoothness) | **0.007, 0.002** |

This directly answers the brief's request to check "movement amplitude," "pauses," and
"periodicity/smoothness" as candidate manner axes, separately from speed. **They are not.**
Every feature the brief hypothesised as a *non*-speed manner axis ranks in the bottom third,
near zero. The entire separable signal is the intensity/energy/duration axis, which is already
in `champ/core.py`'s `PHYS` list and already drives the historical group classifier
(0.603: FAST 0.887, SLOW 0.789, but CARE 0.223, NERV 0.036, NEUT 0.302 — the groups that would
need an amplitude/smoothness/pause axis to separate are exactly the ones this data does not
support separating).

Residual confusion structure (on the frozen `champ/audit_champ.csv` baseline, 75 errors / 809):
62 of 75 keep the block's predicted labels a permutation of the truth (evidence for relative/
within-session methods, but not proof every residual is a two-clip swap — 56 rows are an exact
two-cycle). Only **3 label pairs** repeat as a *balanced* confusion (≥2 errors each direction):
Carefully↔Meticulously, Contently↔Lazily, Rapidly↔Steadily. The last is the only cross-group
pair and has the cleanest physical signature (`tables/residual_pair_physical_map.csv`): 10
matched sessions, `sk_cad_pow` (cadence power) higher for Rapidly with **1.0 sign
consistency** — but `sk_cad_pow` is already in `PHYS`, and 4 symmetric errors is too small a
population to build or validate a specialist on (session 2 already found napp/pool-style
micro-clusters below viable sample size; this is the same shape of dead end).

## 2. Five independent kills on temporal/relative representations

All five use the same evaluation contract: subject-disjoint 5-fold, and — critically — a
**decision-level audit against the immutable `champ/audit_champ.csv` snapshot** (or the
champion's own joint pairwise objective), not just an oracle/true-label accuracy number. This
distinction mattered: several of these show real oracle-level lift that evaporates or reverses
at the decision level.

| # | representation | oracle / standalone result | decision-level result | verdict |
|---|---|---|---|---|
| 1 | radar micro-Doppler + IMU spectral (session 2) | n/a | manner-group acc 0.5958 → 0.6007 | dead |
| 2 | skeleton DTW warping path (session 2) | pairwise orientation 0.9738 → 0.9825 | +0.9pp, not promoted | marginal, not used |
| 3 | raw skeleton + DINO relative pair probe (`emolab/probe_relative_raw.py`) | orientation 0.9378 vs 0.8642 matched control (+7.4pt over prior-only) | best gate: 26 pairs, 13 W→R/36 R→W, **precision 0.227-0.265, net -11 to -36** | killed — oracle lift does not transfer to the champion's actual disagreement distribution |
| 4 | DINOv2-Depth frame-delta trajectory, independent override policy (`probe_dino_trajectory.py`) | orientation acc 0.91-0.96 across most fold/head/regime cells | every regime/head/threshold: **precision 0.0-0.33, net -8 to -127** | killed, uniformly and badly |
| 5 | DINOv2-Depth frame-delta trajectory, joint-objective integration (`probe_joint_integration.py`) | — | champion-only 0.9738 vs combined **0.9738, bit-identical in all 5 folds** | dead — classifier assigns it zero weight |
| 6 | DINOv2-Depth DTW alignment (raw sequence, not pooled), joint-objective integration (`probe_dino_dtw.py`) | standalone 0.0000 (correctly implemented control) | combined **0.9726 vs 0.9738 baseline** (fold 3 regresses) | dead, slightly negative |

Probes 1-2 are from session 2. Probes 3-6 are this session's; #3 and #4 were substantially
complete on disk from earlier work this session and are audited/confirmed above; #5 and #6 are
new.

### A retraction, made and corrected within this session

The first run of `probe_joint_integration.py` reported +1.4pp (0.9738 → 0.9875, 4/5 folds
improved). **This was a label-leakage bug**, not a real result: the trajectory-diff feature was
sign-flipped in lockstep with the swap/no-swap label being predicted for that row, which
trivially encodes the answer. The champion's own convention (used by `dtw_of()` in
`champ/emopair.py`, and by every `PHYS`-derived difference column) keeps a pair's physical
features **fixed** regardless of which manner hypothesis is being scored, and lets the
classifier learn the interaction with the categorical group/slot columns itself. Once corrected
to match that convention, the "trajectory only" arm collapsed to exactly 0.0000 accuracy — the
two rows per pair became indistinguishable — which is itself the diagnostic that caught the
bug. Full writeup and retraction: `research/dino_temporal_style_probe_20260904/
RESULTS_joint_integration.md`. This is recorded here so the mistake is not repeated: **any
future pairwise/relative probe in this codebase must verify that a "trajectory only" or
"feature only" ablation does not collapse to 0.0000**, which would indicate the same class of
leakage rather than a real null result.

### Why this matters beyond "these five things didn't work"

The pattern across #3-#6 is specific: real, measurable *oracle*-level or *standalone*-level
signal (probe #3's control-adjusted lift, probe #4's high fold/regime orientation accuracy)
does not survive contact with (a) the champion's actual, harder disagreement distribution, or
(b) a fair joint-objective comparison against features the champion already has. This is the
same failure mode the brief's own framing warns about ("A model that is globally worse but
wins 85% of a reproducibly identifiable disagreement subset is extremely valuable" — the
inverse also holds: a model that looks globally better can still be worthless on the subset
that matters). It is now observed five times with two different modalities (skeleton+DINO
combined, DINOv2-Depth alone) and three different formulations (independent override, relative
pair probe, joint-objective integration). This is strong, convergent evidence — not proof of a
mathematical impossibility, but enough that further hand-engineered feature variations on these
same caches are very unlikely to clear the bar without a materially different mechanism (a
trained temporal encoder rather than hand-statistics; see §4).

## 3. No-LMT HARn ablation — killed on its own pre-declared bar

`experiments/no_lmt_harn_20260904/` (already substantially built this session before this
writeup; the `--scope all --evaluate` run was completed and reported here). Protocol: raw
`Depth/Depth.mp4` only (no LMT/skeleton/IMU/radar/IR/Depth_Color/`meta.csv`/existing caches),
44-class action classifier trained on all direct-video training descriptors, subject-disjoint,
compared against the exact `oof_v8_final.csv` fallback vector `champ/pipeline.py` already uses
for `harn_no_evidence` rows.

Full-scope result (3098 training descriptors, 562 held-out HARn questions):

| gate (pre-registered) | n disagreements | W→R | R→W | flip precision | net |
|---|---|---|---|---|---|
| confidence ≥ 0.00 | 120 | 59 | 47 | 0.557 | +12 |
| confidence ≥ 0.70 | 92 | 53 | 32 | 0.624 | +21 |
| confidence ≥ 0.80 | 84 | 49 | 28 | 0.636 | +21 |
| **confidence ≥ 0.90** | **74** | **44** | **24** | **0.647** | **+20** |

(All rows restricted to "test-style actionable" — excludes questions the no-wearable
biconditional already resolves.) Best precision at the highest pre-registered gate is
**0.647**, short of the **0.80** minimum the brief and this repository's own precedent (session
2's noise floor / mechanism-U kill) require. The *ungated* per-fold breakdown additionally has
two of five folds net-negative (fold_2 -2, fold_4 -5) before any confidence filter, so even the
positive aggregate is not fold-consistent. **Killed.** No override proposed; the champion's
existing v8 fallback stays. Full report: `experiments/no_lmt_harn_20260904/REPORT_all.md`
(verdict section appended this session).

## 4. What would be needed for the ideal outcome, honestly

The brief's own external-research shortlist (`results/research/
2026-09-04-motion-quality-representations.md`) ranks a **learned** session-pair alignment
scorer (soft-DTW / cross-attention over pose+IMU+DINO token streams, in the spirit of
CoRe/FineDiving action-quality-assessment work) as the highest-value untried direction, with a
frozen/linear DINO-trajectory baseline as its own recommended cheap first probe.

That cheap first probe is exactly what #5 and #6 above are — a linear/tree model over rich
hand-engineered trajectory and DTW-alignment statistics, which is more expressive in its input
features (velocity, acceleration, curvature, 8-phase rhythm, spectral bins, per-frame DTW
residuals) than a "2-layer TCN + linear head" would need to discover a signal a boosted tree
cannot. Both came back null (#5) or slightly negative (#6). Per the research doc's own
recommended order and evaluation contract ("stop unless both full Emotion accuracy and
champion-disagreement precision improve consistently... do not spend a long run on a large
pretrained model before demonstrating that raw temporal information improves the relevant
*relative* decision"), the honest reading is that this session's cheap probes did not clear the
bar to justify building the learned encoder. A trained model *could* still in principle extract
something a fixed statistic cannot (e.g. a specific brief mid-clip hesitation localized to one
body part, which no summary or global DTW-path statistic isolates) — that possibility is not
disproven — but building and properly validating it (soft-DTW/cross-attention, part-aware,
subject-disjoint, with the full decision-level audit contract) is a multi-hour undertaking this
session's evidence does not justify starting blind. It is the correct next thing to try in a
future session if it is tried immediately, before re-deriving any of the above.

## 5. Session-3 deliverable

No new corrections were added to the S/T/W/X correction layer this session — the emotion
representation search found nothing to add. `submission_corrlayer_S_W_T.csv` (30 changes vs
the 0.92105 champion; audit in `submission_corrlayer_S_W_T_audit.csv`) remains the
strongest validated candidate, expected **+6 to +7 public (~321-322/342)**, per session 2's
validation. All new code this session (`champ/emopair.py`'s `W_TRAJ` hook,
`research/dino_temporal_style_probe_20260904/probe_joint_integration.py` and
`probe_dino_dtw.py`, the no-LMT ablation's full-scope evaluation) is either default-off or
pure research/analysis and does not change the champion's behaviour. Regression-checked:
`emopair.W_DTW == 0.0` and `emopair.W_TRAJ == 0.0` by default, matching the pre-existing
default-off convention from session 2's `CHAMP_REPAIR`/`CHAMP_W_SLOT`/`CHAMP_EMO_DTW` flags.

## Relevant paths

- `research/emotion_manner_physics_map_20260904/` — physical/confusion map (RESULTS.md,
  `tables/*.csv`), pinned to `champ/audit_champ.csv`.
- `research/emotion_relative_raw_20260904/` — raw skeleton+DINO relative pair probe, killed.
- `research/dino_temporal_style_probe_20260904/` — DINOv2-Depth trajectory probes:
  `probe_dino_trajectory.py` (override, killed), `probe_joint_integration.py` (joint objective,
  null, with retraction), `probe_dino_dtw.py` (DTW alignment, null/negative).
- `experiments/no_lmt_harn_20260904/` — no-LMT ablation, killed on its own pre-declared bar.
- `results/research/2026-09-04-motion-quality-representations.md` — external literature review
  and the ranked probe order this session largely followed.
