# Phase-conditioned relative manner recognition (PCRME) — session 4

Champion for this entire session: the exact **0.93859 = 321/342, rank #5** artifact
(`submission_093859_SUBMITTED.csv`, a preserved byte-identical copy of
`submission_corrlayer_S_W_T.csv`, Kaggle submission id 56007625). Nothing was submitted to
Kaggle this session; no preserved artifact was modified. All work is in `phaselab/`.

## The question this session answers

> If we compare corresponding ACTION PHASES across sibling trials, rather than whole clips,
> can we learn a representation of HOW an action was performed that generalises across held-out
> users?

**Answer: partially yes at the representation level, no at the decision level.** A real,
reproducible, subject-disjoint accuracy gain exists (+3.3pp average manner-group accuracy, 4/5
folds, on top of the champion's own existing session-relative features) — the first time in two
research sessions that any new feature source has cleared that bar. But it does not translate
into a usable correction: isolated at the decision level, it changes the actual joint-session
assignment on only 22/809 clips, and among the ones that also differ from the champion's real
prediction, it is net **negative** (4 right / 6 wrong). A parallel per-action specialist search
across all 40 actions found zero exploitable niches, uniformly and often catastrophically
negative. Both results are documented in detail with per-fold breakdowns in
`phaselab/RESULTS.md`; this document is the higher-level diagnosis the brief asked for.

## What was built (Stages 1, 2, 2b, 7, and the specialist search)

1. **`build_phase_dataset.py`** — 272/272 emotion sessions have >=2 labelled sibling trials.
   983 aligned (session, action) tuples (914 triples + 69 pairs) from directly HARn-anchored
   segments (no dense-localizer inference mixed in, by design), 40 actions, all 18 users,
   unambiguous alignment (0/2927 segments repeat an action within a trial).
2. **`probe_falsification.py`** — cheap linear/nearest-centroid probes; session-relative phase
   features beat absolute phase features in all 5 folds (+7.4pp). This alone is Gate A on its
   narrowest reading.
3. **`stage2b_fair_comparison.py`** — the important correction. The first comparison had used
   raw absolute PHYS as the "champion" baseline and found +13.5pp, which was correctly treated
   as suspicious (per this session's own leakage-first discipline) and checked two ways: a
   shuffle-label control (passed — no gross leakage) and a fairness audit, which found the real
   bug — the baseline should have been `champ/core.py:block_features()`'s actual session
   z-scored/ranked features, not raw absolute values. Corrected result: **+3.3pp**, 4/5 folds.
4. **`stage7_decision_audit.py`** — decision-level audit against the frozen champion. First run
   returned an implausible 0.0000 accuracy for both arms; diagnosed immediately (0.0000 across
   809 rows is not a real result, it is a comparison bug) and found in minutes: truth was being
   compared as manner TEXT against a predicted LETTER. Fixed. Second issue found on inspecting
   the corrected output: the raw "54 flips vs champion, precision 0.569" number conflated two
   different things (a simplified unary-only reimplementation naturally disagreeing with the
   full production pipeline ~7% of the time regardless of the new feature, versus the new
   feature's own effect). Isolating the latter gives the real number: 22/809 clips changed,
   net +3 vs truth, net **-2** on genuine overrides of the champion (precision 0.40, n=11).
5. **`action_specialist_search.py`** — 0/40 actions clear 0.80 precision; median precision
   ~0.04; several actions at exactly 0.000.

## Two bugs caught and fixed in-session — recorded so the pattern is recognised faster next time

1. **Baseline-fairness bug** (Stage 2): comparing a new session-relative feature against a
   session-BLIND baseline overstates its contribution, because part of the "gain" is just
   "session-relative beats session-blind", which is already known and already shipped. Always
   diff against the production feature set's actual output (`block_features()`), not a
   hand-rolled proxy for it.
2. **Unit-mismatch bug** (Stage 7): comparing a predicted LETTER against a TEXT label. The tell
   was not subtle (exactly 0.0000 across 809 rows, not ~20-25%), and the lesson generalises:
   **an exact 0.0000 (or exact 1.0000) accuracy on a multi-class problem with real data is
   almost never a real result — check units/comparison keys before anything else.**

Neither bug involved information leaking forward from labels into features (the specific thing
session 3's retraction was about); both were comparison/bookkeeping errors that inflated or
zeroed a result. All three failure classes (label leakage, baseline unfairness, unit mismatch)
are now understood and should be checked as a matter of course in this codebase's future
pairwise/relative probes.

## Diagnosis: why does the aggregate gain not transfer? (as the brief explicitly requires)

This is now the **sixth** perceptual-feature probe across two sessions to show real
aggregate/oracle-level signal that does not survive decision-level scrutiny (after: radar
micro-Doppler+IMU spectral, skeleton DTW, raw skeleton+DINO relative pairs, DINOv2-Depth
trajectory as an override, DINOv2-Depth trajectory/DTW integrated into the joint objective).
Ruling through the brief's own candidate limitations:

* **Not sample size in the classic sense.** The phase-relative classifier is fit on 620-650
  training clips across the SAME 5-fold splits used throughout this project, and the accuracy
  gain (+3.3pp) is itself fold-consistent — the representation trains fine. What is small is
  the *decision-relevant* sample: only 22 of 809 clips (2.7%) ever have their final assignment
  changed by the new evidence at all, because the position prior and physical-group evidence
  the champion already has are usually decisive on their own, and the new feature only tips
  genuinely close calls.
* **Not gross label ambiguity.** The physics map (session 3) already identified only 3 confusable
  label pairs with any repeated symmetric confusion (Carefully/Meticulously, Contently/Lazily,
  Rapidly/Steadily), too few to explain a pattern spanning all 40 actions and hundreds of
  disagreement rows.
* **Not alignment quality.** Stage 1's alignment is provably unambiguous (0/2927 duplicate
  actions within a trial) and 999/1000 phase rows extracted successfully; this is not a data
  construction problem.
* **The most likely explanation: signal/hard-case mismatch.** The 22 clips whose assignment
  the new evidence *can* change are, almost by construction, the ones where the champion's
  EXISTING evidence (position prior + physical-group classifier + — in the real pipeline —
  the pairwise emopair term) is close to indifferent between two candidates. Those are not a
  random sample of clips; they are disproportionately likely to be the genuinely hardest,
  most physically ambiguous cases in the whole dataset (the ones the champion is ALREADY
  least confident about) — precisely the population where any single additional feature,
  however good on average, is least likely to be reliably decisive, because the difficulty is
  intrinsic to the case rather than a gap in any one feature family. This is consistent with
  the action-specialist search's finding that the champion is very hard to beat on
  disagreements *in general* (not just for this feature): a small classifier trained on any
  narrow slice loses to the full system's combined evidence almost every time they disagree,
  regardless of which slice.

This reframes the emotion residual precisely: it is not "the champion is missing an
representation that would resolve most of its errors" (six independent, methodologically
distinct probes across two sessions would very likely have found at least one such
representation if it existed cheaply in these modalities); it is closer to "the champion's
remaining errors are concentrated in a hard-case population where no single new evidence
source, however real its aggregate signal, reliably wins." Combining MULTIPLE weak, imperfectly
correlated evidence sources (this session's phase-relative signal plus, say, the historical
physical-group classifier plus position prior — which the champion already does) is the right
structural response to that diagnosis, not searching for one more feature.

## What would be needed to revisit this (for a future session)

Per Gate B and Gate C's own text, a MotionBERT/MoBind-style learned encoder is not justified by
this session's evidence, because the demonstrated problem is not "not enough model capacity to
extract a manner-relevant signal from skeleton/IMU" (real signal was found) but "the signal
that exists does not correlate with the champion's specific hard cases." A future attempt should
either (a) target the identified hard-case population directly — e.g. characterise what makes
those 22-ish always-close-call clips different (confidence/margin of the champion's OWN existing
evidence) and ask whether ANY additional signal, not just phase-relative skeleton motion, moves
them — or (b) accept that the manner residual may be at or near the ceiling reachable from these
modalities and redirect effort to the small remaining validated structural gaps (S/T/W/X have
already captured the large ones).

## Session deliverable

No new correction is proposed. `submission_corrlayer_S_W_T.csv` /
`submission_093859_SUBMITTED.csv` (30 changes vs the 0.92105 champion, already scored 0.93859 /
321/342 on the leaderboard) remains the standing candidate. All new code
(`phaselab/`) is analysis-only and does not touch `champ/pipeline.py`'s default path.

## Relevant paths

- `phaselab/README.md` — stage-by-stage overview.
- `phaselab/RESULTS.md` — full numeric results, including the two in-session bug fixes.
- `phaselab/purity_report.json`, `aligned_phases.csv` — Stage 1 dataset and its provenance.
- `phaselab/stage2b_summary.json`, `stage7_decision_audit.csv`, `action_specialist_results.csv`
  — the three decisive result tables.
