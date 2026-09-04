# PCRME — Phase-Conditioned Relative Manner Encoder

Research question: if we compare corresponding ACTION PHASES across sibling HAU trials,
rather than whole clips, can we learn a representation of HOW an action was performed that
generalises across held-out users? This directory is the session-4 investigation of that
hypothesis, structured as the gated stages below. All work here is training-only /
analysis-only; nothing writes a submission or touches `champ/pipeline.py`'s default path.

The champion for every decision-level comparison is the exact artifact that scored
**0.93859 = 321/342** on the public leaderboard (`submission_093859_SUBMITTED.csv`, a copy of
`submission_corrlayer_S_W_T.csv`, Kaggle submission id 56007625). Its constituent OOF snapshot
for flip-precision auditing is still `champ/audit_champ.csv` (the pre-S/T/W/X pipeline
prediction set that S/T/W/X were themselves validated against); this directory extends that
same convention rather than inventing a new baseline.

## Stage 1 — aligned action-phase dataset (`build_phase_dataset.py`)

Output: `aligned_phases.csv` (2880 rows), `purity_report.json`.

- 272 of 272 emotion sessions have >=2 sibling trials, all labeled.
- 983 aligned (session, action) tuples (914 triples + 69 pairs) from directly HARn-anchored
  segments only (ground-truth folder label, not the dense localizer) -- coverage capped by how
  many sessions/actions have a recorded HARn clip for every sibling, by design (see the
  module docstring for why anchored and dense-inferred evidence are not mixed here).
- Alignment is unambiguous: 0/2927 HARn segments repeat an action within one trial, so
  "which occurrence" never needs to be decided.
- 40 distinct actions, 18/18 users, median ~19.5 aligned tuples per action (Walking dominant
  at 112, Watching TV smallest at 4).

## Stage 2 — cheap falsification probe (`probe_falsification.py`)

Four arms, identical 5-fold subject-disjoint protocol, 5-way manner-group target:

| arm | logreg acc | notes |
| --- | ---: | --- |
| B. phase absolute (per phase row) | 0.385 | |
| C. phase session-relative (per phase row) | **0.459** | beats B in all 5 folds, +7.4pp |
| D. shuffled-label control | 0.273 | correctly collapses toward chance |
| A. whole-clip absolute (champion PHYS, per clip) | 0.501 | different unit of analysis (clip vs phase-row); see clip-level rows below for the fair comparison |

Clip-level aggregation (mean-pool phase rows back to one vector per clip, 776/809 clips
covered):

| arm | logreg acc |
| --- | ---: |
| clip-level mean of phase-absolute | 0.433 |
| clip-level mean of phase-relative | 0.482 |
| (for reference) champion whole-clip PHYS, same clips | 0.500 |

Antisymmetry/grouping sanity check passed (session-relative centering sums to ~0 across each
triple, verifying the (session, action) join is not silently misgrouping rows).

**Gate A read:** action-conditioning (arm C vs arm B) reproducibly helps *within* the phase
representation family, in all 5 folds. It does not, on its own, beat the champion's existing
whole-clip representation. The decisive test is whether it adds anything ON TOP of what the
champion already has -- see `results_joint.md` for that combined test and the verdict.

## Stage 5 (secondary objective) — action x manner-group specialist search (`action_specialist_search.py`)

Per-action, decision-level audit of a phase-relative specialist against the champion's own
predicted manner GROUP (derived from its OOF letter prediction), restricted to sessions
containing that action. Output: `action_specialist_results.csv`.

## Gates (from the brief)

- **Gate A** (action-conditioned beats action-unconditioned): see Stage 2 above.
- **Gate B** (frozen pretrained motion representation shows overall gain or high-precision
  disagreement subset): not yet run: gated on Gate A's combined-feature result.
- **Gate C** (build MoBind-style learned alignment only if earlier stages show latent signal):
  not started.

See `RESULTS.md` for the running verdict as each stage completes.
