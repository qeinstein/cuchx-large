# PCRME — session 4 results (draft, being finalized as remaining stages complete)

Champion for every decision-level comparison: the exact **0.93859 = 321/342, rank #5** artifact
(`submission_093859_SUBMITTED.csv`, Kaggle submission 56007625). OOF flip-precision auditing
uses `champ/audit_champ.csv`, the same frozen snapshot S/T/W/X were themselves validated
against.

## Stage 1 — aligned action-phase dataset

272/272 emotion sessions have >=2 labeled sibling trials. 983 aligned (session, action) tuples
(914 triples + 69 pairs), 2880 rows, 40 actions, all 18 users, directly HARn-anchored (no
dense-localizer inference mixed in). 0/2927 segments repeat an action within one trial, so
alignment is unambiguous. Full detail: `purity_report.json`.

## Stage 2 — cheap falsification probe, and a fairness correction made mid-session

Initial comparison used raw absolute PHYS as the "champion" baseline and found a suspiciously
large +13.5pp combined gain (0.4513 -> 0.5865, HistGradientBoostingClassifier). Per this
session's own leakage-first discipline, a jump that large was treated as a suspected bug before
being trusted. Two checks were run:

1. **Shuffle-label control** on the exact combined arm: real labels 0.5865 vs shuffled-label
   accuracy 0.30 / 0.33 / 0.38 across three seeds. No leakage of the label-encoding kind.
2. **Fairness audit of the baseline itself**: the "champion" arm had used raw absolute PHYS,
   not what `champ/core.py:block_features()` (and hence the actual champion) computes -- a
   TRIPLE feature representation (absolute + within-session z-score + within-session rank).
   That z/rank representation is itself already a session-relative construction, so comparing
   a new session-relative PHASE feature against a session-BLIND whole-clip baseline
   overstated the gain.

**Corrected, fair result** (`stage2b_fair_comparison.py`, same classifier, same protocol):

| arm | 5-fold subject-disjoint accuracy | folds |
| --- | ---: | --- |
| champion `block_features()` [fair baseline] | 0.5713 | 0.584 / 0.508 / 0.560 / 0.560 / 0.645 |
| **+ phase-relative [combined]** | **0.6043** | 0.613 / 0.599 / 0.583 / 0.590 / 0.637 |
| phase-relative alone | 0.5696 | 0.601 / 0.542 / 0.571 / 0.552 / 0.581 |

**Net gain: +3.3pp average, 4/5 folds improved (fold 4 essentially flat, -0.008).** This is
smaller than the initial (unfair) estimate but is a REAL, reproducible signal above the
champion's own existing relative representation -- something none of session 3's five killed
probes (radar/IMU spectral, skeleton DTW, raw skeleton+DINO relative pairs, DINOv2-Depth
trajectory as override or joint-integration) achieved. It also brings 5-way group accuracy to
0.6043, above the previously-quoted historical ceiling of 0.603.

Antisymmetry/grouping sanity check passed (session-relative centering sums to ~0 within each
triple).

**Gate A verdict: PASSED**, on the corrected, fair comparison.

## Stage 7 — decision-level audit against the frozen champion

*(in progress; results below are filled in once `stage7_decision_audit.py` completes)*

This is the metric that actually matters, per the brief: accuracy gains do not by themselves
justify a correction. `stage7_decision_audit.py` re-implements the SAME two unary evidence
terms `champ/core.solve_emotion` uses (physical-group log-likelihood-ratio + position prior,
no pairwise/pool term in either arm, so the comparison isolates only the new classifier's
contribution), fits both arms subject-disjoint, runs the identical bijective/marginalised
assignment search, and compares the resulting letter against `champ/audit_champ.csv`.

### Result, and a bug caught in the first run

The first run of `stage7_decision_audit.py` reported **0.0000 accuracy for both arms**
(baseline and extended alike). Per this session's leakage-first discipline, an implausible
result this extreme (literally zero, not ~25%) was diagnosed before being trusted. The bug:
the truth column was built from the manner **text** (`em["label"]`, e.g. `"Calmly"`) while the
assignment solver's output is a **letter** (`"A"`-`"D"`); every single comparison was therefore
comparing a letter against a sentence and could never match. Fixed by adding an explicit
`truth_letter` column and comparing like-for-like. Rerun gave plausible numbers: unary-only
baseline 0.9122, unary-only extended 0.9159, against the full production pipeline's 0.9073 for
reference (the unary-only reimplementation omits the pairwise/pool terms, so it is not expected
to reproduce the champion number exactly -- it agrees with the champion's actual predicted
letter 92.8%/93.3% of the time, confirming it is a reasonable but not identical proxy).

**A second, more important refinement.** The raw "flips vs champion" count (54, precision
0.569) mixes two different things: rows where the unary-only reimplementation disagrees with
the full champion for reasons that have nothing to do with the new feature (missing
pairwise/pool evidence in the simplified test), and rows where phase-relative evidence itself
changed the call. Isolating the latter (`base_pred != ext_pred`, i.e. holding methodology fixed
and toggling only the new feature):

| comparison | n | W->R | R->W | precision | net |
| --- | ---: | ---: | ---: | ---: | ---: |
| phase-relative changes the call, vs TRUTH | 22 | 12 | 9 | 0.571 | +3 |
| ... restricted to the 11 that ALSO differ from the champion's actual prediction | 11 | 4 | 6 | 0.400 | **-2** |

Per fold (vs truth, n=22 total): fold 0 n=3 prec=0.667; fold 1 n=10 prec=0.556; fold 2 n=0
(no changes); fold 3 n=6 prec=0.333 (net -2); fold 4 n=3 prec=1.000. Small, inconsistent, and
net negative once measured as an actual override against the real champion.

### Verdict: Gate A passed, decision-level promotion FAILS

This explains the earlier accuracy result precisely. The phase-relative feature carries real,
reproducible group-level signal (Stage 2b: +3.3pp average, 4/5 folds) -- but that signal is
diffuse: it nudges many already-close-to-the-boundary group-classification calls, while the
actual joint per-session bijective assignment only changes its final answer on 22 of 809 clips
(2.7%), and on the subset that also disagrees with the champion's real prediction, it is
net-negative (4 W->R vs 6 R->W). This is the SAME failure mode observed for every one of session
3's five killed probes (radar/IMU spectral, skeleton DTW, raw skeleton+DINO relative pairs,
DINOv2-Depth trajectory as override or joint-integration): oracle/aggregate-level lift does not
transfer to the champion's specific hard cases. It is now the sixth time this pattern has
appeared, with a sixth different representation, which is a strong structural finding in its
own right (see the session findings doc for the diagnosis of *why*).

**No correction is proposed from Stage 1/2/7.** Per Gate B's own text ("If neither [meaningful
overall gain or high-precision disagreement subset], do not fine-tune blindly"), and given that
a real, measured accuracy gain from a comparatively rich phase-conditioned skeleton+IMU
representation already failed to clear the decision-level bar, investing further compute in a
more expressive frozen encoder (MotionBERT) over the SAME underlying skeleton modality is not
justified without a new hypothesis for why a better encoder would fix a problem that is not
about representational capacity (Stage 2b already found real signal) but about that signal's
lack of correlation with the champion's SPECIFIC hard residual cases. Gate B / Stage 3 is not
attempted this session; see the findings doc for what would be needed to justify revisiting it.


## Secondary objective — action x manner-group specialist search

`action_specialist_search.py`: per action (40 actions, >=8 sessions each), a small
HistGradientBoostingClassifier trained ONLY on that action's phase-relative rows, compared
against the champion's own predicted manner GROUP (derived from its frozen OOF letter) on
sessions containing that action.

**Result: 0 of 40 actions clear the >=0.80-precision / >=8-disagreement / >=3-fold bar.**
Every single action is net NEGATIVE, and most are far below even chance-level disagreement
precision (median ~0.04, several actions at exactly 0.000 -- Washing dishes, Washing face,
Wiping surface, Brushing teeth, Taking medicine, Standing up, Massaging oneself, Lying down,
Lunges, Listening to music, Checking body temperature, Writing). The single best action
(Sweeping) still only reaches 0.179 precision on 29 disagreements, net -18.

This is not itself evidence of a bug (spot-checked: the champion's derived predicted-group
values exactly reproduce its own labelled manner text on 10/10 sampled correct rows). It is the
expected consequence of comparing a classifier trained on 20-330 tiny per-action rows against
a production system that is already ~85-90%+ accurate in aggregate: on the rare occasions they
disagree, the base rate strongly favours the champion being right and the small, noisy
specialist being wrong. **No action-specific niche exists at any usable precision.** The
brief's hoped-for pattern ("manner can be corrected at 90% precision when the session contains
action X") does not hold for any of the 40 actions in this corpus.

## Gate status — final

- **Gate A** (action-conditioned beats action-unconditioned, reproducibly): **PASSED** after
  the fairness correction (Stage 2b, +3.3pp average, 4/5 folds).
- **Decision-level promotion** (the actual bar every mechanism in this repository must clear):
  **FAILED**. Isolated effect of phase-relative evidence on the real joint assignment: 22/809
  clips changed, net +3 vs truth, but net **-2** when restricted to genuine overrides of the
  champion's actual prediction (precision 0.40, n=11). Six-for-six with every previous
  perceptual-feature probe across two sessions: real aggregate/oracle signal, no transfer to
  the champion's specific hard cases.
- **Secondary objective** (per-action specialist): **FAILED, decisively**. 0/40 actions,
  uniformly and often catastrophically net-negative.
- **Gate B** (frozen MotionBERT): **not attempted**. Both the full-model decision-level test
  and the per-action decomposition failed, and the failure mode in both cases is "the
  underlying signal exists in aggregate but does not correlate with which specific cases are
  hard" -- a problem a more expressive frozen encoder over the same skeleton/IMU modality has
  no clear mechanism to fix, per Gate B's own stated criterion.
- **Gate C** (MoBind-style learned alignment): not started; the Gate B precondition failed.

## Overall verdict

**The phase-conditioned relative manner hypothesis is falsified as a source of a production
correction**, cleanly and on its own terms, after a genuine (initially +13.5pp-looking, then
correctly attributed to +3.3pp) accuracy signal was found and then shown not to survive
decision-level scrutiny. This is the strongest, most carefully falsified attempt at a new
manner representation across both research sessions -- see the main findings doc's diagnosis
of *why* (representation vs. alignment vs. sample size vs. label ambiguity vs. lack of
signal).
