# Residual-error surface of the 0.92105 champion — research session 2

Everything below was measured in this session and is reproducible from the repo.
The 0.92105 submission and the preserved commits/files were not modified.
Nothing was submitted to Kaggle.

---

## Executive summary

Public errors decompose (pair-thinned OOF rates x test counts, halved) into roughly
**11 emotion + 8.5 sequence + 2.9 multi + 1.1 HARn single + 1.1 object** of ~25.
Three mechanisms survived subject-disjoint validation, all of them **selective overrides on
top of the champion**, none of them a global re-fit:

| mech | what it is | OOF flips | W→R | R→W | precision | test flips | expected public |
|---|---|---|---|---|---|---|---|
| **S** | sequence questions of a session decoded jointly as one latent total order | 112 | 64 | 11 | **0.853** | 17 | **+3.5** |
| **W** | object question reads its action off the `single` answer for the same clip | 7 | 7 | 0 | **1.000** | 4 | **+1.5** |
| **X** | reverse direction: no object option fits the single answer → re-solve the pair | 3 | 3 | 0 | **1.000** | 2 | **+0.8** |
| ~~U~~ | learned prior over which protocol slots a two-clip block contains | 15 | 10 | 5 | 0.667 | — | **killed, see §3** |
| **T** | split inferred blocks that cannot be one session | 68 | 46 | 10 | **0.821** | ~4 | **+1.2** |

Mechanism S is positive in all five folds and in both block-size regimes; W and X are
logical constraints with tiny but perfect samples; T is positive in all five folds.
Together: **30 changes, expected +6 to +7 public (≈ 321-322 / 342)**.  That is rank #4-#5,
not yet Top 3 — the honest gap is stated in §6.

Measured noise floor: two runs of the same configuration differ on 3-5 of 3689 rows,
net 0 to -2, so any claimed mechanism must clear about ±2 net.  Mechanism **U** (a learned
prior over which protocol slots a two-clip block contains) did not and was killed; see §3.

### Candidate set (nothing submitted)

| file | changes | mechanisms | notes |
|---|---|---|---|
| **`submission_corrlayer_S_W_T.csv`** | **30** | S 17, T 7, W1 3, W2 1, X 2 | **recommended** |
| `submission_corrlayer_S_W.csv` | 23 | S 17, W1 3, W2 1, X 2 | without the block repair |
| `submission_corrlayer_S_W1.csv` | 20 | S 17, W1 3 | drops the rules that never fired on OOF |
| `submission_corrlayer_S.csv` | 17 | S only | |
| `submission_seqjoint_B_gated.csv` | 14 | S, gated | sequence only, and only where the champion is provably self-inconsistent |

Every file has a per-row audit (`*_audit.csv`) naming the mechanism and its reason.  The
mechanisms touch disjoint rows (S∩T = S∩W = T∩W = 0), and none of them overwrites the 10 rows
the 0.92105 submission already changed — those were themselves validated overrides.

**Attribution note.** The intended regression check (champion flags must reproduce
`submission_090643_regen.csv`) does **not** pass: 12 rows differ.  The cause is in
`champ/log_regen.txt` — that file was produced by a script which substituted v8 answers for 15
`harn_no_evidence` fallbacks, whereas `champ/make_candidate.py` maps a missing prediction to
'A'; the remaining rows are the mechanism H/E settings differing from whatever regen used.
None of it implicates the new code.  Attribution of T was therefore established by a **paired
design** instead: two real-test runs identical except for `CHAMP_REPAIR`, differing in exactly
7 rows, all inside repaired-block scope (6 directly, and `test_0523` because HARn clip 11 nests
in clip **190**, whose block was split).  On all 7 rows the paired baseline agrees with both the
0.92105 champion and the 0.90643 base, so importing them is clean.

Two further mechanisms (**T** block-conformance repair, **U** the two-clip slot prior) are
diagnosed and implemented behind default-off flags but their pipeline validation runs had not
finished when this was written; see §2, §3 and §6.

---

## 0. Error budget, and a pseudo-test protocol that finally matches the public score

Champion 5-fold subject-disjoint OOF (`champ/audit_champ.csv`): **3809 / 4087 = 0.9320**.

| source | category | n | err | test n | exp. test err |
|---|---|---|---|---|---|
| HAU | sequence | 308 | **132** | 39 | 16.7 |
| HAU | emotion | 809 | **75** | 144 | 13.3 |
| HAU | multi | 809 | 29 | 144 | 5.2 |
| HARn | single | 429 | 21 | 51 | 2.5 |
| HARn | object_interaction | 133 | 14 | 21 | 2.2 |
| HAU | combination | 790 | 5 | 139 | 0.9 |
| HAU | single | 809 | 2 | 144 | 0.4 |

That predicts 0.9396 public, but we score 0.92105 — a 6-question public shortfall.
The shortfall is explained: the plain pseudo-test gives every held-out session three trials,
while the real test does not.  Re-running the full pipeline with 38 % of held-out sessions
thinned to two trials (`champ/eval_pairstress.py`) gives

**pair-stress OOF = 3397 / 3689 = 0.9208**, against an actual public 0.92105.

The protocol now matches the leaderboard to within one question, and it localises the loss:

| category | inferred block size 2 | inferred block size 3 |
|---|---|---|
| **emotion** | **143/196 = 0.7296** | 468/522 = **0.8966** |
| multi | 0.9490 | 0.9636 |
| sequence | 0.5211 | 0.5813 |
| combination | 0.9898 | 0.9980 |
| single | 1.0000 | 0.9962 |

Two-clip blocks are the single largest test-specific liability, and emotion carries almost all
of it (-16.7 pp).  The real test has 23 two-clip blocks, i.e. **46 of 144 test emotion
questions** sit in that regime.

---

## 1. MECHANISM S — sequence questions must be decoded jointly (validated, +53 OOF)

### The structure
A sequence answer is a full permutation (`DBCA`), and the three trials of a session are the
same script performed three times.  Measured on training data:

* every session block has a latent **total order** over its action pool, and every sequence
  question is the restriction of that order to its four options:
  **104 / 104 triples are pairwise-order-conflict-free over 1848 observed pairs, zero conflicts.**
* the champion decodes each sequence question **independently** — argsort of the softmax-weighted
  temporal centroid of the four options' dense logits (`champ/pipeline.py:190-204`).

Consequence, checked without any labels: the champion's own predictions inside a block are
frequently mutually incompatible with any single total order.

| champion block self-consistency | blocks | questions | champion accuracy |
|---|---|---|---|
| consistent | 46 | 136 | **0.809** |
| **INCONSISTENT** | 56 | 168 | **0.387** |

On the real test the champion is self-consistent in only **6 of 15** sequence blocks, so it
**provably** errs in at least 9 of them.  All 39 test sequence questions live in blocks whose
siblings also carry a sequence question (9 triples + 6 pairs), so the constraint binds everywhere.

### The evidence
Two order signals the champion never used for ordering:

1. **Cross-clip pairwise order.** Because `answer = options ∩ SessionActionPool` holds per
   *session* (verified here: single 809/809, sequence 308/308, combination 789/790, multi
   808/809), every clip of a block performs the same pool, so *every* clip's dense logits are
   evidence about the *same* order — including for options that only appear in a sibling's
   question.  P(x before y) is taken from the full time marginal (not the centroid) at two
   temperatures, presence-weighted.
2. **A learned pairwise-order model.** The 2927 HARn segments nested in their parent HAU clips
   give the true onset order of every co-occurring action pair — 10788 labelled ordered pairs,
   versus the 1848 implicit in the sequence answers.  The champion consumed those segments only
   as frame labels.  A HistGradientBoosting model over time-marginal statistics + the LOO count
   prior reaches **0.8503 held-out pairwise accuracy**.
   (The count prior alone is 0.784 over 92.5 % of option pairs, and **0.934 on the 27 % of pairs
   with n>=10 and confidence>=0.95** — the earlier "global precedence failed at 110/308" test
   had used only the 1848 sequence-answer pairs, not the 10788 segment pairs.)

### Validation
`score(total order) = 4·crosscliff(τ=0.5) + 4·crossclip(τ=1.0) + 1·learned`, argmax over all
total orders of the block's option union, per-question answers read off as restrictions.

| | n | correct | flips | W→R | R→W | precision | net |
|---|---|---|---|---|---|---|---|
| champion (centroid) | 305 | 172 (0.5639) | — | — | — | — | — |
| **Mechanism S, all flips** | 305 | **225 (0.7377)** | 112 | 64 | 11 | **0.853** | **+53** |
| Mechanism S, gated to inconsistent blocks | 305 | 218 (0.7148) | 91 | 53 | 7 | **0.883** | +46 |

Per fold (champion → S): 34→47, 27→34, 51→62, 24→36, 36→46 — **positive in all five folds**,
per-fold flip precision 0.93 / 0.71 / 0.82 / 0.88 / 1.00.
Weights chosen on a broad plateau (4,4,1)=(8,8,2)=(16,16,4) all give 225; honest nested-CV
weight selection over a 767-point grid gives **209/305**, so the transferable estimate is
0.685-0.738 against the champion's 0.564.

### Validation in the two-clip regime (12 of the 39 test questions, 8 of the 17 flips)
The OOF set has 300 sequence questions in 3-question blocks and only 4 in 2-question blocks, so
the pair regime was simulated by dropping one clip from every training triple **and withholding
that clip's dense logits from the evidence too**:

| regime | n | champion | S | Δ | flips | W→R | R→W | precision |
|---|---|---|---|---|---|---|---|---|
| 3-question blocks | 300 | 0.5633 | 0.7367 | **+17.3 pp** | 109 | 62 | 10 | 0.861 |
| 2-question blocks | 600 | 0.5633 | 0.6800 | **+11.7 pp** | 203 | 97 | 27 | 0.782 |

Both positive in all five folds.  Flip precision holds up in every margin bucket
(<1: 0.812, 1-3: 0.778, 3-8: 0.864, >8: 0.947), and the test flips' margin distribution
matches the OOF one (median 2.14 vs 2.23), so no gating is warranted — ungated maximises
expected correct interventions (+53 vs +46 net).

### Test application
`seqlab/apply_test.py` — blocks from the champion's own DP, evidence from the cached test dense
logits, pairwise model refitted on all 18 training users.  **17 of 39** sequence answers change
(14 in provably-inconsistent blocks, 3 in consistent ones; 9 in triples, 8 in pairs).
Recovered latent orders read as real scripts, e.g.
`Walking < Washing face < Brushing teeth < Combing hair`,
`Walking < Sitting down < Typing on a keyboard < Writing < Checking the time < Turning a page`,
`Grabbing utensils < Eating < Walking < Sweeping < Mopping`.

Regime-weighted expectation: 9 triple-block flips × 0.477 net/flip + 8 pair-block flips × 0.345
= **+7.0 test answers ≈ +3.5 public**.
Candidates: `submission_seqjoint_A_all.csv` (17 changes), `submission_seqjoint_B_gated.csv` (14).

---

## 2. MECHANISM T — four inferred test blocks cannot be sessions

`infer_blocks` is a DP with `kmin=2`; it is structurally incapable of emitting a one-clip block.
The test set contains orphans (sessions with a single released trial), so each orphan is glued
onto a genuine session and corrupts that session's pool and manner candidate set.

Two independent, test-visible signals agree on exactly four blocks:

| inferred block | \|I\| | t0 gaps (s) | verdict |
|---|---|---|---|
| `[101,102,103]` | 0 | 84.6, **1 286 771** | `[101,102]` + orphan `103` |
| `[119,120,121]` | 1 | 70.9, 65.5 | `[119,120]` (\|I\|=3) + orphan `121` |
| `[168,169,170]` | 1 | **79 881**, 88.6 | orphan `168` + `[169,170]` (\|I\|=3) |
| `[189,190]` | 0 | 90.9 | two orphans |

Calibration on training data:
`P(|I| >= size | one session) = 0.990`, `P(|I| >= 3 | different sessions) = 0.0073`
→ **136:1** for same-session when \|I\|>=3 and **158:1** against when \|I\|=0.
Consecutive trials start 61-319 s apart (5th-95th pct); within-session gaps are log-normal
μ=4.75 σ=0.65 versus between-session μ=6.29 σ=2.10.
All 51 conforming test blocks are contiguous index runs, so only contiguous refinements are
considered.  `champ/repair.py` scores every contiguous refinement with the DP's own
intersection/size likelihoods plus the gap term, allowing one-clip parts at a tunable penalty.

Also checked and **rejected**: the non-contiguous alternatives suggested by the option-set graph
(`{168,193,194}`, `{187,188,190}` — the only two triangles in clips 164-208) require clock gaps
of 435 975 s and -94 089 s and are coincidences.  Conversely the pair structure of the test tail
is confirmed: in clips 164-208 the \|I\|>=3 graph is a **perfect matching of 21 adjacent pairs
with 0 contiguous triangles**, against 31 in the three-trial zone 65-163.

### What an impure block costs — measured
`champ/eval_orphan.py` injects orphans (8.5% of held-out sessions reduced to one trial) on top
of the pair thinning, reproducing the exact pathology.  Splitting the resulting questions by
whether their inferred block **is exactly one true session**:

| | n | accuracy |
|---|---|---|
| block is one true session | 2791 | **0.9240** |
| block is NOT one true session | 206 | **0.6117** |

per category on impure blocks: emotion **0.4583** (vs 0.8721), single **0.6875** (vs 1.0000),
combination **0.7021** (vs 0.9920), multi **0.6042** (vs 0.9719).
So an impure block costs about **0.31 accuracy on every question it contains**, and the four
non-conforming test blocks contain **44 questions**.

### Repair, and why conformance must outrank the likelihood
`champ/repair.py` scores every contiguous refinement.  Two modes:

| mode | pseudo-test purity (questions in a correct block) | blocks fixed on the real test |
|---|---|---|
| off | 0.9173 | – |
| likelihood (`CHAMP_SINGLE_PEN` 1-6, flat) | **0.9673** | 2 of 4 |
| conformance-first (`CHAMP_CONFORM_FIRST=1`) | **0.9626** | **4 of 4** |

The likelihood mode looks marginally better on the pseudo-test and is worse on the real one,
and the reason matters: `make_pseudo` randomises block order, so an injected orphan's
index-neighbours are sessions from other users recorded far apart, and the **clock alone
settles it**.  In the real test the orphan's neighbours were recorded minutes away
(`[119,120,121]` gaps 70.9 s / 65.5 s; `[189,190]` 90.9 s), so only the option-set
intersection can separate them.  Conformance-first therefore leans on the statistic that is
actually available (`P(|I| >= size | one session) = 0.990` vs `0.0073`) and uses the
likelihood only to break ties, preferring the fewest one-clip parts.  It is completely
insensitive to the singleton penalty over 1-12, and it recovers exactly the structure derived
by hand: **31 triples + 23 pairs + 5 orphans = 144 clips**.

Label-free check on the test set itself: **59 of 59 repaired blocks conform** (|I| >= size),
against 51 of 55 before.  Recorded in `champ/test_blocks_repaired.csv`.

### Result — validated

| | total | net | flips | W→R | R→W | precision |
|---|---|---|---|---|---|---|
| baseline (orphans injected, no repair) | 3229/3559 = 0.9073 | – | – | – | – | – |
| **repair, likelihood mode** | **3265/3559 = 0.9174** | **+36** | 68 | 46 | 10 | **0.821** |
| repair, conformance-first | 3263/3559 = 0.9168 | +34 | – | – | – | – |

Positive in all five folds (+6 / +9 / +4 / +3 / +14), per-fold flip precision
0.80 / 0.85 / 0.70 / 0.80 / 0.89.  Questions in an impure block fall from **206 to 23**.
Restricted to the questions whose block actually changed (n=755): 0.8543 → 0.9046,
63 flips at **0.852** precision; combination 0.727 → **1.000**, multi 0.622 → **0.867**,
single 0.924 → 0.947, emotion 0.467 → 0.556.

The two modes are statistically tied (+36 vs +34, disagreeing on only 13 rows, of which the
likelihood mode gets 5 right and conformance-first 3 — inside the ±2 noise floor).
**Conformance-first is nevertheless the mode to ship**, and that is a judgement call rather
than a measurement: it is tied on the pseudo-test, it relies on the statistic that is actually
available on the real test rather than on the clock the pseudo-test over-supplies, and it is
the only mode that leaves zero non-conforming test blocks.

### Correcting an earlier over-estimate
An intermediate note in this session reasoned "impure blocks cost 0.31 accuracy x 44 test
questions ≈ +7 public".  That was wrong.  The base accuracy on the blocks the repair actually
changes is **0.854, not 0.611** — a triple that absorbed one orphan still contains two correct
clips whose questions are mostly answered correctly.  The measured rate is **+38 net per 755
changed questions = +0.050/question**, so the real test's 44 changed questions are worth
**+2.2 test answers ≈ +1.1 public** (or +1.4 by the flip-rate route: 8.3% x 44 = 3.7 flips at
0.852).  T is a solid, well-validated **+1.1 to +1.4 public**, not +7.

---

## 3. MECHANISM U — the missing-slot latent in two-clip blocks (diagnosed, fix killed)

Diagnosis on the pair-stress OOF, for the 188 of 196 two-clip emotion questions with \|I\|=3:
**the true manner is in the candidate set 100 % of the time**, and accuracy is still 0.734.
So the loss is entirely in the assignment, and it has a signature:

| manner group | true freq | predicted freq | bias |
|---|---|---|---|
| SLOW | 0.383 | 0.439 | **+0.056** |
| CARE | 0.133 | 0.097 | -0.036 |
| NEUT | 0.143 | 0.107 | -0.036 |

In three-clip blocks the same table is exact to ±0.002, because the bijection over three
manners forces it.  A two-clip block has an extra latent — *which two of the three protocol
slots are present* — which `solve_emotion` marginalises with a flat prior, and the manner-slot
prior then pushes a SLOW manner into slot 0 whether or not slot 0 is present.

`champ/slotprior.py` supplies a learned prior over the slot pair from the two clips' physical
features (duration — training frame counts fall 222 / 178 / 140 with trial index — plus the
recording gap, since a skipped trial roughly doubles it).  Subject-disjoint 3-way accuracy
**0.660** against a 0.333 baseline (0.761 on its confident 68%).
Wired into `solve_emotion` behind `CHAMP_W_SLOT` (default 0.0 = champion behaviour).

### Result: not established — killed

| slot classifier | its own accuracy | emotion, two-clip blocks | net |
|---|---|---|---|
| champion (flat prior) | – | 143/196 = 0.7296 | – |
| first version | 0.540 | 148/196 = 0.7551 | **+5** |
| upgraded (+ clock, + frame counts) | **0.660** | 145/196 = 0.7398 | **+2** |

The **better-calibrated** prior produces the **smaller** gain, flip precision is 0.667, and one
fold is −2.  With 196 questions the binomial sd on the count is ~6, so the spread between the
two arms is inside noise: the +5 was luck, not signal.  Not included in any candidate.

Noise floor, measured while checking this: comparing two runs on the rows the slot prior
*cannot* touch (everything except emotion in a two-clip block) gives only **3-5 differing rows
of 3689, net 0 to -2**.  The pipeline is very nearly deterministic — the residual comes from
`set`-iteration order over action strings under a randomised `PYTHONHASHSEED` — but any claimed
mechanism must clear about ±2 net to be real.

---

## 3b. MECHANISMS W and X — cross-question consistency on HARn clips (validated, precision 1.000)

`Which object is the person interacting with?` is asked about a HARn clip, and given the action
the answer is essentially deterministic: `25_Watch_TV` → 'a remote' in 10/10 training questions,
`22_Turn_pages` → 'a documents' in 10/10.  But the champion's object head re-estimates the action
from its own action classifier / pool / DINOv2 fusion rather than reading it off the **`single`
question asked about the same clip**, which the champion answers at 0.951.

Consequence, measured: `25_Watch_TV` object questions score **3/10** while the action is
identified correctly on 9 of those 10 clips, and 'a phone' (per-action prior count **0**) is
predicted five times over 'a remote' (count **10**).

Rule W1: if the same clip carries a `single` question, map the champion's answer to its HARn
action folder and take the argmax of the per-action object prior — but only override when the
champion's own option has per-action prior exactly 0.

| | n | champion | W1 | flips | W→R | R→W | precision |
|---|---|---|---|---|---|---|---|
| 5-fold subject-disjoint | 7 fired | 0/7 | **7/7** | 7 | **7** | **0** | **1.000** |

(Fires in 3 of 5 folds; on the 34 of 133 object questions whose clip has a sibling `single`
question the champion scores 27/34 = 0.794 and the rule scores 34/34 = 1.000.)

Rule W2, purely logical: an option that is **never** the correct object for any action in the
133 training questions cannot be the answer.  It fires 0 times on OOF (the champion never picks
such an option there) but **twice on the real test**, one of which W1 already covers.

Test: **4 overrides** — `test_0528` A→D, `test_0533` C→A, `test_0534` A→B (all W1, all with the
champion's option at per-action prior 0), and `test_0530` A→B (W2; the champion chose 'a towel',
count 0 of 133 — this one was also independently proposed by the earlier mechanism H at 0.778
flip precision, so it is included, and `submission_corrlayer_S_W1.csv` is the variant without it).
Expected public gain ≈ **+1.5**.

### MECHANISM X — the reverse direction of the same constraint
When **no** object option is compatible with the champion's single answer, the single answer is
the suspect one:

| condition | n | champion `single` acc | champion `object` acc |
|---|---|---|---|
| some object option fits the single answer | 30 | **1.000** | 0.767 |
| **no object option fits** | 8 | **0.375** | 1.000 |

Re-solving the pair jointly — take the (single option, object option) combination with the
largest per-action prior count, and only act when that count is > 0 — fires on 3 of those 8:

    single: 3 flips, W->R 3, R->W 0 ;  object: 0 flips (already correct)

Test: fires once, on clip `LM_test_0013`.  The single answer is 'eating food' but the object
options are {a napkin, a clothes, a cloth, a sponge}; 'wiping a bowl' is option C of the single
question and `14_Wipe_bowls` → 'a cloth' has prior count 10.  So `test_0480` A→C and
`test_0525` B→C, one coherent correction of both answers about the same clip.
Small sample (3 OOF flips); the two test changes are perfectly correlated.

---

## 4. Negative results — measured and killed

| hypothesis | measurement | verdict |
|---|---|---|
| Richer manner sensing: 269 radar micro-Doppler + 5-IMU spectral columns | 5-way manner-group accuracy **0.5958 → 0.6007** | **dead** — the signal is duration+intensity, already in PHYS |
| DTW warping-path features between the trials of a session | pairwise manner orientation 0.9738 → 0.9825 | marginal (+0.9 pp), not worth the complexity |
| Comma order of combination / letter order of multi answers encodes temporal order | 0.487 consistent vs **0.533 in distractor controls** | **dead** |
| Generator artifacts in answer letters | sequence permutations uniform (χ²=12.7, df=23, p=0.96); HARn `single` "never D" is just a 3-option format | **dead** |
| Action pool identifies the scenario `a`, so a canonical script can be transferred | Jaccard 0.105 same-`a` vs 0.081 different-`a`; scenario ID from pool only 0.607 | **dead** |
| Cross-user scenario order prior from sequence answers | 0.786 accurate but only when the vote is unique (38 % of questions) and needs `a` | superseded by §1 |
| `loc_maps` / `loc_depth_maps` / `loc_dino_maps` / `dense2_logits` as extra sequence evidence | +2 to +5 on 305 samples after ~30 sweeps; `loc_dino` has no test coverage | inside the noise floor, **not used** |
| Retrain the dense TCN with frozen DINOv2-depth frames as an extra per-frame stream (16 PCA components + velocity, feat dim 231→263) | OOF **frame** accuracy 0.4962 → **0.5047**, but sequence via Mechanism S 225/305 → **222** blended, **202** DINOv2-only | **dead for ordering** — the extra stream improves per-frame identity and *degrades* onset timing, which is what sequence needs. Cached as `champ/dense_dino_logits.npz`; the champion's `dense_logits.npz` is untouched |
| Non-contiguous test sessions | rejected by the recording clock (§2) | **dead** |
| MECHANISM Y: nested HARn children as direct order observations for Mechanism S (child action from its own `single` answer, correct **0.951**; onset from the frame range; 385 observations over 304 parent clips at test-matched density) | 225/305 → 231 at w_y=8 but 226 at w_y=4 and 229 at w_y=16; flip precision 0.571 / **0.750** / 0.625 across adjacent weights | **not used** — the peak is a weight-sensitivity spike, not a plateau, and the test set offers only ~1.5 children per sequence block |
| Timestamp gaps as a standalone block-recovery signal | within-session p95 = 319 s vs between-session p25 = 138 s; 66 % overlap | too weak alone; **useful only as a likelihood term (§2)** |

---

## 5. Oracle ceilings measured this session

* Emotion with a **perfect 5-way manner-group classifier** + position prior: **800/809 = 0.989**
  (current 0.907).  The group classifier itself is only 0.603, and neither richer sensors nor DTW
  moved it — so this headroom is real but not reachable by feature engineering on these modalities.
* Emotion from the position prior alone, bijective: 668/809 = 0.826.
* Sequence: the total-order assumption is exact (104/104), so the ceiling is set by pairwise
  evidence quality, currently 0.850 held-out.

---

## 6. Honest accounting, and what Top 3 would still require

Validated so far: **+5 to +6 public**, landing at ~320-321 / 342 (from 315).  Qualification
needs 324.  The remaining gap is emotion, and this session established both where it is and
why it is hard:

* 46 of the 144 test emotion questions sit in two-clip blocks where accuracy is **0.7296**
  against 0.8966 in three-clip blocks.  The candidate manner set is right 100% of the time
  there; the loss is the missing-slot latent, whose learned prior reaches only 0.660
  subject-disjoint (0.761 on its confident 68%).  Mechanism U targets exactly this.
* The remaining three-clip emotion error is bounded by the pairwise manner-orientation model.
  A **perfect** 5-way manner-group classifier would give emotion 800/809 = 0.989, but that
  classifier sits at 0.603 and did not move for a 269-column radar micro-Doppler + 5-IMU
  spectral expansion (0.5958 → 0.6007) nor for DTW warping-path features (+0.9 pp on the
  pairwise task).  This is the one place where a genuinely new perceptual signal — not more
  statistics over the same three modalities — would be needed.

The next highest-value experiments, in order:
1. finish the orphan-injected validation of **T** (`champ/eval_orphan.py` with and without
   `CHAMP_REPAIR=1`); 4 test blocks and ~47 test questions are currently solved under a
   session hypothesis that two independent signals reject;
2. finish the **U** sweep (`champ/sweep_emo3.py`, `champ/log_slot_w1.txt`) and, if the slot
   prior helps, re-tune the two-clip fusion weights separately from the three-clip ones —
   they have never been swept in that regime;
3. only then consider new perception for manner.


---

## 7. Targeted search for a fifth mechanism (>=5-10 test overrides at >=80% OOF flip precision)

Searched the five areas requested, excluding radar/IMU/DTW emotion features.  **No candidate
met the bar.**  All ten are recorded here so the ground is not re-covered.

| candidate | area | OOF flips | precision | test overrides | verdict |
|---|---|---|---|---|---|
| assault config, un-gated (mechanisms H+E) | model disagreement | 31 | **0.786** | 3 new | closest to the bar, but not independent — same H/E audit, and only 3 rows are not already covered by W |
| DINOv2-depth-only on the no-LMT clips | metadata | 34 | 0.758 | ~2 | below bar; and the regime has no OOF analogue (see below) |
| nested-child order as a hard gated constraint | cross-question | 21 | 0.611 | – | the pairwise facts are **0.981** accurate, but forcing one pair reshuffles the rest of the permutation |
| Z: set-conditioned manner ordering | structural | 41 | 0.400 | – | real structure (modal ordering covers 0.957 of a set's sessions) but the champion already captures it — see below |
| Z restricted to two-clip blocks | structural | 1 | 1.000 | ~0 | the champion almost never violates the canonical order |
| HARn action must lie in its parent's pool | cross-question | 3 | – | ~1 | theorem holds 387/389, but the champion violates it only 14 times and only 3 are actionable |
| pool inconsistency of a block's answers | structural | – | – | **0** | strong OOF detector (0.759 vs 0.944) but **zero** inconsistent blocks on the real test |
| napp-threshold rule over pool decisions | structural | – | <=0.48 | – | the pool side is saturated: 61 option-level errors in 10864 (0.56%) |
| manifest modality patterns | metadata | – | – | – | the non-(IMU,Radar,Skeleton) patterns are scattered missing data across 14 actions, not a biconditional |
| historical models v6 / v7 / v8 / solver / unified | model disagreement | 1063-1603 | 0.07-0.13 | – | dead; they win only 12-13% of their disagreements with the champion |

### Why Z looked like a breakthrough and was not
The question generator does fix, per manner set, which manner occupies which protocol slot:
sets seen in >=2 training sessions show **1.12 distinct orderings out of 6** and a modal
ordering covering **0.957** of their sessions.  Leave-one-user-out, *prior against prior*, the
set-conditioned ordering scores **0.909** triple-exact against the independent slot prior's
0.747, and 39 of 59 test blocks (107 emotion questions) have their set in training.

That comparison is the trap.  The champion is not the independent prior — it also has the
physical group classifier and the pairwise discriminator, and it scores **0.948** on exactly
those covered questions.  Against the champion, Z is **net -8 at 0.400 flip precision**, and
its support breakdown shows why: it only helps at support >=4 (2 flips, 2-0) and actively
hurts at support 2 (13 flips, 0-12), where the modal ordering is decided by a 2-vote.

### The no-LMT cluster, and a protocol gap worth recording
13 test HARn clips (7, 23, 27, 46, 50-56, 59, 62) have **no LMT directory at all** — no
skeleton, IMU or radar was ever recorded for them.  They carry 15 questions (10 `single`,
5 `object`), and the champion answers all 15 from a v8 fallback (`harn_no_evidence`).  They
do have Depth / Depth_Color / IR video on disk, but they appear in **none** of the cached
feature sets (`dino_frames`, `depth_frames`, `harn_dino`, `harn_clf`), because every cache was
built by iterating `champ/meta.csv`, which is keyed on skeleton availability.

The modality biconditional (no wearables <=> action in {40 Stand on one leg, 41 Peel fruits
with a knife, 42 Throw away, 43 Open the cabinet}) resolves 5 of the 10 `single` questions —
exactly one option is a no-wearable action and the champion already picks it.  For the other
5 **no option is one of the four**, so the biconditional does not transfer to the test set.
Substituting a DINOv2-depth-only classifier for the v8 fallback measures 0.758 flip precision
at the median-margin gate (34 OOF flips, 25-8) — below bar, on ~5 actionable test questions.

Protocol gap: `champ/oof_final.csv` contains only **3** fallback rows, all sequence, so the
pseudo-test never exercises the regime that produces 15 real-test fallbacks.  Any future work
here must ablate wearables on training clips to build an honest baseline.
