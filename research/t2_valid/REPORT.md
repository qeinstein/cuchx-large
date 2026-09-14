# T2 validation: faithful OOF baseline reconstruction of the 332 champion

Date: 2026-09-14. Scope: `research/t2_valid/` only (read-only elsewhere).

## 0. TL;DR

- `submission_final.csv` is byte-identical to `submission_090643_regen.csv` (0/682).
  The "55 hand patches" are the full accumulated lineage 090643 -> 332, not late edits.
- Classification: **46 class-(a)** generalizable, **9 class-(b)** one-offs with
  mechanistic evidence, **0 class-(c)** algebra-only. Full table: `provenance.csv`.
- Implemented CPU-only (no dense logits): joint sequence decoder, W rerun,
  exact-template reruns, CPU emotion OOF + test run, struct-pool OOF + test run,
  precedence-ordered patch-applier (`apply_patches.py`).
- Reproduction from pipeline base (champion used ONLY for scoring):
  sequence 14/21 at 100% flip precision (16/16 under 2/3 hash seeds);
  W+templates 11 rows at 100%; patch-applier total 28/28 correct (recall 28/55);
  emotion test-view 7/17 with 0 spurious (3 PRIORFIX + 4 E-confounded).
- CPU OOF (grouped 5-fold, same protocol as baseline): emotion 758/809,
  bit-identical to baseline (net +0, dense contributes nothing); struct pool on
  HAU rows net -20 (multi -15, comb -4, single -1; clean-exact subset net -3,
  all multi completeness).
- Top validation gap (§5): dense-logit cache missing -> S/H/E/T-letter mechanisms
  (27 class-(a) rows: H3 E5 S15 T4) blocked on data, not code.

## 1. Lineage (verified, not trusted)

Pairwise diffs between consecutive scored submissions (recomputed locally; scores
from `README.md` §3, `state.md`, `research/submission_results_20260905.json`,
`research/submission_results_20260907.json`):

| step | diffs | score |
|---|---|---|
| 090643_regen -> 092105_SUB | 10 | 310 -> 315 (0.90643 -> 0.92105) |
| 092105_SUB -> 093859_SUB | 30 | 315 -> 321 (0.92105 -> 0.93859) |
| 093859_SUB -> 094736_mixed14 | 14 | 321 -> 324 (0.93859 -> 0.94736) |
| mixed14 -> 326max | 6 | 324 -> 326 |
| 326max -> 327cons | 2 | 326 -> 327 (reverts the 2 maximal-only HARn rows) |
| 327 -> 328CH | 3 | 327 -> 328 (0206, 0335, 0647) |
| 328 -> 329CH | 5 | 328 -> 329 (reverts 0206+0647, adds 0452/0459/0461) |
| 329 -> 330CH | 1 | 329 -> 330 (0443) |
| 330 -> 332CH | 9 | 330 -> 332 (0206/0426/0444/0477/0488/0501/0506/0519/0647) |
| 332CH -> submission_final | 55 | final == 090643_regen (0 diffs) |

Composition of the 55 by introducing step: 092105: 8 (H: 0471/0472/0493;
E: 0418/0436/0464/0470/0679). 093859: 24 (S: 15 seq; T: 0137/0289/0431/0523;
W: 0480/0525/0528/0533/0534). mixed14: 9 (SEQTMPL: 0330/0334/0342/0353;
EMOTMPL: 0386/0387; PRIORFIX: 0427/0429/0458). 326max: 3 (0385/0524/0530).
328: 3 (0335/0647/0206). 329: 2 (0459/0461). 330: 1 (0443). 332: 5
(0426/0444/0477/0501/0506). Total 8+24+9+3+3+2+1+5 = 55.

Notable non-55 transitions (context): 093859 flipped 0397 D->C and 0452 A->C
(both later reverted as wrong: 0397 in 326max, 0452 in 329); mixed14 flipped
0488/0519 C->A (restored in 332; the 0488 restoration is a PROVEN public win,
revert probe -1) and reverted 0443 B->D + 0447 D->A (0443 re-flipped D->B by the
330 singleton, PROVEN win; 0447 stayed A, i.e. 092105's E flip there was wrong).

## 2. Provenance + classification

Full per-row table with evidence references: `provenance.csv` in this folder.
Mechanism codes: H = HARn DINO late fusion (`CHAMP_W_HDINO`, OOF prec 0.778);
E = emotion DINO feats (`CHAMP_EMO_DINO`, OOF prec 0.800); S = joint total-order
decode (OOF prec 0.853, needs dense logits); T = block-conformance repair (OOF
prec 0.821); W = object<->single consistency (OOF prec 1.000); PRIORFIX =
`fit_manner` indentation-bug repair (OOF +16: 18 W->R / 2 R->W); SEQTMPL =
exact-pool sequence transfer (nsrc==1, closure coverage 6/6); EMOTMPL/USERTMPL/
OBJTMPL = exact donor/cohort/visible templates; CONSIST = cross-question
mutual-inconsistency repair (testb006/testb021); MULTIMODAL = 4-modal Stirring
evidence (skel/imu 0.9993, dino 0.9983, thermal 33.6x, pool-gap 2.95);
COHORT2/3 = trial-withholding analyses; PHYS = fast-dynamics physics (84f,
speed 0.0523); VISUAL = takeover frame inspection.

Class counts: **(a) 46** [19 CPU-rerunnable + 27 cache-blocked], **(b) 9**
[0426/0444 weakest (b2): cohort hypothesis + zero algebra], **(c) 0**.
The empty (c) is a finding: every patch carries mechanistic evidence
(OOF-validated mechanism, template/donor match, physics, pixels, generator
model). Post-hoc leaderboard probes refined the PACK (reverts, singletons,
proven wins on 0330/0358/0643/0488/0477) but no surviving row rests on
algebra alone.

Per-row (pipe->champ, intro step, mechanism, class):

- multi: 0137 ACD->AC 093859 T a; 0206 D->CD 328 MULTIMODAL b.
- combination: 0289 B->A 093859 T a.
- sequence (21): 0330 m14 SEQTMPL a; 0331/0336/0338/0341/0346/0347/0348/0350/
  0351/0352/0354/0355/0359/0643/0644 093859 S a-blocked; 0334/0353 m14 SEQTMPL a
  (corrected S interims); 0342 m14 SEQTMPL a; 0335/0647 328 CONSIST a.
- emotion (17): 0385 326max USERTMPL a; 0386/0387 m14 EMOTMPL a; 0418/0436/0464/
  0470/0679 092105 E a-blocked (4/5 fixed-manner-recoverable, §4.4);
  0427/0429/0458 m14 PRIORFIX a; 0431 093859 T a;
  0443 330 COHORT2+PHYS b (proven); 0459/0461 329 COHORT3 b; 0426/0444 332
  COHORT2 b2.
- single/HARn (7): 0471/0472/0493 092105 H a-blocked; 0480 093859 W:X a;
  0477/0501/0506 332 VISUAL b (0477 proven win; 0501 neutral-negative e=0;
  0506 neutral).
- object (7): 0523 093859 T a; 0524/0530 326max OBJTMPL a; 0525 093859 W:X a;
  0528/0533/0534 093859 W:W1 a.

## 3. What was implemented (all in `research/t2_valid/`)

All mechanisms run CPU-only with test-visible inputs + already-fitted artifacts.
No dense logits, no harn/dino caches, no video reads. Determinism requires
`PYTHONHASHSEED=0` (the struct pool solver's exact-match decisions are
hash-order-sensitive near ties; see §4.1).

| file | mechanism | input -> output |
|---|---|---|
| `mech_seq_joint.py` | SEQTMPL (faithful port of `choose_closure` gate) + anchored CONSIST joint decode (24^k search, min contradictions/tau/-agreement); struct pools via `CHAMP_POOL_FEATS=struct`; skeleton timing audit-only | base CSV -> repaired CSV + audit + skel JSON |
| `mech_w_rerun.py` | W wrapper around `objlab/object_rule.py` (unchanged upstream code) | base CSV -> overrides + audit |
| `mech_templates_rerun.py` | OBJTMPL (unanimous, object-only) + USERTMPL (signature-exact, emotion), reusing `research/exact_visible_template_audit_20260905.py` and `research/evaluate_user_template_pairs_20260904.py` (snapshot self-reconstructs from tracked atlas) | base CSV -> transfers + audits |
| `oof_emotion_cpu.py` | grouped 5-fold emotion OOF (`fit_manner`+`solve_emotion`+emopair, default weights, folds seed=7, pseudo seed=100+fi) + all-users real-test run; arms: pool_of=None vs struct pools | OOF CSV + test 17-row CSV |
| `oof_pool_struct.py` | grouped 5-fold struct-pool OOF (single/multi/combination) + all-users real-test run; greedy fallback counted on n>20 components | OOF CSV + test letters |
| `apply_patches.py` | precedence merge: SEQ/CONSIST -> W -> TEMPLATES -> PRIORFIX (gated to m14-recipe rows); blocked/hand rows listed as SKIPPED, never applied | base CSV -> patched CSV + audit |
| `provenance.csv` | 55-row provenance + classification table | reference |

Not implemented as appliers (documented with reasons): S/H/E (code exists,
data missing); T letters (repair runnable, dense pool re-solve missing);
class-(b) one-offs (no generalizable code by definition).

Skeleton timing verdict: profiled for 39/39 test sequence clips (motion-energy
centroid fraction) but correctly NOT used as a decision input — without
action-labeled segments, energy timing cannot order actions. The CONSIST
tie-break is exact-pool donor agreement, then historical margin. The mission's
"if useful" clause was tested and answered: not useful for order; the
structural (option-text/contradiction) signal carries the repairs.

## 4. Measured numbers (all protocol-labeled)

Convention: "reproduced X/Y" = flips emitted from the PIPELINE base by the
generalized mechanism that exactly match the champion, with no champion input
to the mechanism. Precision = matches / flips emitted.

### 4.1 Joint sequence decoder (`mech_seq_joint.py`)

- From `submission_final.csv` (pipeline base), `PYTHONHASHSEED=0`: SEQTMPL fires
  9 (0330/0334/0336/0338/0341/0342/0348/0353/0644), anchored CONSIST fires 5
  (0331/0335/0352/0643/0647). **14 flips, 14 match: precision 1.000, recall
  14/21.** Missed: 0346/0347/0350/0351/0354/0355/0359 (all need S/dense).
  Artifacts: `seqjoint_from_submission_final.csv`/`.audit.csv`/`.skel.json`.
- Hash-seed sensitivity (struct-pool exact-match near-tie on block (193,194)):
  seeds 1,2 also fire SEQTMPL on 0350/0351 -> **16/16 match, missed 5**
  (0346/0347/0354/0355/0359). Stable core across seeds {0,1,2}: 14 rows at
  1.000 precision, 0 wrong flips in any run (46/46 flips correct pooled:
  14 + 16 + 16).
- 327-base control: SEQTMPL 0 flips (correct negative control: all template
  rows already set), CONSIST exactly {0335 DBAC->DBCA, 0647 DCAB->DCBA}, both
  match. This is the historical testb006/testb021 repair, reproduced.
- Ablation that shaped v2: unanchored greedy CONSIST (v1) flipped 0334/0350/
  0354/0649 wrongly (overshoot walks through the right answer); all 4 wrong
  repairs were unanchored, all correct ones template-anchored -> anchors are
  REQUIRED (block abstains without one). Matches history: 0335/0647 aligned
  to template-backed siblings 0334/0342.
- Note: from the pipeline base, template+consistency re-derive 8 rows whose
  historical mechanism was S (0331/0336/0338/0341/0348/0352/0643/0644) — the
  mechanisms overlap; S proper (pairwise dense scores) remains blocked.

### 4.2 W rerun (`mech_w_rerun.py`)

From pipeline base: 6 overrides, 5 match champion (0480/0525/0528/0533/0534);
0530 ->B is the historical W2 interim (champion D comes from OBJTMPL, applied
later by precedence). 0 wrong flips. Artifacts: `w_from_submission_final.csv`.

### 4.3 Exact templates (`mech_templates_rerun.py`)

From pipeline base: OBJTMPL fires 6 unanimous object disagreements, ALL match
(0523/0524/0525/0528/0530/0534 — incl. independent agreement with T on 0523 and
W on 0525/0528/0534); USERTMPL fires 3 signature-exact emotion transfers, ALL
match (0385 D->A direct, 0386 A->B, 0387 A->C — the EMOTMPL rows reproduce under
the same whole-cohort machinery). **9/9 match, 0 wrong flips.** 0397 correctly
no-fires (base already D). Artifacts: `templates_from_submission_final*.csv`.

### 4.4 CPU emotion test-view (`oof_emotion_cpu.py --only-test`, all-users fit)

nopool arm, 17 champion-diff emotion rows: **7/17 match, 0 spurious flips**
(the other 10 output pipeline values, i.e. abstain-by-equality):

| row | pipe | champ | cpu | verdict |
|---|---|---|---|---|
| 0427/0429/0458 | D/A/A | B/C/B | B/C/B | PRIORFIX x3 REPRODUCED |
| 0418/0436/0470/0679 | D/A/D/A | B/C/B/B | B/C/B/B | E-rows reproduced WITHOUT dino |
| 0464 | A | D | A | E-specific: not reproduced (needs dino) |
| 0426/0443 | A/D | B/B | C/D | m14 interims reproduced, not 330/332 values (correct: those came from probes) |
| 0385/0386/0387/0431/0444/0459/0461 | pipe | champ | pipe | no spurious flip (template/T/cohort rows untouched) |

The 4/5 E-row reproduction without dino feats confounds the historical E
attribution: for 0418/0436/0470/0679 the fixed-manner solver agrees with the
dino-flavored one, so dino evidence is corroborating, not necessary. Only 0464
is genuinely dino-dependent among the 5. (Historical 092105 ran buggy-manner +
dino; the decomposition is: fixed manner alone recovers 4/5.)
struct-pool arm on the 17 test rows: not measured (the combined fit was OOM-killed
3x under box pressure); the 5-fold OOF shows pool context flips 0/809 decisions
(§4.5, both arms bit-identical), so nopool ≈ structpool on test is expected.

### 4.5 Emotion OOF (grouped 5-fold, protocol = oof_driver.py)

Baseline (`oof_pipeline_base.csv` emotion): 758/809 = 0.9370.
CPU rerun (`oof_emotion_cpu.py`, same folds/seeds/weights, `fit_manner` +
`solve_emotion` + emopair, no dense anywhere): **nopool arm 758/809 = 0.9370,
structpool arm 758/809 = 0.9370, per-row agreement with the dense baseline
809/809 on both arms (W->R 0, R->W 0, net +0).** Artifacts:
`oof_emotion_cpu.csv` + `oof_emotion_cpu.fold{0..4}.csv`.
Interpretation: dense evidence contributes EXACTLY ZERO decisions to the
emotion OOF — the 758/809 baseline is bit-for-bit CPU-reproducible, and pool
context (none vs struct) flips nothing either. All 17 emotion champion-diffs
therefore come from test-side mechanisms (templates, probes, cohorts, T),
not from dense. The OOF baseline is already faithful for emotion.

### 4.6 Struct-pool OOF + test letters (`oof_pool_struct.py`)

Coverage: HAU pool-solved rows only (2408 = 790 combination + 809 multi + 809
HAU-single; HARn single/object use the blocked action-clf path).
Struct-only (`CHAMP_POOL_FEATS=struct`, empty dense statcache) vs the dense
baseline per-row: **combination 782/790 (net -4: 0 W->R / 4 R->W), multi
767/809 (net -15: 4/19), HAU-single 806/809 (net -1: 0/1); total net -20.**
Artifacts: `oof_pool_struct.csv` + `oof_pool_struct.fold{0..4}.csv`.
Decomposition (via `diag_pool_blocks.py` + divergence mapping): exact
enumeration was skipped (greedy fallback) on oversized (n>20) components in
83/272 OOF blocks (19/19/18/11/16 per fold). Of the 29 struct-vs-dense
pred-divergences, 24 sit in greedy blocks (fallback-confounded) and only 5 in
exact blocks — **all 5 multi, net -3 (1 struct win, 4 losses), all
completeness errors (missing/extra letter)**. Fold 0 (19 greedy blocks, 0
divergences) suggests the fallback is near-exact in practice, so most of the
-20 is likely genuine dense evidence, concentrated where expected: multi
completeness in thin-evidence blocks (the known pair-block under-prediction
asymmetry, Session 5).
Test letters (all-users fit): struct pools on DP blocks give 0137=ACD,
0206=D, 0289=B (all = pipeline, not champion); struct pools on REPAIRED
blocks give 0137=ABC (a third, also-wrong answer), 0206=D, 0289=B. Verdict:
**T letters genuinely need the dense pool re-solve** (repair alone moves 0137
but not to the champion answer); 0206 needs dense/multimodal as documented.

### 4.7 Patch-applier totals (`apply_patches.py --score`, T2-validation-only)

From `submission_final.csv`: SEQ/CONSIST 14 + W 6 + TEMPLATES 6 (net of overlap:
0525/0528/0534 agree, 0530 B->D overrides W2) + PRIORFIX 3 = **28 unique flips,
28/28 match champion (precision 1.000, recall 28/55)**. The 27 un-emitted rows
are exactly the blocked/hand set: H3 E5 S-only7 T-letters3 b9 (0137/0289/0431
+ 0206/0459/0461/0443/0426/0444/0477/0501/0506). 0 wrong flips. Artifacts:
`patched_from_final.csv`/`.audit.csv`. (Two applier bugs found and fixed during
validation: full-CSV merge reverting earlier stages; NaN-truthiness in the
PRIORFIX gate — now merges audited flips only.)

## 5. The single most important validation gap

**The dense-logit cache (`champ/dense_logits.npz`) is missing, and 27 of the 46
class-(a) rows cannot be re-derived or OOF-audited without it.** This is a data
gap, not a code gap: `seqlab/apply_test.py` (S), `champ/harn.py` + DINO fusion
(H), `DINO_PHYS` in `champ/core.py` (E), and the dense pool re-solve behind the
T letters all exist and are understood, but their inputs
(`dense_logits.npz`, `harn_clf.npz`, `harn_dino.npz`, dino `feats.csv` columns)
are absent, and the GPU job to regenerate them is pending (other agent).

Consequences, measured where possible:

1. S (15 seq rows): the historical OOF flip precision 0.853 cannot be
   re-audited; 5 rows (0346/0347/0354/0355/0359, +0350/0351 under seed 0) are
   unreachable by any CPU mechanism and would silently stay wrong in a
   dense-free "reconstruction". Template+consistency overlap recovers 8
   S-rows, but that is luck of overlap, not S.
2. H (0471/0472/0493) and E-proper (0464): no CPU signal reproduces them; the
   CPU emotion run abstains (outputs pipeline values). 4/5 E-rows turned out
   to be fixed-manner-recoverable (§4.4), so the true dino-dependent set is
   smaller than history suggests — but 0464 + the 3 H rows still need the
   caches.
3. T letters (0137/0289/0431; 0523 covered independently by OBJTMPL): the
   block repair itself is CPU-rerunnable (`champ/repair.py`), but the letters
   came from a dense pool re-solve. Measured (§4.6): struct pools give
   pipeline letters on DP blocks and move 0137 only to a third wrong answer
   (ABC) on repaired blocks — T stays blocked on dense.
4. The published OOF baseline (`oof_pipeline_base.csv`, 3771/4087) bakes in
   dense evidence the validators cannot regenerate — so "OOF delta vs
   baseline" for any new dense-adjacent model is currently a comparison
   against an unreproducible artifact on the sequence/HARn slices. The CPU
   OOFs in §4.5-4.6 bound the dense-free part only.

Second-order gaps (recorded, not blocking): `test_pool_snapshot.pkl` is gone
(EMOTMPL/SEQTMPL inputs reconstructed via tracked atlas + struct pools);
`seqlab/test_sequence_audit.csv`, `objlab/test_object_overrides.csv`,
`champ/diff_T_repair_cf.csv`, `research/submission_priorfix_base.csv` are gone
(mechanisms re-derived from surviving code + `submission_corrlayer_S_W_T_audit.csv`,
which pins all 30 S/T/W flips with per-row margins).

## 6. Per-category: pipeline-OOF accuracy vs champion-test behavior

| category | pipeline OOF (dense) | CPU OOF (this work) | champion-test behavior on the 55 |
|---|---|---|---|
| single (HAU) | 1216/1238 = 0.9822 (HAU subset 0.9975) | 806/809 = 0.9963 (net -1) | 7 diffs: H3 blocked, W:X 1 repro, VISUAL 3 hand (1 proven win, 2 neutral) |
| multi | 782/809 = 0.9666 | 767/809 = 0.9481 (net -15; clean-exact subset net -3) | 2 diffs: T 1 blocked-letter, MULTIMODAL 1 hand (Δ0 probe) |
| combination | 786/790 = 0.9949 | 782/790 = 0.9899 (net -4) | 1 diff: T blocked-letter |
| emotion | 758/809 = 0.9370 | 758/809 = 0.9370 (agree 809/809, net +0); test-view 7/17 repro, 0 spurious | 17 diffs: 3 PRIORFIX repro, 4 E confounded-repro, 1 E blocked, 3 template repro, 6 probe/cohort/T hand-or-blocked |
| sequence | 116/308 = 0.3766 | n/a (order needs dense; joint decoder is test-side) | 21 diffs: 14-16 re-derived @1.000, 5-7 need S |
| object (+HARn single) | 113/133 = 0.8496 | n/a (needs action clf; W/templates are test-side) | 7 diffs: 7/7 repro (W1/X + OBJTMPL; 0523 is template-cover of a T row) |

Note the OOF-vs-test asymmetry the mission suspected: the OOF baseline's
sequence accuracy (37.7%) is far below the champion's test behavior because S
(a test-side joint decoder over dense scores) is not in the OOF pipeline path
at all — the OOF uses per-question centroid order. Same for W/templates on
object/single. A "faithful OOF baseline of the real 332 champion" must apply
S/W/T/templates INSIDE the OOF loop (pseudo-test blocks), not just the raw
pipeline. The CPU mechanisms here (§3) are OOF-applicable except where
dense-blocked; wiring them into `oof_driver.py` is the recommended next step
once dense logits return.

## 7. Reproduction

```bash
cd /home/fluxx/Workspace/cuchx-large
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 research/t2_valid/mech_seq_joint.py --base submission_final.csv
python3 research/t2_valid/mech_seq_joint.py --base submission_095614_327of342_CHAMPION.csv
python3 research/t2_valid/mech_w_rerun.py --base submission_final.csv
python3 research/t2_valid/mech_templates_rerun.py --base submission_final.csv
python3 research/t2_valid/oof_emotion_cpu.py --only-test --no-struct   # lean; struct arm needs headroom
python3 research/t2_valid/oof_emotion_cpu.py --folds 0 --skip-test    # per-fold (repeat 1..4)
python3 research/t2_valid/oof_pool_struct.py --folds 0 --skip-test      # per-fold (repeat 1..4)
python3 research/t2_valid/oof_pool_struct.py --only-test                # T-letter check (lottery-prone)
python3 research/t2_valid/apply_patches.py --base submission_final.csv \
  --out research/t2_valid/patched_from_final.csv --score   # --score is T2-validation-only
```

Environment notes: box is shared (load 13-19 during this work; sibling GPU/CPU
jobs); full 5-fold OOF in one process was OOM-killed twice (exit 137), hence
per-fold checkpoint CSVs + `--only-test`/`--no-struct` lean modes. All
test-side mechanisms are light (<2 min each). `PYTHONHASHSEED=0` is required
for exact reproduction (struct-pool near-tie, §4.1).
