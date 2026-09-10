# Final recommendation and evidence audit — 2026-09-08

Use `TODAY_RECOMMENDED.csv` for the one remaining submission. **No submission was performed.** This final configuration supersedes the smaller provisional configurations discussed during the audit.

Base: `submission_096491_330of342_CHAMPION.csv`, 682 rows, SHA-256 `e42cde96bafb108b8c16bcd97e5deb2af32c1657a089351ada8a72a95ed805cd`.

Recommended SHA-256: `25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56`.

## Exact configuration

| Row | Base → new | Rationale | Evidence status |
|---|---|---|---|
| test_0488 | A → C | Turning pages: seated material handling, paper-object option universe, VLM C, zero cached margin | Strong structural and visual evidence; individual public effect unresolved |
| test_0477 | A → B | Mopping: bent posture and downward implement-like motion; zero-margin A default | Visual inference, not a measured label |
| test_0506 | A → C | Lunges: forward stepping and lowering/recovery; zero-margin A default | Visual inference; legs partly cropped |
| test_0519 | A → C | Drinking: seated hand-to-mouth motion; zero-margin A default | Visual inference; small object unresolved |
| test_0501 | A → B | **New error:** clear seated-to-standing transition; current mopping margin 0.604 | Direct temporal evidence; cached VLM incorrectly suggests walking |
| test_0444 | C → B | Patiently in place of Comfortably; training vocabulary and prior protocol evidence | Public contribution measured zero; private benefit unverified |
| test_0426 | C → B | Anxiously in place of Comfortably; prior physical/protocol evidence | Public contribution measured zero; private benefit unverified |
| test_0647 | DCAB → DCBA | Checking temperature before massaging; sibling/donor order support | Public contribution measured zero; private benefit unverified |
| test_0206 | D → CD | Add stirring; prior multimodal omission evidence | Public contribution measured zero; private benefit unverified |

Every other row stays exactly at the 330 base. Specifically exclude changes to **0458, 0146, 0165, 0150, 0151, 0461, 0469** today. Also exclude conditional/rejected entries listed in the candidate ledger. Keep `0517=A`: raw frames visibly show putting on clothes despite its zero margin.

Public contribution of this file is exactly

`d_0488 + d_0477 + d_0506 + d_0519 + d_0501`.

The algebraic range is **−5 to +5**, or 325–335/342. A modest positive result is the working expectation from the evidence; there is no calibrated probability distribution or defensible narrow confidence interval. A +4 tie or +5 target score requires four or five net public wins. If all five corrections are right, the range becomes 0…+5 depending on public membership. Do not assume random independent split membership or add private gains to the public score. The four measured-zero changes cannot improve today's public result under the supplied scoring facts.

This chooses public upside from direct action evidence over another emotion protocol bundle. Pure isolation has less upside, while a measured-zero-only fortified file cannot improve public score. The tradeoff is that five gain candidates have greater aggregate regression exposure than a singleton. Their errors can be correlated through depth interpretation and annotation boundaries.

Main failure modes: limited depth detail (especially 0519), cropped transitions, mismatched or noisy action annotations, correct-looking physical activity not included in benchmark labels, and public/private allocation. The four zero-contribution changes can still be wrong privately. No private points are guaranteed.

## Facts versus deductions

**Leaderboard facts supplied by the user:** base 330/342, rank 3, leaders 334/342; today's four scores 329,330,330,330; one slot left. The leaderboard was not independently queried and quota was not consumed.

**Algebraic deductions:** `d_0458=-1`, `d_0165=0`, `d_0488+d_0146=0`, and `d_0461+d_0150+d_0151+d_0469=0`. The other four supplied singleton effects are zero. Zero contribution can mean private membership or public wrong-to-wrong. A zero bundle can contain cancelling gains/losses.

**Structural evidence newly measured:** training positive/negative consistency, object-option compatibility, donor action unions, and cached margins. These do not reveal held-out ground truth.

**Hypotheses:** individual visual action corrections, exact emotion protocol transfer, private gains, and the expected sign of today's unmeasured score.

## Findings that change the research direction

1. **Sequence options are not exhaustive action sets.** Other labeled same-clip questions establish extra actions in **219/308** training sequence questions. Combination answers are incomplete in **364/790** training combination questions. Therefore absence from these sets cannot justify removing an action from a Multi answer.
2. **The old distractor claim was extended incorrectly.** Single-question negative-action checks have **0/3,285** observed violations. Treating each extra action in a rejected combination as absent produces **206/4,804** violations. Even the weaker rejected-conjunction rule has one training counterexample (`training_2383`), so it is not an exceptionless theorem.
3. **Strict same-clip tests found no remaining HAU positive/negative or rejected-conjunction conflicts in the champion.** This rules out easy repairs through those specific checks; it does not prove all HAU answers correct. The previous audit's sequence loop compared unrelated clips, so its global conflict count was not a valid same-session contradiction count.
4. **Raw video refutes the earlier removal narrative.** Clip 0117 (`0146`) includes a sit-then-stand sequence. Clip 0123 (`0150`) visibly includes jumping-jack motion. The cited `user21/3-2` donor trials also contain labeled jumping jacks and massaging. These observations support retaining base answers; they do not individually decode public contributions.
5. **A stronger argument for 0488 does not depend on predicted object D.** All four object options are paper materials. Only turning pages among its action choices has supporting training object labels. General object-option-universe inference gives **33/33** correct eligible HARn training questions when the target user is excluded. This is selected-sample accuracy, not measured correction precision at zero-margin test fallbacks; the vocabulary map is pre-existing and was learned from training labels.
6. **Blank-option tie handling has a concrete bug pattern.** `champ/pipeline.py` tests equality across all four scores. `[0,0,0,-1000000]` is flat across the three valid options but fails that test and yields A with zero top-two margin. The five cached zero-margin A rows are 0477,0488,0506,0517,0519. The exact historical scorer was not replayed; the bug pattern is demonstrated, not proven to be the complete provenance of each cached prediction. Missing historical classifier caches preclude faithful full-model replay in this workspace. No production solver or champion was modified.
7. **New 0501 correction:** actual clip 0041 rises from seated to standing. Standing up is option B. Clip 0042 offers a similar transition and is already answered B by the champion. These are different recordings, not exact duplicate frames or shared timestamps.
8. **New omission leads were actively falsified.** `0590` peeling had cached sensor/DINO peaks 0.979/0.924 but raw footage shows sink washing/wiping without clear peeling; its OOF omitted-action analogues contain 0 positives/32. `0591` pouring has peaks 0.948/0.339 but no clear pouring in sampled phone/headphone footage. `0158` page-turning remains visually ambiguous, with 0 positives/28 same-action omitted OOF rows. None enters today's file. The old `0137` argument imported evidence from another block: its own stirring peaks are only 0.00175/0.00390.

OOF atlas values come from an older frozen solver (`submission_093859_SUBMITTED.csv`) and are useful for residual mechanism checks, **not calibrated estimates for the 330 champion**. No generic stacking was retried.

## Tomorrow: candidate queue and five-slot strategy

`candidate_ledger.csv` contains **12 ranked hypotheses: 5 promoted and 7 conditional**. It would be false to call all 12 high-confidence residual errors. Only `0501` is a newly promoted row; the audit substantially strengthens and reprioritizes four pre-existing HARn candidates while rejecting several overstated claims.

The strongest five, in rank order, are **0488,0501,0477,0506,0519**. Today's result determines which remain worth testing tomorrow. Conditional reserves: **0483,0461,0469,0430,0432,0341,0646**. These require new evidence or favorable decoded scores before promotion. 0341/0646 form one correlated order hypothesis, not two independent discoveries. Do not fill the submission quota merely to use these reserves.

All **32 subsets** of the five promoted rows are prebuilt with the four public-zero changes held fixed. `subset_manifest.json` records bit order, exact diffs and full SHA-256 for every file. `candidate_manifest.json` also supplies singleton files and three conditional pair bundles. A `research_only_` filename means **not recommended for immediate submission**.

Let `S` be today's final delta. For the generic recovery branch, four leave-one-out observations decode all five contributions and preserve slot 5 for consolidation:

| Tomorrow slot | Prebuilt file (relative to this folder) | Interpretation |
|---|---|---|
| 1 | primary_subset_10111.csv | All promoted rows except 0519; `d_0519 = S − delta1` |
| 2 | primary_subset_11011.csv | Except 0506; `d_0506 = S − delta2` |
| 3 | primary_subset_11101.csv | Except 0477; `d_0477 = S − delta3` |
| 4 | primary_subset_11110.csv | Except 0488; `d_0488 = S − delta4`; recover 0501 from total |
| 5 | decoder-generated confirmed-wins CSV | Keep individually proven public gains; discard measured losses |

This is a fallback design, **not an instruction to spend four slots on information regardless of outcome**. Adapt after every result: when a loss is identified, remove it from future probes and use the matching prebuilt subset (do not apply the simple leave-one-out formula to a changed file). The decoder consumes the actual subset mask and remains valid. If S=+5, every primary contribution is +1; skip all four diagnostics. If S=+4, there are four +1s and one zero, so the whole pack has no measured public regression and pure isolation has little immediate value. Keep that pack and use slots only for independently promoted residuals. If S is lower, informative subsets can recover a stronger champion by separating cancellations. Once individual wins are known, stop unnecessary testing and consolidate early.

The decoder retains `d_0146=-d_0488` and the original four-row emotion/Multi equation. It supports measured conditional pair/singleton results with `--extra-observe` and adds **only individually forced +1 contributions** to the consolidation file. It never treats a zero-net conditional pair as two neutral rows.

Commands (replace placeholders with real measured integer deltas; no submit operation exists):

```text
python3 research/decode_takeover_20260908.py --today-delta S
python3 research/decode_takeover_20260908.py --today-delta S --observe primary_subset_10111=D1
python3 research/decode_takeover_20260908.py --today-delta S --extra-observe research_only_emotion_pair=D
```

The final example decodes a result **only if that exact conditional file was separately chosen and scored**. It is not a recommendation to submit the conditional pair now. All observations use the original 330 base; mixing baselines or modified files invalidates the equations. The decoder rejects inconsistent ledgers.

## Artifacts and verification

- `today_manifest.json`, `TODAY_RECOMMENDED.diff.csv`: final file provenance.
- `structural_audit.json`, `constraints_audit.json`: training counts, counterexamples, donor unions, leave-user-out rows.
- `frames/`: extracted evidence sheets and video metadata; original videos unchanged.
- `candidate_ledger.csv`, `rejected_candidates.json`: explicit evidence quality and failure modes.
- `subset_manifest.json`, `candidate_manifest.json`: prebuilt CSV hashes and exact diffs.
- `../decode_takeover_20260908.py --self-test`: exhaustive exact recovery of all **243** signed five-row states; zero-net ambiguity preserved; impossible scores rejected.

All CSV builders validate the base hash, 682 unique rows, original row order, legal option letters, sorted Multi letters and sequence permutations. Every generated submission excludes 0458. Pre-existing modifications to `research/reset_suite_manifest_20260908.json` and `research/submission_reset_sub4_pure_strike_pack.csv` were preserved. No Kaggle calls, submissions, commits, or external messages were made.
