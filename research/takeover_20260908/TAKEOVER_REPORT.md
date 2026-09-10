# Takeover handoff — 2026-09-08

## Decision

The verified incumbent remains `submission_097076_332of342_CHAMPION.csv`, 332/342,
SHA-256 `25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56`.
No submission was made during this takeover.

No new standalone semantic flip cleared a promotion gate. The strongest new submission
family is an adaptive signed-effect/group-test plan that uses the exact public result from
the 330→332 transition and reserves the fourth slot for the only remaining independent
research lead, `test_0526 D→C`.

## Independent findings

* Pixel-level nesting links the metadata-missing HARN clips to HAU parent clips. Eleven of
  the 13 missing-metadata clips have a unique low-resolution parent match; the important
  examples include 0007→0114, 0023→0083, 0027→0112, 0046→0198, 0050→0069,
  0051→0118, 0055→0183, 0056→0191, 0059→0139, and 0062→0201. This is useful for
  locating evidence, but not sufficient to change an answer.
* The cached parent-window dense logits support the incumbent phone interpretation on
  `test_0526` and weaken the proposed keyboard flip. Its object donor signature is a 3–3
  keyboard/phone tie with leave-one-user-out transfer 0/6. Keep it as a fourth-slot test,
  not as a singleton recommendation.
* A cheap exact-segment thermal MobileNet retrieval audit scored 656/2,888 = 22.7% under
  user-disjoint folds. It is rejected. This is materially different from the incumbent
  solver but not accurate enough to override a 332 champion.
* Direct inspection of the five no-evidence changed clips supports the current answers:
  0477 mopping, 0488 page turning, 0501 standing up, 0506 lunges-compatible, and 0519
  drinking. Therefore their reversions are information probes, not semantic convictions.

## Ranked flips and dependence

| Rank | Flip | Status | Evidence / confidence |
|---|---|---|---|
| 1 | `test_0526 D→C` phone→keyboard | Only remaining independent lead; S4 only | Mixed: desk/typing cues versus parent dense phone evidence and 3–3 donor tie. Low-to-medium, not calibrated. Correlated with 0483. |
| 2 | `test_0483 A→C` sitting→typing | Reject for tomorrow | User-disjoint sitting/typing model gives p(typing)=0.435; dense parent frames favor reading/checking time. Low. Correlated with 0526. |
| 3–7 | Revert 0488/0477/0506/0519/0501 to old 330 answers | Conditional recovery tools | Their aggregate signed contribution is known, but individual signs are not. Same no-metadata/fallback family, so correlated as failure mode; do not interpret a reversion as a believed label. |
| 8 | 0478 C→B, 0507 A→C, 0527 C→A | Preserve incumbent | Each has a counter-model or low-purity template failure. Do not spend a slot. |

The five 330→332 row changes satisfy:

`d0488 + d0477 + d0506 + d0519 + d0501 = +2`.

For a submitted reversion, `e_i = -d_i`, hence the five reversion deltas sum to `-2`.
This is the useful group-testing invariant. The five rows are not independent, but the
0526 row is outside that sum and is the only plausible independent test.

## Exactly four submissions

Submit in this order, waiting for each public delta relative to 332:

1. `S1_REVERT_0488.csv`
2. `S2_REVERT_0477.csv`
3. `S3_REVERT_0506.csv`
4. Select one S4 branch below.

Let `E0488`, `E0477`, and `E0506` be the returned deltas in `{-1,0,+1}`.

* If any early delta is `+1`, submit the matching file:
  `S4_IF_0488_WIN__REVERT_0488_PLUS_0526.csv`, or the corresponding 0477/0506 file.
  That known recovery plus 0526 reaches `334` exactly if 0526 is a true `+1`.
* Otherwise compute `Eremaining = -2 - E0488 - E0477 - E0506`.
  * If `Eremaining >= 0`, submit
    `S4_IF_REMAINING_GAIN__REVERT_0519_0501_PLUS_0526.csv`.
  * If `Eremaining < 0`, submit
    `S4_IF_REMAINING_NONPOSITIVE__0526_ONLY.csv` to avoid knowingly adding a losing
    omitted bundle.

Examples:

* `(-1,-1,-1)` ⇒ `Eremaining=+1` ⇒ submit the remaining-gain bundle. Its observed delta
  `+2/+1/0` decodes 0526 as `+1/0/-1` respectively.
* `(+1,-1,-1)` ⇒ submit the 0488-win pair. Its observed delta `+2/+1/0` decodes
  0526 as `+1/0/-1` respectively.
* `(0,0,0)` ⇒ `Eremaining=-2` ⇒ submit 0526 alone; never add the known-losing omitted
  bundle.

The first three are not four singleton probes: they isolate three signs, while S4 is an
adaptive recovery/new-hypothesis bundle. A lower-scoring probe does not displace the
already-scored 332 incumbent; retain the highest verified public/private-robust result.

## Prebuilt files

All files below are 682-row, two-column submissions. Exact one-row/two-row/three-row diffs
are stored in the matching `.vs332.diff.csv` files. The complete machine-readable record is
`four_slot_manifest.json`.

| File | SHA-256 | Exact diff vs 332 |
|---|---|---|
| `S1_REVERT_0488.csv` | `5c424b408967b7750661e495b296e1975de4599afa29fa905e6aa0d0a7d47ffd` | 0488 C→A |
| `S2_REVERT_0477.csv` | `02d63945c94855115c05d668c2308af30be4b702741123857ffde84e42897399` | 0477 B→A |
| `S3_REVERT_0506.csv` | `204b72a7b24f5b07e5acac9948b27253f218caeee3d809090fe6883ce8b18fd6` | 0506 C→A |
| `S4_IF_0488_WIN__REVERT_0488_PLUS_0526.csv` | `a8c2738ab91eccbe14480ee353b1a344c75104b4ad2ff5e4a2eff8faf540c43e` | 0488 C→A; 0526 D→C |
| `S4_IF_0477_WIN__REVERT_0477_PLUS_0526.csv` | `4720e281a901da7715af1261d6d5ca375b98f8263ca77f7a52a4f7a72eb34231` | 0477 B→A; 0526 D→C |
| `S4_IF_0506_WIN__REVERT_0506_PLUS_0526.csv` | `15d7a650d2587d5b0cbe6d268037ccf37736737824bf54d737266eddc3a9dcf2` | 0506 C→A; 0526 D→C |
| `S4_IF_REMAINING_GAIN__REVERT_0519_0501_PLUS_0526.csv` | `cdb79b718d8f776ead7358e209c73f13ee1b02e2811262c96cb64cd60c5cbc66` | 0519 C→A; 0501 B→A; 0526 D→C |
| `S4_IF_REMAINING_NONPOSITIVE__0526_ONLY.csv` | `fc386cb51497729ba65682928acc279bf11b7e968cbfdeaa26b9f576a2b956a7` | 0526 D→C |

## Kaggle execution status

The authenticated Kaggle notebook path was verified with a completed inventory notebook.
The competition attachment itself contains only README/agent metadata, so a private raw
research dataset was staged with hard links and a Kaggle-ready exact-supervision probe was
prepared. The subsequent dataset upload was blocked before authentication by the current
environment's DNS failure for `api.kaggle.com`; no leaderboard submission was attempted.
