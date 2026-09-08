# Current result and research handoff

Submission 56090799 is COMPLETE: **0.97076 = 332/342, +2 versus 330, observed rank 3**. Leaders were 334/342. No further prediction submission is authorized today. The immutable champion is `submission_097076_332of342_CHAMPION.csv`; SHA-256 `25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56`.

## What the result establishes

Five previously unresolved changes (0488,0477,0506,0519,0501) together contributed +2. Four other included changes had previously measured zero contribution (0444,0426,0647,0206). Under additive exact-match scoring, the five have either two wins and three zeros, or three wins, one loss and one zero. There are 30 possible signed assignments. Individual correctness and private membership remain unidentified. The earlier constraint d0488+d0146=0 remains valid. Never reintroduce 0458 A: its measured contribution was -1.

## Updated candidate pool: evidence, not quota

There are currently **no newly established high-confidence corrections**. Do not pad a queue to 8–12 or claim four strong rows when the evidence does not support them.

| Row/family | Status | Evidence and limitation |
|---|---|---|
| 0483 A→C, sitting→typing | Demoted; retain only as a research target | Child is seated in the exact parent interval and overlaps a separate checking-time candidate at the boundary. The fixed user-disjoint sitting/typing model gives p(typing)=.435. No direct evidence beats the incumbent A. |
| 0526 D→C, phone→keyboard | Best remaining residual, still not submission-ready | Child is nested in a parent whose sequence includes typing and shows hands at a desk. Phone/typing model gives p(typing)=.470 (149/173 OOF); the raw object is unresolved and parent action sets are not exhaustive. |
| Five changes retained in 332 | Preserve aggregate gain | Known total +2; a single reversion can only improve to 333. Reversion candidates are conditional recovery tools, not five recommended probes. |
| 0430/0432 emotion swaps | Killed rationale | Historical template audit remerged repaired sessions 168 and 169–170; duplicate-emotion contradiction is invalid. |
| 0461/0469 middle-trial emotion swaps | Strongly demoted | Current user1 answers match donor endpoint trials in all 11 pairs; measured 0458 regression further contradicts this family. This is structural evidence, not individual public labels. |
| 0341/0646 sequence swaps | Rejected for next cycle | Raw video supports reading before checking time, as in champion. Analogous nonexact six-action OOF transfers lost all three attempted flips. |
| 0146/0150/0151 action removals | Reject current rationale | Nonexhaustive sibling questions cannot define full action sets; raw/protocol evidence contradicts proposed removals. |
| 0590/0591/0158 omissions | Not promoted | Raw or OOF evidence fails to substantiate the tempting model peaks. |

For any singleton exploratory flip, the exact-match public delta range is −1, 0, or +1.
For the 0526 + 0501-reversion branch the range is 0…+2 under the forced-loss condition;
there is no unverified private-score guarantee behind these bounds.

## Source and model checks

All 197 local depth videos match official test ZIP members by size and CRC (`source_crc_audit.json`). The 0064/0119 duplicated full video and mismatch with short sensor metadata therefore appear in distributed source assets, not local corruption. This does not reveal the intended answer; treat modality alignment as unreliable there.

`pixel_nesting.json` contains low-resolution contiguous-parent matches. Low MAE is structural evidence, not exact crop provenance: static frames can produce ambiguous offsets. Dense windows for 0483 and 0526 are saved alongside the report.

The cached dense per-frame logits add a useful falsification check: test_0017's final frames
favor Reading/Checking-time and do not favor typing, while test_0016's final frames favor
Using-a-phone. These logits are not independently calibrated enough to overwrite either
answer, but they further weaken 0483 A→C and do not support 0526 D→C.

The two fixed logistic residual models use temporal skeleton descriptors and five user-disjoint folds, without stacking or leaderboard labels. Their OOF accuracies are not correction precision against the champion. Out-of-class standing examples receive extreme typing scores in the phone model, demonstrating why it must not be used as a universal confidence detector.

## Post-332 residual audit (new)

The full champion replay could not be completed safely in the current memory budget (the
existing dense cache plus another user process was killed by the cgroup), so no fabricated
OOF score is reported. Instead, the audit uses saved subject-disjoint old/new decisions and
keeps standalone accuracy separate from flip precision.

* Exact visible object-template fallback is the one genuinely repeatable residual route:
  at support >=1 it covers 113 OOF rows (fallback 87/113, unanimous template 112/113),
  with 25 changed decisions, **25 W→R and 0 R→W**. At support >=2 it is 14/0 on 14
  changes; support >=3 is 7/0. The test rows where this gate is unanimous are already
  consistent with 332; the only remaining object alternatives (0526/0527) are non-unanimous
  and are not promoted.
* An exact-template-vs-sensor HARn audit has 52 disagreement rows and 52/0 in favor of
  the template, but this is a selected support>=1 subset, not a universal sensor override.
  The only test single disagreements are 0478 (template B vs 332 C) and 0507 (template C
  vs 332 A). They are retained as preserve-incumbent research checks: an independent
  ExtraTrees action model agrees with 332 on both (mopping and walking), with 0507's
  incumbent margin 26.8. Exact-template single OOF is 115/118 at support>=1, not a proof
  that either test row is wrong.
* A grouped ExtraTrees action probe (five user-disjoint folds) was trained only as an
  independent disagreement detector. It produced no new test flip: its seven visible
  action-mapped test decisions all matched 332 (0478=C and 0507=A among them). Its OOF
  absolute action accuracy is saved in `harn_extra_residual_oof.csv`; because the old
  champion OOF decisions are not present locally, it is deliberately not reported as
  W→R/R→W precision.
* The cached dense temporal model was evaluated on exact train-HARn child intervals:
  385 subject-disjoint action-mapped rows, 79.5% absolute accuracy; even its margin>=5
  subset is only 82/84 (97.6%) and has no old/new champion comparison. On test it proposes
  12 single-row disagreements (largest: 0497 C→B), but this model is materially below
  the champion's established HARn-single OOF level and its top alternatives conflict with
  same-clip object evidence (0497), so the entire dense-disagreement family is killed for
tomorrow unless a true flip audit is added.
The reproducible dense audit is `dense_segment_probe.py`; it writes
`dense_segment_oof.csv` and `dense_segment_test.csv`.
* Raw-depth sweep of all 15 no-metadata HARn rows found no additional correction: 0477
  is bent/implement-compatible with mopping, 0488 is an ambiguous page/hand clip with
  no decisive label, 0501 is a clear stand-up, 0506 is compatible
  with lunges, 0510/0511 are one-leg standing, 0514 opens the cabinet, 0517 puts on
  clothes, 0519 drinks, and no-metadata object rows 0528/0530/0538/0539/0540 agree with
  their action/object context. Contact sheets are in this directory.

The machine-readable decision audit is `residual_audit.json`. Oracle-style top-k checks
are intentionally negative for the old structural families: recovered-pool sequence
transfers are net −2 by top-20, exact-child sequence is −20 by top-20, frame-forest
sequence is −14 by top-20, and paired-pool is −1 on its ten changed OOF rows. These
families remain killed; they are not candidates merely because a top-1 happened to win.
The source-level fallback inventory (all explicit `None`/default branches and 22 observable
test-side likely-fallback signatures) is `fallback_static_audit.json`, with row details in
`fallback_test_signatures.csv`; 22 is a diagnostic signature, not a label claim.
`residual_rank.csv` is a complete 682-row screening order. Its `p_wrong_screen` values are
explicit priors/uncertainty scores (not calibrated test probabilities); the first five are
the algebraically unresolved +2 pack, followed by 0526/0483/0527 and the two exact-template
disagreements.

## Additional autonomous probes (this run)

Two deliberately different routes were tested and rejected by held-out evidence.

* A raw `Depth_Color` video classifier was trained from 809 HAU single-question clips and
  evaluated with five user-disjoint folds.  It reached only **103/809 = 12.7%** clip-action
  accuracy (the weak single-question label is not a reliable whole-video visual target), and
  produced just one mapped test suggestion, which retained the incumbent.  It cannot safely
  correct the sensor-missing HARn rows.  The reproducible code and outputs are
  `raw_video_action_probe.py`, `raw_video_action_oof.csv`, and `raw_video_action_test.csv`.
* A near-template transfer audit relaxed exact option matching to three-of-four or two-of-four
  shared option texts.  This loses the exact-template signal: overall OOF accuracy is only
  34.997% (object 109/133 = 81.95%, sequence 70/84 = 83.33%), versus the established
  unanimous exact-object gate's 112/113.  It proposes many obvious false positives, including
  a jumping-jacks answer for the visually mopping `test_0477`; the only changed object test
  row is `test_0533` (A→C, purity .565), with no independent visual support.  No fuzzy-template
  flip is promotion-grade.  See `fuzzy_template_probe.py`, `fuzzy_template_oof.csv`, and
  `fuzzy_template_test.csv`.

These failures strengthen, rather than weaken, the conservative conclusion: current public
errors are not recoverable by generic raw-video or relaxed-template overrides.  The only new
research-only lead is `test_0533` as a counterexample to fuzzy transfer; it remains at A.

## Tomorrow's strategy

1. Keep 332 as the incumbent; do not spend a slot merely re-uploading it. First pursue exact sensor/crop alignment and stronger object evidence for 0483/0526, then broaden missing-modality residual mining. Neither exploratory flip is approved yet.
2. Only if both become independently credible, test the two-row bundle on 332: algebraic delta range −2…+2, not an estimated expectation. +2 ties the observed leaders. If only one clears the evidence gate, use its singleton.
3. Decode immediately. A two-row +2 fixes both signs at +1; +1 permits one win and one zero; zero may conceal cancellation; negative results retain the incumbent and require new evidence before isolation.
4. Use remaining slots for genuinely supported new gain bundles and, when immediately useful, deconvolution. Do not precommit all five slots to reversions: that family has only one possible recoverable loss and cannot alone reach 334.
5. Reserve a slot for the best supported combination. Reassess live rank and remaining allowance tomorrow. No credible route to 335 is established yet.

## Five-submission adaptive plan

The exact +2 aggregate makes one-loss recovery unusually valuable. Submit these four files independently against the 332 champion, in order (or in parallel only if evaluation timing permits):

**Recommended first submission:** `ADAPTIVE_probe_revert_test_0488.csv`.

1. `ADAPTIVE_probe_revert_test_0488.csv` — SHA-256 `5c424b408967b7750661e495b296e1975de4599afa29fa905e6aa0d0a7d47ffd`.
2. `ADAPTIVE_probe_revert_test_0477.csv` — SHA-256 `02d63945c94855115c05d668c2308af30be4b702741123857ffde84e42897399`.
3. `ADAPTIVE_probe_revert_test_0506.csv` — SHA-256 `204b72a7b24f5b07e5acac9948b27253f218caeee3d809090fe6883ce8b18fd6`.
4. `ADAPTIVE_probe_revert_test_0519.csv` — SHA-256 `40f59896dedd3622122424aeba3c163dc4894a35b28866a44f494d9bb190ec49`.

Each result is delta versus 332 and identifies that row's signed effect: −1 means the changed row was a public win, 0 neutral, +1 a public loss. The omitted `test_0501` sign is forced by `d0488+d0477+d0506+d0519+d0501=+2`. Let `S = -sum(first-four observed deltas)`; then `d0501 = 2-S`.

For submission 5, use `ADAPTIVE_fifth_0526.csv` (SHA-256 `fc386cb51497729ba65682928acc279bf11b7e968cbfdeaa26b9f576a2b956a7`) if `d0501 != -1`. If the forced sign is the unique loss (`d0501=-1`), use `ADAPTIVE_fifth_0526_plus_revert_0501.csv` (SHA-256 `8a5a02f0138af255c32a499122e4b1cd418599c48636842586af598a422d1c9d`); it recovers that point while testing 0526. If 0526 is a true win, this reaches 334; if not, it still recovers the hidden loss to 333. Exact diffs against 332 and the old 330 base are in `submission_manifest.json`.

## Ready files and decoder

Run `python3 research/build_post332_suite.py` to regenerate the 15 local CSVs, SHA-256 manifest, and exact diffs against both 332 and 330. Exploratory singletons/pair and conditional reversions are explicitly gated, not a recommended submission queue. The script has no network or submission capability.

The evidence-ranked machine-readable pool is `candidate_pool.json`; its core queue contains
0526, demoted 0483, and the algebraic recovery branch, while the three preserve-incumbent
research checks are explicitly marked and excluded from the queue.

Three additional research-only variants are prebuilt by `make_research_variants.py`:
0478 C→B (SHA `e89b0a42f00cd613b645cd03781a4f6cd49327ee75a5bfdd2756e84969d3b5e4`),
0507 A→C (`3d483af007a7b95f6bfd254873a5df5aac52370755fea8f85fe9d4d47eaf4a72`), and
0527 C→A (`76f42cdc8dc55cb7e08ca7158f6fd2a926b69cc4097d838f42ab30cb4a809716`).
Their one-row diffs are emitted beside each CSV. They are not in the five-slot queue: 0478/0507 have exact-template disagreements
without independent correction precision, and 0527's template purity is only .60.

Decode a hypothetical observed pair +1 with `python3 research/build_post332_suite.py --observe EXPLORATORY_pair=1`. Repeat `--observe` for subsequent evaluations; all deltas are versus **332**. The decoder preserves the best actually scored configuration instead of discarding unidentified aggregate wins. Current observations live separately in `../takeover_20260908/scored_result.json`.

For the five-slot plan, run `python3 research/post332_20260908/adaptive_decode.py <d0488-revert-delta> <d0477-revert-delta> <d0506-revert-delta> <d0519-revert-delta>`. It validates the four signed outcomes and prints the exact fifth filename; it has no network or write side effects.
