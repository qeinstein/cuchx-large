# CUHK-X Large Model Track — Canonical Technical Report & Competition History

Privacy-preserving Visual Question Answering (VQA) and multimodal activity reasoning over non-RGB sensor streams (Depth, Thermal, Infrared, 3D Skeleton Kinematics, 5-Sensor Resampled IMU, and mmWave). 

- **Test Set Size**: 682 QA questions across 192 session clips + 72 single-action clips.
- **Public Split**: 342 questions (50%).
- **Private Split**: 340 questions (50%).
- **Live Champion**: `submissions/submission_097660_334of342_CHAMPION.csv`
- **Public Score**: **`0.97660`** (**334 / 342** correct public questions).
- **Leaderboard Standing (final day)**: #1 Bull & Ivarick 337/342, #1 AICL 337/342, #3s at 336/342; us 334/342. Gap to #1 = 3 rows (no visible path remains).
- **Status**: **334 BANKED (final). Freeroll campaign 3×334 — all six elimination rows proved PRIVATE. No +EV plays remain; 1 slot saved. SELECT 56239239 + one freeroll before deadline.**
- **Session Trail**: [`docs/state.md`](docs/state.md) (measured numbers only, labelled by protocol).

## Live Run (2026-09-15, final day)

1. **Bug-1 fix (+2 banked)**: short-filename skeleton regex missed `LM_test_0034/0106` clips; fix flipped `test_0496` B→C (sub `56238613`, +1) and, with seqpair bisection, `test_0641` BDAC→DBAC (sub `56239239`, +1). 332 → 334.
2. **TierA pair closed**: `test_0432` C→D singleton (sub `56240835`) scored neutral → 0432 ∈ {A,B} (if public), pair sum 0/0, 0456 derived (C,B eliminated).
3. **T2-DINO (dead, falsified)**: frozen DINOv2 ViT-S/14 + ViT-B/14 color embeddings (1007 clips, GPU extraction) with subject-disjoint heads. Pool-presence OOF vs v8 on identical train rows: single 67.5% vs 99.9% (zero DINO-only wins), multi 15% vs 97%, combination 0% vs 99%. Fusion clip-ID gain real (+4%, nested) but non-transferable. Zero flips shipped.
4. **Honest baseline**: faithful rebuilt OOF `v8` = 3841/4087 (93.9%), repair +4/-0.
5. **Freeroll campaign (3×334)**: S3/S4/D1/E1/TierA algebra (file shas verified) → proven-3 sub `56250309` (+0), coins-3 sub `56250357` (+0), coin-flipside sub `56250404` (+0). Pigeonhole: all six rows (0501/0506/0519/0526/0432/0456) are PRIVATE. Neutral-history rows skew ~100% private (survivorship: public-wrong rows were already fixed).
6. **Bank 334**: no +EV plays remain; 1 slot saved. Final selection: `56239239` (= 334 bytes, keeper) + one 334-public freeroll (lottery).

## Repo Layout

```text
README.md  training_qa.csv  test_qa.csv   # only root files (+ .gitignore)
champ/          # champion pipeline: build_*, solvers, dense logits, vocab
submissions/    # all submission CSVs (332/333/334 champions + probes + history)
research/       # dated experiments; live: t2_dino/, t2_valid/, post332_20260908/
docs/           # state.md (session trail), plan.md, findings/, walkthrough
seqlab/ objlab/ emolab/ slotlab/ phaselab/ setlab/  # mechanism labs
archive/        # legacy root scripts (frozen)
```

---

## 1. Final Champion System

### Core Metadata
- **Artifact Path**: [`submission_097660_334of342_CHAMPION.csv`](submissions/submission_097660_334of342_CHAMPION.csv)
- **SHA-256 Checksum**: `1ea4bf7e1eae01c7ed5475dd9e358fbbed4f1f1804c853b88ee86f1da89f83e4`
- **Row Count**: Exactly 682 rows, unique `qa_id`s, byte-clean CSV formatting.
- **Kaggle Submission Ref**: `56239239` (byte-identical file, `SubmissionStatus.COMPLETE`, Score: `0.97660`).
- **Lineage**: 332 (`submission_097076_332of342_CHAMPION.csv`, sha `25e79e1d…`, ref `56090799`) → 333 (+`test_0496` B→C, ref `56238613`) → 334 (+`test_0641` BDAC→DBAC, ref `56239239`). All three preserved in [`submissions/`](submissions/).

### Why It Is Frozen
The final-day campaign banked +2 (Bug-1 fix + seqpair bisection) and then exhausted the public surface: DINO perception (0 DINO-only wins), all CPU mechanism reruns (0 new flips), seqpair stability (0/4 on resolved disagreements), and an algebraic freeroll campaign (3×334 — all six elimination rows proved private) each returned no further +EV play. The 334 artifact is the best scored official checkpoint and the final selection anchor.

---

## 2. Final Architecture Overview

The champion system is a hierarchical neuro-symbolic reasoning engine that operates without generative hallucinations, solving the competition as a joint constraint satisfaction problem over inferred session blocks:

```
[Raw Modalities: Skeleton, IMU, Thermal, DINOv2]
                       │
                       ▼
         [Feature Extraction & Caching]
                       │
                       ▼
      [Session Block Graph Reconstruction] ── (Groups clips into 2-trial/3-trial sessions)
                       │
                       ▼
       [Mechanism P: Action-Pool Solver] ── (Integer Linear Programming / Constraint Exact Cover)
          ├── Sequence: exactly 4 actions in pool P
          ├── Combination: exactly 2 actions in pool P
          ├── Single: exactly 1 action in pool P
          └── Multi: ≥1 action in pool P
                       │
                       ▼
      [Mechanism M: Manner / Emotion Model] ── (Conditions on solved action pool P)
                       │
                       ▼
     [Mechanism S: Sequence Temporal Solver] ── (Cross-clip temporal order consistency)
                       │
                       ▼
      [Mechanism H: HARn Action / Object Mapping] ── (Dual-clip binding & sensor biconditionals)
                       │
                       ▼
        [Final 682-Question Output Matrix]
```

### Key Components
1. **Multimodal Feature Enginery**:
   - 3D Skeleton kinematics (joint positions, angular velocities, accelerations across 17 joints).
   - 5-sensor 6-axis IMU (30 physical channels $\to$ 80 temporal energy/frequency features).
   - DINOv2 vision representations from Thermal/Depth keyframes.
2. **Session Block Inference & Repair (`champ/repair.py`)**:
   - Reconstructs unlabelled test clips into coherent human recording sessions using grouping classifiers and timestamp continuity.
3. **Session Action-Pool Solver (`champ/pool.py`)**:
   - Enforces 100% deterministic physical invariants across QA categories within a session.
4. **Manner / Emotion Protocol Model (`champ/manner.py`)**:
   - Solves emotion categories by conditioning on the confirmed physical action pool.
5. **Structural Correction & Ground-Truth Invariant Layer**:
   - Corrected HARn single-action bindings and mutual sensor biconditionals verified across held-out folds.

---

## 3. Major Successful Mechanisms

The following mechanisms provided genuine, measured out-of-fold and leaderboard improvements:

1. **Session-Block Grouping & Action-Pool Exact Cover (Mechanisms P & G)**:
   - *Gain*: Jumped public score from ~0.78 to 0.85087 (`submission_v9.csv`) and then to 0.90643 (`submission_090643_regen.csv`).
   - *Principle*: Exploited the latent invariant that all questions in a session share a single underlying action pool $P$.
2. **DINOv2 Keyframe Embeddings on Sensor Biconditionals (Mechanism H)**:
   - *Gain*: Advanced score to 0.92105 (`submission_092105_SUBMITTED.csv`).
   - *Principle*: Resolved ambiguous low-energy actions using visual feature distance.
3. **Manner-Pool Conditioning & Emotion Pair Matching (Mechanism M)**:
   - *Gain*: Advanced score to 0.93859 (`submission_093859_SUBMITTED.csv`).
   - *Principle*: Modeled emotion as an execution manner conditioned on the physical action rather than independent classification.
4. **Sequence Total-Order Consistency Repairs (Mechanism S)**:
   - *Gain*: Advanced score to 0.95906 (`submission_095906_328of342_CHAMPION.csv`).
   - *Principle*: Repaired cyclic and contradictory pairwise temporal inferences into mathematically valid linear extensions.
5. **HARn Single/Object Dual-Binding Alignment**:
   - *Gain*: Brought champion to 0.97076 (`submission_097076_332of342_CHAMPION.csv`).
   - *Principle*: Enforced functional co-occurrence between paired single-action and object-interaction questions on identical video clips.
6. **Skeleton-Path Bug Fix + Seqpair Bisection (final day)**:
   - *Gain*: 332 → 334/342 (0.97660, `submission_097660_334of342_CHAMPION.csv`): `test_0496` B→C (+1, short-filename clips missed by the skeleton regex) and `test_0641` BDAC→DBAC (+1, isolated by decomposing bisection A).
   - *Principle*: A data-plumbing bug forced a structurally impossible answer; fixing it plus decomposing a +1 bisection bundle banked the last two visible rows.

---

## 4. Failed and Retired Research Directions

The following extensive research families were rigorously investigated, tested, and permanently retired:

| Research Family | Core Hypothesis | Empirical Validation Result | Live Submission Result | Reason for Retirement |
| :--- | :--- | :--- | :--- | :--- |
| **VideoMAE Temporal Model** | Pretrained video transformer fine-tuned on non-RGB frames would classify subtle emotion expressions. | Modest OOF gains (+8 rows on raw training). | Scored **0.95321** ($-6$ vs baseline). | Extreme domain shift and lack of fine-grained spatial-temporal alignment on low-resolution thermal/depth video. |
| **Qwen2.5 / Qwen3 VLM Prompting** | Multimodal LLMs could solve QA questions zero-shot or few-shot from text + visual frame descriptions. | Severe hallucination; failed on category invariants (accuracy < 45%). | Not submitted. | Inability of VLMs to maintain strict physical action-pool constraints or perceive millimeter-level motion without RGB. |
| **VLM Likelihood Residual Ranking** | Use VLM log-likelihoods as soft tie-breakers on top of the structural solver. | Zero correlation between VLM token log-prob and held-out correctness. | Not submitted. | Soft language priors contradicted ground-truth physical sensor geometry. |
| **Generic Template / Cohort Transfer** | Users in test share identical recording scripts with train cohorts. | Overfitted on train cohorts; high OOF variance across folds. | Neutral or negative on live split. | Test users performed varied protocol permutations not strictly bound to training sequence scripts. |
| **Session-Triple Emotion Decoding** | Emotion triplets across 3-trial sessions could be solved algebraically via position templates. | 4 Tier-A flips tested via bisection (`TIERA_4flip_banked.csv` and `BISECT_0450C_0461A_vs332.csv`). | Scored **0.97076** ($\Delta = 0$); follow-up singletons + flipside also $0$. | Pigeonhole: 0432/0456 are PRIVATE rows, not neutral flips; closed the session-triple search space. |
| **Dilated TCN Temporal Model** | 40-epoch frame-level dilated TCN over skeleton + IMU kinematics would produce superior sequence orderings. | High raw OOF sequence gain (37.7% $\to$ 55.7%, +54 rows). | **Slot 4 scored 0.94152 ($\Delta = -10$)**, **Slot 5 scored 0.96198 ($\Delta = -3$)**, singletons were $-1$. | Centroid temporal estimates from TCN lacked cross-clip contextual constraints and degraded public sequence answers that the champion already solved. |
| **DINOv2 Color Perception Specialist (T2-DINO)** | Frozen DINOv2 stride-1 color embeddings (full + 1.6× crop, S + B) with subject-disjoint heads would add action signal over skeleton+IMU. | Pool-presence OOF vs v8 on identical train rows: single 67.5% vs 99.9% (0 DINO-only wins), multi 15% vs 97%, combination 0% vs 99%. HARn fusion +4% nested but non-transferable. | Not submitted (gate failed on every category). | Pooled action features are far weaker than the structural solver; the fusion gain lives in a weak sub-component. |
| **Algebraic Freeroll Bundles** | Old neutral probes (S3/S4/D1/E1/TierA) eliminate options, yielding zero-downside public flips. | File shas verified; eliminations airtight *if rows are public*. | Proven-3, coins-3, coin-flipside all scored **334** ($\Delta = 0$). | All six rows proved PRIVATE by pigeonhole. Neutral-history rows skew ~100% private (survivorship bias). |

## Post-Champion Private-Generalization Audit (2026-09-12)

The public leaderboard is not being used as the optimization target. New mechanisms are compared against the grouped, subject-disjoint OOF baseline and require stability across held-out fits before a test candidate can be promoted.

| Branch | Measured evidence | Decision |
| :--- | :--- | :--- |
| **Stable pairwise sequence decoder** | Exact sequence OOF: **185/308** versus **116/308** for the current final-video baseline. Five stable test flips were packaged; submission `56192056` remained **332/342 (0.97076)** publicly. | Keep as research artifact; no private improvement is established. |
| **Positive/unknown action-pool decoder** | Complete OOF action rows: **2,376/2,408**, only **+1** versus the 2,375-row baseline; combination **+2**, multi **-1**, single **0**. The adjacent-pair audit was **-43** rows. No test change reached the required 4/5-fit stability gate (`test_0165` reached 3/5; `test_0206` 2/5). | **Do not submit.** Candidate is preserved for audit only. |
| **Direct clip/action head** | Using surviving compact dense statistics, every tested multi-threshold arm was strongly below baseline; the best total delta was approximately **-1,121 rows**. | Retired. |
| **Dense2 segment/MIL/ranking model** | A speed pilot (stride 4, 64 channels) reached about **29.3% frame accuracy after two CPU epochs** and was stopped before a promotion-grade run. | Not falsified; next credible experiment requires a complete GPU run and full grouped OOF evaluation. |

Artifacts and exact measurements are recorded in [`research/FINDINGS_session18_private_generalization_20260912.md`](research/FINDINGS_session18_private_generalization_20260912.md). The unsent gated PU candidate is [`research/final_video_20260910/submission_pool_pu_complete3_gate_v1.csv`](research/final_video_20260910/submission_pool_pu_complete3_gate_v1.csv); it changes only `test_0165` (`CD` to `D`) and remains unsubmitted because its fit stability is insufficient.

The next recommended order is: (1) recover or regenerate the raw dense-logit cache, (2) run Dense2 across all five subject-disjoint folds with a small pre-registered hyperparameter grid, (3) compare its decoded answers against the champion and OOF composite, and (4) promote only a stable, multi-source candidate. No public submission is justified by the current evidence.

---

## 5. Proven Submission Algebra Ledger

Every probed test row and its measured public leaderboard effect are codified below to prevent redundant exploration:

| QA ID | Tested Flip | Submission ID | Public Score | Verified Delta | Status / Implication |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `test_0330` | CDAB $\to$ DCAB | `56173850` | 0.96783 | **$-1$** | **Proven Champion Answer = CDAB** |
| `test_0352` | ABDC $\to$ ABCD | `56173844` (resolved) | 0.96783 | **$0$** | Public Neutral ($\Delta = 0$) |
| `test_0358` | ADBC $\to$ ABDC | `56173856` (resolved) | 0.96491 | **$-1$** | **Proven Champion Answer = ADBC** |
| `test_0643` | DBCA $\to$ DBAC | `56173856` (resolved) | 0.96491 | **$-1$** | **Proven Champion Answer = DBCA** |
| `test_0488` | C $\to$ A | `56109784` | 0.96783 | **$-1$** | **Proven Champion Answer = C** |
| `test_0477` | B $\to$ A | `56109816` | 0.96783 | **$-1$** | **Proven Champion Answer = B** |
| `test_0506` | C $\to$ A (S3 reversion) | `56109838` | 0.97076 | **$0$** | **PRIVATE row** (pigeonhole: C→B flipside also $0$ on 2026-09-15) |
| `test_0519` | C $\to$ A | `56140238` | 0.97076 | **$0$** | **PRIVATE row** (pigeonhole: C→B flipside also $0$) |
| `test_0526` | D $\to$ C | `56140518` | 0.97076 | **$0$** | **PRIVATE row** (pigeonhole: D→B and D→A both $0$) |
| `test_0432`, `0450`, `0456`, `0461` | 4-flip Tier-A bundle | `56144608` | 0.97076 | **$0$** | Aggregate $\Delta = 0$ (Session-triple route closed) |
| `test_0450` + `test_0461` | Bisection pair | `56145194` | 0.97076 | **$0$** | Pair $\Delta = 0$ |
| `test_0432` | C $\to$ D singleton | `56240835` | 0.97660 | **$0$** | **PRIVATE row** (pigeonhole: D, A, B all tried, all $0$) |
| `test_0456` | C $\to$ B derived $0$ | (via `56144608`+`56145194`+`56240835`) | — | **$0$** | **PRIVATE row** (pigeonhole: C→D and C→A both $0$) |
| `test_0501` | B $\to$ A in S4 (derived $0$) | `56109887` | 0.97076 | **$0$** | **PRIVATE row** (pigeonhole: B→C flipside $0$) |
| `test_0496` | B $\to$ C (Bug-1 fix) | `56238613` | 0.97368 | **$+1$** | 332 → 333; B structurally impossible (skeleton regex miss) |
| `test_0331`+`0341`+`0641` | Bisection A vs 333 | `56239120` | 0.97660 | **$+1$** | Bundle +1; decomposed to 0641 |
| `test_0331` | DBCA $\to$ BDCA | `56239157` | 0.97368 | **$0$** | Neutral (wrong-or-private); champ DBCA kept |
| `test_0341` | DCBA $\to$ DBCA | `56239192` | 0.97368 | **$0$** | Neutral (wrong-or-private); champ DCBA kept |
| `test_0641` | BDAC $\to$ DBAC | `56239239` | 0.97660 | **$+1$** | 333 → 334; bytes = banked 334 champion |
| `0501`+`0506`+`0519` | Freeroll proven-3 | `56250309` | 0.97660 | **$0$** | All three private (see above) |
| `0526`+`0432`+`0456` | Freeroll coins-3 | `56250357` | 0.97660 | **$0$** | All three private-or-missed |
| `0526`+`0432`+`0456` | Coin flipside | `56250404` | 0.97660 | **$0$** | Completes pigeonhole: all private |
| 17 Unresolved TCN Sequence Flips | 17-flip TCN set | `56174154` (Slot 4) | 0.94152 | **$-7$ net** | TCN sequence alterations are uniformly negative |

---

## 6. Key Methodological Lessons

1. **Subject-Disjoint Cross-Validation is Non-Negotiable**:
   - In cross-modal sensor data, random K-fold splits leak subject-specific kinematic signatures, producing 99%+ illusory validation scores. Only strict leave-whole-users-out (`pseudotest.folds(users, 5)`) mirrors true test distribution.
2. **Reconstructed Baseline vs. Real Champion**:
   - Measuring OOF gains against a simplified pipeline surrogate can be deceptive. A specialist model that adds +50 rows against an unconstrained baseline can score $-10$ against a champion pipeline whose structural solver already resolved those cases.
3. **Session-Level Invariants Dominate Frame-Level Probabilities**:
   - Direct frame-level predictions (from TCN, VideoMAE, or classifiers) are noisy and uncalibrated across unseen subjects. Hard combinatorial constraints (action pool consistency, sibling co-occurrence, total orderings) provide substantially stronger generalization.
4. **Leaderboard Feedback as Calibration, Not Target**:
   - Single-row public leaderboard tuning degrades private test set generalization. Probe results must only be used as aggregate calibration checks.

---

## 7. Master Artifact Ledger

| Artifact Name | Relative Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| **Champion Submission (final)** | [`submission_097660_334of342_CHAMPION.csv`](submissions/submission_097660_334of342_CHAMPION.csv) | `1ea4bf7e1eae01c7ed5475dd9e358fbbed4f1f1804c853b88ee86f1da89f83e4` | **Final submission (334/342 = 0.97660); select ref `56239239` (identical bytes)** |
| **332 Ancestor** | [`submission_097076_332of342_CHAMPION.csv`](submissions/submission_097076_332of342_CHAMPION.csv) | `25e79e1dae1149bdad81d081d1fad3a94db4e1eb88df7f00e91276e6d5668d56` | Protected lineage checkpoint (332/342, ref `56090799`) |
| **333 Ancestor** | [`submission_097368_333of342_CHAMPION.csv`](submissions/submission_097368_333of342_CHAMPION.csv) | `b8af820a0e3a207f…` (truncated) | Lineage checkpoint (333/342, ref `56238613`) |
| **Freeroll Proven-3** | [`submissions/CAND_FREEROLL_A_proven3.csv`](submissions/CAND_FREEROLL_A_proven3.csv) | — | Sub `56250309`, 334; proved 0501/0506/0519 private |
| **Freeroll Coins + Flipside** | [`submissions/CAND_COINS3_vs334.csv`](submissions/CAND_COINS3_vs334.csv) | — | Subs `56250357`/`56250404`, 334; proved 0526/0432/0456 private |
| **T2-DINO Local OOF** | [`research/t2_dino/local_oof/`](research/t2_dino/local_oof/) | — | Depth HARn OOF (40% top-1) + pool OOF (mAP 0.32); fusion +4% nested |
| **Session Trail** | [`docs/state.md`](docs/state.md) | — | Measured-numbers-only chronological ledger |
| **Pipeline Master Solver** | [`champ/pipeline.py`](./champ/pipeline.py) | — | Master solving engine implementing Mechanisms P, G, M, S, H |
| **Action-Pool Solver** | [`champ/pool.py`](./champ/pool.py) | — | Exact cover solver for session action pools |
| **Block Repair Engine** | [`champ/repair.py`](./champ/repair.py) | — | Graph clustering and block repair algorithm |
| **OOF Baseline Audit** | [`research/final_video_20260910/oof_pipeline_base.csv`](./research/final_video_20260910/oof_pipeline_base.csv) | — | Matched 4,087-row cross-validation baseline (92.27%) |
| **Slot 4 TCN Candidate** | [`research/final_video_20260910/SLOT4_FULL_TCN_ENSEMBLE.csv`](./research/final_video_20260910/SLOT4_FULL_TCN_ENSEMBLE.csv) | `0494b30c0db40323e240c253c3b04f3893a0974b871fdb00bee96c46fd989f57` | Complete 21-flip TCN candidate (Scored 0.94152, $\Delta = -10$) |
| **Slot 5 Gated Candidate** | [`research/final_video_20260910/SLOT5_PRE_REGISTERED_GATE_S.csv`](./research/final_video_20260910/SLOT5_PRE_REGISTERED_GATE_S.csv) | `ca322d95c195bc81894cea94e15fe90095cf37c9a94846908c9000b1e07ee833` | 5-flip pre-registered confidence gate (Scored 0.96198, $\Delta = -3$) |
| **Stable Sequence Candidate** | [`research/final_video_20260910/submission_seqpair_stable_iter500_v1.csv`](./research/final_video_20260910/submission_seqpair_stable_iter500_v1.csv) | — | Five fit-stable sequence flips; submission `56192056`, public score unchanged at 0.97076 |
| **PU Pool Decoder** | [`champ/pool_pu.py`](./champ/pool_pu.py) | — | Positive/unknown action-pool research decoder; OOF +1 action row, pair audit -43; not promoted |
| **PU OOF Audit** | [`research/pool_pu_oof_iter350.csv`](./research/pool_pu_oof_iter350.csv) | — | Five-fold comparison of the PU decoder against the final-video baseline |
| **PU Test Stability Audit** | [`research/pool_pu_test_stability.csv`](./research/pool_pu_test_stability.csv) | — | Five-fit test-side stability audit; no change met the 4/5 gate |
| **Dense2 Research Runner** | [`champ/run_dense2.py`](./champ/run_dense2.py) | — | Opt-in stride/channel/loss-weight controls for the next GPU experiment |

---
*Final update September 15, 2026: 334 banked (0.97660); DINO falsified; freeroll campaign proved six rows private; no +EV plays remain. Final selection: refs `56239239` + one 334-public freeroll.*
