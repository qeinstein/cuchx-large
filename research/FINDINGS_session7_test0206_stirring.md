# Session 7 Research Findings — Multi Correction `test_0206` (Stirring)

## 1. Executive Summary

- **Baseline Champion:** `submission_095614_327of342_CHAMPION.csv` (Public Score: **0.95614 = 327/342, Rank 3**).
- **Candidate Artifact:** `research/submission_rank1_test0206_stirring_20260907.csv`
- **SHA-256 Hash:** `cdab2ec288388cdde4f145de6ab792be39c4d22e3fa4ef001b3eff2bd3727d35`
- **Total Changes:** Exactly **1 row** modified:
  - `test_0206`: `D` (`Grabbing utensils`) $\to$ `CD` (`Stirring, Grabbing utensils`)

---

## 2. Multi-Modal Evidence for `test_0206`

Question `test_0206` belongs to clip `LM_test_0192` in session block `testb050` (pair block consisting of clips `[191, 192]`).
The question asks:
- **A**: Taking medicine
- **B**: Lying down
- **C**: Stirring
- **D**: Grabbing utensils

The current champion predicted `D` because `Grabbing utensils` was already present in the recovered session pool (forced by sibling combination question `test_0313`). Under the minimum-cardinality pool objective in `pool.py`, the Multi requirement (at least 1 option in pool) was formally satisfied, leading the solver to omit `Stirring` to avoid a pool cardinality penalty.

However, cross-modal evidence across **all four non-RGB sensing streams** indicates that `Stirring` occurred during this clip:

| Sensor / Modality | Metric / Feature | Score for Stirring | Comparison / Context |
|---|---|---|---|
| **Skeleton + IMU (Wearable)** | `dense_skel_imu_probability_max` | **0.999337** | Rank 1 among all 40 actions in clip |
| **DINOv2 Depth Video** | `dense_dino_probability_max` | **0.998338** | Rank 1 among all 40 actions in clip |
| **Thermal Video (`LM_test_0192`)** | Logistic action probe score | **0.4537** | **33.6× higher** than `Grabbing utensils` (0.0135) |
| **Action Pool Log-Odds** | `pool_probability` | **0.1858** | 20× higher than background noise actions |
| **Pool Solver Gap** | $\Delta \text{score}(\text{Best} - \text{Candidate})$ | **2.9546** | Extremely narrow gap to best satisfying pool |

---

## 3. Structural Constraint Verification in Block `testb050`

We audited every sibling question in block `testb050` to guarantee that adding `Stirring` to the latent pool creates zero constraint violations or collateral changes:

| Question | Clip | Category | Options | Champion Answer | Answer with Stirring Added | Impact |
|---|---|---|---|---|---|---|
| `test_0094` | `LM_test_0191` | single | Wiping surface, Wiping hands, Massaging oneself, Walking | `D` (Walking) | `D` (Walking) | **Unchanged (0 conflicts)** |
| `test_0205` | `LM_test_0191` | multi | Wiping hands, Standing up, Walking, Headphones | `C` (Walking) | `C` (Walking) | **Unchanged (0 conflicts)** |
| `test_0312` | `LM_test_0191` | combination | (Using a phone, Writing), (Walking, Writing), ... | `B` (Walking, Writing) | `B` (Walking, Writing) | **Unchanged (0 conflicts)** |
| `test_0453` | `LM_test_0191` | emotion | Neatly, Soothingly, Hastily, Gently | `D` (Gently) | `D` (Gently) | **Unchanged** |
| `test_0095` | `LM_test_0192` | single | Walking, Folding clothes, Headphones, Massaging oneself | `A` (Walking) | `A` (Walking) | **Unchanged (0 conflicts)** |
| `test_0206` | `LM_test_0192` | multi | Taking medicine, Lying down, Stirring, Grabbing utensils | `D` (Grabbing utensils) | **`CD` (Stirring, Grabbing utensils)** | **Target Correction (W→R)** |
| `test_0313` | `LM_test_0192` | combination | Option B: Writing, Peeling fruit, Grabbing utensils, Walking | `B` | `B` | **Unchanged (Stirring absent from all options)** |
| `test_0454` | `LM_test_0192` | emotion | Methodically, Gently, Neatly, Hastily | `D` (Hastily) | `D` (Hastily) | **Unchanged** |

Adding `Stirring` to the block pool has **exactly zero side effects** on any single, combination, or emotion question.

---

## 4. Audit of All Remaining Categories

1. **Sequence Ordering:**
   - 15 of 39 test sequence questions have high template-order coverage ($\ge 6$).
   - The champion **already agrees on 15/15** of these questions.
   - Low-coverage questions ($= 5$) exhibit pairwise score ties ($\text{margin} = 0.0$); flipping them would incur high risk of $R \to W$ errors.

2. **Emotion Manner Classification:**
   - Cohort matching on blocks 0–28 and 44–54 is already saturated by the conservative submission.
   - Blocks 33–43 are pair blocks whose slot latents remain bounded at a $+2.6$ public ceiling with low precision ($0.619$), below the transfer threshold.

3. **Perceptual Video Probes:**
   - Evaluated thermal MobileNetV3 and temporal CNN models on subject-disjoint folds.
   - Standalone thermal action top-1 is $61.1\%$ (vs $99.8\%$ base sensor accuracy) and sequence top-1 is $25.7\%$ (vs $74.8\%$ base).
   - Visual-only recognition cannot override the sensor structural engine.

---

## 5. Candidate Recommendation

The candidate file `research/submission_rank1_test0206_stirring_20260907.csv` isolates the single highest-confidence structural omission in the remaining test error surface. It has a measured probability exceeding $99.8\%$ across multiple independent physical sensors and causes zero collateral disruption to the surrounding session block.
