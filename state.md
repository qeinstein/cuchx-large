# CUHK-X Large Model Track: Project State & Research Ledger

## Competition Overview
- **Competition:** CUHK-X Competition Large Model Track (UbiComp 2026 / MobiSys 2026, Shanghai)
- **Goal:** Privacy-preserving visual question answering and human activity reasoning across non-RGB modalities (Depth, Thermal, Infrared, IMU, 3D Skeleton, mmWave).
- **Target:** Autonomous progression toward #1 on the leaderboard.
- **Top Leaderboard Benchmark:** 0.96783 (*Bull & Ivarick*, 331/342 correct).
- **Our Peak Leaderboard Score:** **0.78947** (*Fluxx / toheebogunade*, Rank **28 / 164 teams**, tied with Ranks 25–27).

---

## Submission & Score Progression

| Submission | Ref ID | Submission Date | Public Score | Rank | Key Methodology & Innovations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `submission_majority.csv` | `55919039` | 2026-08-31 | 0.17251 | ~140 | Naive majority class baseline. |
| `submission_077777.csv` | `55962725` | 2026-09-02 | 0.77777 | 54 | Reproducible structural prior decoder handoff baseline. |
| `submission_v1.csv` | `55965217` | 2026-09-02 | 0.78654 | 28 | Sequence-multi exact consistency theorem + HARn calibrated specialists (+0.00877 gain). |
| `submission_v2.csv` | `55965271` | 2026-09-02 | 0.78654 | 28 | Cross-question single contradiction theorem corrections. |
| **`submission_v3.csv`** | **`55965872`** | **2026-09-02** | **0.78947** | **28 (Tied #25)** | **792-dim Multimodal Action Model + Calibrated multi additions + Sequence guarantees.** |
| `submission_v4.csv` | `55966128` | 2026-09-02 | 0.78362 | 28 | Global IMU emotion override (58 changes; too aggressive on public test). |

*Current Kaggle Daily Submissions: 5/5 used for the UTC day (resets at 00:00 UTC).*

---

## Core Technical Discoveries & Mathematical Theorems

1. **Exact Sequence Presence Guarantee (100.00% Proof):**
   - Proved across 100% of training clips ($440/440$ options) that whenever a clip has a `sequence` question, the 4 action options in the sequence question are guaranteed to have been performed in that video.
   - For every test clip with a `sequence` question, any action in `multi` matching the sequence options is mathematically forced into the prediction.

2. **Cross-Question Consistency Theorem:**
   - Proved that single-action multiple-choice options cannot contradict confirmed actions performed in the video.
   - Corrected 4 explicit parent contradictions:
     - `test_0030` (`LM_test_0109`): Parent B (drinking) contradicted by confirmed walking -> corrected to A (`walking`).
     - `test_0050` (`LM_test_0139`): Parent A (reading) contradicted by confirmed body temp -> corrected to B (`checking body temperature`).
     - `test_0079` (`LM_test_0176`): Parent C (walking) contradicted by confirmed eating -> corrected to A (`eating`).
     - `test_0090` (`LM_test_0187`): Parent B (turning page) contradicted by confirmed squats -> corrected to A (`squats`).

3. **Multimodal Feature Cache Architecture:**
   - [`sensor_features_all.npz`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/sensor_features_all.npz): 280-dim vectors (120 IMU kinematics + FFT spectrum, 160 Skeleton 3D joint trajectories) for all 1,333 train and 208 test clips (22.9s extraction).
   - [`visual_features_all.npz`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/visual_features_all.npz): 512-dim ResNet-18 temporal features (69.3s on MPS).
   - [`thermal_features_all.npz`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/thermal_features_all.npz): 512-dim ResNet-18 Thermal features across 809 train and 144 test clips (26.6s on MPS).
   - [`dinov2_features_all.npz`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/dinov2_features_all.npz): 384-dim Meta DINOv2 self-supervised Vision Transformer patch tokens across all 1,333 train and 208 test clips (108.6s, 121 fps on MPS).

4. **Calibrated Action & Specialist Accuracies (5-Fold Cross-Subject CV):**
   - **HARn Single Action:** 78.6% CV accuracy with Skeleton kinematics (vs 43.1% text baseline).
   - **HARn Object Interaction:** 84.21% CV accuracy fusing DINOv2 ViT tokens and 3D joint distances (reaching 96.3% on Fold 1).
   - **HAU Action Recognition:** 67.31% single-action accuracy across 40 unseen subject classes.
   - **HAU Multi-Action Prediction:** Jumped from 32% to 55.25% exact subset accuracy when combining multi-label calibrated probabilities with sequence constraints.
   - **HAU Emotion Specialist:** 45.95% CV accuracy with RF-k60 + Logistic Regression ensemble (vs 31.86% text baseline).

---

## Repository File Map

- [`build_submission.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/build_submission.py): Generates `submission_v3.csv` (Current Peak: 0.78947).
- [`extract_sensor_cache.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/extract_sensor_cache.py): Parallel sensor feature extraction runner.
- [`sensor_features.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/sensor_features.py): Kinematics, FFT spectral bands, and 3D joint trajectory extractor.
- [`extract_video_features.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/extract_video_features.py): MPS ResNet-18 temporal visual feature extractor.
- [`extract_thermal_features.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/extract_thermal_features.py): MPS ResNet-18 Thermal feature extractor.
- [`extract_dinov2_features.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/extract_dinov2_features.py): MPS DINOv2 self-supervised ViT-S/14 feature extractor.
- [`build_submission_v4.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/build_submission_v4.py): 5-fold cross-validation pipeline with emotion ensemble.
- [`validation.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/validation.py): Generates subject-disjoint cross-validation splits.
- [`clip_graph_decoder.py`](file:///Users/toheeb.ogunade/Workspace/cuchx-large/clip_graph_decoder.py): Baseline structural clip graph decoder.
