# Motion-quality representation research: an execution-ranked shortlist

## Scope and local facts

The target is **how** an approximately fixed HAU script was performed, not ordinary action recognition.  This makes the session a valuable reference set rather than a nuisance: compare its trials, then assign the manner candidates jointly.

Local compatibility check (read-only, 2026-09-04):

- Emotion has 809 HAU clips, 18 subjects, 272 sessions, 56 manner words, and five existing groups.  There are 265 true three-clip sessions and only seven true two-clip sessions; the test-like two-clip regime must therefore be simulated by thinning.
- `champ/skel_seq.npz` contains 3,905 variable-length, 10 Hz `(T, 17, 3)` skeleton sequences; `champ/imu_seq.npz` contains 3,917 frame-aligned `(T, 30)` IMU sequences.  They cover 796/809 Emotion clips.
- `champ/dino_frames.npz` contains a full per-frame `(T, 384)` frozen Depth-DINO trajectory for every 809 Emotion clip.  It is a particularly useful modality-complete first probe.
- Existing relative work is not a reason to call raw relative learning exhausted: `champ/emopair.py` uses summary-feature differences, and `emolab/dtw_feats.py` reduces a pose alignment to a small set of hand-made path statistics.  The recorded DTW increment was only +0.9 percentage points on pair orientation.  A learned, part-aware temporal comparison has not been tested.

The following proposals do **not** justify another generic action localizer, a global re-fit, or a claim based on ordinary clip accuracy alone.

## 1. Learned session-pair alignment and assignment scorer — do this first

**External basis.** [CoRe / Group-Aware Contrastive Regression](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Group-Aware_Contrastive_Regression_for_Action_Quality_Assessment_ICCV_2021_paper.html) recasts action-quality assessment as comparison to a reference with shared attributes rather than an absolute prediction; its [official code](https://github.com/yuxumin/CoRe) samples same-group query/reference pairs.  [FineDiving](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FineDiving_A_Fine-Grained_Dataset_for_Procedure-Aware_Action_Quality_Assessment_CVPR_2022_paper.html) and its [official implementation](https://github.com/xujinglin/FineDiving) make pairwise temporal procedure correspondence central to fine-grained quality assessment.  [FineParser](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FineParser_A_Fine-grained_Spatio-temporal_Action_Parser_for_Human-centric_Action_Quality_CVPR_2024_paper.pdf) strengthens that idea with human-centric spatial-temporal alignment.

**Mechanism.** For each pair of clips in one inferred session, build root-centered, torso-scale-normalized pose channels plus velocity/acceleration, aligned IMU channels, and optionally projected DINO-frame deltas.  A small shared 1-D TCN (not a large Transformer) produces temporal tokens.  Use banded monotonic cross-attention or differentiable soft-DTW between the two token streams; retain the alignment map, local speed/jerk residuals, and an unwarped duration channel.  Score the two hypotheses

`(clip i -> candidate a, clip j -> candidate b)` versus `(clip i -> b, clip j -> a)`.

The output is a pairwise log-odds term that can replace or be added to `champ/emopair.py`'s summary-feature `pair_logodds`, while the existing constrained assignment still handles label-level priors and missing slots.

**Why it is a real new test.** Current DTW compares unit-normalized pose vectors and discards almost all aligned residual structure.  This scorer learns which time regions and body parts distinguish a manner, so it is closer to FineDiving's procedure comparison than another duration/spectral feature expansion.

**Small honest probe.** Use only HAU session pairs and groups first (not the sparse 56-word target).  Train a <=1M-parameter model on four subject folds; evaluate the fifth on (a) held-out pair orientation, (b) full candidate assignment, and (c) the 38%-pair-thinned pseudo-test.  Add modalities one at a time: pose only, IMU only, DINO trajectory only, then late fusion.  Stop unless both full Emotion accuracy and champion-disagreement precision improve consistently in at least four folds.  This is the highest-value experiment because it preserves the actual relative decision structure.

**Important guard.** Do not train or align arbitrary clips across sessions: that lets action/script identity dominate.  Within-session pairs are the positive design; all same-session grouping at validation/test must come only from the existing test-visible block inference.

## 2. Frozen DINO trajectory encoder with temporal/contextual contrast — cheap modality-complete probe

**External basis.** [TS-TCC](https://mlanthology.org/ijcai/2021/eldele2021ijcai-time/) learns time-series representations by temporal and contextual contrast across augmented views; the [official repository](https://github.com/emadeldeen24/TS-TCC) is small PyTorch code.  FineDiving/FineParser provide the relevant quality-assessment rationale: preserve temporal correspondence rather than mean-pool a video.

**Mechanism.** Treat each existing `(T,384)` DINO sequence as a time series.  Per training fold, fit PCA only on training frames (e.g., 48 or 64 dimensions), concatenate first and second frame differences, and train a tiny temporal-convolution encoder.  Use self-supervised contrastive pretraining across two weak views of the *same* trajectory, then use its pooled embedding and session-centred embedding as inputs to a five-group head and/or the pair scorer in proposal 1.

**Manner-preserving augmentations.** Use feature dropout, small Gaussian noise, masked patches/time spans, and mild brightness-independent perturbation.  Do **not** use time warping, arbitrary temporal crops, or per-clip time normalization as the only view: pace, pause placement, and duration are target information.  Keep native-rate length/log-duration as explicit channels.

**Why it is distinct.** The current DINO contribution is fixed summary motion statistics (`dino_v_*`, cadence, path).  This probe asks whether the order and local changes of a depth-appearance trajectory contain a style signal, and it works on all Emotion clips even where skeleton/IMU is absent.

**Small honest probe.** First run a frozen/linear benchmark: one 2-layer TCN on the DINO trajectory versus the existing DINO summary baseline, with identical folds and session-centering.  Only add TS-TCC pretraining if the linear temporal model improves at least two fold-level group accuracies and changes a nontrivial number of champion answers.  Report triple and pair-thinned regimes separately.

## 3. Content/style factorization conditioned on the script — promising, but keep it small

**External basis.** [MoST: Motion Style Transformer between Diverse Action Contents](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_MoST_Motion_Style_Transformer_Between_Diverse_Action_Contents_CVPR_2024_paper.pdf) explicitly separates style and content with Siamese encoders and a style-disentanglement loss; [official code](https://github.com/Boeun-Kim/MoST) is available.  It is a motion-generation paper, not a directly transferable classifier, but its central failure mode matches this task: content and style otherwise get entangled.

**Mechanism.** Split a compact pose/IMU/DINO encoder into `z_content` and `z_style`.

- Train `z_content` to predict a training-only action-pool representation or a dense action histogram.
- Train `z_style` to predict the manner group and pair orientation.
- Add a gradient-reversal action-pool head on `z_style`, plus cross-covariance/orthogonality penalty between the two codes.
- Apply assignment only through `z_style`; leave the existing action-pool solver in charge of action identity.

The action-pool target must be calculated from training answers for training samples, and from the champion's test-visible pool prediction at inference.  Never use ground-truth pools on held-out subjects.

**Compatibility and caveat.** The raw skeleton is COCO-like 17-joint data (the local extractor uses nose=0, shoulders=5/6, hips=11/12), but full MoST expects a different motion-capture/BVH ecosystem and CUDA.  Reimplement only the two-code loss/head around the repository's existing small temporal encoder; importing the generative model is disproportionate and likely incompatible.

**Small honest probe.** Starting from the best encoder in proposals 1 or 2, compare `lambda_adv in {0, .03, .1, .3}`.  Require (i) held-out action-pool prediction from `z_style` to fall, (ii) held-out group/assignment accuracy not to fall, and (iii) a positive net of champion flips in at least four folds.  If any condition fails, it is evidence that action identity is helping the actual manner decision and this route should be killed.

## 4. Motion-aware masked skeleton pretraining, then a small relative head

**External basis.** [MAMP](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html) predicts masked joints' *temporal motion*, not merely their coordinates; [official code](https://github.com/maoyunyao/MAMP) is configurable in joint count and frame length.  [MotionBERT](https://motionbert.github.io/) uses a dual-stream spatio-temporal Transformer pretrained to recover 3-D motion from noisy/partial observations; its [official code](https://github.com/Walter0807/MotionBERT) has a 17-joint `DSTformer` interface.

**Mechanism.** Pretrain an MAMP-like encoder on the 3,713 available raw *training* HAU+HARn skeleton sequences without manner labels, then freeze most layers and train only a small group/pair head on HAU Emotion.  Use 96 or 128 frames (divisible by the four-frame patch size), `num_joints=17`, root-relative coordinates, velocity prediction, and motion-aware masking.  Feed original duration and unresampled velocity-energy summaries beside the embedding so time resampling does not erase pace.

**Compatibility and caveat.** MAMP's reference defaults are 25 joints/120 frames, but its implementation exposes both values; therefore train a local 17-joint version rather than attempting to load incompatible 25-joint positional embeddings.  MotionBERT is structurally closer to the local 17-joint layout, but external pretrained weights may assume different camera geometry, joint scale, and coordinate normalization.  Treat MotionBERT as a frozen-feature ablation, not assumed free performance.

**Small honest probe.** First compare: random small encoder, MAMP-pretrained small encoder, and frozen MotionBERT + linear head.  Train on four subject folds, score group accuracy and the final constrained assignment on the fifth.  The pretraining corpus must exclude held-out subjects for a strict subject-disjoint representation claim; an all-subject unlabeled pretrain can be reported separately as transductive but must not be mixed with the primary result.

## 5. Native-rate video-motion encoder on Depth/IR — only after the cache-based probes

**External basis.** [VideoMAE](https://arxiv.org/abs/2203.12602) is a self-supervised video masked-autoencoder framework; [official code/checkpoints](https://github.com/MCG-NJU/VideoMAE) support frozen-video feature extraction.  Its relevance is temporal visual representation, not its action-classification benchmark score.

**Mechanism.** Extract native-rate 16-frame Depth and IR snippets, replicate grayscale to three channels only for the pretrained input, pool snippet embeddings with a lightweight temporal attention layer, and retain number of snippets/true duration.  Compare clips within a session by embedding difference/cross-attention, not a global clip classifier.  This creates a motion-sensitive visual signal that static DINO frames cannot provide.

**Why this is lower priority.** A RGB-pretrained video model may not transfer to Depth/IR; a uniformly sampled fixed-length clip would also accidentally remove the speed signal.  It is materially more expensive than using `dino_frames.npz`, so start frozen and do not fine-tune a large backbone.

**Small honest probe.** Evaluate only a frozen 16-frame VideoMAE-S/ViT-style embedding plus logistic/pair head, against DINO-trajectory proposal 2 on the same subject folds.  For the 13 no-LMT HARn test clips, use a distinct wearable-removal training protocol and compare to the exact fallback; do not borrow ordinary OOF rows as evidence for that regime.  Kill this route if its matched missing-wearable flip precision remains below 0.80.

## Evaluation contract for every candidate

1. Fit every scaler, PCA, self-supervised encoder, nuisance target, and classifier using only training subjects in each fold.  Do not use true session/user IDs or answer-derived action pools after the hidden-answer boundary.
2. Run both normal subject-disjoint pseudo-test and the existing 38% pair-thinned protocol.  Split results into triple, two-clip, and singleton/repaired regimes.
3. Save OOF posterior/answer rows and compare directly with `champ/oof_final.csv`: overall and champion accuracy, disagreement count, W→R, R→W, flip precision, net/100 flips, per-fold values, confidence strata, and predicted real-test overrides.
4. A higher standalone group accuracy is insufficient.  Keep only a specialist with reproducible, selective champion wins (target at least 0.80 flip precision and positive, non-noise-floor net across folds).
5. Do not replace the constrained session assignment with independent clip argmax.  The relative representation should supply an additional unary/pairwise term to the established solver.

## Recommended order

1. Proposal 1 with DINO trajectory only, then pose/IMU late fusion.
2. Proposal 2 as a low-cost trajectory-only baseline/ablation.
3. Proposal 4 (local MAMP) only if raw pose contains signal beyond proposal 1.
4. Proposal 3 only if an encoder shows content leakage that hurts held-out transfer.
5. Proposal 5 only if cache-based probes fail or the no-LMT ablation is actively pursued.

This order avoids spending a long run on a large pretrained model before demonstrating that raw temporal information improves the relevant *relative* decision.
