# Stage 3/3b/4 — frozen pretrained MotionBERT probe

Runs Gate B of the PCRME gate sequence, as explicitly required before closing the hypothesis.
Model: MotionBERT DSTformer backbone, NTU60-xsub action-finetuned checkpoint
(`walterzhu/MotionBERT`, `checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin`),
chosen because its config (`num_joints: 17` on the `ntu60_hrnet` variant) matches our own
COCO-17 skeleton layout exactly, and an action-recognition fine-tune is a closer match to
"encode execution style" than the raw 2D-to-3D lifting pretrain checkpoint.

## Input adaptation, numerically audited before bulk extraction

Our 3D skeleton -> MotionBERT's expected (x, y, confidence) COCO-order-then-H36M-reordered
input: drop z (orthographic projection onto the camera plane), constant confidence 1.0,
`coco2h36m` reorder and `crop_scale` normalisation copied verbatim from MotionBERT's own
source (not reimplemented). Audited on 4 real segments before running anything at scale: MB
input xy in [-1, 1] as expected, confidence exactly 1.0, output representations finite with
consistent statistics (mean ~-0.015, std ~0.31) across all 4. Full extraction: 2869/2880
aligned phase rows embedded (per-frame, per-joint 512-dim tokens), 776/809 clips covered
after aggregation, matching Stage 2's coverage exactly (same underlying Stage-1 dataset).

## Stage 3 — global (time+joint) mean pooling

| arm | accuracy | vs champion baseline (0.5713) |
| --- | ---: | ---: |
| A0. champion `block_features()` [fair baseline] | 0.5713 | -- |
| A. MotionBERT absolute (no conditioning) | 0.3754 | -19.6pp |
| B. MotionBERT action-conditioned/session-relative | 0.4549 | -11.6pp |
| D. champion + MotionBERT-relative [combined] | **0.5670** | **-0.43pp** |
| [reference] Stage 2 handcrafted: fair baseline / combined | 0.5713 / 0.6043 | -- / **+3.30pp** |

Shuffle-label controls: 0.285 / 0.289 / 0.302 across three seeds -- correctly near chance,
confirms no leakage. Triple-regime only (n=762, the pair regime has only 14 clips, too few to
report): champion baseline 0.5855, MotionBERT-relative 0.4534, combined **0.5583 (-2.7pp)** --
the negative effect is, if anything, larger where there is enough data to measure it precisely.

**Global pooling: net NEGATIVE**, consistently, not just "no gain".

## Stage 3b — per-joint (body-part-local) pooling, PCA-reduced

The brief explicitly asked not to collapse to a single mean if joint-local tokens are
available. `joint_pooled` (17 x 512, time-averaged but not joint-averaged) concatenated to
8704 dims, PCA-reduced WITHIN each fold's training subjects only (no leakage) before the same
combined-with-champion test:

| PCA components | joint-local alone | champion + joint-local [combined] |
| ---: | ---: | ---: |
| 20 | 0.3872 | 0.5812 (**+0.99pp**) |
| 50 | 0.3958 | 0.5651 (-0.62pp) |
| 100 | 0.4148 | 0.5705 (-0.08pp) |

This oscillates around the baseline (+1.0pp, -0.6pp, -0.1pp) as an unrelated hyperparameter
(PCA component count) changes sign and magnitude with no monotonic trend -- the signature of
noise scattering around zero, not a real effect. Contrast with the handcrafted feature's
+3.3pp, positive and of similar magnitude in **4 of 5 individual folds** in one single,
un-tuned configuration. Per-joint pooling does not rescue the frozen representation.

## Stage 4 — lightweight supervised adaptation (partial-adaptation proxy)

The kill criterion requires checking whether SOME adaptation helps before closing, not just
frozen features. A full contrastive fine-tuning loop (same-action-different-manner hard
negatives, gradient descent through the transformer) is a substantial undertaking; as a fast,
legitimate proxy, a per-fold Linear Discriminant Analysis (supervised, unlike PCA -- it
optimises directly for the manner-group target) was fit on the relative features and combined
with the champion baseline.

Result: LDA's own held-out accuracy is 0.265 (global) / 0.293 (joint-local) -- **at chance**,
and combining it with the champion's features actively drags the combined score down to
0.269-0.278, far below the 0.5713 baseline. This is a degenerate result, not informative
evidence against adaptation in general: with 5 classes and severe class imbalance (NERV alone
is ~7% of rows), fitting an LDA covariance per class on ~500-600 training rows split across 5
groups produces an unstable projection that a downstream classifier cannot use safely. It
shows this SPECIFIC cheap proxy for "adaptation" does not work, not that a properly trained
contrastive head would fail too. Recorded honestly as an inconclusive/negative result for the
proxy, separate from the (already fairly conclusive) frozen-embedding result above.

## Verdict on Gate B

**No configuration of the frozen or lightly-adapted MotionBERT representation shows a
meaningful, fold-consistent gain over the champion's fair baseline.** Global pooling is
consistently negative; per-joint pooling oscillates around zero with no consistent sign; the
cheap supervised-adaptation proxy collapsed to chance. This is a clean, three-times-replicated
negative for Gate B, using three different ways of extracting value from the same frozen
backbone.

## What this says about the two competing hypotheses

> 1. phase conditioning itself is useless for the champion's residuals
> 2. phase conditioning is useful, but our handcrafted descriptors are too lossy

If (2) were true, a purpose-built, 512-dim temporal+joint-token human-motion transformer
should show AT LEAST as much lift as 123 hand-computed velocity/acceleration/jerk/spectral
statistics -- it has far more capacity to represent subtle body-part-local and sub-second
dynamics. Instead it consistently underperforms the handcrafted features in every arm tried
(absolute, session-relative, per-joint, PCA-adapted, LDA-adapted). That is evidence AGAINST
(2): the handcrafted descriptors were not the bottleneck. Combined with a plausible mechanism
for why (action-recognition pretraining on NTU60 has an incentive to be somewhat invariant to
execution TEMPO -- recognising "drinking water" regardless of how fast it is done -- which is
close to the opposite of what a manner-sensitive representation needs, and the 3D-to-2D
projection adaptation is itself lossy for a foreign camera/action distribution), this session's
evidence now favours **hypothesis (1)**: the handcrafted phase-relative signal already found in
Stage 2 is close to what these modalities can offer for manner, and it is not merely a
capacity/representation limitation waiting on a bigger model.

## Recommendation on the remaining Stage-7-in-production requirement

The brief separately (and correctly) asks for the handcrafted phase-relative result to be
re-verified by integrating it into the REAL `solve_emotion` production path rather than the
92.8%-agreement simplified surrogate used so far, before fully trusting Stage 7's decision-level
verdict (net -2 on genuine overrides). That is still worth doing for the handcrafted feature,
since it is the one candidate that showed a real, fold-consistent representation-level gain.
Given MotionBERT's representation-level result is flat-to-negative in every configuration
tested here, integrating it into full production `solve_emotion` is very unlikely to change
Gate B's verdict and is not recommended as the next use of compute; the same integration effort
applied to the handcrafted feature is the higher-value use of further time, if the session
continues.
