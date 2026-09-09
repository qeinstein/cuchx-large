# End-to-end video-bundle challenger

This experiment is intentionally independent of the 332 prediction vector. It fine-tunes a
video-language model on complete clip-level QA bundles and validates on subjects excluded from
all training. The first run is a fold-0 feasibility screen; it does not create a competition
submission.

Promotion gates before five-fold training:

1. At least 0.88 overall held-out HAU accuracy on fold 0.
2. At least 0.75 on both emotion and sequence, the two categories where the zero-shot VLM
   failed most severely.
3. Parse rate at least 0.995.
4. If the standalone model clears those gates, train all five folds and add a session-level
   decoder over its answer likelihoods. If it does not, inspect category failures before
   increasing model size; do not generate a submission.

The final architecture, if the screen passes, is Qwen video likelihoods plus raw
skeleton/IMU/radar temporal encoders feeding a trainable session factor graph. The scored 332
champion is used only as a comparator and fallback benchmark, never as a training label.
