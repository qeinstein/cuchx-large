# VideoMAE structured emotion challenger

Validation-only fold-0 screen of a task-aligned Kinetics-pretrained VideoMAE backbone.  The
last two transformer blocks are fine-tuned jointly for broad manner-group recognition,
within-session ordinal speed, and exact candidate-permutation likelihood.  Full and
pair-thinned predictions are emitted for strict decision-level comparison; the kernel has no
competition submission code.
