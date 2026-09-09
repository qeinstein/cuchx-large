# Exact-segment VideoMAE action timeline

Validation-only fold-0 screen.  A Kinetics-pretrained VideoMAE is fine-tuned on exact HARn
intervals cropped from their HAU parent videos plus background windows.  It then produces a
sliding action timeline for held-subject videos and scores all sequence permutations with
CTC, mapping every non-option action into blank.
