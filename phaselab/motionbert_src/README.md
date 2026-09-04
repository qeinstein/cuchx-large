# Vendored MotionBERT source (subset)

These files are copied verbatim from https://github.com/Walter0807/MotionBERT (accessed
2026-09-04) — specifically the model definition (`lib/model/DSTformer.py`, `drop.py`,
`model_action.py`) and the pure data-transform utilities (`lib/utils/utils_data.py`, `tools.py`,
`lib/data/dataset_action.py`, `dataset_wild.py`, `augmentation.py`) needed to load the
pretrained checkpoint and replicate its exact input normalisation (`coco2h36m`, `crop_scale`).

Not modified from upstream. Vendored rather than pip-installed because the upstream project is
not packaged for pip; only these files (not the full training/eval CLI) are needed for frozen
inference. See `phaselab/motionbert_extract.py` for how they are used and
`phaselab/RESULTS_stage3_motionbert.md` for what was found.

The pretrained checkpoint itself (`checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch
.bin`, from `walterzhu/MotionBERT` on Hugging Face) is not committed here (240MB); it is
downloaded by a one-off script and gitignored (`phaselab/motionbert_weights/`).
