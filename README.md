# CUHK-X Large Model Track

This repository contains the solution pipeline for the CUHK-X Large Model Track competition on Kaggle.

## Architecture
- Direct Vision-Language Model (VLM) baseline using LLaVA-1.5-7B in 4-bit precision.
- Modality-agnostic temporal sparse sampling (extracting 4 evenly spaced RGB frames).
- Category-conditioned instruction tuning for zero-shot inference.
- Robust cross-subject validation splitting to prevent data leakage.
- Automated fallback and submission generation module.

## Setup
1. Create a virtual environment and install dependencies: `pip install -r requirements.txt` (or install manually).
2. Download the Kaggle dataset to `hf_data_manual/`.
3. Run `validation.py` to generate the validation splits.
4. Run `vlm_pipeline.py` to execute the baseline on the local validation split.

## Structure
- `validation.py`: Generates rigorous cross-subject validation folds.
- `vlm_pipeline.py`: The core inference loop.
- `submission_builder.py`: Validates and structures VLM output for Kaggle submission.
- `baseline_majority.py`: A simple rule-based baseline calculating the category mode.
