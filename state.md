# CUHK-X Large Model Track: Project State

## Goal
Execute the Kaggle CUHK-X Large Model Track project autonomously to achieve a Top 15 placement on the Private Leaderboard. The project revolves around zero-shot multimodal action recognition using Vision Language Models (VLMs) on sensor data (Depth, IR, Thermal).

## Current Status & Achievements
- **Environment:** Initialized perfectly. Dependencies (`transformers`, `accelerate`, `opencv-python`, `pandas`, `tqdm`, `python-dotenv`) installed. Git is tracking cleanly with a robust `.gitignore`.
- **Data Acquisition (COMPLETED):** The massive 8GB multimodal video datasets were successfully downloaded and extracted into `hf_data_manual/`.
- **Model Acquisition (COMPLETED):** The 15GB `LLaVA-1.5-7B` model weights were fully downloaded to a local directory (`llava_local/`).
- **Validation Pipeline:** We created `validation.py` to ensure strictly leak-free cross-subject validation, generating `splits/fold_0_val.csv`.
- **Inference Pipeline:** We built `vlm_pipeline.py` which loads the local VLM, extracts 4 frames from the `Depth_Color` (RGB-aligned) videos, applies the specific Kaggle Category prompt, and outputs predictions.

## Current Blocker / Performance Bottleneck
We successfully launched `vlm_pipeline.py`, but hit a catastrophic performance bottleneck.
- **The Issue:** Generating predictions on macOS Apple Silicon (MPS) using the standard `transformers` library takes **~4 minutes per video**.
- **The Impact:** Evaluating the 913-sample validation fold takes **~62 hours**. This is far too slow for rapid Kaggle experimentation.
- **Context:** The original Kaggle solution relied on `bitsandbytes` 4-bit quantization, which is completely unsupported on macOS. We are currently running native `float16` weights on the MPS backend.

## Environment Quirks (CRITICAL FOR NEXT AGENT)
- **Networking Bug:** Do **NOT** use `huggingface_hub.snapshot_download()` or standard `curl` to download large files. It triggers a LibreSSL bad decrypt bug on this machine and hangs infinitely. 
- **The Fix:** Always use `aria2c` for downloads. We wrote `download_dataset.py` and `download_model.py` which dynamically generate `aria2c` input files (reading the token from `.env`) and download files with 8 parallel connections perfectly.
- **Security:** The Hugging Face token is stored in `.env` (ignored by git). Never hardcode it.

## Next Immediate Steps
1. **Micro-Batch Testing:** Modify `vlm_pipeline.py` to evaluate only 5 samples (`df.head(5)`) just to verify that the model correctly outputs the "A", "B", "C" multiple-choice format demanded by Kaggle.
2. **Inference Acceleration:** To solve the 62-hour bottleneck, completely refactor `vlm_pipeline.py` to utilize **Apple MLX** (`mlx-lm` / `mlx-vlm`). MLX is deeply optimized for Apple Silicon unified memory and will drastically accelerate LLaVA inference compared to the generic PyTorch/Transformers MPS backend.
3. **Modality Iteration:** Once inference is fast, start testing if swapping `Depth_Color` for raw `Depth` or `Thermal` frames improves the zero-shot accuracy score on the validation split.
