# Competition Solution Map: CUHK-X Large Model Track

This document details the reverse engineering of top public Kaggle submissions, tracking their models, preprocessing techniques, and overall strategies to reach high scores.

## 1. LLava-1.5 7B Baseline (VLM)
- **Model**: `llava-hf/llava-1.5-7b-hf` loaded in 4-bit precision via `bitsandbytes`.
- **Modality Handling**: It treats the sensor videos (Depth, Thermal, etc.) as standard `.mp4` visual sequences. 
- **Preprocessing (Frame Extraction)**: Extracts exactly **4 evenly spaced frames** from the video using OpenCV (`cv2.VideoCapture`). Converts BGR to RGB. 
- **Prompt Engineering**: 
  - Passes all 4 frames directly into the VLM: `<image>\n<image>\n<image>\n<image>\nBased on these sequential frames from a video...`
  - Strongly conditions the prompt on the question category:
    - `single`, `emotion`, `object_interaction`, `combination`: "Select the single correct option. Reply strictly with only one letter (e.g., A, B, C, or D)."
    - `multi`: "Select all correct options. Reply strictly with the letters combined (e.g., AB, ACD)."
    - `sequence`: "Order the actions chronologically. Reply strictly with the sequence of letters (e.g., DBCA)."

## 2. Category-Mode Baseline
- **Model**: None (Rule-based).
- **Strategy**: Groups the training set by `category` and calculates the mode (most frequent answer) for each. Predicts the mode for the test set based on its category.
- **Outcome**: We have independently reproduced this as Baseline B and submitted it to the leaderboard to establish our floor.

## Insights for our Pipeline (Phase 5 Prep)
- **Actionable Takeaway 1 (Temporal Sparsity)**: A 4-frame temporal sampling strategy is sufficient for a baseline VLM approach. We don't need to feed 30fps video into the model.
- **Actionable Takeaway 2 (Modality Conversion)**: Standard VLMs (like LLaVA or Qwen-VL) can process Depth/Thermal/IR mapped directly to RGB matrices.
- **Actionable Takeaway 3 (Strict Prompting)**: Instruction-tuning the final prompt based on the specific `category` (e.g. sequence vs multi-choice) is critical for format compliance.
