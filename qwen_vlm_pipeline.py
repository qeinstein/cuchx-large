"""Qwen2.5-VL-3B VLM Pipeline for CUHK-X Large Model Track.

Uses mlx-vlm for native Apple Silicon inference on Depth/Thermal videos.
Processes each test question by extracting 8 keyframes and prompting
the VLM with category-specific instructions.
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "hf_data_manual"

MODEL_PATH = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced frames from a video file."""
    if not os.path.exists(video_path):
        return [Image.new("RGB", (224, 224), color="black")] * num_frames

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return [Image.new("RGB", (224, 224), color="black")] * num_frames

    indices = np.linspace(0, max(0, total - 1), num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(Image.new("RGB", (224, 224), color="black"))
    cap.release()
    return frames


def build_prompt(row):
    """Build a category-specific VLM prompt for a question."""
    question = row["question"]
    category = row["category"]

    options = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}"
    if pd.notna(row.get("D")) and str(row["D"]).strip():
        options += f"\nD) {row['D']}"

    if category == "single":
        instruction = (
            "You are analyzing depth camera video frames showing a person performing daily activities. "
            "Based on the visual evidence in these frames, identify the SINGLE action the person is performing. "
            "Answer with ONLY the letter of the correct option (A, B, C, or D). Nothing else."
        )
    elif category == "combination":
        instruction = (
            "You are analyzing depth camera video frames showing a person performing multiple daily activities. "
            "Based on the visual evidence, identify which combination of actions the person performs in the video. "
            "Answer with ONLY the letter of the correct option (A, B, C, or D). Nothing else."
        )
    elif category == "multi":
        instruction = (
            "You are analyzing depth camera video frames showing a person performing multiple daily activities. "
            "Select ALL actions that the person performs in this video. "
            "Answer with ONLY the letters of all correct options combined (e.g., AB, ACD, ABCD). Nothing else."
        )
    elif category == "sequence":
        instruction = (
            "You are analyzing depth camera video frames showing a person performing 4 actions in sequence. "
            "Based on the temporal order visible in the frames (from earliest to latest), "
            "arrange these actions in chronological order. "
            "Answer with ONLY the letters in the correct temporal order (e.g., DBCA). Nothing else."
        )
    elif category == "emotion":
        instruction = (
            "You are analyzing depth camera video frames showing a person performing daily activities. "
            "Based on the person's movement speed, body language, and physical rhythm, "
            "determine how the person is performing the actions (their manner/emotion). "
            "Answer with ONLY the letter of the correct option (A, B, C, or D). Nothing else."
        )
    elif category == "object_interaction":
        instruction = (
            "You are analyzing depth camera video frames showing a person interacting with objects. "
            "Based on the visual evidence, identify which object the person is interacting with. "
            "Answer with ONLY the letter of the correct option (A, B, or C). Nothing else."
        )
    else:
        instruction = "Answer with ONLY the correct letter(s). Nothing else."

    prompt = f"{instruction}\n\nQuestion: {question}\n\nOptions:\n{options}"
    return prompt


def get_video_path(clip_path):
    """Find the best available video for a clip."""
    base = DATA_DIR / clip_path
    # Prefer Depth_Color (RGB-like), then Depth, then Thermal
    for subdir in ["Depth_Color/Depth_Color.mp4", "Depth/Depth.mp4", "Thermal/Thermal.mp4"]:
        p = base / subdir
        if p.exists():
            return str(p)
    return str(base / "Depth/Depth.mp4")


def clean_answer(raw, category):
    """Extract valid answer letters from VLM output."""
    # Extract only valid letters
    valid = "".join(c for c in raw.upper() if c in "ABCD")
    if not valid:
        return "A"

    if category in ["single", "emotion", "object_interaction", "combination"]:
        # Single letter answer
        return valid[0]
    elif category == "multi":
        # Multiple unique sorted letters
        return "".join(sorted(set(valid)))
    elif category == "sequence":
        # Exactly 4 unique letters in order
        seen = []
        for c in valid:
            if c not in seen:
                seen.append(c)
        if len(seen) == 4:
            return "".join(seen)
        # Pad missing letters
        for c in "ABCD":
            if c not in seen:
                seen.append(c)
        return "".join(seen[:4])
    return valid


def run_vlm_on_test(output_path=None, limit=None):
    """Run Qwen2.5-VL on the full test set."""
    test_df = pd.read_csv(ROOT / "test_qa.csv").drop(columns=["prediction"], errors="ignore")

    print(f"Loading {MODEL_PATH}...")
    model, processor = load(MODEL_PATH)
    print("Model loaded successfully!")

    config = model.config if hasattr(model, 'config') else {}

    predictions = {}
    total = len(test_df) if limit is None else min(limit, len(test_df))

    t0 = time.time()
    for i, (idx, row) in enumerate(test_df.iterrows()):
        if limit and i >= limit:
            break

        video_path = get_video_path(row["path"])
        frames = extract_frames(video_path, num_frames=8)
        prompt_text = build_prompt(row)

        # Save frames as temp files for mlx-vlm
        temp_paths = []
        for j, frame in enumerate(frames):
            tp = f"/tmp/vlm_frame_{j}.jpg"
            frame.save(tp)
            temp_paths.append(tp)

        # Format for mlx-vlm chat template
        formatted_prompt = apply_chat_template(
            processor,
            config,
            prompt_text,
            images=temp_paths,
        )

        try:
            output = generate(
                model,
                processor,
                formatted_prompt,
                images=temp_paths,
                max_tokens=20,
                temp=0.1,
                verbose=False,
            )
            answer = clean_answer(output, row["category"])
        except Exception as e:
            print(f"  Error on {row['qa_id']}: {e}")
            answer = "A"

        predictions[row["qa_id"]] = answer

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            qps = (i + 1) / elapsed
            eta = (total - i - 1) / qps if qps > 0 else 0
            print(f"  [{i+1}/{total}] {row['qa_id']} ({row['category']}): {answer} | {qps:.2f} q/s | ETA: {eta/60:.1f}min")

        # Clean up temp files
        for tp in temp_paths:
            try:
                os.remove(tp)
            except:
                pass

    elapsed = time.time() - t0
    print(f"\nCompleted {len(predictions)} predictions in {elapsed:.1f}s ({len(predictions)/elapsed:.2f} q/s)")

    # Build submission
    out_df = pd.DataFrame(
        {"qa_id": test_df["qa_id"], "prediction": test_df["qa_id"].map(predictions)}
    )
    # Fill any missing with "A"
    out_df["prediction"] = out_df["prediction"].fillna("A")

    if output_path is None:
        output_path = ROOT / "submission_vlm.csv"
    out_df.to_csv(output_path, index=False)
    print(f"Saved VLM submission to {output_path}")

    return predictions


def run_vlm_on_validation(fold=0, limit=None):
    """Run Qwen2.5-VL on a validation fold to measure accuracy."""
    val_df = pd.read_csv(ROOT / f"splits/fold_{fold}_val.csv")

    print(f"Loading {MODEL_PATH}...")
    model, processor = load(MODEL_PATH)
    print("Model loaded!")

    config = model.config if hasattr(model, 'config') else {}

    correct = 0
    total = 0
    cat_correct = {}
    cat_total = {}

    n = len(val_df) if limit is None else min(limit, len(val_df))

    t0 = time.time()
    for i, (idx, row) in enumerate(val_df.iterrows()):
        if limit and i >= limit:
            break

        video_path = get_video_path(row["path"])
        frames = extract_frames(video_path, num_frames=8)
        prompt_text = build_prompt(row)

        temp_paths = []
        for j, frame in enumerate(frames):
            tp = f"/tmp/vlm_frame_{j}.jpg"
            frame.save(tp)
            temp_paths.append(tp)

        formatted_prompt = apply_chat_template(
            processor,
            config,
            prompt_text,
            images=temp_paths,
        )

        try:
            output = generate(
                model,
                processor,
                formatted_prompt,
                images=temp_paths,
                max_tokens=20,
                temp=0.1,
                verbose=False,
            )
            answer = clean_answer(output, row["category"])
        except Exception as e:
            answer = "A"

        cat = row["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1

        gt = str(row["answer"]).strip()
        if answer == gt:
            correct += 1
            cat_correct[cat] = cat_correct.get(cat, 0) + 1

        total += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}] Acc={correct/total:.4f} | {(i+1)/elapsed:.2f} q/s")

        for tp in temp_paths:
            try:
                os.remove(tp)
            except:
                pass

    elapsed = time.time() - t0
    print(f"\n=== VLM VALIDATION RESULTS (Fold {fold}) ===")
    print(f"Overall: {correct}/{total} = {correct/total:.4f}")
    for cat in sorted(cat_total.keys()):
        cc = cat_correct.get(cat, 0)
        ct = cat_total[cat]
        print(f"  {cat:25s}: {cc}/{ct} = {cc/ct:.4f}")
    print(f"Speed: {total/elapsed:.2f} q/s ({elapsed:.1f}s total)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_vlm_on_validation(fold=0, limit=limit)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_vlm_on_test(limit=limit)
    else:
        # Quick test on 20 validation questions
        run_vlm_on_validation(fold=0, limit=20)
