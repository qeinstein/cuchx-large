"""Grandmaster VLM Oracle Engine.

Leverages Qwen-2.5-VL-72B-Instruct on OpenRouter to visually audit and resolve
all high-ambiguity test questions (Sequence ordering, fine-grained Emotion/Adverbs,
and unconstrained Multi-action sets) using sequential Depth video keyframes.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
import json
import os
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "hf_data_manual"
CACHE_FILE = ROOT / "vlm_predictions_cache.json"

API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct"


def extract_video_frames(rel_path: str, num_frames: int = 6) -> list:
    """Extract num_frames evenly spaced JPEG base64 images from depth video."""
    video_path = DATA_ROOT / rel_path
    if not video_path.exists():
        alt_path = DATA_ROOT / rel_path.replace("Depth/Depth.mp4", "Depth_Color/Depth_Color.mp4")
        if alt_path.exists():
            video_path = alt_path
        else:
            return []

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    b64_frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((336, 336))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            b64_frames.append(b64_str)
    cap.release()
    return b64_frames


def query_qwen_vl(qa_id: str, category: str, question: str, options: dict, b64_frames: list) -> str:
    """Query Qwen-2.5-VL-72B for a specific question."""
    if not b64_frames:
        return ""

    image_contents = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in b64_frames
    ]

    opt_text = "\n".join([f"{l}: {options[l]}" for l in "ABCD"])

    if category == "sequence":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show actions performed chronologically from first to last.\n"
            f"Question: {question}\n"
            f"{opt_text}\n\n"
            "Task: Arrange all 4 letters in exact chronological order from first performed to last performed (e.g. DBCA or ABCD).\n"
            "Reply strictly with only the 4 uppercase letters, nothing else."
        )
    elif category == "emotion":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n"
            f"{opt_text}\n\n"
            "Task: Determine the physical pace, manner, or emotion expressed by the person in these frames.\n"
            "Reply strictly with only the single uppercase letter of the correct option (A, B, C, or D), nothing else."
        )
    elif category == "multi":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n"
            f"{opt_text}\n\n"
            "Task: Identify ALL actions from the choices that actually appear in these frames.\n"
            "Reply strictly with only the uppercase letters of the correct actions with no spaces (e.g. AB or CD or BCD), nothing else."
        )
    elif category == "combination":
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n"
            f"{opt_text}\n\n"
            "Task: Determine which combination of actions is performed in these frames.\n"
            "Reply strictly with only the single uppercase letter of the correct option (A, B, C, or D), nothing else."
        )
    else:
        prompt_text = (
            f"These {len(b64_frames)} sequential depth camera frames show a person performing activities.\n"
            f"Question: {question}\n"
            f"{opt_text}\n\n"
            "Task: Answer the question based on what is visible in these frames.\n"
            "Reply strictly with only the single uppercase letter of the correct option (A, B, C, or D), nothing else."
        )

    content = [{"type": "text", "text": prompt_text}] + image_contents

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 12,
        "temperature": 0.05,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            ans = resp.json()["choices"][0]["message"]["content"].strip()
            # Clean up response to keep only valid letters A, B, C, D
            clean_ans = "".join([c for c in ans.upper() if c in "ABCD"])
            if category == "sequence" and len(clean_ans) == 4 and len(set(clean_ans)) == 4:
                return clean_ans
            elif category in ("emotion", "single", "combination", "object_interaction") and len(clean_ans) >= 1:
                return clean_ans[0]
            elif category == "multi" and len(clean_ans) >= 1:
                # Deduplicate while preserving order
                seen = set()
                dedup = "".join([c for c in clean_ans if not (c in seen or seen.add(c))])
                return "".join(sorted(dedup))
            return clean_ans
    except Exception as e:
        return ""
    return ""


def main():
    print(f"=== Running Grandmaster VLM Oracle ({MODEL_NAME}) ===")
    test_df = pd.read_csv(ROOT / "test_qa.csv")

    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print(f"Loaded existing cache with {len(cache)} entries.")
        except Exception:
            cache = {}

    # Target ALL categories across the entire test set (single, multi, combination, sequence, emotion, object_interaction)
    target_df = test_df.copy()
    print(f"Total test questions: {len(target_df)} across {len(target_df.path.unique())} clips.")

    # Filter out already cached
    pending = [r for _, r in target_df.iterrows() if r.qa_id not in cache]
    print(f"Pending queries to execute: {len(pending)}")

    # Pre-extract frames per unique video path to avoid duplicate I/O
    unique_paths = list({r.path for r in pending})
    print(f"Extracting video frames for {len(unique_paths)} unique videos...")
    path_frames = {}
    for p in tqdm(unique_paths, desc="Extracting Frames"):
        path_frames[p] = extract_video_frames(p, num_frames=6)

    def process_item(row):
        frames = path_frames.get(row.path, [])
        options = {l: row[l] for l in "ABCD"}
        ans = query_qwen_vl(row.qa_id, row.category, row.question, options, frames)
        return row.qa_id, ans

    # Execute queries with thread pool (5 concurrent workers)
    print(f"Querying Qwen-2.5-VL-72B concurrently (5 workers)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_item, r): r.qa_id for r in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="VLM Oracle"):
            qa_id, ans = future.result()
            if ans:
                cache[qa_id] = ans
                # Periodic checkpoint save every 10 queries
                if len(cache) % 10 == 0:
                    with open(CACHE_FILE, "w") as f:
                        json.dump(cache, f, indent=2)

    # Final save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\nOracle complete! Total cached answers: {len(cache)}")


if __name__ == "__main__":
    main()
