"""Fold-0 screening run for a genuinely end-to-end video-QA challenger.

The unit of training is a complete clip bundle, not an isolated question.  A single video
is shown with every question attached to it, and Qwen2.5-VL is QLoRA-tuned to emit all exact
answer strings jointly.  Subjects are held out exactly as in the production pseudo-test
protocol.  This script never reads leaderboard labels and never creates a Kaggle submission.
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


gpu_name = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    check=False,
    capture_output=True,
    text=True,
).stdout.strip()

# Kaggle can assign a Pascal P100.  Its current default torch wheel has dropped sm_60,
# so install the last CUDA 12.1 wheel that still executes on Pascal before importing torch.
if "P100" in gpu_name:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch==2.5.1",
            "torchvision==0.20.1",
        ]
    )

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.51.3",
        "accelerate>=1.2.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "qwen-vl-utils[decord]>=0.0.8",
    ]
)

import cv2
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


SEED = 20260909
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
N_FRAMES = 12
MAX_SIDE = 224
EPOCHS = 2
GRAD_ACCUM = 8
LR = 1.5e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_input() -> Path:
    roots = list(Path("/kaggle/input").glob("cuchx-hau-video-training-corpus*"))
    if not roots:
        raise FileNotFoundError("cuchx-hau-video-training-corpus is not mounted")
    roots.sort()
    return roots[0]


DATA = find_input()
OUT = Path("/kaggle/working")
archive_path = DATA / "hau_videos.tar"
mounted_video_root = DATA / "hau_videos"
if mounted_video_root.is_dir():
    # Kaggle expands uploaded tar archives into their member paths.
    VIDEO_ROOT = mounted_video_root
else:
    VIDEO_ROOT = OUT / "hau_videos"
    VIDEO_ROOT.mkdir(exist_ok=True)
if not (VIDEO_ROOT / "hau__user1__1-1-1.mp4").exists() and archive_path.exists():
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (VIDEO_ROOT / member.name).resolve()
            if VIDEO_ROOT.resolve() not in target.parents:
                raise RuntimeError(f"unsafe archive member {member.name}")
        archive.extractall(VIDEO_ROOT)
if not (VIDEO_ROOT / "hau__user1__1-1-1.mp4").exists():
    raise FileNotFoundError(f"HAU videos not found under {DATA}")


def user_of(path: str) -> str | None:
    m = re.search(r"HAU/(user\d+)/", path)
    return m.group(1) if m else None


def folds(users: list[str], n: int = 5, seed: int = 7) -> list[list[str]]:
    users = sorted(users, key=lambda value: int(value[4:]))
    rng = np.random.default_rng(seed)
    users = list(rng.permutation(users))
    return [users[i::n] for i in range(n)]


def train_video(path: str) -> Path:
    m = re.fullmatch(r"HAU/(user\d+)/(\d+-\d+-\d+)", path)
    if not m:
        raise ValueError(path)
    return VIDEO_ROOT / f"hau__{m.group(1)}__{m.group(2)}.mp4"


def sample_frames(path: Path, n: int = N_FRAMES) -> tuple[list[Image.Image], float]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read {path}")
    indices = np.linspace(0, total - 1, min(n, total)).round().astype(int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded zero frames from {path}")
    # Qwen's temporal mRoPE consumes the supplied frame rate.  Our frames span the complete
    # recording, so use the effective sampling rate rather than pretending every clip is 12 s.
    if len(frames) % 2:
        frames.append(frames[-1].copy())
    duration = (total - 1) / native_fps if native_fps > 0 and total > 1 else len(frames) - 1
    effective_fps = (len(frames) - 1) / max(duration, 1e-3)
    return frames, float(effective_fps)


def option_text(row: pd.Series) -> str:
    values = []
    for letter in "ABCD":
        value = row.get(letter)
        if pd.notna(value) and str(value).strip():
            values.append(f"{letter}) {str(value).strip()}")
    return "  ".join(values)


SYSTEM = (
    "You solve privacy-preserving video multiple-choice questions. The video is a temporal "
    "Depth_Color recording. Answer every listed question jointly. For single, emotion, "
    "object_interaction and combination return one option letter. For multi return every "
    "correct letter in alphabetical order. For sequence return all four letters in temporal "
    "order. Output only lines of the form Q1=A, Q2=BC."
)


def prompt_for(rows: pd.DataFrame) -> str:
    chunks = []
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        chunks.append(
            f"Q{i} [{row.category}] {str(row.question).strip()}\n{option_text(row)}"
        )
    return "\n\n".join(chunks)


def target_for(rows: pd.DataFrame) -> str:
    return "\n".join(f"Q{i}={row.answer}" for i, (_, row) in enumerate(rows.iterrows(), 1))


def messages_for(rows: pd.DataFrame, frames: list[Image.Image], include_answer: bool):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 1.0},
                {"type": "text", "text": prompt_for(rows)},
            ],
        },
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_for(rows)}]}
        )
    return messages


def processor_inputs(processor, rows: pd.DataFrame, frames, effective_fps: float, training: bool):
    prompt_messages = messages_for(rows, frames, False)
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt = processor(
        text=[prompt_text], videos=[frames], fps=[effective_fps], padding=True,
        return_tensors="pt"
    )
    if not training:
        return prompt
    full_messages = messages_for(rows, frames, True)
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full = processor(
        text=[full_text], videos=[frames], fps=[effective_fps], padding=True,
        return_tensors="pt"
    )
    prefix = prompt.input_ids.shape[1]
    if full.input_ids.shape[1] <= prefix:
        raise RuntimeError("assistant target was not appended by the chat template")
    labels = full.input_ids.clone()
    labels[:, :prefix] = -100
    full["labels"] = labels
    return full


def to_device(batch):
    return {key: value.to("cuda") if torch.is_tensor(value) else value for key, value in batch.items()}


def parse_answers(text: str, n: int) -> dict[int, str]:
    found = {}
    for idx, answer in re.findall(r"Q\s*(\d+)\s*=\s*([A-D]+)", text.upper()):
        i = int(idx)
        if 1 <= i <= n:
            found[i] = answer
    return found


qa = pd.read_csv(DATA / "training_qa.csv")
qa = qa[qa.source == "HAU"].copy()
qa["user"] = qa.path.map(user_of)
users = sorted(qa.user.dropna().unique())
hold = set(folds(users)[0])
train_qa = qa[~qa.user.isin(hold)]
valid_qa = qa[qa.user.isin(hold)]
train_groups = [(path, group.sort_values("qa_id")) for path, group in train_qa.groupby("path")]
valid_groups = [(path, group.sort_values("qa_id")) for path, group in valid_qa.groupby("path")]
random.Random(SEED).shuffle(train_groups)
print(json.dumps({
    "data": str(DATA),
    "train_users": sorted(set(users) - hold),
    "hold_users": sorted(hold),
    "train_clips": len(train_groups),
    "valid_clips": len(valid_groups),
    "train_questions": len(train_qa),
    "valid_questions": len(valid_qa),
}, indent=2), flush=True)

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
)
model.print_trainable_parameters()
model.config.use_cache = False
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

step = 0
optimizer.zero_grad(set_to_none=True)
for epoch in range(EPOCHS):
    random.Random(SEED + epoch).shuffle(train_groups)
    running = []
    model.train()
    for item, (path, rows) in enumerate(train_groups, 1):
        frames, effective_fps = sample_frames(train_video(path))
        batch = to_device(processor_inputs(processor, rows, frames, effective_fps, True))
        loss = model(**batch).loss / GRAD_ACCUM
        loss.backward()
        running.append(float(loss.detach().cpu()) * GRAD_ACCUM)
        if item % GRAD_ACCUM == 0 or item == len(train_groups):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        if item % 25 == 0:
            print(
                f"epoch={epoch} clip={item}/{len(train_groups)} step={step} "
                f"loss={np.mean(running[-25:]):.4f}",
                flush=True,
            )
        del batch, loss, frames
        if item % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

adapter_dir = OUT / "qwen_fold0_adapter"
model.save_pretrained(adapter_dir)
processor.save_pretrained(adapter_dir)

model.eval()
model.config.use_cache = True
pred_rows = []
with torch.inference_mode():
    for item, (path, rows) in enumerate(valid_groups, 1):
        frames, effective_fps = sample_frames(train_video(path))
        batch = to_device(processor_inputs(processor, rows, frames, effective_fps, False))
        generated = model.generate(
            **batch,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
        )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        parsed = parse_answers(text, len(rows))
        for i, (_, row) in enumerate(rows.iterrows(), 1):
            pred = parsed.get(i, "")
            pred_rows.append(
                {
                    "qa_id": row.qa_id,
                    "path": path,
                    "user": row.user,
                    "category": row.category,
                    "truth": row.answer,
                    "prediction": pred,
                    "correct": int(pred == row.answer),
                    "parsed": int(i in parsed),
                    "raw_output": text,
                }
            )
        if item % 10 == 0:
            done = pd.DataFrame(pred_rows)
            print(
                f"valid={item}/{len(valid_groups)} acc={done.correct.mean():.4f} "
                f"parse={done.parsed.mean():.4f}",
                flush=True,
            )
        del batch, generated, new_tokens, frames

pred = pd.DataFrame(pred_rows)
pred.to_csv(OUT / "fold0_predictions.csv", index=False)
summary = {
    "model": MODEL_ID,
    "fold": 0,
    "hold_users": sorted(hold),
    "frames": N_FRAMES,
    "epochs": EPOCHS,
    "questions": len(pred),
    "correct": int(pred.correct.sum()),
    "accuracy": float(pred.correct.mean()),
    "parse_rate": float(pred.parsed.mean()),
    "per_category": {
        category: {
            "n": len(group),
            "correct": int(group.correct.sum()),
            "accuracy": float(group.correct.mean()),
        }
        for category, group in pred.groupby("category")
    },
}
(OUT / "fold0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
