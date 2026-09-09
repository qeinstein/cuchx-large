"""Subject-disjoint Qwen3-VL fine-tuning on complete HAU session bundles.

Unlike an isolated video-QA model, one example contains every visible sibling clip and every
question in that session.  Random clip withholding during training matches the real test's
two-clip regime.  This script only produces validation artifacts; it cannot submit.
"""

from __future__ import annotations

import gc
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


SEED = 20260909
FOLD = 0
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
N_FRAMES = 12
MAX_SIDE = 224
EPOCHS = 5
GRAD_ACCUM = 4
LR = 1.2e-4
PAIR_AUGMENT_PROB = 0.45
VALID_PAIR_FRAC = 0.38


gpu_name = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    check=False,
    capture_output=True,
    text=True,
).stdout.strip()
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
        "transformers>=4.57.0,<5",
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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_input() -> Path:
    roots = sorted(Path("/kaggle/input").glob("cuchx-hau-video-training-corpus*"))
    if roots:
        return roots[0]
    # Dataset mount names occasionally receive a version suffix.  Resolve from the unique
    # supervision file instead of coupling the run to Kaggle's presentation slug.
    qa_files = sorted(Path("/kaggle/input").glob("**/training_qa.csv"))
    if len(qa_files) == 1:
        return qa_files[0].parent
    mounted = [str(path) for path in sorted(Path("/kaggle/input").iterdir())]
    raise FileNotFoundError(
        f"training corpus is not mounted; /kaggle/input contains {mounted}"
    )


DATA = find_input()
OUT = Path("/kaggle/working")
VIDEO_ROOT = DATA / "hau_videos"
if not VIDEO_ROOT.is_dir():
    raise FileNotFoundError(f"expanded HAU videos not found under {DATA}")


def path_parts(path: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"HAU/(user\d+)/(\d+-\d+)-(\d+)", path)
    if not match:
        raise ValueError(path)
    return match.group(1), match.group(2), int(match.group(3))


def user_of(path: str) -> str:
    return path_parts(path)[0]


def session_of(path: str) -> str:
    user, session, _ = path_parts(path)
    return f"{user}/{session}"


def trial_of(path: str) -> int:
    return path_parts(path)[2]


def folds(users: list[str], n: int = 5, seed: int = 7) -> list[list[str]]:
    users = sorted(users, key=lambda value: int(value[4:]))
    users = list(np.random.default_rng(seed).permutation(users))
    return [users[i::n] for i in range(n)]


def video_path(path: str) -> Path:
    user, session, trial = path_parts(path)
    return VIDEO_ROOT / f"hau__{user}__{session}-{trial}.mp4"


def sample_frames(path: Path) -> tuple[list[Image.Image], dict]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read {path}")
    indices = np.linspace(0, total - 1, min(N_FRAMES, total)).round().astype(int)
    frames = []
    selected_indices = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(Image.fromarray(frame))
        selected_indices.append(int(index))
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded zero frames from {path}")
    if len(frames) % 2:
        frames.append(frames[-1].copy())
        selected_indices.append(selected_indices[-1])
    duration = (total - 1) / native_fps if native_fps > 0 and total > 1 else len(frames) - 1
    metadata = {
        "total_num_frames": total,
        "fps": native_fps if native_fps > 0 else (len(frames) - 1) / max(duration, 1e-3),
        "duration": duration,
        "frames_indices": selected_indices,
    }
    return frames, metadata


def option_text(row: pd.Series) -> str:
    values = []
    for letter in "ABCD":
        value = row.get(letter)
        if pd.notna(value) and str(value).strip():
            values.append(f"{letter}) {str(value).strip()}")
    return "  ".join(values)


SYSTEM = (
    "You compare how the same person performs related activities in sibling videos from one "
    "recording session. Assign one visible candidate manner to each video. Assignments are "
    "one-to-one: never reuse a candidate index. Return only the candidate indices in video "
    "order, separated by |, with no explanation."
)


def ordered_clips(rows: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    clips = []
    for path, group in rows.groupby("path"):
        clips.append((path, group.sort_values("qa_id")))
    clips.sort(key=lambda item: trial_of(item[0]))
    return clips


def candidate_manners(clips) -> list[str]:
    sets = []
    for _, rows in clips:
        row = rows.iloc[0]
        sets.append({str(row[letter]).strip() for letter in "ABCD"})
    common = set.intersection(*sets)
    if len(common) < len(clips):
        raise ValueError(f"candidate intersection too small: {len(common)} for {len(clips)}")
    return sorted(common)


def keys_for(clips: list[tuple[str, pd.DataFrame]]) -> list[tuple[str, pd.Series, str]]:
    keyed = []
    for video_index, (_, rows) in enumerate(clips, 1):
        for question_index, (_, row) in enumerate(rows.iterrows(), 1):
            keyed.append((f"V{video_index}Q{question_index}", row, row.qa_id))
    return keyed


def prompt_for(clips: list[tuple[str, pd.DataFrame]], candidates: list[str]) -> str:
    listed = "  ".join(f"{i}) {value}" for i, value in enumerate(candidates, 1))
    return (
        f"Candidate manners: {listed}\n"
        f"Assign exactly one candidate to each of Videos 1-{len(clips)}. "
        f"Return exactly {len(clips)} distinct indices."
    )


def target_for(clips: list[tuple[str, pd.DataFrame]], candidates: list[str]) -> str:
    answers = []
    for _, rows in clips:
        row = rows.iloc[0]
        truth = str(row[str(row.answer)]).strip()
        answers.append(str(candidates.index(truth) + 1))
    return "|".join(answers)


def messages_for(clips, frame_sets, candidates, include_answer: bool):
    content = []
    for video_index, frames in enumerate(frame_sets, 1):
        content.extend(
            [
                {"type": "text", "text": f"Video {video_index}:"},
                {"type": "video", "video": frames},
            ]
        )
    content.append({"type": "text", "text": prompt_for(clips, candidates)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": content},
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_for(clips, candidates)}]}
        )
    return messages


def processor_inputs(processor, clips, frame_sets, video_metadata, candidates, training: bool):
    prompt_messages = messages_for(clips, frame_sets, candidates, False)
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt = processor(
        text=[prompt_text],
        videos=frame_sets,
        video_metadata=video_metadata,
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    )
    if not training:
        return prompt
    full_messages = messages_for(clips, frame_sets, candidates, True)
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full = processor(
        text=[full_text],
        videos=frame_sets,
        video_metadata=video_metadata,
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    )
    prefix = prompt.input_ids.shape[1]
    if full.input_ids.shape[1] <= prefix:
        raise RuntimeError("assistant target was not appended")
    labels = full.input_ids.clone()
    labels[:, :prefix] = -100
    full["labels"] = labels
    return full


def load_media(clips):
    frame_sets = []
    video_metadata = []
    for path, _ in clips:
        frames, metadata = sample_frames(video_path(path))
        frame_sets.append(frames)
        video_metadata.append(metadata)
    return frame_sets, video_metadata


def to_device(batch):
    return {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def parse_answers(text: str, clips, candidates) -> dict[str, str]:
    expected_keys = [key for key, _, _ in keys_for(clips)]
    answers = [int(value) for value in re.findall(r"(?<!\d)([1-4])(?!\d)", text)]
    if len(answers) != len(expected_keys) or len(set(answers)) != len(answers):
        return {}
    if any(value > len(candidates) for value in answers):
        return {}
    return {key: candidates[value - 1] for key, value in zip(expected_keys, answers)}


qa = pd.read_csv(DATA / "training_qa.csv")
qa = qa[(qa.source == "HAU") & (qa.category == "emotion")].copy()
qa["user"] = qa.path.map(user_of)
qa["session"] = qa.path.map(session_of)
users = sorted(qa.user.unique())
hold = set(folds(users)[FOLD])


def valid_session_rows(group: pd.DataFrame) -> bool:
    clips = ordered_clips(group)
    try:
        candidates = set(candidate_manners(clips))
    except ValueError:
        return False
    truths = {
        str(rows.iloc[0][str(rows.iloc[0].answer)]).strip()
        for _, rows in clips
    }
    return truths <= candidates


train_sessions = [
    (session, group.copy())
    for session, group in qa[~qa.user.isin(hold)].groupby("session")
    if valid_session_rows(group)
]
valid_sessions = [
    (session, group.copy())
    for session, group in qa[qa.user.isin(hold)].groupby("session")
    if valid_session_rows(group)
]
print(
    json.dumps(
        {
            "gpu": gpu_name,
            "train_users": sorted(set(users) - hold),
            "hold_users": sorted(hold),
            "train_sessions": len(train_sessions),
            "valid_sessions": len(valid_sessions),
            "train_questions": int((~qa.user.isin(hold)).sum()),
            "valid_questions": int(qa.user.isin(hold).sum()),
        },
        indent=2,
    ),
    flush=True,
)

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant,
    device_map="auto",
    dtype=torch.float16,
)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    # Reentrant checkpointing drops gradients when a frozen visual block receives pixel
    # inputs with requires_grad=False.  Non-reentrant mode still computes gradients for the
    # LoRA parameters inside those blocks.
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    ),
)
model.print_trainable_parameters()
model.config.use_cache = False
optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad), lr=LR
)

step = 0
optimizer.zero_grad(set_to_none=True)
for epoch in range(EPOCHS):
    epoch_sessions = list(train_sessions)
    random.Random(SEED + epoch).shuffle(epoch_sessions)
    model.train()
    losses = []
    for item, (_, session_rows) in enumerate(epoch_sessions, 1):
        clips = ordered_clips(session_rows)
        aug_rng = random.Random(SEED + epoch * 10000 + item)
        if len(clips) == 3 and aug_rng.random() < PAIR_AUGMENT_PROB:
            del clips[aug_rng.randrange(3)]
        candidates = candidate_manners(clips)
        aug_rng.shuffle(candidates)
        frame_sets, video_metadata = load_media(clips)
        batch = to_device(
            processor_inputs(processor, clips, frame_sets, video_metadata, candidates, True)
        )
        loss = model(**batch).loss / GRAD_ACCUM
        loss.backward()
        losses.append(float(loss.detach().cpu()) * GRAD_ACCUM)
        if item % GRAD_ACCUM == 0 or item == len(epoch_sessions):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        if item % 10 == 0:
            print(
                f"epoch={epoch} session={item}/{len(epoch_sessions)} step={step} "
                f"loss={np.mean(losses[-10:]):.4f}",
                flush=True,
            )
        del batch, loss, frame_sets
        if item % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

adapter_dir = OUT / f"qwen3_emotion_assignment_fold{FOLD}_adapter"
model.save_pretrained(adapter_dir)
processor.save_pretrained(adapter_dir)
model.eval()
model.config.use_cache = True


def validation_view(session_rows: pd.DataFrame, session_index: int, regime: str):
    clips = ordered_clips(session_rows)
    if regime == "pair" and len(clips) == 3:
        rng = random.Random(SEED + 50000 + session_index)
        if rng.random() < VALID_PAIR_FRAC:
            del clips[rng.randrange(3)]
    return clips


pred_rows = []
with torch.inference_mode():
    for regime in ["full", "pair"]:
        for item, (session, session_rows) in enumerate(valid_sessions, 1):
            clips = validation_view(session_rows, item, regime)
            candidates = candidate_manners(clips)
            frame_sets, video_metadata = load_media(clips)
            batch = to_device(
                processor_inputs(processor, clips, frame_sets, video_metadata, candidates, False)
            )
            generated = model.generate(
                **batch,
                max_new_tokens=16,
                do_sample=False,
                use_cache=True,
            )
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            raw_output = processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            parsed = parse_answers(raw_output, clips, candidates)
            for key, row, qa_id in keys_for(clips):
                predicted_manner = parsed.get(key, "")
                prediction = next(
                    (
                        letter
                        for letter in "ABCD"
                        if str(row[letter]).strip() == predicted_manner
                    ),
                    "",
                )
                pred_rows.append(
                    {
                        "regime": regime,
                        "session": session,
                        "qa_id": qa_id,
                        "user": row.user,
                        "category": row.category,
                        "truth": row.answer,
                        "prediction": prediction,
                        "correct": int(prediction == row.answer),
                        "parsed": int(key in parsed),
                        "raw_output": raw_output,
                    }
                )
            if item % 5 == 0:
                done = pd.DataFrame(pred_rows)
                current = done[done.regime == regime]
                print(
                    f"regime={regime} valid={item}/{len(valid_sessions)} "
                    f"acc={current.correct.mean():.4f} parse={current.parsed.mean():.4f}",
                    flush=True,
                )
            del batch, generated, new_tokens, frame_sets

pred = pd.DataFrame(pred_rows)
pred.to_csv(OUT / f"fold{FOLD}_session_predictions.csv", index=False)
summary = {
    "model": MODEL_ID,
    "fold": FOLD,
    "hold_users": sorted(hold),
    "frames_per_video": N_FRAMES,
    "epochs": EPOCHS,
    "pair_augment_probability": PAIR_AUGMENT_PROB,
    "task": "one_to_one_candidate_manner_assignment",
    "regimes": {},
}
for regime, regime_rows in pred.groupby("regime"):
    summary["regimes"][regime] = {
        "questions": len(regime_rows),
        "correct": int(regime_rows.correct.sum()),
        "accuracy": float(regime_rows.correct.mean()),
        "parse_rate": float(regime_rows.parsed.mean()),
        "per_category": {
            category: {
                "n": len(group),
                "correct": int(group.correct.sum()),
                "accuracy": float(group.correct.mean()),
            }
            for category, group in regime_rows.groupby("category")
        },
    }
(OUT / f"fold{FOLD}_session_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
