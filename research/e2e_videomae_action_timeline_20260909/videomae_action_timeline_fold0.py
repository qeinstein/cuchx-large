"""Exact-segment VideoMAE action localizer and CTC sequence decoder.

The model is trained on the 2,927 HARn intervals nested inside HAU parent videos, rather than
using noisy whole-video labels.  At validation it slides over a complete held-subject HAU
video, converts non-option actions to CTC blank, and scores every 4! answer permutation.
This kernel only emits validation artifacts and contains no submission API.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Kaggle's Python 3.12 image can ship a torch build that has dropped sm_60 even when
# the assigned accelerator is a P100.  Install the last known-compatible cu121 wheel
# before importing either torch or transformers.  Other GPU types keep the image build.
gpu_name = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip()
if "P100" in gpu_name:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torch==2.5.1", "torchvision==0.20.1",
    ])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.48.0,<5", "accelerate>=1.2.0", "opencv-python-headless",
])

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


SEED = 20260909
FOLD = 0
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
N_FRAMES = 16
EPOCHS = 3
BATCH = 2
GRAD_ACCUM = 8
LR = 2.0e-5
HEAD_LR = 2.0e-4
UNFREEZE_BLOCKS = 2
N_WINDOWS = 24
WINDOW_SECONDS = 3.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_input() -> Path:
    roots = sorted(Path("/kaggle/input").glob("cuchx-hau-video-training-corpus*"))
    if roots:
        return roots[0]
    meta = sorted(Path("/kaggle/input").glob("**/meta.csv"))
    if len(meta) == 1:
        return meta[0].parent
    raise FileNotFoundError(f"cannot identify HAU corpus: {meta}")


DATA = find_input()
VIDEO_ROOT = DATA / "hau_videos"


def folds(users: list[str], n: int = 5, seed: int = 7) -> list[list[str]]:
    ordered = sorted(users, key=lambda value: int(value[4:]))
    shuffled = list(np.random.default_rng(seed).permutation(ordered))
    return [shuffled[index::n] for index in range(n)]


def flattened_video(user: str, trial: str) -> Path:
    result = VIDEO_ROOT / f"hau__{user}__{trial}.mp4"
    if not result.exists():
        raise FileNotFoundError(result)
    return result


meta = pd.read_csv(DATA / "meta.csv")
qa = pd.read_csv(DATA / "training_qa.csv")
vocab_file = DATA / "vocab.json"
if vocab_file.exists():
    vocab = json.loads(vocab_file.read_text())["HARN2HAU"]
else:
    # The first private dataset version predates vocab.json.  This is the immutable 40-class
    # benchmark mapping, included explicitly so the run does not depend on an unrelated mount.
    vocab = {
        "0_Wash_face": "Washing face", "1_Brush_teeth": "Brushing teeth",
        "2_Comb_hair": "Combing hair", "3_Take_off_clothes": "Undressing",
        "4_Wipe_hands": "Wiping hands", "5_Put_on_clothes": "Getting dressed",
        "6_Drink_water": "Drinking", "7_Eat_food": "Eating",
        "8_Take_and_use_tableware": "Grabbing utensils", "9_Pour_drinks": "Pouring",
        "10_Stir_drinks": "Stirring", "11_Peel_fruits": "Peeling fruit",
        "12_Sweep_the_floor": "Sweeping", "13_Mop_the_floor": "Mopping",
        "14_Wipe_bowls": "Washing dishes", "15_Wipe_windows_and_tables": "Wiping surface",
        "16_Fold_clothes": "Folding clothes", "17_Tap_the_keyboard": "Typing on a keyboard",
        "18_Write": "Writing", "19_Make_a_phone_call": "Calling",
        "20_Check_the_time": "Checking the time", "21_Read_documents": "Reading",
        "22_Turn_pages": "Turning a page",
        "23_Listen_to_music_with_headphones": "Listening to the music with headphones",
        "24_Use_a_mobile_phone": "Using a phone", "25_Watch_TV": "Watching TV",
        "26_Play_games": "Playing games", "27_Take_a_selfie": "Taking a selfie",
        "28_Jog_in_place": "Running", "29_Do_squats": "Squats",
        "30_Do_jumping_jacks": "Jumping jacks", "31_Do_stretching_exercises": "Stretching",
        "32_Stand_up": "Standing up", "33_Lie_down": "Lying down",
        "34_Sit_down": "Sitting down", "35_Do_lunges": "Lunges", "36_Walk": "Walking",
        "37_Take_medicine": "Taking medicine", "38_Massage_oneself": "Massaging oneself",
        "39_Take_body_temperature": "Checking body temperature",
    }
ACTIONS = sorted(vocab)
A2I = {action: index for index, action in enumerate(ACTIONS)}
TEXT2I = {text: A2I[action] for action, text in vocab.items()}
HARN_SELF = {
    "0_Wash_face": "washing their face", "1_Brush_teeth": "brushing their teeth",
    "2_Comb_hair": "combing their hair", "3_Take_off_clothes": "taking off their clothes",
    "4_Wipe_hands": "wiping their hands", "5_Put_on_clothes": "putting on clothes",
    "6_Drink_water": "drinking water", "7_Eat_food": "eating food",
    "8_Take_and_use_tableware": "using tableware", "9_Pour_drinks": "pouring a drink",
    "10_Stir_drinks": "stirring a drink", "11_Peel_fruits": "peeling fruit",
    "12_Sweep_the_floor": "sweeping the floor", "13_Mop_the_floor": "mopping the floor",
    "14_Wipe_bowls": "wiping a bowl", "15_Wipe_windows_and_tables": "wiping surfaces",
    "16_Fold_clothes": "folding clothes", "17_Tap_the_keyboard": "typing on the keyboard",
    "18_Write": "writing", "19_Make_a_phone_call": "making a phone call",
    "20_Check_the_time": "checking the time", "21_Read_documents": "reading documents",
    "22_Turn_pages": "turning pages",
    "23_Listen_to_music_with_headphones": "putting on headphones",
    "24_Use_a_mobile_phone": "using a smartphone", "25_Watch_TV": "using a remote",
    "26_Play_games": "using a smartphone", "27_Take_a_selfie": "taking a selfie",
    "28_Jog_in_place": "jogging in place", "29_Do_squats": "doing squats",
    "30_Do_jumping_jacks": "doing jumping jacks",
    "31_Do_stretching_exercises": "doing stretching exercises",
    "32_Stand_up": "standing up", "33_Lie_down": "lying down",
    "34_Sit_down": "sitting down", "35_Do_lunges": "doing lunges", "36_Walk": "walking",
    "37_Take_medicine": "taking medicine", "38_Massage_oneself": "massaging themselves",
    "39_Take_body_temperature": "taking their body temperature",
}
SELF2IS = defaultdict(list)
for action, text in HARN_SELF.items():
    if action in A2I:
        SELF2IS[text].append(A2I[action])
BG = len(ACTIONS)
N_CLASS = BG + 1

parents = {
    (row.user, row.trial): row for row in meta[meta.kind == "train_hau"].itertuples()
}
segments = []
by_parent = defaultdict(list)
for row in meta[meta.kind == "train_harn"].itertuples():
    if row.action not in A2I or not np.isfinite(row.f0) or (row.user, row.trial) not in parents:
        continue
    item = {
        "user": row.user, "trial": row.trial, "action": A2I[row.action],
        "f0": float(row.f0), "f1": float(row.f1), "qa_path": row.qa_path,
    }
    segments.append(item)
    by_parent[(row.user, row.trial)].append(item)
users = sorted({item["user"] for item in segments})
hold = set(folds(users)[FOLD])


def global_to_video(parent, frame10: float, video_fps: float) -> float:
    # meta.f0 and HARn intervals share the global 10-Hz skeleton frame axis.
    seconds = (frame10 - float(parent.f0)) / max(float(parent.fps), 1e-6)
    return seconds * video_fps


def decode_interval(item: dict, augment: bool, seed: int) -> list[np.ndarray]:
    parent = parents[(item["user"], item["trial"])]
    path = flattened_video(item["user"], item["trial"])
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    left = global_to_video(parent, item["f0"], video_fps)
    right = global_to_video(parent, item["f1"], video_fps)
    rng = random.Random(seed)
    if augment:
        span = max(right - left, N_FRAMES)
        left -= rng.uniform(0.0, 0.15) * span
        right += rng.uniform(0.0, 0.15) * span
    left = max(0.0, left)
    right = min(float(max(total - 1, 0)), max(left + 1, right))
    indices = np.linspace(left, right, N_FRAMES).round().astype(int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            frame = frames[-1].copy() if frames else np.zeros((224, 224, 3), np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if augment and rng.random() < 0.5:
        frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    return frames


def background_items() -> list[dict]:
    result = []
    for key, parent in parents.items():
        if key not in by_parent:
            continue
        intervals = sorted((value["f0"], value["f1"]) for value in by_parent[key])
        bounds = [(float(parent.f0), intervals[0][0])]
        bounds += [(intervals[i][1], intervals[i + 1][0]) for i in range(len(intervals) - 1)]
        bounds += [(intervals[-1][1], float(parent.f1))]
        candidates = [(b - a, a, b) for a, b in bounds if b - a >= 12]
        if not candidates:
            continue
        _, left, right = max(candidates)
        center = (left + right) / 2
        half = min(15.0, (right - left) / 2)
        result.append({
            "user": key[0], "trial": key[1], "action": BG,
            "f0": center - half, "f1": center + half,
            "qa_path": f"background/{key[0]}/{key[1]}",
        })
    return result


all_items = segments + background_items()
train_items = [item for item in all_items if item["user"] not in hold]
valid_items = [item for item in segments if item["user"] in hold]
counts = Counter(item["action"] for item in train_items)
class_weight = torch.tensor([
    math.sqrt(len(train_items) / max(counts.get(index, 1), 1)) for index in range(N_CLASS)
], dtype=torch.float32, device=DEVICE)
class_weight /= class_weight.mean()
print(json.dumps({
    "device": str(DEVICE), "hold_users": sorted(hold), "actions": len(ACTIONS),
    "train_items": len(train_items), "valid_segments": len(valid_items),
    "background_items": sum(item["action"] == BG for item in train_items),
}, indent=2), flush=True)

seed_all(SEED)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_ID, num_labels=N_CLASS, ignore_mismatched_sizes=True,
).to(DEVICE)
for parameter in model.videomae.parameters():
    parameter.requires_grad = False
for block in model.videomae.encoder.layer[-UNFREEZE_BLOCKS:]:
    for parameter in block.parameters():
        parameter.requires_grad = True
# Kinetics VideoMAE uses mean pooling: its active final norm is on the classification
# wrapper (`fc_norm`), while `videomae.layernorm` is intentionally None.  Handle both
# configuration variants rather than testing only whether the attribute exists.
final_norm = model.fc_norm if model.fc_norm is not None else model.videomae.layernorm
if final_norm is not None:
    for parameter in final_norm.parameters():
        parameter.requires_grad = True
for parameter in model.classifier.parameters():
    parameter.requires_grad = True
backbone = [parameter for parameter in model.videomae.parameters() if parameter.requires_grad]
head = list(model.classifier.parameters())
if final_norm is not None:
    head += list(final_norm.parameters())
optimizer = torch.optim.AdamW([
    {"params": backbone, "lr": LR}, {"params": head, "lr": HEAD_LR},
], weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")


def pixel_batch(items: list[dict], augment: bool, seed: int) -> torch.Tensor:
    values = []
    for offset, item in enumerate(items):
        frames = decode_interval(item, augment, seed + offset)
        values.append(processor(frames, return_tensors="pt").pixel_values[0])
    return torch.stack(values).to(DEVICE)


optimizer.zero_grad(set_to_none=True)
step = 0
for epoch in range(EPOCHS):
    order = list(train_items)
    random.Random(SEED + epoch).shuffle(order)
    model.train()
    losses = []
    for start in range(0, len(order), BATCH):
        batch = order[start:start + BATCH]
        pixels = pixel_batch(batch, True, SEED + epoch * 100000 + start)
        target = torch.tensor([item["action"] for item in batch], device=DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda", dtype=torch.float16):
            logits = model(pixel_values=pixels).logits
            loss = F.cross_entropy(logits, target, weight=class_weight, label_smoothing=0.03)
        scaler.scale(loss / GRAD_ACCUM).backward()
        losses.append(float(loss.detach()))
        batch_number = start // BATCH + 1
        if batch_number % GRAD_ACCUM == 0 or start + BATCH >= len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(backbone + head, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        if batch_number % 50 == 0:
            print(f"epoch={epoch} batch={batch_number}/{math.ceil(len(order)/BATCH)} "
                  f"step={step} loss={np.mean(losses[-50:]):.4f}", flush=True)


@torch.inference_mode()
def predict_segments(items: list[dict]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    truth, pred = [], []
    cache = {}
    for start in range(0, len(items), BATCH):
        batch = items[start:start + BATCH]
        pixels = pixel_batch(batch, False, SEED + 900000 + start)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda", dtype=torch.float16):
            logits = model(pixel_values=pixels).logits
        truth.extend(item["action"] for item in batch)
        pred.extend(logits.argmax(1).cpu().tolist())
        for item, value in zip(batch, logits.float().cpu().numpy()):
            cache[item["qa_path"]] = value
    return np.asarray(truth), np.asarray(pred), cache


segment_truth, segment_pred, segment_logits = predict_segments(valid_items)
print(f"held-segment accuracy={(segment_truth == segment_pred).mean():.4f} "
      f"n={len(segment_truth)}", flush=True)

# Multiple-choice action accuracy is the decision-level quantity relevant to HARn single
# questions.  Reuse the exact held-segment forward passes and restrict only by visible options.
harn_qa = qa[qa.source == "HARn"].copy()
harn_qa["user"] = harn_qa.path.str.extract(r"HARn/[^/]+/(user\d+)/")[0]
harn_rows = []
for _, row in harn_qa[harn_qa.user.isin(hold)].iterrows():
    logits = segment_logits.get(row.path)
    if logits is None or row.category != "single":
        continue
    option_classes = [
        (letter, SELF2IS.get(str(row[letter]).strip(), []))
        for letter in "ABCD" if pd.notna(row[letter]) and str(row[letter]).strip()
    ]
    if not option_classes or all(not values for _, values in option_classes):
        continue
    # The four extra HARN-only labels have no parent-video segment supervision.  In this
    # held-known-action audit they are visible distractors, so leave them at -inf rather than
    # dropping otherwise valid questions.  Test-time use will separately gate on modality
    # availability before applying the 40-class head.
    option_scores = [float(np.max(logits[values])) if values else -1e9
                     for _, values in option_classes]
    prediction = option_classes[int(np.argmax(option_scores))][0]
    harn_rows.append({
        "regime": "full", "qa_id": row.qa_id, "fold": FOLD, "user": row.user,
        "path": row.path, "category": "single", "truth": row.answer,
        "prediction": prediction, "correct": int(prediction == row.answer),
        "option_logits": json.dumps(option_scores),
    })
harn_predictions = pd.DataFrame(harn_rows)
harn_predictions.to_csv(OUT / f"videomae_action_fold{FOLD}_mc_predictions.csv", index=False)
print(f"held-HARn MC accuracy={harn_predictions.correct.mean():.4f} "
      f"n={len(harn_predictions)}", flush=True)


def decode_window(path: Path, center_seconds: float, duration_seconds: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    center = center_seconds * fps
    half = duration_seconds * fps / 2
    indices = np.linspace(max(0, center - half), min(total - 1, center + half), N_FRAMES)
    frames = []
    for index in indices.round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            frame = frames[-1].copy() if frames else np.zeros((224, 224, 3), np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


@torch.inference_mode()
def timeline_logits(user: str, trial: str) -> np.ndarray:
    model.eval()
    path = flattened_video(user, trial)
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.release()
    duration = max(total / fps, WINDOW_SECONDS)
    centers = np.linspace(WINDOW_SECONDS / 2, max(WINDOW_SECONDS / 2, duration - WINDOW_SECONDS / 2), N_WINDOWS)
    values = []
    for start in range(0, len(centers), BATCH):
        videos = [decode_window(path, float(center), WINDOW_SECONDS) for center in centers[start:start+BATCH]]
        pixels = torch.stack([
            processor(video, return_tensors="pt").pixel_values[0] for video in videos
        ]).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda", dtype=torch.float16):
            logits = model(pixel_values=pixels).logits
        values.append(logits.float().cpu().numpy())
    return np.concatenate(values)


CTC = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)


def permutation_scores(logits: np.ndarray, option_classes: list[int]) -> list[tuple[float, tuple[int, ...]]]:
    chosen = set(option_classes)
    other = [index for index in range(N_CLASS) if index not in chosen]
    option_logits = logits[:, option_classes]
    blank_logits = torch.logsumexp(torch.tensor(logits[:, other]), dim=1).numpy()
    local_logits = np.concatenate([blank_logits[:, None], option_logits], axis=1)
    lp = torch.tensor(local_logits, dtype=torch.float32).log_softmax(-1)[:, None, :]
    input_len = torch.tensor([len(lp)], dtype=torch.long)
    scores = []
    for permutation in itertools.permutations(range(4)):
        target = torch.tensor([index + 1 for index in permutation], dtype=torch.long)
        loss = CTC(lp, target, input_len, torch.tensor([4], dtype=torch.long))[0]
        scores.append((-float(loss), permutation))
    return sorted(scores, reverse=True)


sequence = qa[(qa.source == "HAU") & (qa.category == "sequence")].copy()
sequence["user"] = sequence.path.str.extract(r"HAU/(user\d+)/")[0]
sequence = sequence[sequence.user.isin(hold)]
cache = {}
rows = []
for item, row in sequence.iterrows():
    match = re.fullmatch(r"HAU/(user\d+)/(\d+-\d+-\d+)", row.path)
    if not match:
        continue
    key = (match.group(1), match.group(2))
    if key not in cache:
        cache[key] = timeline_logits(*key)
        print(f"timeline {len(cache)} path={row.path}", flush=True)
    option_classes = [TEXT2I.get(str(row[letter]).strip()) for letter in "ABCD"]
    if any(value is None for value in option_classes) or len(set(option_classes)) != 4:
        continue
    scored = permutation_scores(cache[key], option_classes)
    permutation = scored[0][1]
    prediction = "".join("ABCD"[index] for index in permutation)
    rows.append({
        "regime": "full", "qa_id": row.qa_id, "fold": FOLD, "user": row.user, "path": row.path,
        "category": "sequence", "truth": row.answer, "prediction": prediction,
        "correct": int(prediction == row.answer), "margin": scored[0][0] - scored[1][0],
        "permutation_scores": json.dumps({
            "".join("ABCD"[index] for index in perm): score for score, perm in scored
        }),
    })

predictions = pd.DataFrame(rows)
predictions.to_csv(OUT / f"videomae_action_timeline_fold{FOLD}_predictions.csv", index=False)
summary = {
    "model": MODEL_ID, "fold": FOLD, "hold_users": sorted(hold),
    "train_items": len(train_items), "valid_segments": len(valid_items),
    "held_segment_correct": int((segment_truth == segment_pred).sum()),
    "held_segment_accuracy": float((segment_truth == segment_pred).mean()),
    "harn_mc_n": len(harn_predictions),
    "harn_mc_correct": int(harn_predictions.correct.sum()),
    "harn_mc_accuracy": float(harn_predictions.correct.mean()),
    "sequence_n": len(predictions), "sequence_correct": int(predictions.correct.sum()),
    "sequence_accuracy": float(predictions.correct.mean()),
    "frames_per_window": N_FRAMES, "timeline_windows": N_WINDOWS,
    "epochs": EPOCHS, "unfrozen_blocks": UNFREEZE_BLOCKS,
}
(OUT / f"videomae_action_timeline_fold{FOLD}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
