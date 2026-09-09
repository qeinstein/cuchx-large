"""Subject-disjoint, task-aligned VideoMAE manner model.

This is a validation-only challenger.  It fine-tunes the final VideoMAE blocks on complete
HAU session triples, jointly predicting the five manner groups and an ordinal trial-speed
score.  At validation time, raw clip logits are decoded under the exact one-to-one candidate
assignment constraint.  No competition submission API is present in this file.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

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
EPOCHS = 5
LR = 2.0e-5
HEAD_LR = 2.0e-4
GRAD_ACCUM = 4
PAIR_AUGMENT_PROB = 0.45
VALID_PAIR_FRAC = 0.38
UNFREEZE_BLOCKS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working")

FAST = {
    "Quickly", "quickly", "Rapidly", "Hastily", "Hasitly", "Hurriedly", "Swiftly",
    "Briskly", "Urgently", "Frantically", "Impatiently", "Eagerly", "Forcefully",
}
SLOW = {
    "Slowly", "Leisurely", "Unhurriedly", "Calmly", "Peacefully", "Relaxedly", "Lazily",
    "Gently", "Softly", "Quietly", "Comfortably", "Soothingly", "Patiently", "Lightly",
    "Contently", "Absentmindedly",
}
CARE = {
    "Carefully", "Cautiously", "Meticulously", "Precisely", "Thoroughly", "Deliberately",
    "Methodically", "Attentively", "Intently", "Diligently", "Earnestly", "Neatly",
    "Orderly", "Seriously", "Serioiusly",
}
NERV = {"Nervously", "Anxiously", "Tensely", "Tensly", "Restlessly"}
GROUPS = ["SLOW", "CARE", "NEUT", "NERV", "FAST"]
G2I = {value: index for index, value in enumerate(GROUPS)}


def manner_group(value: str) -> str:
    if value in FAST:
        return "FAST"
    if value in SLOW:
        return "SLOW"
    if value in CARE:
        return "CARE"
    if value in NERV:
        return "NERV"
    return "NEUT"


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
    qa = sorted(Path("/kaggle/input").glob("**/training_qa.csv"))
    if len(qa) == 1:
        return qa[0].parent
    raise FileNotFoundError(f"cannot identify HAU corpus: {qa}")


DATA = find_input()
VIDEO_ROOT = DATA / "hau_videos"


def path_parts(path: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"HAU/(user\d+)/(\d+-\d+)-(\d+)", path)
    if not match:
        raise ValueError(path)
    return match.group(1), match.group(2), int(match.group(3))


def session_of(path: str) -> str:
    user, session, _ = path_parts(path)
    return f"{user}/{session}"


def video_path(path: str) -> Path:
    user, session, trial = path_parts(path)
    result = VIDEO_ROOT / f"hau__{user}__{session}-{trial}.mp4"
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def folds(users: list[str], n: int = 5, seed: int = 7) -> list[list[str]]:
    ordered = sorted(users, key=lambda value: int(value[4:]))
    shuffled = list(np.random.default_rng(seed).permutation(ordered))
    return [shuffled[index::n] for index in range(n)]


def ordered_rows(group: pd.DataFrame) -> list[pd.Series]:
    rows = [part.iloc[0] for _, part in group.groupby("path")]
    return sorted(rows, key=lambda row: path_parts(row.path)[2])


def candidate_manners(rows: list[pd.Series]) -> list[str]:
    sets = [{str(row[letter]).strip() for letter in "ABCD"} for row in rows]
    return sorted(set.intersection(*sets)) if sets else []


def truth_manner(row: pd.Series) -> str:
    return str(row[str(row.answer)]).strip()


def valid_session(group: pd.DataFrame) -> bool:
    rows = ordered_rows(group)
    candidates = set(candidate_manners(rows))
    return len(rows) >= 2 and len(candidates) >= len(rows) and {
        truth_manner(row) for row in rows
    } <= candidates


def decode_frames(path: Path, rng: random.Random | None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"zero frames: {path}")
    edges = np.linspace(0, total, N_FRAMES + 1)
    indices = []
    for left, right in zip(edges[:-1], edges[1:]):
        lo = int(math.floor(left))
        hi = max(lo, int(math.ceil(right)) - 1)
        index = (rng.randint(lo, hi) if rng is not None else (lo + hi) // 2)
        indices.append(min(index, total - 1))
    result = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            frame = result[-1].copy() if result else np.zeros((224, 224, 3), np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result.append(frame)
    cap.release()
    if rng is not None and rng.random() < 0.5:
        result = [np.ascontiguousarray(frame[:, ::-1]) for frame in result]
    return result


qa = pd.read_csv(DATA / "training_qa.csv")
qa = qa[(qa.source == "HAU") & (qa.category == "emotion")].copy()
qa["user"] = qa.path.map(lambda value: path_parts(value)[0])
qa["session"] = qa.path.map(session_of)
users = sorted(qa.user.unique())
hold = set(folds(users)[FOLD])
train_sessions = [
    (name, group.copy()) for name, group in qa[~qa.user.isin(hold)].groupby("session")
    if valid_session(group)
]
valid_sessions = [
    (name, group.copy()) for name, group in qa[qa.user.isin(hold)].groupby("session")
    if valid_session(group)
]


def position_prior(sessions) -> dict[tuple[str, int, int], float]:
    exact = Counter()
    totals = Counter()
    group = Counter()
    group_totals = Counter()
    for _, frame in sessions:
        rows = ordered_rows(frame)
        k = len(rows)
        for pos, row in enumerate(rows):
            manner = truth_manner(row)
            exact[(manner, pos, k)] += 1
            totals[(manner, k)] += 1
            grp = manner_group(manner)
            group[(grp, pos, k)] += 1
            group_totals[(grp, k)] += 1
    result = {}
    manners = {truth_manner(row) for _, frame in sessions for row in ordered_rows(frame)}
    for manner in manners:
        grp = manner_group(manner)
        for k in (2, 3):
            for pos in range(k):
                back = (group[(grp, pos, k)] + 1.0) / (group_totals[(grp, k)] + k)
                result[(manner, pos, k)] = (
                    exact[(manner, pos, k)] + 3.0 * back
                ) / (totals[(manner, k)] + 3.0)
    return result


POS_PRIOR = position_prior(train_sessions)
print(json.dumps({
    "device": str(DEVICE), "hold_users": sorted(hold),
    "train_sessions": len(train_sessions), "valid_sessions": len(valid_sessions),
}, indent=2), flush=True)

seed_all(SEED)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_ID, num_labels=len(GROUPS), ignore_mismatched_sizes=True,
).to(DEVICE)
for parameter in model.videomae.parameters():
    parameter.requires_grad = False
for block in model.videomae.encoder.layer[-UNFREEZE_BLOCKS:]:
    for parameter in block.parameters():
        parameter.requires_grad = True
for parameter in model.videomae.layernorm.parameters():
    parameter.requires_grad = True
for parameter in model.classifier.parameters():
    parameter.requires_grad = True
rank_head = torch.nn.Linear(model.config.hidden_size, 1).to(DEVICE)

backbone_params = [p for p in model.videomae.parameters() if p.requires_grad]
head_params = list(model.classifier.parameters()) + list(rank_head.parameters())
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR},
    {"params": head_params, "lr": HEAD_LR},
], weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")


def pixels_for(rows: list[pd.Series], rng: random.Random | None) -> torch.Tensor:
    values = []
    for row in rows:
        frames = decode_frames(video_path(row.path), rng)
        values.append(processor(frames, return_tensors="pt").pixel_values[0])
    return torch.stack(values).to(DEVICE)


def session_loss(rows: list[pd.Series], rng: random.Random):
    pixel_values = pixels_for(rows, rng)
    targets = torch.tensor(
        [G2I[manner_group(truth_manner(row))] for row in rows],
        dtype=torch.long, device=DEVICE,
    )
    output = model.videomae(pixel_values)
    pooled = output.last_hidden_state.mean(1)
    logits = model.classifier(pooled)
    ce = F.cross_entropy(logits, targets, label_smoothing=0.05)
    speed = rank_head(pooled).flatten()
    rank_losses = [F.softplus(0.25 - (speed[j] - speed[i]))
                   for i in range(len(rows)) for j in range(i + 1, len(rows))]
    rank = torch.stack(rank_losses).mean() if rank_losses else ce.new_zeros(())
    # Direct structured assignment loss.  Candidate texts sharing a broad group remain
    # distinguishable through the learned text-position prior at decode time.
    candidates = candidate_manners(rows)
    truth = [candidates.index(truth_manner(row)) for row in rows]
    assignments = list(itertools.permutations(range(len(candidates)), len(rows)))
    scores = []
    for assignment in assignments:
        value = ce.new_zeros(())
        for pos, candidate_index in enumerate(assignment):
            manner = candidates[candidate_index]
            value = value + F.log_softmax(logits[pos], -1)[G2I[manner_group(manner)]]
            value = value + 0.30 * math.log(max(POS_PRIOR.get((manner, pos, len(rows)), 1 / len(rows)), 1e-6))
        scores.append(value)
    structured = F.cross_entropy(
        torch.stack(scores)[None],
        torch.tensor([assignments.index(tuple(truth))], device=DEVICE),
    )
    return ce + 0.35 * rank + 0.50 * structured, ce.detach(), rank.detach(), structured.detach()


optimizer.zero_grad(set_to_none=True)
global_step = 0
for epoch in range(EPOCHS):
    shuffled = list(train_sessions)
    random.Random(SEED + epoch).shuffle(shuffled)
    model.train()
    rank_head.train()
    metrics = []
    for item, (_, frame) in enumerate(shuffled, 1):
        rows = ordered_rows(frame)
        rng = random.Random(SEED + epoch * 10000 + item)
        if len(rows) == 3 and rng.random() < PAIR_AUGMENT_PROB:
            del rows[rng.randrange(3)]
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda", dtype=torch.float16):
            loss, ce, rank, structured = session_loss(rows, rng)
            scaled_loss = loss / GRAD_ACCUM
        scaler.scale(scaled_loss).backward()
        metrics.append((float(loss.detach()), float(ce), float(rank), float(structured)))
        if item % GRAD_ACCUM == 0 or item == len(shuffled):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(backbone_params + head_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        if item % 10 == 0:
            mean = np.mean(metrics[-10:], axis=0)
            print(f"epoch={epoch} session={item}/{len(shuffled)} step={global_step} "
                  f"loss={mean[0]:.4f} ce={mean[1]:.4f} rank={mean[2]:.4f} struct={mean[3]:.4f}", flush=True)


@torch.inference_mode()
def clip_logits(rows: list[pd.Series]) -> np.ndarray:
    model.eval()
    values = []
    # One clip at a time materially reduces P100 peak memory during validation.
    for row in rows:
        pixel_values = pixels_for([row], None)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda", dtype=torch.float16):
            result = model(pixel_values=pixel_values).logits
        values.append(result[0].float().cpu().numpy())
    return np.stack(values)


def decode_assignment(rows: list[pd.Series], logits: np.ndarray) -> tuple[list[str], float]:
    candidates = candidate_manners(rows)
    k = len(rows)
    slots = len(candidates) if k < len(candidates) <= 4 else k
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    scored = []
    if slots > k:
        present_sets = list(itertools.combinations(range(slots), k))
        for assignment in itertools.permutations(range(len(candidates)), k):
            manners = [candidates[index] for index in assignment]
            if any(manners[i] not in {str(rows[i][letter]).strip() for letter in "ABCD"}
                   for i in range(k)):
                continue
            for present in present_sets:
                value = sum(logp[i, G2I[manner_group(manners[i])]] for i in range(k))
                value += 0.30 * sum(math.log(max(POS_PRIOR.get(
                    (manners[i], present[i], slots), 1 / slots), 1e-6)) for i in range(k))
                scored.append((float(value), manners))
    else:
        for assignment in itertools.permutations(range(len(candidates)), k):
            manners = [candidates[index] for index in assignment]
            if any(manners[i] not in {str(rows[i][letter]).strip() for letter in "ABCD"}
                   for i in range(k)):
                continue
            value = sum(logp[i, G2I[manner_group(manners[i])]] for i in range(k))
            value += 0.30 * sum(math.log(max(POS_PRIOR.get(
                (manners[i], i, k), 1 / k), 1e-6)) for i in range(k))
            scored.append((float(value), manners))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return ["" for _ in rows], float("nan")
    margin = scored[0][0] - scored[1][0] if len(scored) > 1 else float("inf")
    return scored[0][1], margin


predictions = []
for regime in ("full", "pair"):
    for session_index, (session, frame) in enumerate(valid_sessions, 1):
        rows = ordered_rows(frame)
        if regime == "pair" and len(rows) == 3:
            rng = random.Random(SEED + 50000 + session_index)
            if rng.random() < VALID_PAIR_FRAC:
                del rows[rng.randrange(3)]
        logits = clip_logits(rows)
        assigned, margin = decode_assignment(rows, logits)
        for row, manner, raw_logits in zip(rows, assigned, logits):
            prediction = next((letter for letter in "ABCD" if str(row[letter]).strip() == manner), "")
            predictions.append({
                "regime": regime, "session": session, "qa_id": row.qa_id,
                "user": row.user, "category": "emotion", "truth": row.answer,
                "prediction": prediction, "correct": int(prediction == row.answer),
                "assignment_margin": margin,
                "group_logits": json.dumps([float(value) for value in raw_logits]),
            })
        if session_index % 5 == 0:
            current = pd.DataFrame(predictions)
            current = current[current.regime == regime]
            print(f"regime={regime} valid={session_index}/{len(valid_sessions)} "
                  f"acc={current.correct.mean():.4f}", flush=True)

pred = pd.DataFrame(predictions)
pred.to_csv(OUT / f"videomae_emotion_fold{FOLD}_predictions.csv", index=False)
summary = {
    "model": MODEL_ID, "fold": FOLD, "hold_users": sorted(hold),
    "frames": N_FRAMES, "epochs": EPOCHS, "unfrozen_blocks": UNFREEZE_BLOCKS,
    "task": "manner_group_plus_ordinal_rank_plus_structured_assignment",
    "regimes": {
        regime: {"n": len(group), "correct": int(group.correct.sum()),
                 "accuracy": float(group.correct.mean())}
        for regime, group in pred.groupby("regime")
    },
}
(OUT / f"videomae_emotion_fold{FOLD}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
