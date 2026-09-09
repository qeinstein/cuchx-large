"""Subject-disjoint CTC temporal-order challenger.

This model differs from the existing dense/loc models in the supervision target.  It trains
directly on each HAU sequence answer as an ordered transcript and marginalises over every
monotonic frame-to-action alignment with CTC.  Exact HARn intervals remain an auxiliary
frame-localisation signal; they are not required for the 305 sequence-labelled HAU clips.

The first run is one held-out subject fold.  It writes row-level predictions and all 24
permutation log-likelihoods so complementarity with the current structured decoder can be
audited before spending compute on five folds.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 20260909
FOLD = int(os.environ.get("CTC_FOLD", "0"))
EPOCHS = int(os.environ.get("CTC_EPOCHS", "55"))
BATCH_SIZE = int(os.environ.get("CTC_BATCH", "6"))
CUDA_USABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
DEVICE = torch.device("cuda" if CUDA_USABLE else "cpu")
HERE = Path(__file__).resolve().parent
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else HERE / "output_local"


def locate(name: str) -> Path:
    candidates = [
        HERE / name,
        HERE.parent.parent / name,
        HERE.parent.parent / "champ" / name,
    ]
    root = Path("/kaggle/input")
    if root.exists():
        candidates.extend(root.glob(f"**/{name}"))
    for path in candidates:
        if path.exists() and path.stat().st_size:
            return path
    raise FileNotFoundError(name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


VOCAB = json.loads(locate("vocab.json").read_text())
ACTIONS = sorted(VOCAB["HARN2HAU"])
A2I = {a: i for i, a in enumerate(ACTIONS)}
HAU2I = {VOCAB["HARN2HAU"][a]: i for i, a in enumerate(ACTIONS)}
N_ACTION = len(ACTIONS)
BLANK = N_ACTION


def frame_feats(k: np.ndarray) -> np.ndarray:
    hip = k[:, [11, 12]].mean(1, keepdims=True)
    torso = np.linalg.norm(k[:, 5] - k[:, 11], axis=-1) + np.linalg.norm(
        k[:, 6] - k[:, 12], axis=-1
    )
    scale = np.median(torso[torso > 1e-3]) if np.any(torso > 1e-3) else 1.0
    pose = (k - hip) / max(float(scale), 1e-3)
    vel = np.zeros_like(pose)
    vel[1:] = np.diff(pose, axis=0)
    acc = np.zeros_like(pose)
    acc[1:] = np.diff(vel, axis=0)
    hipn = (hip[:, 0] - hip[0, 0]) / max(float(scale), 1e-3)
    hipv = np.zeros_like(hipn)
    hipv[1:] = np.diff(hipn, axis=0)
    speed = np.linalg.norm(vel, axis=-1)
    out = np.concatenate(
        [pose.reshape(len(k), -1), vel.reshape(len(k), -1), acc.reshape(len(k), -1), hipn, hipv, speed],
        axis=1,
    )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def resample(a: np.ndarray, length: int) -> np.ndarray:
    if len(a) == length:
        return a.astype(np.float32)
    if not len(a):
        return np.zeros((length, 30), np.float32)
    idx = np.clip(np.arange(length) * len(a) // max(length, 1), 0, len(a) - 1)
    return a[idx].astype(np.float32)


def imu_feats(a: np.ndarray, length: int) -> np.ndarray:
    a = resample(a, length)
    acc = np.stack([np.linalg.norm(a[:, d * 6 : d * 6 + 3], axis=1) for d in range(5)], 1)
    gyr = np.stack([np.linalg.norm(a[:, d * 6 + 3 : d * 6 + 6], axis=1) for d in range(5)], 1)
    kernel = np.ones(9, np.float32) / 9.0

    def smooth(m: np.ndarray) -> np.ndarray:
        if len(m) < 9:
            return m
        return np.stack([np.convolve(m[:, j], kernel, mode="same") for j in range(m.shape[1])], 1)

    dacc = np.abs(np.diff(acc, axis=0, prepend=acc[:1]))
    out = np.concatenate([a, acc, gyr, smooth(acc), smooth(gyr), smooth(dacc)], 1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def options(row) -> list[str]:
    return [str(row[x]).strip() for x in "ABCD"]


def build_items() -> tuple[list[dict], pd.DataFrame]:
    tr = pd.read_csv(locate("training_qa.csv"))
    meta = pd.read_csv(locate("meta.csv"))
    skel = np.load(locate("skel_seq.npz"), allow_pickle=True)
    imu = np.load(locate("imu_seq.npz"), allow_pickle=True)

    segments: dict[tuple[str, str], list] = {}
    for row in meta[meta.kind == "train_harn"].itertuples():
        if row.action in A2I and np.isfinite(row.f0):
            segments.setdefault((row.user, row.trial), []).append((A2I[row.action], row.f0, row.f1))

    action_rows = tr[(tr.source == "HAU") & tr.category.isin(["single", "multi", "combination", "sequence"])]
    pools: dict[str, set[int]] = {}
    orders: dict[str, list[int]] = {}
    qids: dict[str, str] = {}
    truths: dict[str, str] = {}
    for _, row in action_rows.iterrows():
        pool = pools.setdefault(row.path, set())
        for letter in str(row.answer):
            if letter in "ABCD":
                for value in str(row[letter]).split(","):
                    if value.strip() in HAU2I:
                        pool.add(HAU2I[value.strip()])
        if row.category == "sequence":
            oo = options(row)
            target = [HAU2I.get(oo["ABCD".index(letter)]) for letter in str(row.answer)]
            if len(target) == 4 and all(value is not None for value in target) and len(set(target)) == 4:
                orders[row.path] = target
                qids[row.path] = row.qa_id
                truths[row.path] = str(row.answer)

    items = []
    for row in meta[meta.kind == "train_hau"].itertuples():
        kk = row.unit_dir + "|K"
        fk = row.unit_dir + "|F"
        if kk not in skel or fk not in skel:
            continue
        k = np.asarray(skel[kk], np.float32)
        frames = np.asarray(skel[fk])
        ia = np.asarray(imu[row.unit_dir], np.float32) if row.unit_dir in imu else np.zeros((len(k), 30), np.float32)
        x = np.concatenate([frame_feats(k), imu_feats(ia, len(k))], 1)
        y = np.full(len(k), -100, np.int64)
        seg_classes = set()
        for cls, f0, f1 in segments.get((row.user, row.trial), []):
            mask = (frames >= f0) & (frames <= f1)
            y[mask] = cls
            seg_classes.add(cls)
        pool = pools.get(row.qa_path, set())
        # Background is supervised only when every QA-known action has an exact segment.
        if pool and pool <= seg_classes:
            y[y == -100] = BLANK
        items.append(
            dict(
                path=row.qa_path,
                user=row.user,
                x=x,
                y=y,
                pool=sorted(pool),
                order=orders.get(row.qa_path),
                qa_id=qids.get(row.qa_path),
                truth=truths.get(row.qa_path),
            )
        )
    return items, tr


class DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class TranscriptTCN(nn.Module):
    """Two-stage temporal encoder; stage two refines boundaries from class posteriors."""

    def __init__(self, din: int, channels: int = 192, layers: int = 8):
        super().__init__()
        self.input = nn.Conv1d(din, channels, 1)
        self.stage1 = nn.Sequential(*[DilatedBlock(channels, 2 ** (i % 6)) for i in range(layers)])
        self.head1 = nn.Conv1d(channels, N_ACTION + 1, 1)
        self.refine_in = nn.Conv1d(N_ACTION + 1, channels, 1)
        self.stage2 = nn.Sequential(*[DilatedBlock(channels, 2 ** (i % 6)) for i in range(6)])
        self.head2 = nn.Conv1d(channels, N_ACTION + 1, 1)

    def forward(self, x):
        h = self.stage1(self.input(x))
        first = self.head1(h)
        refined = self.head2(self.stage2(self.refine_in(first.softmax(1))))
        return first, refined


def population_stats(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = sum(len(item["x"]) for item in items)
    sx = np.zeros(items[0]["x"].shape[1], np.float64)
    sx2 = np.zeros_like(sx)
    for item in items:
        value = item["x"].astype(np.float64)
        sx += value.sum(0)
        sx2 += np.square(value).sum(0)
    mean = sx / n
    std = np.sqrt(np.maximum(sx2 / n - mean * mean, 0.0)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


CTC = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)


def transcript_loss(logits: torch.Tensor, batch: list[dict], lengths: list[int]) -> torch.Tensor:
    total = logits.new_zeros(())
    count = 0
    for i, item in enumerate(batch):
        if not item["order"]:
            continue
        option_classes = sorted(item["order"])
        local = {cls: j + 1 for j, cls in enumerate(option_classes)}
        target = torch.tensor([local[cls] for cls in item["order"]], device=logits.device, dtype=torch.long)
        cols = torch.tensor([BLANK] + option_classes, device=logits.device)
        lp = logits[i, cols, : lengths[i]].transpose(0, 1).log_softmax(-1)[:, None, :]
        total = total + CTC(
            lp,
            target,
            torch.tensor([lengths[i]], device=logits.device),
            torch.tensor([len(target)], device=logits.device),
        )
        count += 1
    return total / max(count, 1)


def mil_loss(logits: torch.Tensor, batch: list[dict], lengths: list[int]) -> torch.Tensor:
    losses = []
    for i, item in enumerate(batch):
        if not item["pool"]:
            continue
        score = logits[i, :N_ACTION, : lengths[i]].logsumexp(-1) - math.log(lengths[i])
        target = torch.zeros(N_ACTION, device=logits.device)
        target[item["pool"]] = 1.0
        losses.append(F.binary_cross_entropy_with_logits(score, target))
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def compute_loss(outputs, y, batch, lengths):
    first, refined = outputs
    weights = torch.ones(N_ACTION + 1, device=first.device)
    weights[BLANK] = 0.15
    if (y != -100).any():
        frame1 = F.cross_entropy(first, y, weight=weights, ignore_index=-100)
        frame2 = F.cross_entropy(refined, y, weight=weights, ignore_index=-100)
    else:
        frame1 = first.new_zeros(())
        frame2 = refined.new_zeros(())
    ctc1 = transcript_loss(first, batch, lengths)
    ctc2 = transcript_loss(refined, batch, lengths)
    mil = mil_loss(refined, batch, lengths)
    # Penalise frame-to-frame posterior jitter without discouraging real sharp boundaries.
    logp = refined.log_softmax(1)
    smooth = torch.clamp((logp[:, :, 1:] - logp.detach()[:, :, :-1]).square(), max=16).mean()
    return 0.25 * frame1 + 0.75 * frame2 + 0.35 * ctc1 + ctc2 + 0.25 * mil + 0.02 * smooth


def collate(batch: list[dict], mean: np.ndarray, std: np.ndarray, augment: bool):
    lengths = [len(item["x"]) for item in batch]
    tmax = max(lengths)
    din = len(mean)
    x = np.zeros((len(batch), tmax, din), np.float32)
    y = np.full((len(batch), tmax), -100, np.int64)
    for i, item in enumerate(batch):
        value = (item["x"] - mean) / std
        if augment:
            value = value + np.random.normal(0, 0.015, value.shape).astype(np.float32)
            # Whole sensor-feature dropout is safer than temporal cropping for CTC transcripts.
            if random.random() < 0.25:
                start = random.randrange(value.shape[1])
                width = random.randint(1, min(12, value.shape[1] - start))
                value[:, start : start + width] = 0
        x[i, : len(value)] = value
        y[i, : len(value)] = item["y"]
    return (
        torch.tensor(x, device=DEVICE).transpose(1, 2),
        torch.tensor(y, device=DEVICE),
        lengths,
    )


def train_model(train_items: list[dict], seed: int):
    seed_all(seed)
    mean, std = population_stats(train_items)
    model = TranscriptTCN(len(mean)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    steps = EPOCHS * math.ceil(len(train_items) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2.5e-3, total_steps=steps)
    rng = np.random.default_rng(seed)
    for epoch in range(EPOCHS):
        model.train()
        # Length buckets retain randomness while avoiding extreme padding waste.
        order = sorted(range(len(train_items)), key=lambda j: len(train_items[j]["x"]))
        buckets = [order[j : j + BATCH_SIZE] for j in range(0, len(order), BATCH_SIZE)]
        rng.shuffle(buckets)
        losses = []
        for indices in buckets:
            batch = [train_items[j] for j in indices]
            x, y, lengths = collate(batch, mean, std, augment=True)
            loss = compute_loss(model(x), y, batch, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch={epoch + 1:02d} loss={np.mean(losses):.5f}", flush=True)
    return model, mean, std


@torch.no_grad()
def predict_logits(model, items, mean, std):
    model.eval()
    result = {}
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        x, _, lengths = collate(batch, mean, std, augment=False)
        _, logits = model(x)
        for i, item in enumerate(batch):
            result[item["path"]] = logits[i, :, : lengths[i]].float().cpu().numpy()
    return result


def permutation_scores(logits: np.ndarray, order: list[int]):
    option_classes = sorted(order)
    local = {cls: i + 1 for i, cls in enumerate(option_classes)}
    cols = [BLANK] + option_classes
    lp = torch.tensor(logits[cols].T, dtype=torch.float32).log_softmax(-1)[:, None, :]
    ilen = torch.tensor([len(lp)], dtype=torch.long)
    scored = []
    for perm in itertools.permutations(option_classes):
        target = torch.tensor([local[x] for x in perm], dtype=torch.long)
        nll = CTC(lp, target, ilen, torch.tensor([4], dtype=torch.long))
        scored.append((perm, -float(nll)))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def pairwise_accuracy(pred: list[int], truth: list[int]) -> tuple[int, int]:
    ppos = {value: i for i, value in enumerate(pred)}
    tpos = {value: i for i, value in enumerate(truth)}
    correct = total = 0
    for a, b in itertools.combinations(truth, 2):
        correct += (ppos[a] < ppos[b]) == (tpos[a] < tpos[b])
        total += 1
    return correct, total


def main():
    print(f"device={DEVICE} fold={FOLD} epochs={EPOCHS}", flush=True)
    items, _ = build_items()
    users = sorted({item["user"] for item in items}, key=lambda x: int(str(x).replace("user", "")))
    shuffled = list(np.random.default_rng(7).permutation(users))
    folds = [shuffled[i::5] for i in range(5)]
    hold = set(folds[FOLD])
    train_items = [item for item in items if item["user"] not in hold]
    valid_items = [item for item in items if item["user"] in hold]
    print(
        f"items={len(items)} train={len(train_items)} valid={len(valid_items)} "
        f"train_transcripts={sum(x['order'] is not None for x in train_items)} "
        f"valid_transcripts={sum(x['order'] is not None for x in valid_items)} hold={sorted(hold)}",
        flush=True,
    )
    model, mean, std = train_model(train_items, SEED + FOLD)
    logits = predict_logits(model, valid_items, mean, std)
    rows = []
    exact = pair_c = pair_n = 0
    for item in valid_items:
        if not item["order"]:
            continue
        scored = permutation_scores(logits[item["path"]], item["order"])
        pred_order, best = scored[0]
        option_classes = sorted(item["order"])
        option_letter = {cls: "ABCD"[i] for i, cls in enumerate(option_classes)}
        # Reconstruct letters relative to the original question options, not sorted classes.
        truth_letters = item["truth"]
        cls_to_letter = {cls: truth_letters[i] for i, cls in enumerate(item["order"])}
        pred = "".join(cls_to_letter[cls] for cls in pred_order)
        second = scored[1][1]
        pc, pn = pairwise_accuracy(list(pred_order), item["order"])
        pair_c += pc
        pair_n += pn
        exact += pred == truth_letters
        rows.append(
            dict(
                qa_id=item["qa_id"],
                path=item["path"],
                user=item["user"],
                truth=truth_letters,
                prediction=pred,
                correct=int(pred == truth_letters),
                margin=best - second,
                best_loglik=best,
                pair_correct=pc,
                pair_total=pn,
                permutation_scores=json.dumps({"".join(cls_to_letter[c] for c in p): s for p, s in scored}),
            )
        )
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"ctc_temporal_fold{FOLD}_predictions.csv"
    out.to_csv(out_path, index=False)
    summary = {
        "fold": FOLD,
        "held_users": sorted(hold),
        "n": len(out),
        "exact_correct": int(exact),
        "exact_accuracy": float(exact / max(len(out), 1)),
        "pair_correct": int(pair_c),
        "pair_total": int(pair_n),
        "pair_accuracy": float(pair_c / max(pair_n, 1)),
        "epochs": EPOCHS,
        "device": str(DEVICE),
    }
    (OUT / f"ctc_temporal_fold{FOLD}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
