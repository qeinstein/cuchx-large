"""Nonlinear temporal probe on exact HARn-aligned thermal MobileNet sequences."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
from pseudotest import folds  # noqa: E402


VOCAB = json.load(open(os.path.join(ROOT, "champ", "vocab.json")))["HARN2HAU"]
ACTIONS = sorted(VOCAB)
A2I = {action: index for index, action in enumerate(ACTIONS)}
LENGTH = 16


def interpolate(sequence: np.ndarray, length: int = LENGTH) -> np.ndarray:
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0)
    position = np.linspace(0, len(sequence) - 1, length)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(sequence) - 1)
    weight = (position - lower)[:, None]
    return sequence[lower] * (1 - weight) + sequence[upper] * weight


def examples(meta: pd.DataFrame, cache):
    segment_cache = os.environ.get("THERMAL_SEGMENT_CACHE")
    if segment_cache:
        direct = np.load(segment_cache)
        output = []
        for row in meta[meta.kind == "train_harn"].itertuples():
            if row.qa_path not in direct or row.action not in A2I:
                continue
            x = direct[row.qa_path].astype(np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            output.append((row.qa_path, row.user, A2I[row.action], interpolate(x)))
        return output
    parent = {
        (row.user, row.trial): row
        for row in meta[meta.kind == "train_hau"].itertuples()
        if row.qa_path + "|X" in cache and np.isfinite(row.f0)
    }
    output = []
    for row in meta[meta.kind == "train_harn"].itertuples():
        p = parent.get((row.user, row.trial))
        if p is None or row.action not in A2I or not np.isfinite(row.f0):
            continue
        frame25 = cache[p.qa_path + "|F"].astype(float)
        global10 = p.f0 + frame25 * (p.fps / 25.0)
        mask = (global10 >= row.f0) & (global10 <= row.f1)
        x = cache[p.qa_path + "|X"][mask].astype(np.float32)
        if not len(x):
            continue
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        output.append((row.qa_path, row.user, A2I[row.action], interpolate(x)))
    return output


class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(576, 128), nn.LayerNorm(128), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1), nn.GELU(), nn.Dropout(0.15),
            nn.Conv1d(128, 128, 3, padding=1), nn.GELU(),
        )
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
                                        nn.Linear(128, len(ACTIONS)))

    def forward(self, x):
        z = self.temporal(self.project(x).transpose(1, 2))
        return self.classifier(torch.cat([z.mean(2), z.amax(2)], 1))


def train_model(x: torch.Tensor, y: torch.Tensor, seed: int):
    torch.manual_seed(seed)
    model = TemporalHead()
    counts = torch.bincount(y, minlength=len(ACTIONS)).float()
    weight = torch.sqrt(counts.sum() / counts.clamp_min(1))
    weight /= weight.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    epochs = int(os.environ.get("THERMAL_TEMPORAL_EPOCHS", "25"))
    model.train()
    for epoch in range(epochs):
        total = correct = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        if epoch in {0, epochs - 1}:
            print(" epoch", epoch + 1, "train", correct / total, flush=True)
    return model.eval()


def main() -> None:
    torch.set_num_threads(int(os.environ.get("THERMAL_TORCH_THREADS", "8")))
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    cache = np.load(os.path.join(ROOT, "champ", "thermal_mnv3_frames.npz"))
    data = examples(meta, cache)
    paths = np.asarray([row[0] for row in data])
    users = np.asarray([row[1] for row in data])
    y = torch.tensor([row[2] for row in data], dtype=torch.long)
    x = torch.tensor(np.stack([row[3] for row in data]), dtype=torch.float32)
    print("segments", tuple(x.shape), "classes", len(set(y.tolist())))
    only = os.environ.get("THERMAL_ONLY_FOLD")
    rows = []
    for fold, held in enumerate(folds(sorted(set(users)), 5)):
        if only is not None and fold != int(only):
            continue
        train = torch.from_numpy(~np.isin(users, held))
        valid = ~train
        model = train_model(x[train], y[train], 20260906 + fold)
        with torch.inference_mode():
            logits = []
            for start in range(0, int(valid.sum()), 128):
                logits.append(model(x[valid][start:start + 128]))
            pred = torch.cat(logits).argmax(1)
        truth = y[valid]
        correct = pred == truth
        for path, user, target, guess, ok in zip(
            paths[valid], users[valid], truth.tolist(), pred.tolist(), correct.tolist()
        ):
            rows.append(dict(
                fold=fold, path=path, user=user, truth=ACTIONS[target],
                prediction=ACTIONS[guess], correct=int(ok),
            ))
        print("fold", fold, "held", held, "top1", int(correct.sum()), "/", len(correct),
              "=", float(correct.float().mean()), flush=True)
    out = pd.DataFrame(rows)
    suffix = f"_fold{only}" if only is not None else ""
    out.to_csv(os.path.join(
        ROOT, "research", f"thermal_temporal_action_oof_20260906{suffix}.csv"
    ), index=False)
    print("TOTAL", int(out.correct.sum()), "/", len(out), "=", out.correct.mean())


if __name__ == "__main__":
    main()
