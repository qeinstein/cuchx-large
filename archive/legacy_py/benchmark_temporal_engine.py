"""Benchmark temporal models across subject-held-out validation folds.

Evaluates:
1. TemporalConvNet (TCN)
2. BiGRUNet
3. TemporalTransformerNet
4. DualStreamTemporalNet

Measures:
- Sequence 4-letter exact permutation accuracy
- Sequence pairwise ordering agreement (out of 6 pairs)
- Action classification F1 and Single action accuracy
"""

import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from temporal_models import TemporalConvNet, BiGRUNet, TemporalTransformerNet, DualStreamTemporalNet

ROOT = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class MultimodalTemporalDataset(Dataset):
    def __init__(self, clip_paths, X_skel_all, X_imu_all, path_to_idx, clip_labels, num_classes, seq_targets=None):
        self.items = []
        self.num_classes = num_classes
        for p in clip_paths:
            if p not in path_to_idx:
                continue
            idx = path_to_idx[p]
            skel = X_skel_all[idx] # (64, 166)
            imu = X_imu_all[idx]   # (64, 50)
            comb = np.concatenate([skel, imu], axis=-1) # (64, 216)
            y_clip = clip_labels.get(p, np.zeros(num_classes, dtype=np.float32))

            # Frame-level sequence supervision if available
            has_seq = 0.0
            y_frame = np.zeros((64, num_classes), dtype=np.float32)
            if seq_targets and p in seq_targets:
                ordered_acts = seq_targets[p] # list of 4 action indices
                if len(ordered_acts) == 4:
                    has_seq = 1.0
                    y_frame[0:16, ordered_acts[0]] = 1.0
                    y_frame[16:32, ordered_acts[1]] = 1.0
                    y_frame[32:48, ordered_acts[2]] = 1.0
                    y_frame[48:64, ordered_acts[3]] = 1.0

            self.items.append((comb, skel, imu, y_clip, y_frame, has_seq, p))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        comb, skel, imu, y_clip, y_frame, has_seq, p = self.items[idx]
        return (
            torch.from_numpy(comb).float(),
            torch.from_numpy(skel).float(),
            torch.from_numpy(imu).float(),
            torch.from_numpy(y_clip).float(),
            torch.from_numpy(y_frame).float(),
            torch.tensor(has_seq, dtype=torch.float32),
            p
        )


def build_clip_labels(df, vocab):
    act_to_idx = {a: i for i, a in enumerate(vocab)}
    clip_labels = {}
    seq_targets = {}

    for p, g in df.groupby("path"):
        vec = np.zeros(len(vocab), dtype=np.float32)
        s = g[g.category == "single"]
        if len(s) and s.iloc[0]["answer"] in "ABCD":
            a = str(s.iloc[0][s.iloc[0]["answer"]]).strip().lower()
            if a in act_to_idx:
                vec[act_to_idx[a]] = 1.0

        m = g[g.category == "multi"]
        if len(m):
            for l in str(m.iloc[0]["answer"]):
                if l in "ABCD":
                    a = str(m.iloc[0][l]).strip().lower()
                    if a in act_to_idx:
                        vec[act_to_idx[a]] = 1.0

        c = g[g.category == "combination"]
        if len(c) and c.iloc[0]["answer"] in "ABCD":
            for x in str(c.iloc[0][c.iloc[0]["answer"]]).split(","):
                a = x.strip().lower()
                if a in act_to_idx:
                    vec[act_to_idx[a]] = 1.0

        seq = g[g.category == "sequence"]
        if len(seq):
            seq_r = seq.iloc[0]
            ans = str(seq_r["answer"]).strip().upper()
            if len(ans) == 4 and set(ans) == set("ABCD"):
                ordered_idx = []
                for l in ans:
                    a = str(seq_r[l]).strip().lower()
                    if a in act_to_idx:
                        vec[act_to_idx[a]] = 1.0
                        ordered_idx.append(act_to_idx[a])
                if len(ordered_idx) == 4:
                    seq_targets[p] = ordered_idx

        clip_labels[p] = vec

    return clip_labels, seq_targets


def decode_sequence_from_centroids(frame_probs: np.ndarray, seq_row: pd.Series, act_to_idx: dict) -> str:
    """Predict 4-letter sequence order by computing temporal centroids of the 4 candidate actions."""
    T = frame_probs.shape[0] # 64
    t_steps = np.arange(T, dtype=np.float32)

    centroids = {}
    for letter in "ABCD":
        act_text = str(seq_row[letter]).strip().lower()
        if act_text in act_to_idx:
            idx = act_to_idx[act_text]
            probs_t = frame_probs[:, idx]
            weight_sum = np.sum(probs_t)
            if weight_sum > 1e-4:
                tau = np.sum(t_steps * probs_t) / weight_sum
            else:
                tau = 32.0 # Neutral midpoint
            centroids[letter] = tau
        else:
            centroids[letter] = 32.0

    # Sort letters by centroid timestamp (smallest centroid = happened first)
    sorted_letters = sorted(["A", "B", "C", "D"], key=lambda l: (centroids[l], l))
    return "".join(sorted_letters)


def count_pairwise_agreement(true_seq: str, pred_seq: str) -> int:
    if len(true_seq) != 4 or len(pred_seq) != 4 or set(true_seq) != set("ABCD") or set(pred_seq) != set("ABCD"):
        return 0
    agree = 0
    for i in range(4):
        for j in range(i + 1, 4):
            t_a, t_b = true_seq[i], true_seq[j]
            if pred_seq.index(t_a) < pred_seq.index(t_b):
                agree += 1
    return agree


def train_and_eval_model(model_name: str, train_loader, val_loader, val_df, act_to_idx, num_epochs: int = 15):
    num_classes = len(act_to_idx)
    print(f"\n--- Training {model_name} on {DEVICE} ({num_epochs} epochs, {num_classes} classes) ---")
    if model_name == "TemporalConvNet":
        model = TemporalConvNet(in_dim=216, num_classes=num_classes, hidden_dim=128).to(DEVICE)
    elif model_name == "BiGRUNet":
        model = BiGRUNet(in_dim=216, num_classes=num_classes, hidden_dim=128).to(DEVICE)
    elif model_name == "TemporalTransformerNet":
        model = TemporalTransformerNet(in_dim=216, num_classes=num_classes, d_model=128, nhead=4).to(DEVICE)
    elif model_name == "DualStreamTemporalNet":
        model = DualStreamTemporalNet(skel_dim=166, imu_dim=50, num_classes=num_classes, d_model=96).to(DEVICE)
    else:
        raise ValueError(model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_clip = nn.BCEWithLogitsLoss()
    criterion_frame = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(num_epochs):
        tot_loss = 0.0
        for comb, skel, imu, y_clip, y_frame, has_seq, _ in train_loader:
            comb = comb.to(DEVICE)
            skel = skel.to(DEVICE)
            imu = imu.to(DEVICE)
            y_clip = y_clip.to(DEVICE)
            y_frame = y_frame.to(DEVICE)
            has_seq = has_seq.to(DEVICE)

            optimizer.zero_grad()
            if model_name == "DualStreamTemporalNet":
                f_logits, c_logits = model(skel, imu)
            else:
                f_logits, c_logits = model(comb)

            loss_c = criterion_clip(c_logits, y_clip)
            # Frame loss on clips with sequence supervision
            seq_mask = has_seq > 0.5
            if seq_mask.sum() > 0:
                loss_f = criterion_frame(f_logits[seq_mask], y_frame[seq_mask])
            else:
                loss_f = torch.tensor(0.0, device=DEVICE)

            loss = loss_c + 2.0 * loss_f
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

    # Evaluation on Validation Set
    model.eval()
    clip_frame_probs = {}
    clip_action_probs = {}

    with torch.no_grad():
        for comb, skel, imu, _, _, _, paths in val_loader:
            comb = comb.to(DEVICE)
            skel = skel.to(DEVICE)
            imu = imu.to(DEVICE)

            if model_name == "DualStreamTemporalNet":
                f_logits, c_logits = model(skel, imu)
            else:
                f_logits, c_logits = model(comb)

            f_probs = torch.sigmoid(f_logits).cpu().numpy()
            c_probs = torch.sigmoid(c_logits).cpu().numpy()

            for i, p in enumerate(paths):
                clip_frame_probs[p] = f_probs[i]
                clip_action_probs[p] = c_probs[i]

    # Evaluate Sequence predictions via Temporal Centroids
    seq_val = val_df[val_df.category == "sequence"].copy()
    seq_correct = 0
    pairwise_agreements = []

    for _, row in seq_val.iterrows():
        p = row["path"]
        f_probs = clip_frame_probs.get(p, np.zeros((64, 40), dtype=np.float32))
        pred_seq = decode_sequence_from_centroids(f_probs, row, act_to_idx)
        true_seq = str(row["answer"]).strip().upper()

        if pred_seq == true_seq:
            seq_correct += 1
        pairwise_agreements.append(count_pairwise_agreement(true_seq, pred_seq))

    seq_acc = seq_correct / max(1, len(seq_val)) * 100
    mean_pairs = np.mean(pairwise_agreements) if pairwise_agreements else 0.0

    print(f"Results for {model_name}:")
    print(f"  Sequence Exact Acc: {seq_correct}/{len(seq_val)} ({seq_acc:.2f}%) [Baseline was 27.69%]")
    print(f"  Pairwise Agreement: {mean_pairs:.2f} / 6.0 ({mean_pairs/6.0*100:.2f}%)")

    return {
        "model": model_name,
        "seq_acc": seq_acc,
        "pairwise_acc": mean_pairs / 6.0 * 100,
        "clip_action_probs": clip_action_probs,
        "clip_frame_probs": clip_frame_probs,
    }


def main():
    print("=== MULTI-ARCHITECTURE TEMPORAL BENCHMARK (FOLD 0) ===")
    cache = np.load(ROOT / "temporal_streams_all.npz")
    X_skel_all = cache["X_skel_train"]
    X_imu_all = cache["X_imu_train"]
    all_paths = list(cache["train_paths"])
    path_to_idx = {p: i for i, p in enumerate(all_paths)}

    train_df = pd.read_csv(ROOT / "splits" / "fold_0_train.csv")
    val_df = pd.read_csv(ROOT / "splits" / "fold_0_val.csv")

    vocab = sorted(list(set(
        str(r[r.answer]).strip().lower()
        for _, r in train_df[train_df.category == "single"].iterrows()
        if r.answer in "ABCD"
    )))
    act_to_idx = {a: i for i, a in enumerate(vocab)}
    print(f"Loaded {len(vocab)} actions in vocabulary.")

    clip_labels_tr, seq_targets_tr = build_clip_labels(train_df, vocab)
    clip_labels_val, _ = build_clip_labels(val_df, vocab)

    tr_paths = [p for p in train_df["path"].unique() if p in path_to_idx]
    val_paths = [p for p in val_df["path"].unique() if p in path_to_idx]

    ds_tr = MultimodalTemporalDataset(tr_paths, X_skel_all, X_imu_all, path_to_idx, clip_labels_tr, len(vocab), seq_targets_tr)
    ds_val = MultimodalTemporalDataset(val_paths, X_skel_all, X_imu_all, path_to_idx, clip_labels_val, len(vocab))

    loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=32, shuffle=False)

    models_to_test = [
        "TemporalConvNet",
        "BiGRUNet",
        "TemporalTransformerNet",
        "DualStreamTemporalNet"
    ]

    results = []
    for m in models_to_test:
        t0 = time.time()
        res = train_and_eval_model(m, loader_tr, loader_val, val_df, act_to_idx, num_epochs=20)
        res["time_s"] = time.time() - t0
        results.append(res)

    print("\n" + "=" * 75)
    print(f"{'Model Architecture':25s} | {'Seq Exact Acc':15s} | {'Pairwise Agree':16s} | {'Train Time':10s}")
    print("-" * 75)
    print(f"{'Baseline ClipGraphDecoder':25s} | {'27.69% (18/65)':15s} | {'75.92%':16s} | {'-':10s}")
    for r in results:
        print(f"{r['model']:25s} | {r['seq_acc']:6.2f}%        | {r['pairwise_acc']:6.2f}%         | {r['time_s']:6.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
