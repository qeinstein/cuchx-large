"""Generate verified Championship v7:
1. Re-evaluate 5-fold Sequence model to produce oof_v7_final.csv.
2. Train sequence model on full training set and update sequence predictions in submission_v7.csv.
3. Validate format and print complete 5-fold OOF table vs v6 and v5.
"""

from collections import Counter
import itertools
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from temporal_models import TemporalConvNet

ROOT = Path(__file__).parent
device = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib"


def main():
    print("=== PRODUCING VERIFIED CHAMPIONSHIP v7 ===")
    cache_t = np.load(ROOT / "temporal_streams_all.npz", allow_pickle=True)
    all_paths = list(cache_t["train_paths"])
    test_paths = list(cache_t["test_paths"])
    p_to_idx = {p: i for i, p in enumerate(all_paths)}
    te_p_to_idx = {p: i for i, p in enumerate(test_paths)}
    X_skel = cache_t["X_skel_train"]
    X_imu = cache_t["X_imu_train"]
    X_skel_te = cache_t["X_skel_test"]
    X_imu_te = cache_t["X_imu_test"]

    v6_df = pd.read_csv(ROOT / "oof_v6_final.csv")
    v7_df = v6_df.copy()

    t_steps = np.arange(64, dtype=np.float32)
    seq_preds = {}

    # 1. Evaluate 5 Folds for Sequence
    print("\n1. Running 5-Fold Cross-Validation for Sequence Model...")
    for f in range(5):
        train_df = pd.read_csv(ROOT / "splits" / f"fold_{f}_train.csv")
        val_df = pd.read_csv(ROOT / "splits" / f"fold_{f}_val.csv")

        hau = train_df[train_df.source == "HAU"]
        vocab = sorted(
            list(
                set(
                    str(r[r.answer]).strip().lower()
                    for _, r in hau[hau.category == "single"].iterrows()
                    if r.answer in "ABCD"
                )
            )
        )
        act_to_idx = {a: i for i, a in enumerate(vocab)}
        num_classes = len(vocab)

        before_counts = Counter()
        pair_total = Counter()
        for _, r in train_df[train_df.category == "sequence"].iterrows():
            ans = str(r["answer"]).strip().upper()
            if len(ans) == 4 and set(ans) == set("ABCD"):
                names = [str(r[l]).strip().lower() for l in ans]
                for i in range(4):
                    for j in range(i + 1, 4):
                        before_counts[(names[i], names[j])] += 1
                        pair_total[tuple(sorted([names[i], names[j]]))] += 1

        clip_items = []
        for p, g in hau.groupby("path"):
            if p not in p_to_idx:
                continue
            idx = p_to_idx[p]
            comb = np.concatenate([X_skel[idx], X_imu[idx]], axis=-1)
            y_clip = np.zeros(num_classes, dtype=np.float32)
            y_frame = np.zeros((64, num_classes), dtype=np.float32)
            frame_weight = 0.0
            seq_r = g[g.category == "sequence"]
            if len(seq_r):
                ans = str(seq_r.iloc[0]["answer"]).strip().upper()
                if len(ans) == 4 and set(ans) == set("ABCD"):
                    ordered = [str(seq_r.iloc[0][l]).strip().lower() for l in ans]
                    for i, a in enumerate(ordered):
                        if a in act_to_idx:
                            c = act_to_idx[a]
                            y_clip[c] = 1.0
                            y_frame[i * 16 : (i + 1) * 16, c] = 1.0
                    frame_weight = 3.0
            clip_items.append((comb, y_clip, y_frame, frame_weight))

        class SeqDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                c, yc, yf, fw = self.items[i]
                return (
                    torch.from_numpy(c).float(),
                    torch.from_numpy(yc).float(),
                    torch.from_numpy(yf).float(),
                    torch.tensor(fw).float(),
                )

        loader = torch.utils.data.DataLoader(SeqDataset(clip_items), batch_size=32, shuffle=True)
        torch.manual_seed(42 + f)
        tcn = TemporalConvNet(in_dim=216, num_classes=num_classes, hidden_dim=128).to(device)
        opt = torch.optim.AdamW(tcn.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()

        tcn.train()
        for ep in range(25):
            for bx, byc, byf, bfw in loader:
                bx, byc, byf, bfw = bx.to(device), byc.to(device), byf.to(device), bfw.to(device)
                opt.zero_grad()
                f_logits, c_logits = tcn(bx)
                loss = crit(c_logits, byc)
                mask = bfw > 0
                if mask.sum() > 0:
                    loss += (
                        bfw[mask].unsqueeze(1).unsqueeze(2)
                        * nn.functional.binary_cross_entropy_with_logits(
                            f_logits[mask], byf[mask], reduction="none"
                        )
                    ).mean()
                loss.backward()
                opt.step()
        tcn.eval()

        val_seq = val_df[val_df.category == "sequence"]
        with torch.no_grad():
            for _, r in val_seq.iterrows():
                p = r["path"]
                qid = r["qa_id"]
                if p not in p_to_idx:
                    continue
                comb = (
                    torch.from_numpy(np.concatenate([X_skel[p_to_idx[p]], X_imu[p_to_idx[p]]], axis=-1))
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                f_log, _ = tcn(comb)
                f_prob = torch.sigmoid(f_log)[0].cpu().numpy()

                opts = {l: str(r[l]).strip().lower() for l in "ABCD"}
                centroids = {}
                for l in "ABCD":
                    a = opts[l]
                    pt = f_prob[:, act_to_idx[a]] if a in act_to_idx else np.ones(64) / 64
                    w_sum = np.sum(pt)
                    tau = np.sum(t_steps * pt) / w_sum if w_sum > 1e-4 else 32.0
                    centroids[l] = tau

                best_perm, best_score = None, -np.inf
                for perm in itertools.permutations("ABCD"):
                    score = 0.0
                    for i in range(4):
                        for j in range(i + 1, 4):
                            kin_margin = (centroids[perm[j]] - centroids[perm[i]]) / 16.0
                            a, b = opts[perm[i]], opts[perm[j]]
                            cnt = before_counts[(a, b)]
                            tot = pair_total[tuple(sorted([a, b]))]
                            p_prior = (cnt + 1.0) / (tot + 2.0)
                            score += kin_margin + 1.0 * math.log(p_prior)
                    if score > best_score:
                        best_score = score
                        best_perm = "".join(perm)
                seq_preds[qid] = best_perm

    # Update sequence predictions in v7_df
    seq_mask = v7_df.category == "sequence"
    for idx in v7_df[seq_mask].index:
        qid = v7_df.loc[idx, "qa_id"]
        if qid in seq_preds:
            v7_df.loc[idx, "pred"] = seq_preds[qid]
    v7_df["correct"] = (v7_df["pred"] == v7_df["answer"]).astype(int)
    v7_df.to_csv(ROOT / "oof_v7_final.csv", index=False)

    print("\n" + "=" * 85)
    print("                 CHAMPIONSHIP v7 5-FOLD BENCHMARK REPORT (4,087 SAMPLES)")
    print("=" * 85)
    tot_corr = v7_df["correct"].sum()
    tot_samples = len(v7_df)
    v7_acc = tot_corr / tot_samples * 100
    print(f"Total Correct: {tot_corr} / {tot_samples} ({v7_acc:.2f}%)")
    print(f"v5 Baseline:   3045 / 4087 (74.50%)")
    print(f"v6 Baseline:   3080 / 4087 (75.36%)")
    print(f"Net Gain vs v5: +{tot_corr - 3045} correct answers (+{v7_acc - 74.50:.2f}%)")
    print(f"Net Gain vs v6: +{tot_corr - 3080} correct answers (+{v7_acc - 75.36:.2f}%)")

    print("\n=== Category Performance Comparison ===")
    print(f"{'Category':22s} | {'v6 Baseline':16s} | {'v7 Verified':16s} | {'Delta vs v6':12s}")
    print("-" * 75)
    for cat in sorted(v7_df["category"].unique()):
        sub_v6 = v6_df[v6_df.category == cat]
        sub_v7 = v7_df[v7_df.category == cat]
        c6 = sub_v6["correct"].sum()
        c7 = sub_v7["correct"].sum()
        tot = len(sub_v6)
        print(f"{cat:22s} | {c6:4d}/{tot:4d} ({c6/tot*100:5.2f}%) | {c7:4d}/{tot:4d} ({c7/tot*100:5.2f}%) | {c7 - c6:+d} ({(c7 - c6)/tot*100:+.2f}%)")

    # 2. Train Full TCN and Build submission_v7.csv
    print("\n2. Training Sequence TCN on Full Dataset for submission_v7.csv...")
    train_df = pd.read_csv(ROOT / "training_qa.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv")
    hau = train_df[train_df.source == "HAU"]
    vocab = sorted(
        list(
            set(
                str(r[r.answer]).strip().lower()
                for _, r in hau[hau.category == "single"].iterrows()
                if r.answer in "ABCD"
            )
        )
    )
    act_to_idx = {a: i for i, a in enumerate(vocab)}
    num_classes = len(vocab)

    before_counts = Counter()
    pair_total = Counter()
    for _, r in train_df[train_df.category == "sequence"].iterrows():
        ans = str(r["answer"]).strip().upper()
        if len(ans) == 4 and set(ans) == set("ABCD"):
            names = [str(r[l]).strip().lower() for l in ans]
            for i in range(4):
                for j in range(i + 1, 4):
                    before_counts[(names[i], names[j])] += 1
                    pair_total[tuple(sorted([names[i], names[j]]))] += 1

    clip_items = []
    for p, g in hau.groupby("path"):
        if p not in p_to_idx:
            continue
        idx = p_to_idx[p]
        comb = np.concatenate([X_skel[idx], X_imu[idx]], axis=-1)
        y_clip = np.zeros(num_classes, dtype=np.float32)
        y_frame = np.zeros((64, num_classes), dtype=np.float32)
        frame_weight = 0.0
        seq_r = g[g.category == "sequence"]
        if len(seq_r):
            ans = str(seq_r.iloc[0]["answer"]).strip().upper()
            if len(ans) == 4 and set(ans) == set("ABCD"):
                ordered = [str(seq_r.iloc[0][l]).strip().lower() for l in ans]
                for i, a in enumerate(ordered):
                    if a in act_to_idx:
                        c = act_to_idx[a]
                        y_clip[c] = 1.0
                        y_frame[i * 16 : (i + 1) * 16, c] = 1.0
                frame_weight = 3.0
        clip_items.append((comb, y_clip, y_frame, frame_weight))

    loader = torch.utils.data.DataLoader(SeqDataset(clip_items), batch_size=32, shuffle=True)
    torch.manual_seed(42)
    tcn_full = TemporalConvNet(in_dim=216, num_classes=num_classes, hidden_dim=128).to(device)
    opt = torch.optim.AdamW(tcn_full.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    tcn_full.train()
    for ep in range(25):
        for bx, byc, byf, bfw in loader:
            bx, byc, byf, bfw = bx.to(device), byc.to(device), byf.to(device), bfw.to(device)
            opt.zero_grad()
            f_logits, c_logits = tcn_full(bx)
            loss = crit(c_logits, byc)
            mask = bfw > 0
            if mask.sum() > 0:
                loss += (
                    bfw[mask].unsqueeze(1).unsqueeze(2)
                    * nn.functional.binary_cross_entropy_with_logits(
                        f_logits[mask], byf[mask], reduction="none"
                    )
                ).mean()
            loss.backward()
            opt.step()
    tcn_full.eval()

    # Load submission_v6 as base
    sub_v7 = pd.read_csv(ROOT / "submission_v6.csv")
    pred_map = dict(zip(sub_v7["qa_id"], sub_v7["prediction"]))
    changes = 0

    te_seq = test_df[test_df.category == "sequence"]
    with torch.no_grad():
        for _, r in te_seq.iterrows():
            p = r["path"]
            qid = r["qa_id"]
            if p not in te_p_to_idx:
                continue
            comb = (
                torch.from_numpy(np.concatenate([X_skel_te[te_p_to_idx[p]], X_imu_te[te_p_to_idx[p]]], axis=-1))
                .unsqueeze(0)
                .float()
                .to(device)
            )
            f_log, _ = tcn_full(comb)
            f_prob = torch.sigmoid(f_log)[0].cpu().numpy()

            opts = {l: str(r[l]).strip().lower() for l in "ABCD"}
            centroids = {}
            for l in "ABCD":
                a = opts[l]
                pt = f_prob[:, act_to_idx[a]] if a in act_to_idx else np.ones(64) / 64
                w_sum = np.sum(pt)
                tau = np.sum(t_steps * pt) / w_sum if w_sum > 1e-4 else 32.0
                centroids[l] = tau

            best_perm, best_score = None, -np.inf
            for perm in itertools.permutations("ABCD"):
                score = 0.0
                for i in range(4):
                    for j in range(i + 1, 4):
                        kin_margin = (centroids[perm[j]] - centroids[perm[i]]) / 16.0
                        a, b = opts[perm[i]], opts[perm[j]]
                        cnt = before_counts[(a, b)]
                        tot = pair_total[tuple(sorted([a, b]))]
                        p_prior = (cnt + 1.0) / (tot + 2.0)
                        score += kin_margin + 1.0 * math.log(p_prior)
                if score > best_score:
                    best_score = score
                    best_perm = "".join(perm)
            if best_perm != pred_map[qid]:
                pred_map[qid] = best_perm
                changes += 1

    print(f"Total Sequence changes in test over v6: {changes}")
    sub_v7["prediction"] = sub_v7["qa_id"].map(pred_map)

    assert len(sub_v7) == 682
    assert list(sub_v7.columns) == ["qa_id", "prediction"]
    assert sub_v7["prediction"].isna().sum() == 0

    out_file = ROOT / "submission_v7.csv"
    sub_v7.to_csv(out_file, index=False)
    print(f"Successfully saved verified {out_file} (682 rows)!")


if __name__ == "__main__":
    main()
