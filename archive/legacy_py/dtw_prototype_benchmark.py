"""DTW and Nearest-Neighbor Action Prototype benchmark for sequence ordering and action recognition."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

ROOT = Path(__file__).parent


def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Computes Dynamic Time Warping distance between two multivariate time series."""
    # s1: (T1, D), s2: (T2, D)
    T1, T2 = len(s1), len(s2)
    cost = cdist(s1, s2, metric="cosine") # (T1, T2)
    dtw = np.zeros((T1 + 1, T2 + 1))
    dtw[0, :] = np.inf
    dtw[:, 0] = np.inf
    dtw[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[T1, T2])


def main():
    print("=== DTW / NEAREST-NEIGHBOR ACTION PROTOTYPE BENCHMARK (FOLD 0) ===")
    cache = np.load(ROOT / "temporal_streams_all.npz")
    X_skel_all = cache["X_skel_train"]
    all_paths = list(cache["train_paths"])
    p_to_idx = {p: i for i, p in enumerate(all_paths)}

    train_df = pd.read_csv(ROOT / "splits" / "fold_0_train.csv")
    val_df = pd.read_csv(ROOT / "splits" / "fold_0_val.csv")

    # 1. Extract single action prototypes from single-action clips in train_df
    # In HARn, each clip is a single action!
    harn_single = train_df[(train_df.source == "HARn") & (train_df.category == "single")].copy()
    harn_single["act_name"] = harn_single.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)

    action_prototypes = {}
    for act, grp in harn_single.groupby("act_name"):
        trajs = [X_skel_all[p_to_idx[p]] for p in grp.path if p in p_to_idx]
        if trajs:
            # Mean prototype across all single-action demonstrations
            action_prototypes[act] = np.mean(np.array(trajs), axis=0) # (64, 166)

    print(f"Constructed prototypes for {len(action_prototypes)} actions from HARn.")

    # Also extract prototypes from HAU single-action clips if missing
    hau_single = train_df[(train_df.source == "HAU") & (train_df.category == "single")].copy()
    hau_single["act_name"] = hau_single.apply(lambda r: str(r[r.answer]).strip().lower(), axis=1)
    for act, grp in hau_single.groupby("act_name"):
        if act not in action_prototypes:
            trajs = [X_skel_all[p_to_idx[p]] for p in grp.path if p in p_to_idx]
            if trajs:
                action_prototypes[act] = np.mean(np.array(trajs), axis=0)

    print(f"Total action prototypes constructed: {len(action_prototypes)}")

    # 2. Evaluate Sequence questions on Fold 0 Val using 4-quarter DTW / Prototype matching
    # In each clip, divide the 64-step sequence into 4 quarters: [0:16], [16:32], [32:48], [48:64]
    seq_val = val_df[val_df.category == "sequence"].copy()
    correct_dtw = 0
    pairwise_agree = 0
    total_pairs = 0

    for _, row in seq_val.iterrows():
        p = row["path"]
        if p not in p_to_idx:
            continue
        clip_stream = X_skel_all[p_to_idx[p]] # (64, 166)

        # 4 quarters of clip
        quarters = [
            clip_stream[0:16],
            clip_stream[16:32],
            clip_stream[32:48],
            clip_stream[48:64]
        ]

        # For each candidate action A, B, C, D: find best matching quarter
        quarter_match = {}
        for l in "ABCD":
            act_text = str(row[l]).strip().lower()
            if act_text in action_prototypes:
                proto = action_prototypes[act_text]
                # Downsample prototype to 16 steps
                proto_16 = np.mean(proto.reshape(4, 16, -1), axis=1) # (4, 166)
                # Compute distance to each of the 4 quarters
                distances = [np.linalg.norm(np.mean(q, axis=0) - np.mean(proto, axis=0)) for q in quarters]
                best_q = int(np.argmin(distances))
                quarter_match[l] = best_q + 0.1 * distances[best_q]
            else:
                quarter_match[l] = 2.0

        # Sort letters by best quarter
        pred_seq = "".join(sorted(["A", "B", "C", "D"], key=lambda l: (quarter_match[l], l)))
        true_seq = str(row["answer"]).strip().upper()

        if pred_seq == true_seq:
            correct_dtw += 1

        for i in range(4):
            for j in range(i + 1, 4):
                total_pairs += 1
                if pred_seq.index(true_seq[i]) < pred_seq.index(true_seq[j]):
                    pairwise_agree += 1

    acc = correct_dtw / max(1, len(seq_val)) * 100
    pair_acc = pairwise_agree / max(1, total_pairs) * 100
    print(f"\nDTW / Prototype Action Matching on Fold 0:")
    print(f"  Sequence Exact Acc: {correct_dtw}/{len(seq_val)} ({acc:.2f}%)")
    print(f"  Pairwise Agreement: {pairwise_agree}/{total_pairs} ({pair_acc:.2f}%)")


if __name__ == "__main__":
    main()
