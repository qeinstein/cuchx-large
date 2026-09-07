"""Subject-disjoint thermal action probe against the exact shipped mechanisms.

Exact nested HARn intervals provide frame supervision inside HAU thermal videos.  Each outer
fold holds out entire users.  The probe reports standalone action recognition and, critically,
decision-level W->R / R->W transitions against the frozen pool and sequence predictions.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "champ"))
from pseudotest import folds  # noqa: E402


VOCAB = json.load(open(os.path.join(ROOT, "champ", "vocab.json")))["HARN2HAU"]
ACTIONS = sorted(VOCAB)
A2I = {action: index for index, action in enumerate(ACTIONS)}
TEXT2I = {text: A2I[action] for action, text in VOCAB.items() if action in A2I}
BG = len(ACTIONS)
MAX_SEGMENT_SAMPLES = int(os.environ.get("THERMAL_SEGMENT_SAMPLES", "8"))
CACHE_PATH = os.environ.get(
    "ACTION_FRAME_CACHE", os.path.join(ROOT, "champ", "thermal_mnv3_frames.npz")
)
SOURCE_FPS = float(os.environ.get("ACTION_SOURCE_FPS", "25"))
OUTPUT_PATH = os.environ.get(
    "ACTION_EVAL_OUT",
    os.path.join(ROOT, "research", "thermal_action_sequence_oof_20260906.csv"),
)


def answer(row) -> str:
    return "".join(letter for letter in str(row.answer) if letter in "ABCD")


def option_actions(text: str) -> list[int]:
    return [TEXT2I[x.strip()] for x in str(text).split(",") if x.strip() in TEXT2I]


def transition(base: str, new: str, truth: str) -> str:
    if base == new:
        return "same"
    if base != truth and new == truth:
        return "W->R"
    if base == truth and new != truth:
        return "R->W"
    return "W->W"


def choose_even(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit).round().astype(int)]


def action_score(prob: np.ndarray) -> np.ndarray:
    """Top-three mean probability for each action over a full HAU clip."""
    k = min(3, len(prob))
    return np.partition(prob[:, :BG], -k, axis=0)[-k:].mean(0)


def build_examples(meta: pd.DataFrame, cache):
    parents = {
        (row.user, row.trial): row
        for row in meta[meta.kind == "train_hau"].itertuples()
        if row.qa_path + "|X" in cache
    }
    segments = defaultdict(list)
    for row in meta[meta.kind == "train_harn"].itertuples():
        if row.action in A2I and np.isfinite(row.f0):
            segments[(row.user, row.trial)].append(row)

    features, labels, users = [], [], []
    segment_rows = []
    for key, parent in parents.items():
        frame25 = cache[parent.qa_path + "|F"].astype(float)
        x = cache[parent.qa_path + "|X"].astype(np.float32)
        # Convert retained source-video indices to the global skeleton/HARn frame axis.
        global10 = float(parent.f0) + frame25 * (float(parent.fps) / SOURCE_FPS)
        covered = np.zeros(len(frame25), bool)
        for segment in segments.get(key, []):
            mask = (global10 >= segment.f0) & (global10 <= segment.f1)
            ids = choose_even(np.flatnonzero(mask), MAX_SEGMENT_SAMPLES)
            if len(ids) == 0:
                continue
            covered[np.flatnonzero(mask)] = True
            features.append(x[ids])
            labels.extend([A2I[segment.action]] * len(ids))
            users.extend([parent.user] * len(ids))
            segment_rows.append((segment.qa_path, parent.user, A2I[segment.action], x[ids].mean(0)))
        # Explicit background prevents the frame model from hallucinating an action in every
        # quiet interval. Keep its weight comparable to one action segment per clip.
        ids = choose_even(np.flatnonzero(~covered), MAX_SEGMENT_SAMPLES)
        if len(ids):
            features.append(x[ids])
            labels.extend([BG] * len(ids))
            users.extend([parent.user] * len(ids))
    return (
        np.concatenate(features), np.asarray(labels), np.asarray(users), segment_rows, parents
    )


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "training_qa.csv")).set_index("qa_id", drop=False)
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    cache = np.load(CACHE_PATH)
    x, y, users, segments, parents = build_examples(meta, cache)
    print("frame samples", x.shape, "classes", len(np.unique(y)), "segments", len(segments))

    pool = pd.read_csv(
        os.path.join(ROOT, "research", "pool_full_subblocks_paired_oof_20260905.csv")
    ).drop_duplicates("qa_id").set_index("qa_id")
    seq = pd.read_csv(os.path.join(ROOT, "seqlab", "audit_run18.csv")).set_index("qa")
    base = {qid: str(row.pred_base) for qid, row in pool.iterrows()}
    base.update({qid: str(row.base) for qid, row in seq.iterrows()})

    rows = []
    segment_correct = 0
    segment_total = 0
    all_users = sorted(set(users))
    for fold, held in enumerate(folds(all_users, 5)):
        train_mask = ~np.isin(users, held)
        scaler = StandardScaler().fit(x[train_mask])
        clf = LogisticRegression(
            C=float(os.environ.get("THERMAL_C", "0.1")),
            max_iter=int(os.environ.get("THERMAL_MAX_ITER", "300")),
            class_weight="balanced",
            random_state=100 + fold,
        ).fit(scaler.transform(x[train_mask]), y[train_mask])

        held_segments = [item for item in segments if item[1] in held]
        if held_segments:
            sx = np.stack([item[3] for item in held_segments])
            pred = clf.predict(scaler.transform(sx))
            truth = np.asarray([item[2] for item in held_segments])
            segment_correct += int((pred == truth).sum())
            segment_total += len(truth)

        held_paths = [
            parent.qa_path for parent in parents.values()
            if parent.user in held and parent.qa_path + "|X" in cache
        ]
        for path in held_paths:
            frame = cache[path + "|F"].astype(int)
            feature = cache[path + "|X"].astype(np.float32)
            logits = clf.decision_function(scaler.transform(feature))
            prob = softmax(logits, axis=1)
            score = action_score(prob)
            for qid, row in tr[tr.path == path].iterrows():
                if qid not in base or row.category not in {"single", "multi", "combination", "sequence"}:
                    continue
                option_ids = [option_actions(row[letter]) for letter in "ABCD"]
                if any(not ids for ids in option_ids):
                    continue
                truth = answer(row)
                record = dict(
                    qa_id=qid, fold=fold, user=path.split("/")[1],
                    path=path, category=row.category, truth=truth, base=base[qid],
                    score=json.dumps([float(np.mean(score[ids])) for ids in option_ids]),
                )
                if row.category == "single":
                    values = [float(np.mean(score[ids])) for ids in option_ids]
                    record["visual"] = "ABCD"[int(np.argmax(values))]
                    record["confidence"] = float(
                        np.sort(values)[-1] - np.sort(values)[-2]
                    )
                elif row.category == "combination":
                    values = [float(np.mean(np.log(score[ids] + 1e-8))) for ids in option_ids]
                    record["visual"] = "ABCD"[int(np.argmax(values))]
                    record["confidence"] = float(
                        np.sort(values)[-1] - np.sort(values)[-2]
                    )
                elif row.category == "sequence":
                    timeline = [prob[:, ids].mean(1) for ids in option_ids]
                    centroid = [
                        float((p * frame).sum() / (p.sum() + 1e-12)) for p in timeline
                    ]
                    order = np.argsort(centroid)
                    record["visual"] = "".join("ABCD"[int(i)] for i in order)
                    record["confidence"] = float(np.min(np.diff(np.sort(centroid))))
                else:
                    # Store scores; multi decisions are swept after all outer folds complete.
                    record["visual"] = ""
                    record["confidence"] = np.nan
                rows.append(record)
        print("fold", fold, "held", held, "done", flush=True)

    out = pd.DataFrame(rows)
    for category in ("single", "combination", "sequence"):
        group = out[out.category == category].copy()
        group["transition"] = [
            transition(b, n, t) for b, n, t in zip(group.base, group.visual, group.truth)
        ]
        print(
            category, "n", len(group), "base", int((group.base == group.truth).sum()),
            "visual", int((group.visual == group.truth).sum()),
            "changes", group[group.base != group.visual].transition.value_counts().to_dict(),
        )
        for cutoff in np.quantile(group.confidence, [0.5, 0.75, 0.9]):
            gated = group[(group.base != group.visual) & (group.confidence >= cutoff)]
            print("  confidence >=", round(float(cutoff), 4), gated.transition.value_counts().to_dict())

    multi = out[out.category == "multi"].copy()
    scores = np.stack(multi.score.map(json.loads))
    for ratio in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70):
        pred = []
        conf = []
        for values in scores:
            chosen = np.flatnonzero(values >= ratio * values.max())
            pred.append("".join("ABCD"[int(i)] for i in chosen))
            boundary = ratio * values.max()
            conf.append(float(np.min(np.abs(values - boundary))))
        multi["visual"] = pred
        multi["confidence"] = conf
        multi["transition"] = [
            transition(b, n, t) for b, n, t in zip(multi.base, multi.visual, multi.truth)
        ]
        changed = multi[multi.base != multi.visual]
        print(
            "multi ratio", ratio, "visual", int((multi.visual == multi.truth).sum()),
            "changed", len(changed), changed.transition.value_counts().to_dict(),
        )
    print("segment top1", segment_correct, "/", segment_total,
          "=", segment_correct / max(segment_total, 1))
    out.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
