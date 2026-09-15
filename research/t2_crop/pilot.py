"""Pilot: full-frame vs actor-crop DINOv2 transfer, subject-disjoint protocol.

Pool: 6 actions x 4 users = 24 clips (train HAU), labels from training_qa.csv
  ("Which action is performed in this video?"). 4 frames/clip, fractional sampling
  (aligns Depth@10fps with Thermal@25fps by timestamp fraction).
Backbone: frozen facebook/dinov2-small (ViT-S/14) CLS, 384-d, CPU.
Protocol: leave-1-user-out nearest-centroid (cosine), 4 folds.
  Clip embedding = L2-normalized mean of 4 frame CLS tokens.
Compares: full-frame vs crop margins 1.0 / 1.3 / 1.6, per modality.

Outputs: research/t2_crop/pilot_results.json, pilot_emb.npz (cache).
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/tmp/t2libs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_crops import boxes_for_margins, compute_box, crop_frame, sample_frames  # noqa: E402

ROOT = "/home/fluxx/Workspace/cuchx-large"
OUT = os.path.join(ROOT, "research", "t2_crop")
ACTIONS = ["Walking", "Eating", "Sitting down", "Peeling fruit",
           "Listening to the music with headphones", "Pouring"]
N_USERS = 4
N_FRAMES = 4
MARGINS = (1.0, 1.3, 1.6)
FOLDS = [(0,), (1,), (2,), (3,)]  # leave-1-user-out x4 over sorted pool users


def clip_path(qa_path: str, modality: str) -> str:
    return os.path.join(ROOT, "hf_data_manual", qa_path, modality, f"{modality}.mp4")


def build_pool() -> pd.DataFrame:
    tr = pd.read_csv(os.path.join(ROOT, "training_qa.csv"))
    q = tr[tr["question"] == "Which action is performed in this video?"].copy()
    q["label"] = q.apply(lambda r: r[r["answer"]], axis=1)
    q = q[q["label"].isin(ACTIONS)]
    q["user"] = q["path"].str.split("/").str[1]
    # users with the best action coverage
    cov = q.groupby("user")["label"].nunique().sort_values(ascending=False)
    users = cov.index.tolist()[:N_USERS]
    rows = []
    for a in ACTIONS:
        have = set()
        cands = q[(q["label"] == a) & (q["user"].isin(users))]
        for _, r in cands.iterrows():  # pass 1: distinct users
            if r["user"] in have:
                continue
            if os.path.exists(clip_path(r["path"], "Depth_Color")) and \
               os.path.exists(clip_path(r["path"], "Thermal")):
                rows.append({"qa_path": r["path"], "user": r["user"], "label": a})
                have.add(r["user"])
            if len(have) == N_USERS:
                break
        # pass 2: fill empty (action,user) cells with a 2nd clip of the same
        # action from a pool user (none expected for top-4 users x top-6 actions)
        used = {d["qa_path"] for d in rows}
        for _, r in cands.iterrows():
            if sum(1 for d in rows if d["label"] == a) >= N_USERS:
                break
            if r["path"] in used:
                continue
            if os.path.exists(clip_path(r["path"], "Depth_Color")) and \
               os.path.exists(clip_path(r["path"], "Thermal")):
                rows.append({"qa_path": r["path"], "user": r["user"],
                             "label": a, "fill": True})
                used.add(r["path"])
    pool = pd.DataFrame(rows)
    assert len(pool) == len(ACTIONS) * N_USERS, f"pool short: {len(pool)}"
    return pool


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_backbone():
    import os as _os
    _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _os.environ.setdefault("TQDM_DISABLE", "1")
    from transformers import Dinov2Model
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = Dinov2Model.from_pretrained(
        "facebook/dinov2-small", low_cpu_mem_usage=True).eval()
    return model


def preprocess(frames_bgr: list[np.ndarray]) -> torch.Tensor:
    """cv2 resize + normalize; same math as champ/build_dino_frames.py (RGB kept)."""
    arr = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
                    for f in frames_bgr]).astype(np.float32)  # BGR
    arr = arr[..., ::-1] / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


@torch.no_grad()
def embed_batch(model, batch: torch.Tensor) -> np.ndarray:
    out = model(pixel_values=batch).last_hidden_state[:, 0]
    return out.cpu().numpy().astype(np.float32)


def get_embeddings(pool, model) -> dict:
    """{(qa_path, modality, variant): (N_FRAMES, 384)} with npz cache."""
    cache_p = os.path.join(OUT, "pilot_emb.npz")
    cached = {}
    if os.path.exists(cache_p):
        with np.load(cache_p) as z:
            cached = {k: z[k] for k in z.files}
        print(f"loaded {len(cached)} cached embeddings")
    import gc
    emb = {tuple(k.rsplit("|", 2)): v for k, v in cached.items()}
    boxinfo_p = os.path.join(OUT, "pilot_boxes.json")
    boxinfo = json.load(open(boxinfo_p)) if os.path.exists(boxinfo_p) else {}
    todo = [(r.qa_path, m) for r in pool.itertuples()
            for m in ("Depth_Color", "Thermal")]
    done = sum(1 for qa, mod in todo
               if all((qa, mod, v) in emb for v in ("full", "1.0", "1.3", "1.6")))
    print(f"resuming: {done}/{len(todo)} clip-modalities done", flush=True)
    for ci, (qa, mod) in enumerate(todo):
        if all((qa, mod, v) in emb for v in ("full", "1.0", "1.3", "1.6")):
            continue
        frames = sample_frames(clip_path(qa, mod), max(N_FRAMES, 12))
        info = compute_box(frames, mod)
        W, H = info["size"]
        boxes = boxes_for_margins(info["box"], MARGINS, W, H)
        boxinfo[qa + "|" + mod] = {**info, "boxes": {str(k): v for k, v in boxes.items()}}
        idx = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
        sel = [frames[int(i)] for i in idx]
        variants = {"full": sel}
        for m, b in boxes.items():
            variants[f"{m:.1f}"] = [crop_frame(f, b) for f in sel]
        import time as _t
        t_start = _t.time()
        for v, frs in variants.items():
            xs = preprocess(frs)
            emb[(qa, mod, v)] = embed_batch(model, xs)  # N_FRAMES, one batch
            del xs
        del frames, sel, variants
        gc.collect()
        done += 1
        if done % 8 == 0 or done == len(todo):  # checkpoint: survives OOM kills
            np.savez(cache_p, **{f"{a}|{b}|{c}": v for (a, b, c), v in emb.items()})
            with open(boxinfo_p, "w") as f:
                json.dump(boxinfo, f)
        dt = _t.time() - t_start
        print(f"  embedded {done}/{len(todo)} {qa}|{mod} {dt:.0f}s", flush=True)
    return emb


def clip_vec(E: np.ndarray) -> np.ndarray:
    v = E.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def run_eval(pool, emb) -> dict:
    users = sorted(pool["user"].unique().tolist())
    labels = sorted(pool["label"].unique().tolist())
    res = {}
    for mod in ("Depth_Color", "Thermal"):
        for v in ("full", "1.0", "1.3", "1.6"):
            accs = []
            for fold in FOLDS:
                held = {users[i] for i in fold}
                tr = pool[~pool["user"].isin(held)]
                te = pool[pool["user"].isin(held)]
                cents = {}
                for lab in labels:
                    M = np.stack([clip_vec(emb[(r.qa_path, mod, v)])
                                  for r in tr[tr["label"] == lab].itertuples()])
                    c = M.mean(axis=0)
                    cents[lab] = c / (np.linalg.norm(c) + 1e-9)
                C = np.stack([cents[lab] for lab in labels])
                ok = tot = 0
                for r in te.itertuples():
                    s = C @ clip_vec(emb[(r.qa_path, mod, v)])
                    if labels[int(np.argmax(s))] == r.label:
                        ok += 1
                    tot += 1
                accs.append(ok / tot)
            res[f"{mod}|{v}"] = {"folds": accs, "mean": float(np.mean(accs))}
    return {"users": users, "labels": labels, "n_clips": len(pool),
            "chance": 1 / len(labels), "results": res}


def main() -> None:
    pool = build_pool()
    print(f"pool: {len(pool)} clips, users={sorted(pool['user'].unique())}")
    print(pool.groupby("label").size().to_string())
    model = load_backbone()
    print("backbone: facebook/dinov2-small",
          sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)
    emb = get_embeddings(pool, model)
    out = run_eval(pool, emb)
    with open(os.path.join(OUT, "pilot_results.json"), "w") as f:
        json.dump({"pool": pool.to_dict("records"), **out}, f, indent=1)
    print(f"\nleave-users-out nearest-centroid, chance={out['chance']:.3f}")
    for mod in ("Depth_Color", "Thermal"):
        row = "  " + mod + " " + " ".join(
            f"{v}={out['results'][f'{mod}|{v}']['mean']:.3f}"
            for v in ("full", "1.0", "1.3", "1.6"))
        print(row)
        for v in ("full", "1.0", "1.3", "1.6"):
            print(f"    {v}: folds={out['results'][f'{mod}|{v}']['folds']}")


if __name__ == "__main__":
    main()
