"""Full-scale frozen DINOv2 (ViT-S/14) on full-frame + 1.6x actor crop.

Follows champ/build_dino_frames.py conventions (RES=224, ImageNet normalize,
(torch.hub dinov2_vits14), per-clip (T,384) float16 arrays) but embeds TWO
streams per clip: full frame and the research/t2_crop actor box at 1.6x margin.

Box: make_crops.compute_box on 12 probe frames (motion + modality prior);
  single box per clip. Videos resolved for train_hau + test, Depth_Color and
  (when present) Thermal.

Output: champ/dino_full_crop16_<mod>.npz with keys <qa_path>|full and
  <qa_path>|crop1.6 (+ resume support like build_dino_frames.py).

Kaggle GPU usage:
  python3 research/t2_crop/build_dino_crops.py --mod Depth_Color
  python3 research/t2_crop/build_dino_crops.py --mod Thermal
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_crops import boxes_for_margins, compute_box, crop_frame  # noqa: E402

ROOT = "/home/fluxx/Workspace/cuchx-large"
RES = 224
BATCH = 64
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def video_path(qa_path: str, kind: str, mod: str) -> str:
    if kind == "train_hau":
        return os.path.join(ROOT, "hf_data_manual", qa_path, mod, f"{mod}.mp4")
    return os.path.join(ROOT, "hf_data_manual", "large_model_track_test",
                        qa_path, mod, f"{mod}.mp4")


def load_model(dev: str):
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                       pretrained=True, trust_repo=True, verbose=False)
    return m.eval().to(dev)


def probe_box(path: str, mod: str) -> list[int]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), min(12, max(total, 1))).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    info = compute_box(frames, mod)
    W, H = info["size"]
    return boxes_for_margins(info["box"], (1.6,), W, H)[1.6]


@torch.no_grad()
def encode_video(model, path: str, box, dev: str, stride: int = 1):
    cap = cv2.VideoCapture(path)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    full_f, crop_f, buf_f, buf_c, kept = [], [], [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if kept % stride:
            kept += 1
            continue
        kept += 1
        buf_f.append(cv2.resize(fr, (RES, RES), interpolation=cv2.INTER_AREA))
        buf_c.append(cv2.resize(crop_frame(fr, box), (RES, RES),
                               interpolation=cv2.INTER_AREA))
        if len(buf_f) == BATCH:
            full_f.append(_flush(model, buf_f, mean, std, dev)); buf_f = []
            crop_f.append(_flush(model, buf_c, mean, std, dev)); buf_c = []
    if buf_f:
        full_f.append(_flush(model, buf_f, mean, std, dev))
        crop_f.append(_flush(model, buf_c, mean, std, dev))
    cap.release()
    if not full_f:
        return None, None
    return (np.concatenate(full_f, 0).astype(np.float16),
            np.concatenate(crop_f, 0).astype(np.float16))


def _flush(model, buf, mean, std, dev):
    x = torch.from_numpy(np.stack(buf)).to(dev).float().div_(255.0)
    x = torch.flip(x, dims=[3]).permute(0, 3, 1, 2).contiguous()  # BGR -> RGB
    x = (x - mean) / std
    return model(x).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", default="Depth_Color", choices=["Depth_Color", "Thermal"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N clips only")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    todo = [(r.qa_path, r.kind) for r in meta.itertuples()
            if r.kind in ("train_hau", "test")]
    if a.limit:
        todo = todo[:a.limit]
    out_p = os.path.join(ROOT, "champ", f"dino_full_crop16_{a.mod}.npz")
    store: dict = {}
    if os.path.exists(out_p):
        z = np.load(out_p)
        store = {k: z[k] for k in z.files}
        print("resuming with", len(store) // 2, "clips already done", flush=True)
    model = load_model(dev)
    t0, nfr = time.time(), 0
    for i, (p, kind) in enumerate(todo):
        if f"{p}|full" in store and f"{p}|crop1.6" in store:
            continue
        vp = video_path(p, kind, a.mod)
        if not os.path.exists(vp):
            continue
        try:
            box = probe_box(vp, a.mod)
            F, C = encode_video(model, vp, box, dev, a.stride)
        except Exception as e:  # noqa: BLE001 - batch job should continue
            print("  ERR", p, e, flush=True)
            continue
        if F is None:
            continue
        store[f"{p}|full"], store[f"{p}|crop1.6"] = F, C
        nfr += len(F)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)} frames {nfr} "
                  f"{nfr / max(1e-9, el):.1f} fps elapsed {el / 60:.1f}m", flush=True)
        if (i + 1) % 200 == 0:
            np.savez(out_p, **store)
    np.savez(out_p, **store)
    print(f"done: {len(store) // 2} clips, {nfr} frames in "
          f"{(time.time() - t0) / 60:.1f} min -> {out_p}")


if __name__ == "__main__":
    main()
