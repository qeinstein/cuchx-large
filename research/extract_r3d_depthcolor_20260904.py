"""Frozen torchvision R3D features for the HAU Depth_Color videos.

This is an isolated research cache.  It deliberately never imports or modifies the
championship solver.  The input is the native-rate video sampled at 16 evenly-spaced
frames; duration and frame count remain available separately through champ/meta.csv.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "r3d_depthcolor_features.npz"
NFRAMES = 16
SIZE = 112


def video_path(qa_path):
    if qa_path.startswith("HAU/"):
        return ROOT / "hf_data_manual" / qa_path / "Depth_Color" / "Depth_Color.mp4"
    if qa_path.startswith("LM_test_"):
        return (ROOT / "hf_data_manual" / "large_model_track_test" / qa_path
                / "Depth_Color" / "Depth_Color.mp4")
    raise ValueError(qa_path)


def read_clip(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 1:
        cap.release()
        raise RuntimeError(f"no frames: {path}")
    picks = np.linspace(0, max(0, n - 1), NFRAMES).round().astype(int)
    frames = []
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            # A few codecs report an optimistic frame count at the tail.
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(i - 1))))
            ok, fr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"decode failed at {i}/{n}: {path}")
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    x = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.43216, 0.394666, 0.37645], np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def main():
    meta = pd.read_csv(ROOT / "champ" / "meta.csv")
    qa = meta[meta.kind.isin(["train_hau", "test"])].qa_path.tolist()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    out = {}
    missing = []
    with torch.inference_mode():
        for start in range(0, len(qa), 4):
            keys = qa[start:start + 4]
            clips = []
            good = []
            for key in keys:
                p = video_path(key)
                try:
                    clips.append(read_clip(p))
                    good.append(key)
                except Exception as exc:
                    missing.append((key, str(exc)))
            if clips:
                # read_clip returns T,C,H,W; torchvision video models consume B,C,T,H,W.
                x = torch.stack(clips).permute(0, 2, 1, 3, 4).to(device)
                z = model(x).detach().float().cpu().numpy()
                for key, row in zip(good, z):
                    out[key] = row.astype(np.float32)
            if (start + len(keys)) % 40 == 0 or start + len(keys) == len(qa):
                print(f"{start + len(keys)}/{len(qa)} features", flush=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT} keys={len(out)} missing={len(missing)} device={device}")
    for row in missing[:20]:
        print("MISSING", row)


if __name__ == "__main__":
    main()
