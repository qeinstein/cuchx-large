"""Build sparse pretrained MobileNetV3 features for HAU thermal videos.

Thermal video is recorded at 25 fps while skeleton/HARn frame coordinates use 10 fps.
The cache retains original thermal frame indices so downstream code can align exact HARn
intervals without assuming equal frame rates.  Only every fifth frame is encoded (5 fps).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from decord import VideoReader

import torch

# The CPU torchvision wheel does not ship custom detection ops in this environment.  Models
# do not need NMS, but torchvision registers a fake NMS kernel during import.
try:
    _TV_LIB = torch.library.Library("torchvision", "DEF")
    _TV_LIB.define(
        "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
    )
except RuntimeError:
    _TV_LIB = None
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cpu"
BATCH = 64
STRIDE = int(os.environ.get("CHAMP_THERMAL_STRIDE", "5"))
OUT = os.environ.get(
    "CHAMP_THERMAL_OUT", os.path.join(ROOT, "champ", "thermal_mnv3_frames.npz")
)


def video_path(qa_path: str, kind: str) -> str:
    if kind == "train_hau":
        return os.path.join(ROOT, "hf_data_manual", qa_path, "Thermal", "Thermal.mp4")
    return os.path.join(
        ROOT, "hf_data_manual", "large_model_track_test", qa_path,
        "Thermal", "Thermal.mp4",
    )


def model_and_normalizer():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval().to(DEVICE)
    mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
    return model, mean, std


@torch.inference_mode()
def encode(model, mean, std, path: str):
    video = VideoReader(path, num_threads=1)
    indices = np.arange(0, len(video), STRIDE, dtype=np.int32)
    chunks = []
    for start in range(0, len(indices), BATCH):
        frames = video.get_batch(indices[start:start + BATCH]).asnumpy()
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        # 160px is the native low-resolution MobileNetV3 recipe and is sufficient for the
        # 320x240 thermal source.
        x = torch.nn.functional.interpolate(
            x, size=(160, 160), mode="bilinear", align_corners=False
        )
        x = (x - mean) / std
        chunks.append(model(x).cpu().numpy().astype(np.float16))
    return indices, np.concatenate(chunks) if chunks else np.empty((0, 576), np.float16)


def main() -> None:
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    todo = [
        (row.qa_path, row.kind)
        for row in meta.itertuples()
        if row.kind in {"train_hau", "test"}
    ]
    store = {}
    if os.path.exists(OUT):
        old = np.load(OUT)
        store = {key: old[key] for key in old.files}
        print("resuming", len(store) // 2, "clips", flush=True)
    model, mean, std = model_and_normalizer()
    started = time.time()
    encoded = 0
    for index, (qa_path, kind) in enumerate(todo, 1):
        if qa_path + "|X" in store:
            continue
        path = video_path(qa_path, kind)
        if not os.path.exists(path):
            continue
        frames, features = encode(model, mean, std, path)
        store[qa_path + "|F"] = frames
        store[qa_path + "|X"] = features
        encoded += len(frames)
        if index % 25 == 0:
            elapsed = time.time() - started
            print(
                f"{index}/{len(todo)} new_frames={encoded} "
                f"rate={encoded / max(elapsed, 1e-6):.1f}/s",
                flush=True,
            )
        if index % 200 == 0:
            np.savez_compressed(OUT, **store)
    np.savez_compressed(OUT, **store)
    print("wrote", OUT, "clips", len(store) // 2, "new_frames", encoded)


if __name__ == "__main__":
    main()
