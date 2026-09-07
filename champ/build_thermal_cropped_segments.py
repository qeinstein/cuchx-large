"""Encode person-centered thermal crops for exact nested HARn segments."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image

import torch

try:
    _TV_LIB = torch.library.Library("torchvision", "DEF")
    _TV_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except RuntimeError:
    _TV_LIB = None
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "champ", "thermal_mnv3_cropped_segments.npz")
BATCH = 64
SAMPLES = 12


def crop_person(frame: np.ndarray) -> np.ndarray:
    """Crop around the high-temperature mass, suppressing room/background shortcuts."""
    value = frame[:, :, 0].astype(float) - 0.45 * frame[:, :, 2] + 0.15 * frame[:, :, 1]
    threshold = max(float(np.percentile(value, 92)), 50.0)
    weight = np.maximum(value - threshold, 0.0)
    yy, xx = np.indices(value.shape)
    total = weight.sum()
    if total > 1e-6:
        cx = int((weight * xx).sum() / total)
        cy = int((weight * yy).sum() / total)
    else:
        cy, cx = frame.shape[0] // 2, frame.shape[1] // 2
    # A 200px square retains limbs and manipulated objects while removing most of the room.
    radius = 100
    x0 = max(0, min(frame.shape[1] - 2 * radius, cx - radius))
    y0 = max(0, min(frame.shape[0] - 2 * radius, cy - radius))
    crop = frame[y0:y0 + 2 * radius, x0:x0 + 2 * radius]
    return np.asarray(Image.fromarray(crop).resize((160, 160), Image.Resampling.BILINEAR))


def load_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
    return model, mean, std


@torch.inference_mode()
def encode(model, mean, std, frames: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(frames), BATCH):
        x = torch.from_numpy(frames[start:start + BATCH]).permute(0, 3, 1, 2).float() / 255.0
        output.append(model((x - mean) / std).numpy().astype(np.float16))
    return np.concatenate(output)


def main() -> None:
    meta = pd.read_csv(os.path.join(ROOT, "champ", "meta.csv"))
    parents = {
        (row.user, row.trial): row
        for row in meta[meta.kind == "train_hau"].itertuples()
        if np.isfinite(row.f0)
    }
    by_parent = {}
    for row in meta[meta.kind == "train_harn"].itertuples():
        if (row.user, row.trial) in parents and np.isfinite(row.f0):
            by_parent.setdefault((row.user, row.trial), []).append(row)
    model, mean, std = load_model()
    store = {}
    started = time.time()
    nframe = 0
    for index, (key, segments) in enumerate(by_parent.items(), 1):
        parent = parents[key]
        path = os.path.join(
            ROOT, "hf_data_manual", parent.qa_path, "Thermal", "Thermal.mp4"
        )
        if not os.path.exists(path):
            continue
        video = VideoReader(path, num_threads=1)
        wanted = {}
        all_indices = []
        for segment in segments:
            lower = max(0, int(np.ceil((segment.f0 - parent.f0) / parent.fps * 25.0)))
            upper = min(len(video) - 1, int(np.floor((segment.f1 - parent.f0) / parent.fps * 25.0)))
            if upper < lower:
                continue
            ids = np.linspace(lower, upper, min(SAMPLES, upper - lower + 1)).round().astype(int)
            wanted[segment.qa_path] = ids
            all_indices.extend(ids.tolist())
        unique = np.asarray(sorted(set(all_indices)), dtype=int)
        if not len(unique):
            continue
        decoded = video.get_batch(unique).asnumpy()
        cropped = np.stack([crop_person(frame) for frame in decoded])
        features = encode(model, mean, std, cropped)
        lookup = {frame: features[i] for i, frame in enumerate(unique)}
        for segment, ids in wanted.items():
            store[segment] = np.stack([lookup[int(frame)] for frame in ids])
            nframe += len(ids)
        if index % 50 == 0:
            print(index, "/", len(by_parent), "frames", nframe,
                  "rate", round(nframe / (time.time() - started), 1), flush=True)
    np.savez_compressed(OUT, **store)
    print("wrote", OUT, "segments", len(store), "frames", nframe)


if __name__ == "__main__":
    main()
