"""Pool sparse thermal MobileNet frame features into clip-level manner descriptors."""
from __future__ import annotations

import os

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "champ", "thermal_mnv3_frames.npz")
OUTPUT = os.path.join(ROOT, "research", "thermal_mnv3_clip_features_20260906.npz")


def descriptor(value: np.ndarray) -> np.ndarray:
    x = value.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    velocity = np.abs(np.diff(x, axis=0))
    acceleration = np.abs(np.diff(x, n=2, axis=0))
    if len(velocity) == 0:
        velocity = np.zeros_like(x[:1])
    if len(acceleration) == 0:
        acceleration = np.zeros_like(x[:1])
    scalar_v = np.linalg.norm(np.diff(x, axis=0), axis=1)
    if len(scalar_v) == 0:
        scalar_v = np.zeros(1, np.float32)
    scalar_a = np.abs(np.diff(scalar_v))
    if len(scalar_a) == 0:
        scalar_a = np.zeros(1, np.float32)
    scalar = np.array(
        [
            scalar_v.mean(), scalar_v.std(), np.median(scalar_v),
            np.quantile(scalar_v, 0.9), scalar_v.max(),
            scalar_a.mean(), np.quantile(scalar_a, 0.9),
            len(x), np.linalg.norm(x[-1] - x[0]),
        ],
        np.float32,
    )
    # Appearance variance plus direction-aware feature-space motion.  Absolute mean
    # appearance is deliberately excluded to reduce room/action shortcutting.
    return np.concatenate([x.std(0), velocity.mean(0), acceleration.mean(0), scalar])


def main() -> None:
    source = np.load(SOURCE)
    output = {
        key[:-2]: descriptor(source[key])
        for key in source.files if key.endswith("|X") and len(source[key])
    }
    np.savez_compressed(OUTPUT, **output)
    print("wrote", OUTPUT, "clips", len(output), "dim", len(next(iter(output.values()))))


if __name__ == "__main__":
    main()
