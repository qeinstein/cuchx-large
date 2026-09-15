"""Actor-centric crop boxes for CUHK-X Depth_Color / Thermal clips.

Strategy (no calibration exists -- skeleton is root-relative 3D, see REPORT.md):
clip-level box from foreground MOTION + a per-modality appearance prior.

- motion: mean abs diff of N probe frames from their temporal median (room is static).
- prior: Thermal -> hot mass (R-0.45B+0.15G, cf champ/build_thermal_cropped_segments.py);
          Depth_Color -> near mass (B-R on the rainbow colorization, relative per clip).
- mask = z(motion)+0.5*z(prior) > p90, largest connected component;
  fallbacks: motion-only > p90, prior-only > p90, center box (0.7 x min(H,W)).
- keep largest connected component, square-ify, expand by margin, clamp to frame.

One box per clip (stable across frames), resolution/fps agnostic
(train 320x240, test Depth 640x480, Depth 10fps, Thermal 25fps).

CLI: writes PNGs (full + crop variants) + boxes.json for one clip or a list file.
Library: compute_box(frames_bgr, modality), crop_frame(frame, box), sample_frames(path).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

MARGINS = (1.0, 1.3, 1.6)
N_PROBE = 12
MIN_MASS_FRAC = 0.005


def sample_frames(path: str, n: int) -> list[np.ndarray]:
    """Evenly spaced BGR frames (fractional positions -> fps agnostic)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    return frames


def _prior_map(probe_rgb: np.ndarray, modality: str) -> np.ndarray:
    """Mean appearance-prior map over probe frames. probe_rgb: (N,H,W,3) RGB float."""
    if modality == "Thermal":
        m = probe_rgb[..., 0] - 0.45 * probe_rgb[..., 2] + 0.15 * probe_rgb[..., 1]
    elif modality == "Depth_Color":
        m = probe_rgb[..., 2] - probe_rgb[..., 0]  # near=blue, far=red (relative)
    else:
        raise ValueError(f"unknown modality {modality}")
    return m.mean(axis=0)


def _motion_map(probe_gray: np.ndarray) -> np.ndarray:
    """Mean abs diff from temporal median. probe_gray: (N,H,W) float."""
    med = np.median(probe_gray, axis=0)
    return np.abs(probe_gray - med).mean(axis=0)


def _largest_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest connected component bbox as (x0, y0, x1, y1) exclusive; None if empty."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    x, y, w, h = (int(stats[i, k]) for k in
                  (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return (x, y, x + w, y + h)


def _square_expand(box, margin: float, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * margin / 2
    half = min(half, min(W, H) / 2)
    x0n = int(np.clip(round(cx - half), 0, W - 1))
    y0n = int(np.clip(round(cy - half), 0, H - 1))
    x1n = int(np.clip(round(cx + half), 1, W))
    y1n = int(np.clip(round(cy + half), 1, H))
    # keep square when clamped on one side
    side = min(x1n - x0n, y1n - y0n)
    x1n, y1n = x0n + side, y0n + side
    return (x0n, y0n, x1n, y1n)


def compute_box(frames_bgr: list[np.ndarray], modality: str,
                n_probe: int = N_PROBE) -> dict:
    """Clip-level actor box. Returns dict with base box, source, mass fracs, frame size."""
    sel = frames_bgr[::max(1, len(frames_bgr) // n_probe)][:n_probe]
    if len(sel) < 3:
        sel = frames_bgr
    stack = np.stack(sel).astype(np.float32)          # (N,H,W,3) BGR
    rgb = stack[..., ::-1]
    gray = (0.114 * stack[..., 0] + 0.587 * stack[..., 1] + 0.299 * stack[..., 2])
    H, W = gray.shape[1:]
    mot = _motion_map(gray)
    prior = _prior_map(rgb, modality)

    def _z(a):
        s = float(a.std())
        return (a - float(a.mean())) / (s + 1e-6)

    fused = _z(mot) + 0.5 * _z(prior)
    fmask = fused > float(np.percentile(fused, 90))
    mass = float(fmask.mean())
    source = "fused"
    box = _largest_box(fmask) if mass >= MIN_MASS_FRAC else None
    prior_mass = None
    if box is None:  # fallback 1: motion only
        mthr = float(np.percentile(mot, 90))
        mmask = mot > max(mthr, 1e-6)
        mass = float(mmask.mean())
        source = "motion"
        box = _largest_box(mmask) if mass >= MIN_MASS_FRAC else None
    if box is None:  # fallback 2: appearance prior only
        pthr = float(np.percentile(prior, 90))
        pmask = prior > pthr
        prior_mass = float(pmask.mean())
        source = "prior"
        box = _largest_box(pmask) if prior_mass >= MIN_MASS_FRAC else None
    if box is None:  # final fallback: center box
        source = "center"
        s = int(0.7 * min(H, W))
        box = ((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2)
    return {"box": [int(v) for v in box], "source": source,
            "motion_mass": mass, "prior_mass": prior_mass, "size": [W, H]}


def boxes_for_margins(base_box, margins, W: int, H: int) -> dict[float, list[int]]:
    return {m: list(_square_expand(base_box, m, W, H)) for m in margins}


def crop_frame(frame: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    return frame[y0:y1, x0:x1]


def process_clip(path: str, modality: str, out_dir: str, tag: str,
                 margins=MARGINS, n_frames: int = 8) -> dict:
    frames = sample_frames(path, max(n_frames, N_PROBE))
    info = compute_box(frames, modality)
    W, H = info["size"]
    boxes = boxes_for_margins(info["box"], margins, W, H)
    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
    os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
    for j, i in enumerate(idx):
        fr = frames[int(i)]
        cv2.imwrite(os.path.join(out_dir, tag, f"full_f{j}.png"), fr)
        for m, b in boxes.items():
            cv2.imwrite(os.path.join(out_dir, tag, f"crop{m:.1f}_f{j}.png"), crop_frame(fr, b))
    info.update({"boxes": {str(k): v for k, v in boxes.items()},
                 "clip": path, "modality": modality})
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Actor-centric crops for a clip.")
    ap.add_argument("clip", help="mp4 path OR @listfile (lines: <mp4> <modality> <tag>)")
    ap.add_argument("--modality", default="Depth_Color", choices=["Depth_Color", "Thermal"])
    ap.add_argument("--out", default="research/t2_crop/crops")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--margins", default="1.0,1.3,1.6")
    ap.add_argument("--n-frames", type=int, default=8)
    a = ap.parse_args()
    margins = tuple(float(x) for x in a.margins.split(","))
    jobs = []
    if a.clip.startswith("@"):
        for line in open(a.clip[1:]):
            p, m, t = line.split()
            jobs.append((p, m, t))
    else:
        tag = a.tag or os.path.splitext(os.path.basename(a.clip))[0]
        jobs.append((a.clip, a.modality, tag))
    all_info = {}
    for path, mod, tag in jobs:
        try:
            all_info[tag] = process_clip(path, mod, a.out, tag, margins, a.n_frames)
            print(f"OK {tag} box={all_info[tag]['box']} src={all_info[tag]['source']}")
        except Exception as e:  # noqa: BLE001 - batch job should continue
            print(f"FAIL {tag}: {e}")
    with open(os.path.join(a.out, "boxes.json"), "w") as f:
        json.dump(all_info, f, indent=1)
    print("wrote", os.path.join(a.out, "boxes.json"))


if __name__ == "__main__":
    sys.path.insert(0, "/tmp/t2libs")
    main()
