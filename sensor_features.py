"""Sensor feature extraction pipeline for CUHK-X Large Model Track.

Extracts:
1. IMU kinematics and spectrum (accelerometer, gyroscope, FFT bands, dominant frequency, jerk)
2. Skeleton kinematics (centered 17 joints, velocities, angles, temporal segments)
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).parent
LMT_ROOT = ROOT / "hf_data_manual" / "LMT_(IMU,Radar,Skeleton)"


def find_sensor_dir(path: str, source: str, split: str = "train") -> Optional[Path]:
    p = str(path).replace("\\", "/")
    if split == "test" or "large_model_track_test" in p:
        parts = p.split("/")
        for part in parts:
            if part.startswith("LM_test_"):
                candidate = LMT_ROOT / "Testing" / "large_model_track_test" / part
                if candidate.exists():
                    return candidate
    else:
        candidate = LMT_ROOT / "Training" / p
        if candidate.exists():
            return candidate
    return None


def extract_imu_features(sensor_dir: Path) -> np.ndarray:
    """Extract 120-dim spectral and kinematic summary from 5 IMU sensors."""
    imu_dir = sensor_dir / "IMU"
    if not imu_dir.exists():
        return np.zeros(120, dtype=np.float32)

    dfs = []
    for csv_name in ["up(LA+RA+C).csv", "down(LL+RL).csv"]:
        fpath = imu_dir / csv_name
        if fpath.exists() and fpath.stat().st_size > 50:
            try:
                df = pd.read_csv(fpath)
                dev_col = None
                for c in df.columns:
                    if "设备" in c or "device" in c.lower():
                        dev_col = c
                        break
                acc_cols = [c for c in df.columns if "加速度" in c or "acc" in c.lower()][:3]
                gyro_cols = [c for c in df.columns if "角速度" in c or "gyro" in c.lower()][:3]

                if dev_col and len(acc_cols) == 3:
                    sub_df = df[[dev_col] + acc_cols + (gyro_cols if len(gyro_cols) == 3 else [])].copy()
                    sub_df.columns = ["device", "ax", "ay", "az"] + (["gx", "gy", "gz"] if len(gyro_cols) == 3 else [])
                    dfs.append(sub_df)
            except Exception:
                pass

    if not dfs:
        return np.zeros(120, dtype=np.float32)

    all_data = pd.concat(dfs, ignore_index=True)
    for num_col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        if num_col in all_data.columns:
            all_data[num_col] = pd.to_numeric(all_data[num_col], errors="coerce").fillna(0.0)
        else:
            all_data[num_col] = 0.0

    target_sensors = ["WTC", "WTRA", "WTLA", "WTRL", "WTLL"]
    sensor_feats = []

    for s_name in target_sensors:
        s_mask = all_data["device"].astype(str).str.startswith(s_name)
        s_df = all_data[s_mask]
        if len(s_df) < 5:
            sensor_feats.extend([0.0] * 24)
            continue

        ax, ay, az = s_df["ax"].values, s_df["ay"].values, s_df["az"].values
        gx, gy, gz = s_df["gx"].values, s_df["gy"].values, s_df["gz"].values

        a_mag = np.sqrt(ax**2 + ay**2 + az**2)
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        a_mean, a_std, a_max = float(np.mean(a_mag)), float(np.std(a_mag)), float(np.max(a_mag))
        g_mean, g_std, g_max = float(np.mean(g_mag)), float(np.std(g_mag)), float(np.max(g_mag))
        a_jerk = float(np.std(np.diff(a_mag))) if len(a_mag) > 1 else 0.0
        g_jerk = float(np.std(np.diff(g_mag))) if len(g_mag) > 1 else 0.0

        n = len(a_mag)
        fft_a = np.abs(np.fft.rfft(a_mag - a_mean))
        fft_g = np.abs(np.fft.rfft(g_mag - g_mean))

        dom_freq_a = float(np.argmax(fft_a[1:]) + 1) / max(1, n) if len(fft_a) > 1 else 0.0
        dom_freq_g = float(np.argmax(fft_g[1:]) + 1) / max(1, n) if len(fft_g) > 1 else 0.0

        def band_energies(fft_vals):
            L = len(fft_vals)
            if L < 3:
                return 0.0, 0.0, 0.0
            p1, p2 = L // 3, 2 * L // 3
            e1 = float(np.sum(fft_vals[:p1] ** 2)) / max(1, p1)
            e2 = float(np.sum(fft_vals[p1:p2] ** 2)) / max(1, p2 - p1)
            e3 = float(np.sum(fft_vals[p2:] ** 2)) / max(1, L - p2)
            return e1, e2, e3

        e1_a, e2_a, e3_a = band_energies(fft_a)
        e1_g, e2_g, e3_g = band_energies(fft_g)

        def autocorr_features(series):
            if len(series) < 10:
                return 0.0, 0.0
            norm_s = series - np.mean(series)
            std_s = np.std(series)
            if std_s < 1e-6:
                return 0.0, 0.0
            ac = signal.correlate(norm_s, norm_s, mode="full")[len(norm_s) - 1 :]
            ac = ac / (ac[0] + 1e-9)
            peaks, _ = signal.find_peaks(ac[1:], distance=2)
            if len(peaks) > 0:
                first_peak_lag = float(peaks[0] + 1)
                first_peak_val = float(ac[peaks[0] + 1])
                return first_peak_lag, first_peak_val
            return 0.0, 0.0

        lag_a, peak_a = autocorr_features(a_mag)
        lag_g, peak_g = autocorr_features(g_mag)

        q25, q50, q75, q90 = np.percentile(a_mag, [25, 50, 75, 90])

        feats = [
            a_mean, a_std, a_max, a_jerk,
            g_mean, g_std, g_max, g_jerk,
            dom_freq_a, dom_freq_g, e1_a, e2_a, e3_a, e1_g, e2_g, e3_g,
            lag_a, peak_a, lag_g, peak_g,
            float(q25), float(q50), float(q75), float(q90),
        ]
        sensor_feats.extend(feats)

    return np.array(sensor_feats, dtype=np.float32)


def extract_skeleton_features(sensor_dir: Path) -> np.ndarray:
    """Extract 160-dim kinematic features from 3D/2D skeleton joint trajectories."""
    skel_dir = sensor_dir / "Skeleton" / "predictions"
    if not skel_dir.exists():
        return np.zeros(160, dtype=np.float32)

    files = sorted(glob.glob(str(skel_dir / "*.json")))
    if not files:
        return np.zeros(160, dtype=np.float32)

    step = max(1, len(files) // 60)
    sampled_files = files[::step][:60]

    all_kpts = []
    for f in sampled_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                if data and isinstance(data, list) and "keypoints" in data[0]:
                    kpts = np.array(data[0]["keypoints"])
                    all_kpts.append(kpts)
        except Exception:
            continue

    if not all_kpts:
        return np.zeros(160, dtype=np.float32)

    kpts_arr = np.array(all_kpts)
    T, num_kpts, _ = kpts_arr.shape

    hip_center = (kpts_arr[:, 11:12, :] + kpts_arr[:, 12:13, :]) / 2.0
    centered_kpts = kpts_arr - hip_center

    focus_joints = [0, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16]

    feats = []
    for j in focus_joints:
        j_pos = centered_kpts[:, j, :]
        feats.extend(np.mean(j_pos, axis=0))
        feats.extend(np.std(j_pos, axis=0))

    if T > 1:
        j_vel = np.diff(centered_kpts, axis=0)
        j_speeds = np.linalg.norm(j_vel, axis=-1)
        for j in focus_joints:
            s = j_speeds[:, j]
            feats.extend([float(np.mean(s)), float(np.std(s)), float(np.max(s))])
    else:
        feats.extend([0.0] * 33)

    upper_joints = centered_kpts[:, [0, 5, 6, 7, 8, 9, 10], :]
    lower_joints = centered_kpts[:, [11, 12, 13, 14, 15, 16], :]
    for group in [upper_joints, lower_joints]:
        mins = np.min(group, axis=(0, 1))
        maxs = np.max(group, axis=(0, 1))
        ranges = maxs - mins
        feats.extend(ranges.tolist())

    if T >= 3:
        t3 = T // 3
        early = centered_kpts[:t3, focus_joints, :]
        late = centered_kpts[-t3:, focus_joints, :]
        diff = np.mean(late, axis=0) - np.mean(early, axis=0)
        diff_mag = np.linalg.norm(diff, axis=-1)
        feats.extend(diff_mag.tolist())
        if T > 1:
            early_speed = np.mean(j_speeds[:max(1, t3), focus_joints], axis=0)
            late_speed = np.mean(j_speeds[-max(1, t3):, focus_joints], axis=0)
            speed_ratio = (late_speed - early_speed).tolist()
            feats.extend(speed_ratio)
        else:
            feats.extend([0.0] * 11)
    else:
        feats.extend([0.0] * 22)

    l_arm_speed = j_speeds[:, [7, 9]].mean() if T > 1 else 0.0
    r_arm_speed = j_speeds[:, [8, 10]].mean() if T > 1 else 0.0
    l_leg_speed = j_speeds[:, [13, 15]].mean() if T > 1 else 0.0
    r_leg_speed = j_speeds[:, [14, 16]].mean() if T > 1 else 0.0

    feats.extend([
        float(l_arm_speed), float(r_arm_speed), float(l_leg_speed), float(r_leg_speed),
        float(abs(l_arm_speed - r_arm_speed)), float(abs(l_leg_speed - r_leg_speed)),
        float(np.mean(j_speeds)) if T > 1 else 0.0,
        float(np.std(j_speeds)) if T > 1 else 0.0,
        float(np.max(j_speeds)) if T > 1 else 0.0,
        float(T),
        float(np.max(centered_kpts[:, :, 2]) - np.min(centered_kpts[:, :, 2])),
    ])

    out = np.zeros(160, dtype=np.float32)
    L = min(len(feats), 160)
    out[:L] = np.array(feats[:L], dtype=np.float32)
    return np.nan_to_num(out)


def get_clip_sensor_features(path: str, source: str, split: str = "train") -> np.ndarray:
    """Combines IMU (120-dim) and Skeleton (160-dim) into a 280-dim feature vector."""
    sensor_dir = find_sensor_dir(path, source, split)
    if sensor_dir is None:
        return np.zeros(280, dtype=np.float32)
    imu_f = extract_imu_features(sensor_dir)
    skel_f = extract_skeleton_features(sensor_dir)
    return np.concatenate([imu_f, skel_f])
