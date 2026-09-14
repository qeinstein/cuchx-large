"""Extract and cache raw 64-timestep continuous temporal streams for Skeleton and IMU.

Outputs temporal_streams_all.npz:
- X_skel_train: (1333, 64, 166)
- X_skel_test:  (208, 64, 166)
- X_imu_train:  (1333, 64, 50)
- X_imu_test:   (208, 64, 50)
- train_paths, test_paths
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).parent
LMT_ROOT = ROOT / "hf_data_manual" / "LMT_(IMU,Radar,Skeleton)"
TARGET_T = 64


def find_sensor_dir(path: str, source: str, split: str = "train") -> Path | None:
    p = str(path).replace("\\", "/")
    if split == "test" or "large_model_track_test" in p:
        parts = p.split("/")
        for part in parts:
            if part.startswith("LM_test_"):
                cand = LMT_ROOT / "Testing" / "large_model_track_test" / part
                if cand.exists():
                    return cand
    else:
        cand = LMT_ROOT / "Training" / p
        if cand.exists():
            return cand
    return None


def calculate_joint_angle(p1, p2, p3):
    """Angle between p1-p2 and p3-p2 at vertex p2."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def extract_clip_skeleton_stream(sensor_dir: Path | None) -> np.ndarray:
    """Extract (T=64, 166) continuous skeleton trajectory."""
    out = np.zeros((TARGET_T, 166), dtype=np.float32)
    if sensor_dir is None:
        return out

    pred_dir = sensor_dir / "Skeleton" / "predictions"
    if not pred_dir.exists():
        return out

    json_files = sorted(list(pred_dir.glob("*.json")))
    if len(json_files) < 4:
        return out

    raw_frames = []
    for f in json_files:
        try:
            with open(f) as jf:
                d = json.load(jf)
            if not d or not isinstance(d, list) or "keypoints" not in d[0]:
                continue
            kp = np.array(d[0]["keypoints"], dtype=np.float32)
            if kp.shape[0] < 17 or kp.shape[1] < 3:
                continue
            raw_frames.append(kp[:17, :3])
        except Exception:
            continue

    if len(raw_frames) < 4:
        return out

    raw_frames = np.array(raw_frames)  # (N, 17, 3)
    N = len(raw_frames)

    # 1. Mid-hip centering
    mid_hips = (raw_frames[:, 11, :] + raw_frames[:, 12, :]) / 2.0  # (N, 3)
    centered = raw_frames - mid_hips[:, np.newaxis, :]  # (N, 17, 3)

    # Flatten coordinates: (N, 51)
    coords = centered.reshape(N, 51)

    # 2. Velocities: (N, 51)
    vel = np.zeros_like(coords)
    vel[1:] = (coords[1:] - coords[:-1]) * 10.0
    vel[0] = vel[1] if N > 1 else 0.0

    # 3. Accelerations: (N, 51)
    acc = np.zeros_like(coords)
    acc[1:] = (vel[1:] - vel[:-1]) * 10.0
    acc[0] = acc[1] if N > 1 else 0.0

    # 4. Torso tilt: (N, 1)
    mid_sh = (raw_frames[:, 5, :] + raw_frames[:, 6, :]) / 2.0
    torso_v = mid_sh - mid_hips
    norms = np.linalg.norm(torso_v, axis=1)
    norms = np.where(norms > 1e-4, norms, 1e-4)
    cos_tilt = np.clip(np.abs(torso_v[:, 2]) / norms, 0.0, 1.0)
    torso_tilt = np.degrees(np.arccos(cos_tilt))[:, np.newaxis]  # (N, 1)

    # 5. Pairwise distances: (N, 8)
    pw_dist = np.zeros((N, 8), dtype=np.float32)
    pw_dist[:, 0] = np.linalg.norm(raw_frames[:, 9, :] - raw_frames[:, 10, :], axis=1)  # wrist-wrist
    pw_dist[:, 1] = np.linalg.norm(raw_frames[:, 15, :] - raw_frames[:, 16, :], axis=1) # ankle-ankle
    pw_dist[:, 2] = np.linalg.norm(raw_frames[:, 9, :] - raw_frames[:, 0, :], axis=1)   # l_wrist-nose
    pw_dist[:, 3] = np.linalg.norm(raw_frames[:, 10, :] - raw_frames[:, 0, :], axis=1)  # r_wrist-nose
    pw_dist[:, 4] = np.linalg.norm(raw_frames[:, 9, :] - mid_hips, axis=1)             # l_wrist-midhip
    pw_dist[:, 5] = np.linalg.norm(raw_frames[:, 10, :] - mid_hips, axis=1)            # r_wrist-midhip
    pw_dist[:, 6] = np.linalg.norm(raw_frames[:, 9, :] - raw_frames[:, 13, :], axis=1) # l_wrist-l_knee
    pw_dist[:, 7] = np.linalg.norm(raw_frames[:, 10, :] - raw_frames[:, 14, :], axis=1)# r_wrist-r_knee

    # 6. Joint angles: (N, 4)
    angles = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        angles[i, 0] = calculate_joint_angle(raw_frames[i, 5], raw_frames[i, 7], raw_frames[i, 9])   # L elbow
        angles[i, 1] = calculate_joint_angle(raw_frames[i, 6], raw_frames[i, 8], raw_frames[i, 10])  # R elbow
        angles[i, 2] = calculate_joint_angle(raw_frames[i, 11], raw_frames[i, 13], raw_frames[i, 15])# L knee
        angles[i, 3] = calculate_joint_angle(raw_frames[i, 12], raw_frames[i, 14], raw_frames[i, 16])# R knee

    features_raw = np.concatenate([coords, vel, acc, torso_tilt, pw_dist, angles], axis=1) # (N, 166)

    # Resample to TARGET_T=64
    x_orig = np.linspace(0, 1, N)
    x_target = np.linspace(0, 1, TARGET_T)
    f_interp = interp1d(x_orig, features_raw, axis=0, kind="linear", fill_value="extrapolate")
    resampled = f_interp(x_target).astype(np.float32)
    return resampled


def extract_clip_imu_stream(sensor_dir: Path | None) -> np.ndarray:
    """Extract (T=64, 50) continuous IMU signals across 5 sensors."""
    out = np.zeros((TARGET_T, 50), dtype=np.float32)
    if sensor_dir is None:
        return out

    imu_dir = sensor_dir / "IMU"
    if not imu_dir.exists():
        return out

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
        return out

    all_data = pd.concat(dfs, ignore_index=True)
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        if col in all_data.columns:
            all_data[col] = pd.to_numeric(all_data[col], errors="coerce").fillna(0.0)
        else:
            all_data[col] = 0.0

    target_sensors = ["WTC", "WTRA", "WTLA", "WTRL", "WTLL"]
    sensor_streams = []

    for s_name in target_sensors:
        s_mask = all_data["device"].astype(str).str.startswith(s_name)
        s_df = all_data[s_mask]
        if len(s_df) < 5:
            sensor_streams.append(np.zeros((TARGET_T, 10), dtype=np.float32))
            continue

        ax = s_df["ax"].values.astype(np.float32)
        ay = s_df["ay"].values.astype(np.float32)
        az = s_df["az"].values.astype(np.float32)
        gx = s_df["gx"].values.astype(np.float32)
        gy = s_df["gy"].values.astype(np.float32)
        gz = s_df["gz"].values.astype(np.float32)

        a_mag = np.sqrt(ax**2 + ay**2 + az**2)
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        a_jerk = np.zeros_like(a_mag)
        a_jerk[1:] = np.abs(a_mag[1:] - a_mag[:-1])

        g_jerk = np.zeros_like(g_mag)
        g_jerk[1:] = np.abs(g_mag[1:] - g_mag[:-1])

        raw_feats = np.stack([ax, ay, az, gx, gy, gz, a_mag, g_mag, a_jerk, g_jerk], axis=1) # (N, 10)
        N = len(raw_feats)

        x_orig = np.linspace(0, 1, N)
        x_target = np.linspace(0, 1, TARGET_T)
        f_interp = interp1d(x_orig, raw_feats, axis=0, kind="linear", fill_value="extrapolate")
        resampled = f_interp(x_target).astype(np.float32)
        sensor_streams.append(resampled)

    out = np.concatenate(sensor_streams, axis=1) # (64, 50)
    return out


def process_clip(args):
    path, source, split = args
    sensor_dir = find_sensor_dir(path, source, split)
    skel_stream = extract_clip_skeleton_stream(sensor_dir)
    imu_stream = extract_clip_imu_stream(sensor_dir)
    return path, skel_stream, imu_stream


def main():
    print("=== EXTRACTING RAW TEMPORAL STREAMS (SKELETON & IMU, T=64) ===")
    train_df = pd.read_csv(ROOT / "training_qa.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv")

    train_clips = train_df[["path", "source"]].drop_duplicates().reset_index(drop=True)
    test_clips = test_df[["path", "source"]].drop_duplicates().reset_index(drop=True)

    print(f"Total train clips: {len(train_clips)}, Total test clips: {len(test_clips)}")

    items_tr = [(r.path, r.source, "train") for _, r in train_clips.iterrows()]
    items_te = [(r.path, r.source, "test") for _, r in test_clips.iterrows()]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        tr_results = list(executor.map(process_clip, items_tr, chunksize=16))
    print(f"Train temporal extraction completed in {time.time() - t0:.1f}s")

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        te_results = list(executor.map(process_clip, items_te, chunksize=16))
    print(f"Test temporal extraction completed in {time.time() - t1:.1f}s")

    train_paths = [r[0] for r in tr_results]
    X_skel_tr = np.array([r[1] for r in tr_results], dtype=np.float32)
    X_imu_tr = np.array([r[2] for r in tr_results], dtype=np.float32)

    test_paths = [r[0] for r in te_results]
    X_skel_te = np.array([r[1] for r in te_results], dtype=np.float32)
    X_imu_te = np.array([r[2] for r in te_results], dtype=np.float32)

    cache_path = ROOT / "temporal_streams_all.npz"
    np.savez_compressed(
        cache_path,
        X_skel_train=X_skel_tr,
        X_skel_test=X_skel_te,
        X_imu_train=X_imu_tr,
        X_imu_test=X_imu_te,
        train_paths=np.array(train_paths),
        test_paths=np.array(test_paths),
    )

    print(f"\nSaved {cache_path} ({os.path.getsize(cache_path)/1e6:.2f} MB)")
    print(f"X_skel_train: {X_skel_tr.shape}, Non-zero clips: {(X_skel_tr != 0).any(axis=(1,2)).sum()}/{len(X_skel_tr)}")
    print(f"X_skel_test:  {X_skel_te.shape}, Non-zero clips: {(X_skel_te != 0).any(axis=(1,2)).sum()}/{len(X_skel_te)}")
    print(f"X_imu_train:  {X_imu_tr.shape}, Non-zero clips: {(X_imu_tr != 0).any(axis=(1,2)).sum()}/{len(X_imu_tr)}")
    print(f"X_imu_test:   {X_imu_te.shape}, Non-zero clips: {(X_imu_te != 0).any(axis=(1,2)).sum()}/{len(X_imu_te)}")


if __name__ == "__main__":
    main()
