"""Extract 28-dim frame-level temporal kinematics across 4 quarters of each clip.

Features per quarter (7 features x 4 quarters = 28 dims):
1. mean_wrist_height_rel: Max wrist height above mid-hip
2. mean_wrist_speed: 3D velocity of hands (m/s)
3. mean_ankle_speed: 3D velocity of feet/ankles (m/s)
4. mean_head_z: Absolute height of head
5. std_head_z: Vertical head variation (squats, sitting down, standing up)
6. mean_torso_tilt: Angle of torso from vertical (0=standing, 90=lying down)
7. mean_arm_spread: Distance between left and right wrist
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent
LMT_ROOT = ROOT / "hf_data_manual" / "LMT_(IMU,Radar,Skeleton)"


def extract_clip_temporal_kinematics(pred_dir: Path) -> np.ndarray:
    out = np.zeros(28, dtype=np.float32)
    if not pred_dir.exists():
        return out

    files = sorted(list(pred_dir.glob("*.json")))
    if len(files) < 10:
        return out

    head_z = []
    wrist_h = []
    wrist_v = []
    ankle_v = []
    torso_tilt = []
    arm_spread = []

    prev_wrist = None
    prev_ankle = None

    for f in files:
        try:
            with open(f) as jf:
                d = json.load(jf)
            if not d or not isinstance(d, list) or "keypoints" not in d[0]:
                continue
            pts = np.array(d[0]["keypoints"])
            if len(pts) < 17:
                continue

            # Keypoints COCO 17:
            # 0: nose, 5: l_sh, 6: r_sh, 9: l_wri, 10: r_wri, 11: l_hip, 12: r_hip, 15: l_ank, 16: r_ank
            nose = pts[0]
            mid_hip = (pts[11] + pts[12]) / 2.0
            mid_sh = (pts[5] + pts[6]) / 2.0
            l_wri, r_wri = pts[9], pts[10]
            l_ank, r_ank = pts[15], pts[16]

            # 1. Head Z
            h_z = float(nose[2])

            # 2. Wrist Height Relative to Mid-Hip
            w_h = float(max(l_wri[2], r_wri[2]) - mid_hip[2])

            # 3. Speeds (m/s assuming 10 Hz)
            curr_wrist = (l_wri + r_wri) / 2.0
            curr_ankle = (l_ank + r_ank) / 2.0
            w_spd = (
                float(np.linalg.norm(curr_wrist - prev_wrist) * 10.0)
                if prev_wrist is not None
                else 0.0
            )
            a_spd = (
                float(np.linalg.norm(curr_ankle - prev_ankle) * 10.0)
                if prev_ankle is not None
                else 0.0
            )
            prev_wrist = curr_wrist
            prev_ankle = curr_ankle

            # 4. Torso Tilt Angle (degrees from vertical Z)
            torso_vec = mid_sh - mid_hip
            norm_torso = np.linalg.norm(torso_vec)
            if norm_torso > 1e-4:
                cos_angle = abs(torso_vec[2]) / norm_torso
                tilt = float(np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0))))
            else:
                tilt = 0.0

            # 5. Arm Spread
            spread = float(np.linalg.norm(l_wri - r_wri))

            head_z.append(h_z)
            wrist_h.append(w_h)
            wrist_v.append(w_spd)
            ankle_v.append(a_spd)
            torso_tilt.append(tilt)
            arm_spread.append(spread)
        except Exception:
            continue

    n = len(head_z)
    if n < 8:
        return out

    q_len = n // 4
    for q in range(4):
        s_idx = q * q_len
        e_idx = (q + 1) * q_len if q < 3 else n

        q_head = head_z[s_idx:e_idx]
        q_wrist_h = wrist_h[s_idx:e_idx]
        q_wrist_v = wrist_v[s_idx:e_idx]
        q_ankle_v = ankle_v[s_idx:e_idx]
        q_tilt = torso_tilt[s_idx:e_idx]
        q_spread = arm_spread[s_idx:e_idx]

        base = q * 7
        out[base + 0] = float(np.mean(q_wrist_h))
        out[base + 1] = float(np.mean(q_wrist_v))
        out[base + 2] = float(np.mean(q_ankle_v))
        out[base + 3] = float(np.mean(q_head))
        out[base + 4] = float(np.std(q_head))
        out[base + 5] = float(np.mean(q_tilt))
        out[base + 6] = float(np.mean(q_spread))

    return out


def process_clip(args):
    path, is_test = args
    if is_test:
        candidates = [p for p in path.split("/") if p.startswith("LM_test_")]
        clip_name = candidates[0] if candidates else path.split("/")[-1]
        pdir = (
            LMT_ROOT
            / "Testing"
            / "large_model_track_test"
            / clip_name
            / "Skeleton"
            / "predictions"
        )
    else:
        pdir = LMT_ROOT / "Training" / path / "Skeleton" / "predictions"
    return path, extract_clip_temporal_kinematics(pdir)


def main():
    print("=== Extracting 28-dim Temporal Kinematic Matrices ===")
    train_df = pd.read_csv(ROOT / "training_qa.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv")

    train_paths = sorted(train_df["path"].unique())
    test_paths = sorted(test_df["path"].unique())

    print(f"Processing {len(train_paths)} train clips and {len(test_paths)} test clips...")

    tasks_tr = [(p, False) for p in train_paths]
    tasks_te = [(p, True) for p in test_paths]

    with ProcessPoolExecutor() as executor:
        results_tr = list(
            tqdm(executor.map(process_clip, tasks_tr), total=len(tasks_tr), desc="Train Kinematics")
        )
    tr_map = dict(results_tr)
    X_train = np.array([tr_map[p] for p in train_paths], dtype=np.float32)

    with ProcessPoolExecutor() as executor:
        results_te = list(
            tqdm(executor.map(process_clip, tasks_te), total=len(tasks_te), desc="Test Kinematics")
        )
    te_map = dict(results_te)
    X_test = np.array([te_map[p] for p in test_paths], dtype=np.float32)

    out_file = ROOT / "temporal_kinematics_all.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train,
        train_paths=np.array(train_paths),
        X_test=X_test,
        test_paths=np.array(test_paths),
    )
    print(f"\nSaved {out_file}:")
    print(f"  X_train: {X_train.shape} (non-zero: {(np.abs(X_train).sum(axis=1) > 0).sum()} / {len(X_train)})")
    print(f"  X_test:  {X_test.shape} (non-zero: {(np.abs(X_test).sum(axis=1) > 0).sum()} / {len(X_test)})")


if __name__ == "__main__":
    main()
