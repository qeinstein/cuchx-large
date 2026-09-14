"""Extract 32-dim mmWave Doppler Radar features across all train and test clips.

Captures:
- Doppler velocity statistics (mean, std, max, percentiles, positive/negative energy)
- 3D point cloud dispersion, spatial volume, and distance
- SNR and noise floor
- 4-quarter temporal Doppler trajectory for sequence ordering
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent
LMT_ROOT = ROOT / "hf_data_manual" / "LMT_(IMU,Radar,Skeleton)"


def extract_radar_32d(csv_path: Path) -> np.ndarray:
    out = np.zeros(32, dtype=np.float32)
    if not csv_path.exists():
        return out
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 5:
            return out

        v = pd.to_numeric(df["v"], errors="coerce").fillna(0.0).values
        v_abs = np.abs(v)
        x = pd.to_numeric(df["x"], errors="coerce").fillna(0.0).values
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).values
        z = pd.to_numeric(df["z"], errors="coerce").fillna(0.0).values
        snr = pd.to_numeric(df["snr"], errors="coerce").fillna(0.0).values

        # Doppler
        out[0] = float(np.mean(v_abs))
        out[1] = float(np.std(v))
        out[2] = float(np.max(v_abs))
        out[3:8] = np.percentile(v_abs, [25, 50, 75, 90, 95])
        out[8] = float(np.mean(v[v > 0] ** 2)) if np.sum(v > 0) else 0.0
        out[9] = float(np.mean(v[v < 0] ** 2)) if np.sum(v < 0) else 0.0
        out[10] = float(np.mean(v_abs < 1e-4))

        # Spatial Point Cloud
        pts_per_frame = (
            df.groupby("frame").size().values
            if "frame" in df.columns
            else [len(df)]
        )
        out[11] = float(np.mean(pts_per_frame))
        out[12] = float(np.std(pts_per_frame))
        out[13] = float(np.max(pts_per_frame))
        out[14] = float(np.std(x))
        out[15] = float(np.std(y))
        out[16] = float(np.std(z))
        out[17] = float(np.ptp(x))
        out[18] = float(np.ptp(y))
        out[19] = float(np.ptp(z))

        # Radial Distance
        r_dist = np.sqrt(x**2 + y**2 + z**2)
        out[20] = float(np.mean(r_dist))
        out[21] = float(np.std(r_dist))
        out[22] = float(np.min(r_dist))
        out[23] = float(np.max(r_dist))

        # SNR
        out[24] = float(np.mean(snr))
        out[25] = float(np.std(snr))
        out[26] = float(np.max(snr))

        # 4 Temporal Quarters Doppler
        n = len(v_abs)
        q = n // 4
        if q > 0:
            out[27] = float(np.mean(v_abs[:q]))
            out[28] = float(np.mean(v_abs[q : 2 * q]))
            out[29] = float(np.mean(v_abs[2 * q : 3 * q]))
            out[30] = float(np.mean(v_abs[3 * q :]))
        out[31] = float(np.std(np.diff(v))) if n > 1 else 0.0
    except Exception:
        pass
    return out


def process_clip(args):
    path, is_test = args
    if is_test:
        clip_candidates = [part for part in path.split("/") if part.startswith("LM_test_")]
        clip_name = clip_candidates[0] if clip_candidates else path.split("/")[-1]
        csv_path = LMT_ROOT / "Testing" / "large_model_track_test" / clip_name / "Radar" / "Radar.csv"
    else:
        # Training/HAU/... or Training/HARn/...
        csv_path = LMT_ROOT / "Training" / path / "Radar" / "Radar.csv"
    return path, extract_radar_32d(csv_path)


def main():
    print("=== Extracting 32-dim mmWave Radar Features ===")
    train_df = pd.read_csv(ROOT / "training_qa.csv")
    test_df = pd.read_csv(ROOT / "test_qa.csv")

    train_paths = sorted(train_df["path"].unique())
    test_paths = sorted(test_df["path"].unique())

    print(f"Processing {len(train_paths)} train clips and {len(test_paths)} test clips...")

    tasks_tr = [(p, False) for p in train_paths]
    tasks_te = [(p, True) for p in test_paths]

    X_train = []
    with ProcessPoolExecutor() as executor:
        results_tr = list(tqdm(executor.map(process_clip, tasks_tr), total=len(tasks_tr), desc="Train Radar"))
    tr_map = dict(results_tr)
    X_train = np.array([tr_map[p] for p in train_paths], dtype=np.float32)

    with ProcessPoolExecutor() as executor:
        results_te = list(tqdm(executor.map(process_clip, tasks_te), total=len(tasks_te), desc="Test Radar"))
    te_map = dict(results_te)
    X_test = np.array([te_map[p] for p in test_paths], dtype=np.float32)

    out_file = ROOT / "radar_features_all.npz"
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
