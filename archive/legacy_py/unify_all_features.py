"""Unify all extracted multimodal feature caches into a single consolidated feature array."""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


def main():
    print("Loading individual feature caches...")
    sensor_cache = np.load(ROOT / "sensor_features_all.npz", allow_pickle=True)
    visual_cache = np.load(ROOT / "visual_features_all.npz", allow_pickle=True)
    dinov2_cache = np.load(ROOT / "dinov2_features_all.npz", allow_pickle=True)
    thermal_cache = np.load(ROOT / "thermal_features_all.npz", allow_pickle=True)
    radar_cache = np.load(ROOT / "radar_features_all.npz", allow_pickle=True)

    train_paths = list(sensor_cache["train_paths"])
    test_paths = list(sensor_cache["test_paths"])

    X_sens_tr = sensor_cache["X_train"]  # (1333, 280)
    X_sens_te = sensor_cache["X_test"]  # (208, 280)

    X_vis_tr = visual_cache["X_train"]  # (1333, 512)
    X_vis_te = visual_cache["X_test"]  # (208, 512)

    X_dino_tr = dinov2_cache["X_train"]  # (1333, 384)
    X_dino_te = dinov2_cache["X_test"]  # (208, 384)

    X_rad_tr = radar_cache["X_train"]  # (1333, 32)
    X_rad_te = radar_cache["X_test"]  # (208, 32)

    # Thermal features (809 train, 144 test)
    thm_tr_paths = list(thermal_cache["train_paths"])
    thm_te_paths = list(thermal_cache["test_paths"])
    thm_tr_map = {p: i for i, p in enumerate(thm_tr_paths)}
    thm_te_map = {p: i for i, p in enumerate(thm_te_paths)}

    X_thm_tr_full = []
    for p in train_paths:
        if p in thm_tr_map:
            X_thm_tr_full.append(thermal_cache["X_train"][thm_tr_map[p]])
        else:
            X_thm_tr_full.append(np.zeros(512, dtype=np.float32))
    X_thm_tr_full = np.array(X_thm_tr_full, dtype=np.float32)

    X_thm_te_full = []
    for p in test_paths:
        if p in thm_te_map:
            X_thm_te_full.append(thermal_cache["X_test"][thm_te_map[p]])
        else:
            X_thm_te_full.append(np.zeros(512, dtype=np.float32))
    X_thm_te_full = np.array(X_thm_te_full, dtype=np.float32)

    # Concatenate: 280 (sensor) + 384 (DINOv2) + 512 (ResNet) + 512 (Thermal) + 32 (Radar) = 1720 dimensions
    X_train_unified = np.concatenate([X_sens_tr, X_dino_tr, X_vis_tr, X_thm_tr_full, X_rad_tr], axis=1)
    X_test_unified = np.concatenate([X_sens_te, X_dino_te, X_vis_te, X_thm_te_full, X_rad_te], axis=1)

    out_file = ROOT / "multimodal_features_all.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train_unified.astype(np.float32),
        train_paths=np.array(train_paths),
        X_test=X_test_unified.astype(np.float32),
        test_paths=np.array(test_paths),
    )

    print(f"Successfully created unified multimodal cache at {out_file}!")
    print(f"  X_train: {X_train_unified.shape}")
    print(f"  X_test:  {X_test_unified.shape}")


if __name__ == "__main__":
    main()
