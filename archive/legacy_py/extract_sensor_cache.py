"""Extract and cache all sensor features for train and test clips."""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sensor_features import get_clip_sensor_features

ROOT = Path(__file__).parent


def main():
    train = pd.read_csv(ROOT / "training_qa.csv")
    test = pd.read_csv(ROOT / "test_qa.csv")

    train_clips = train[["path", "source"]].drop_duplicates().reset_index(drop=True)
    test_clips = test[["path", "source"]].drop_duplicates().reset_index(drop=True)

    print(f"Extracting {len(train_clips)} train clips and {len(test_clips)} test clips...")

    def process_item(item):
        path, source, split = item
        return get_clip_sensor_features(path, source, split)

    items = (
        [(r.path, r.source, "train") for _, r in train_clips.iterrows()]
        + [(r.path, r.source, "test") for _, r in test_clips.iterrows()]
    )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(process_item, items))

    all_features = np.array(results, dtype=np.float32)
    X_train = all_features[: len(train_clips)]
    X_test = all_features[len(train_clips) :]

    cache_path = ROOT / "sensor_features_all.npz"
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        train_paths=train_clips["path"].values,
        train_sources=train_clips["source"].values,
        X_test=X_test,
        test_paths=test_clips["path"].values,
        test_sources=test_clips["source"].values,
    )

    print(f"Successfully cached sensor features in {time.time() - t0:.1f}s!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Non-zero train: {(X_train != 0).any(axis=1).sum()} / {len(X_train)}")
    print(f"Non-zero test: {(X_test != 0).any(axis=1).sum()} / {len(X_test)}")


if __name__ == "__main__":
    main()
