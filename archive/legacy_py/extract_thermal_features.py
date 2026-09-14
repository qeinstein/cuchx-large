"""Extract ResNet-18 features from Thermal.mp4 videos for HAU emotion recognition."""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).parent


def get_thermal_path(p: str, split: str = "train") -> Path:
    p = str(p).replace("\\", "/")
    if split == "test" or "large_model_track_test" in p:
        parts = p.split("/")
        for part in parts:
            if part.startswith("LM_test_"):
                candidate = ROOT / "hf_data_manual" / "large_model_track_test" / part / "Thermal" / "Thermal.mp4"
                if candidate.exists():
                    return candidate
    else:
        candidate = ROOT / "hf_data_manual" / p / "Thermal" / "Thermal.mp4"
        if candidate.exists():
            return candidate
    return None


def extract_frames(video_path: Path, n_frames: int = 8, transform=None, device="mps") -> torch.Tensor:
    if video_path is None or not video_path.exists():
        black = Image.new("RGB", (224, 224), "black")
        return torch.stack([transform(black) for _ in range(n_frames)]).to(device)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        indices = np.linspace(0, max(0, total - 1), n_frames).astype(int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    if not frames:
        frames = [Image.new("RGB", (224, 224), "black")]

    while len(frames) < n_frames:
        frames.append(frames[-1])
    frames = frames[:n_frames]

    return torch.stack([transform(f) for f in frames]).to(device)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    weights_path = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    transform = ResNet18_Weights.DEFAULT.transforms()

    train = pd.read_csv(ROOT / "training_qa.csv")
    test = pd.read_csv(ROOT / "test_qa.csv")

    train_clips = train[train.source == "HAU"][["path", "source"]].drop_duplicates().reset_index(drop=True)
    test_clips = test[test.source == "HAU"][["path", "source"]].drop_duplicates().reset_index(drop=True)

    print(f"Extracting Thermal features for {len(train_clips)} train HAU clips and {len(test_clips)} test HAU clips...")

    t0 = time.time()
    train_features = []
    with torch.inference_mode():
        for i, row in train_clips.iterrows():
            vpath = get_thermal_path(row["path"], "train")
            tensor = extract_frames(vpath, n_frames=8, transform=transform, device=device)
            feat = model(tensor).mean(0).float().cpu().numpy()
            train_features.append(feat)
            if (i + 1) % 150 == 0 or (i + 1) == len(train_clips):
                print(f"  Train: {i + 1} / {len(train_clips)} clips ({time.time() - t0:.1f}s)")

    test_features = []
    t1 = time.time()
    with torch.inference_mode():
        for i, row in test_clips.iterrows():
            vpath = get_thermal_path(row["path"], "test")
            tensor = extract_frames(vpath, n_frames=8, transform=transform, device=device)
            feat = model(tensor).mean(0).float().cpu().numpy()
            test_features.append(feat)
            if (i + 1) % 50 == 0 or (i + 1) == len(test_clips):
                print(f"  Test: {i + 1} / {len(test_clips)} clips ({time.time() - t1:.1f}s)")

    X_train_thm = np.array(train_features, dtype=np.float32)
    X_test_thm = np.array(test_features, dtype=np.float32)

    cache_file = ROOT / "thermal_features_all.npz"
    np.savez_compressed(
        cache_file,
        X_train=X_train_thm,
        train_paths=train_clips["path"].values,
        X_test=X_test_thm,
        test_paths=test_clips["path"].values,
    )
    print(f"Successfully saved Thermal features to {cache_file} in {time.time() - t0:.1f}s!")
    print(f"Train: {X_train_thm.shape}, Test: {X_test_thm.shape}")


if __name__ == "__main__":
    main()
