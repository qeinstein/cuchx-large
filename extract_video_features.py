"""Extract and cache ResNet-18 visual features for all train and test clips on Apple Silicon MPS."""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).parent


def get_video_path(p: str, source: str, split: str = "train", modality: str = "Depth_Color") -> Path:
    p = str(p).replace("\\", "/")
    if split == "test" or "large_model_track_test" in p:
        # e.g. large_model_track_test/LM_test_0066/Depth/Depth.mp4
        parts = p.split("/")
        for part in parts:
            if part.startswith("LM_test_"):
                candidate = ROOT / "hf_data_manual" / "large_model_track_test" / part / modality / f"{modality}.mp4"
                if candidate.exists():
                    return candidate
                # Try raw Depth if Depth_Color missing
                fallback = ROOT / "hf_data_manual" / "large_model_track_test" / part / "Depth" / "Depth.mp4"
                if fallback.exists():
                    return fallback
    else:
        # train: HAU/user1/1-1-1 or HARn/0_Wash_face/user16/1-1-2
        candidate = ROOT / "hf_data_manual" / p / modality / f"{modality}.mp4"
        if candidate.exists():
            return candidate
        fallback = ROOT / "hf_data_manual" / p / "Depth" / "Depth.mp4"
        if fallback.exists():
            return fallback
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

    # Repeat or trim to n_frames
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

    train_clips = train[["path", "source"]].drop_duplicates().reset_index(drop=True)
    test_clips = test[["path", "source"]].drop_duplicates().reset_index(drop=True)

    print(f"Extracting visual features for {len(train_clips)} train clips and {len(test_clips)} test clips...")

    t0 = time.time()
    train_features = []
    with torch.inference_mode():
        for i, row in train_clips.iterrows():
            vpath = get_video_path(row["path"], row["source"], "train")
            tensor = extract_frames(vpath, n_frames=8, transform=transform, device=device)
            feat = model(tensor).mean(0).float().cpu().numpy()
            train_features.append(feat)
            if (i + 1) % 200 == 0 or (i + 1) == len(train_clips):
                print(f"  Train: {i + 1} / {len(train_clips)} clips ({time.time() - t0:.1f}s)")

    test_features = []
    t1 = time.time()
    with torch.inference_mode():
        for i, row in test_clips.iterrows():
            vpath = get_video_path(row["path"], row["source"], "test")
            tensor = extract_frames(vpath, n_frames=8, transform=transform, device=device)
            feat = model(tensor).mean(0).float().cpu().numpy()
            test_features.append(feat)
            if (i + 1) % 50 == 0 or (i + 1) == len(test_clips):
                print(f"  Test: {i + 1} / {len(test_clips)} clips ({time.time() - t1:.1f}s)")

    X_train_vis = np.array(train_features, dtype=np.float32)
    X_test_vis = np.array(test_features, dtype=np.float32)

    cache_file = ROOT / "visual_features_all.npz"
    np.savez_compressed(
        cache_file,
        X_train=X_train_vis,
        train_paths=train_clips["path"].values,
        X_test=X_test_vis,
        test_paths=test_clips["path"].values,
    )
    print(f"Successfully saved visual features to {cache_file} in {time.time() - t0:.1f}s!")
    print(f"Train: {X_train_vis.shape}, Test: {X_test_vis.shape}")


if __name__ == "__main__":
    main()
