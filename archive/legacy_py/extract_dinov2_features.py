"""Extract dense Vision Transformer features using Meta's DINOv2 (ViT-S/14) on Apple Silicon MPS."""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).parent


def get_depth_path(p: str, split: str = "train") -> Path:
    p = str(p).replace("\\", "/")
    if split == "test" or "large_model_track_test" in p:
        parts = p.split("/")
        for part in parts:
            if part.startswith("LM_test_"):
                cand1 = ROOT / "hf_data_manual" / "large_model_track_test" / part / "Depth" / "Depth.mp4"
                if cand1.exists():
                    return cand1
                cand2 = ROOT / "hf_data_manual" / "large_model_track_test" / part / "Depth_Color" / "Depth_Color.mp4"
                if cand2.exists():
                    return cand2
    else:
        cand1 = ROOT / "hf_data_manual" / p / "Depth" / "Depth.mp4"
        if cand1.exists():
            return cand1
        cand2 = ROOT / "hf_data_manual" / p / "Depth_Color" / "Depth_Color.mp4"
        if cand2.exists():
            return cand2
    return None


def extract_clip_frames(vpath: Path, n_frames: int = 8, transform=None) -> torch.Tensor:
    if vpath is None or not vpath.exists():
        black = Image.new("RGB", (224, 224), "black")
        return torch.stack([transform(black) for _ in range(n_frames)])

    cap = cv2.VideoCapture(str(vpath))
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

    return torch.stack([transform(f) for f in frames])


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading DINOv2 ViT-S/14 onto {device}...")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train = pd.read_csv(ROOT / "training_qa.csv")
    test = pd.read_csv(ROOT / "test_qa.csv")

    train_clips = train[["path", "source"]].drop_duplicates().reset_index(drop=True)
    test_clips = test[["path", "source"]].drop_duplicates().reset_index(drop=True)

    print(f"Extracting DINOv2 features for {len(train_clips)} train clips and {len(test_clips)} test clips...")

    t0 = time.time()
    train_features = []
    batch_tensors = []
    batch_indices = []

    with torch.inference_mode():
        for i, row in train_clips.iterrows():
            vpath = get_depth_path(row["path"], "train")
            t_frames = extract_clip_frames(vpath, n_frames=8, transform=transform)
            batch_tensors.append(t_frames)
            batch_indices.append(i)

            if len(batch_tensors) == 4 or (i + 1) == len(train_clips):
                # (B, 8, 3, 224, 224) -> (B*8, 3, 224, 224)
                B = len(batch_tensors)
                inp = torch.cat(batch_tensors, dim=0).to(device)
                feats = model(inp)  # (B*8, 384)
                feats_reshaped = feats.view(B, 8, -1).mean(dim=1)  # (B, 384)
                train_features.extend(feats_reshaped.cpu().numpy())
                batch_tensors = []
                batch_indices = []

            if (i + 1) % 200 == 0 or (i + 1) == len(train_clips):
                elapsed = time.time() - t0
                fps = (i + 1) * 8 / elapsed
                print(f"  Train: {i + 1} / {len(train_clips)} clips ({elapsed:.1f}s, {fps:.1f} fps)")

    t1 = time.time()
    test_features = []
    with torch.inference_mode():
        for i, row in test_clips.iterrows():
            vpath = get_depth_path(row["path"], "test")
            t_frames = extract_clip_frames(vpath, n_frames=8, transform=transform)
            batch_tensors.append(t_frames)

            if len(batch_tensors) == 4 or (i + 1) == len(test_clips):
                B = len(batch_tensors)
                inp = torch.cat(batch_tensors, dim=0).to(device)
                feats = model(inp)
                feats_reshaped = feats.view(B, 8, -1).mean(dim=1)
                test_features.extend(feats_reshaped.cpu().numpy())
                batch_tensors = []

            if (i + 1) % 50 == 0 or (i + 1) == len(test_clips):
                elapsed = time.time() - t1
                fps = (i + 1) * 8 / elapsed
                print(f"  Test: {i + 1} / {len(test_clips)} clips ({elapsed:.1f}s, {fps:.1f} fps)")

    X_train_dino = np.array(train_features, dtype=np.float32)
    X_test_dino = np.array(test_features, dtype=np.float32)

    cache_file = ROOT / "dinov2_features_all.npz"
    np.savez_compressed(
        cache_file,
        X_train=X_train_dino,
        train_paths=train_clips["path"].values,
        X_test=X_test_dino,
        test_paths=test_clips["path"].values,
    )
    print(f"Successfully saved DINOv2 features to {cache_file} in {time.time() - t0:.1f}s!")
    print(f"Train: {X_train_dino.shape}, Test: {X_test_dino.shape}")


if __name__ == "__main__":
    main()
