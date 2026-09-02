"""Fast supervised sensor-video baseline using frozen ImageNet features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).parent
LABELS = "ABCD"


def clip_id(path: str) -> str:
    parts = str(path).replace("\\", "/").split("/")
    if parts[-1].lower().split(".")[0] in {"depth", "depth_color", "ir", "thermal"}:
        parts = parts[:-2]
    return "/".join(parts)


def media_path(path: str, source: str, modality: str = "Depth_Color") -> Path:
    p = str(path).replace("\\", "/")
    if source == "HAU":
        if p.startswith("HAU/"):
            p = p[4:]
        base = ROOT / "hf_data_manual" / "HAU" / p
    elif source == "HARn":
        if p.startswith("HARn/"):
            p = p[5:]
        base = ROOT / "hf_data_manual" / "HARn" / p
    else:
        base = ROOT / "hf_data_manual" / "large_model_track_test" / p
    if base.suffix == ".mp4":
        return base
    return base / modality / f"{modality}.mp4"


def frame_tensor(video: Path, n: int = 8) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        frames = [Image.new("RGB", (224, 224), "black")]
    transform = ResNet18_Weights.DEFAULT.transforms()
    return torch.stack([transform(x) for x in frames])


def extract_features(paths: list[tuple[str, str, str]], cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)["features"]
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=None)
    state = torch.load(Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    all_features = []
    with torch.inference_mode():
        for i, (path, source, modality) in enumerate(paths):
            x = frame_tensor(media_path(path, source, modality)).to(device)
            z = model(x).mean(0).float().cpu().numpy()
            all_features.append(z)
            if (i + 1) % 25 == 0:
                print(f"features {i + 1}/{len(paths)}", flush=True)
    features = np.asarray(all_features, dtype=np.float32)
    np.savez_compressed(cache, features=features)
    return features


def correct_text(row: pd.Series) -> str:
    return str(row[str(row.answer)])


def clip_target(frame: pd.DataFrame, key: str, category: str) -> str | None:
    rows = frame[frame.path.map(clip_id) == key]
    hit = rows[rows.category == category]
    if len(hit):
        return correct_text(hit.iloc[0])
    return None


def build_clip_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(df.path.map(clip_id), sort=False):
        row = g.iloc[0].copy()
        row["clip"] = key
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def choose_option(options: list[str], target: str) -> str:
    target = target.lower()
    scores = [sum(w in x.lower() for w in target.split() if len(w) > 3) for x in options]
    return LABELS[int(np.argmax(scores))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--modality", default="Depth_Color")
    args = ap.parse_args()
    train = pd.read_csv(ROOT / "training_qa.csv")
    test = pd.read_csv(ROOT / "test_qa.csv")
    fit = train if args.fold < 0 else pd.read_csv(ROOT / f"splits/fold_{args.fold}_train.csv")
    val = test if args.fold < 0 else pd.read_csv(ROOT / f"splits/fold_{args.fold}_val.csv")
    train_clips = build_clip_frame(fit)
    pred_clips = build_clip_frame(val)
    keys = list(zip(train_clips.path, train_clips.source, [args.modality] * len(train_clips))) + list(zip(pred_clips.path, pred_clips.source, [args.modality] * len(pred_clips)))
    cache = ROOT / f"features_{args.modality.lower()}_{'all' if args.fold < 0 else 'fold'+str(args.fold)}.npz"
    X = extract_features(keys, cache)
    Xtr, Xte = X[:len(train_clips)], X[len(train_clips):]
    train_clips["single_target"] = train_clips.clip.map(lambda k: clip_target(fit, k, "single"))
    train_clips["emotion_target"] = train_clips.clip.map(lambda k: clip_target(fit, k, "emotion"))
    for col in ["single_target", "emotion_target"]:
        model = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
        keep = train_clips[col].notna()
        model.fit(Xtr[keep.to_numpy()], train_clips.loc[keep, col])
        pred_clips[col] = model.predict(Xte)
    # Train exact sequence permutation and use the predicted action text for other heads.
    seq = fit[fit.category == "sequence"].copy()
    seq["target"] = seq.answer
    seq = seq[seq.path.map(clip_id).isin(set(train_clips.clip))]
    seq_model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    seq_model.fit(Xtr[[train_clips.clip.tolist().index(c) for c in seq.path.map(clip_id)]], seq.target)
    seq_pred = seq_model.predict(Xte)
    pred_clips["sequence_target"] = seq_pred
    by_clip = {r.clip: r for _, r in pred_clips.iterrows()}
    out = []
    for _, row in val.iterrows():
        c = by_clip[clip_id(row.path)]
        if row.category == "single":
            out.append(choose_option([row[x] for x in LABELS], c.single_target))
        elif row.category == "emotion":
            out.append(choose_option([row[x] for x in LABELS], c.emotion_target))
        elif row.category == "sequence":
            order = c.sequence_target
            # The classifier predicts the order of the four action texts seen in training;
            # map by option position when exact text is unavailable.
            out.append(order)
        else:
            out.append(LABELS[0])
    result = pd.DataFrame({"qa_id": val.qa_id, "prediction": out})
    output = ROOT / ("submission_visual.csv" if args.fold < 0 else f"pred_visual_fold{args.fold}.csv")
    result.to_csv(output, index=False)
    if "answer" in val:
        print("accuracy", (result.prediction == val.answer).mean())
        print(pd.crosstab(val.category, result.prediction == val.answer))
    print(output)


if __name__ == "__main__":
    main()
