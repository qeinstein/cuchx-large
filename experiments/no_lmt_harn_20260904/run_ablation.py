#!/usr/bin/env python3
"""Depth-only, subject-disjoint HARn missing-LMT ablation.

This experiment deliberately does *not* use ``champ/meta.csv`` as a model
input, LMT feature caches, skeletons, IMU, radar, parent-frame alignment, or
test predictions in the video specialist.  ``meta.csv`` is read once only to
identify/audit which real-test clip IDs have no entry; the model reads only the
raw ``Depth/Depth.mp4`` clip available for every one of those 13 clips.

There are two separate comparison objects:

* ``oof_v8_final.csv`` is the held-out vector for the exact v8 fallback that
  ``champ/final.py`` uses when ``pipeline.solve`` returns no evidence.
* ``submission_v8.csv`` is that same fallback on the 15 real-test rows.

The classifier is trained on folder action labels from *other subjects*, then
mapped to a question answer using only training-subject phrase/object priors.
It is intentionally an ablation, not a submission builder.

Usage (from repository root):

    ./venv/bin/python experiments/no_lmt_harn_20260904/run_ablation.py \\
      --scope all --extract --evaluate --resume

``--scope qa`` is a faster diagnostic that encodes just the QA-referenced
training clips.  ``--scope all`` is the primary protocol: it uses all labelled
training HARn clips, excluding every held-out subject for each fold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
RAW_HARN = ROOT / "hf_data_manual" / "HARn"
RAW_TEST = ROOT / "hf_data_manual" / "large_model_track_test"
DEPTH = "Depth"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
RES = 224
BATCH = 64
GATES = (0.00, 0.70, 0.80, 0.90)  # fixed before looking at this run's results
NO_SENSOR_ACTIONS = {
    "40_Stand_on_one_leg_(balance)",
    "41_Peel_fruits_with_a_knife",
    "42_Throw_away_(rubbish)",
    "43_Open_the_cabinet_(to_get_things)",
}


def normalise(text: Any) -> str:
    """The QA vocabulary is stable but has harmless spacing/case variation."""
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def qpath_to_video(qpath: str, *, is_test: bool = False) -> Path:
    if is_test:
        return RAW_TEST / qpath / DEPTH / f"{DEPTH}.mp4"
    parts = qpath.split("/")
    if len(parts) != 4 or parts[0] != "HARn":
        raise ValueError(f"not a HARn path: {qpath}")
    _, action, user, trial = parts
    return RAW_HARN / action / user / trial / DEPTH / f"{DEPTH}.mp4"


@dataclass(frozen=True)
class Clip:
    key: str
    qpath: str
    user: str
    action: str | None
    split: str  # train or test
    video: Path


def load_frames(video: Path) -> np.ndarray | None:
    """Read exactly the one modality common to all real no-LMT clips."""
    cap = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        frames.append(cv2.resize(gray, (RES, RES), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return None
    return np.stack(frames)


def pool(frame_features: np.ndarray) -> np.ndarray:
    """Frozen DINO frame trajectory -> fixed descriptor, as in champ/harn_dino.py."""
    x = frame_features.astype(np.float32, copy=False)
    n = len(x)
    # A few valid HARn clips contain one frame.  Keep every pooled bin non-empty rather
    # than manufacturing NaNs for their second/third temporal bins.
    i1 = min(n, max(1, n // 3))
    i2 = min(n, max(i1 + 1, 2 * n // 3))
    thirds = (
        x[:i1],
        x[i1:i2] if i2 > i1 else x[-1:],
        x[i2:] if i2 < n else x[-1:],
    )
    delta = np.abs(np.diff(x, axis=0)) if n > 1 else np.zeros((1, x.shape[1]), np.float32)
    return np.concatenate(
        [
            x.mean(0), x.max(0), x.std(0),
            thirds[0].mean(0), thirds[1].mean(0), thirds[2].mean(0),
            delta.mean(0), delta.max(0),
        ]
    ).astype(np.float32)


@torch.no_grad()
def encode_one(model: torch.nn.Module, device: str, video: Path) -> np.ndarray | None:
    frames = load_frames(video)
    if frames is None:
        return None
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    pieces = []
    for start in range(0, len(frames), BATCH):
        chunk = torch.from_numpy(frames[start : start + BATCH]).to(device).float().div_(255.0)
        chunk = chunk[:, None].repeat(1, 3, 1, 1)
        pieces.append(model((chunk - mean) / std).float().cpu().numpy())
    return pool(np.concatenate(pieces, axis=0))


def raw_train_clips(scope: str) -> list[Clip]:
    """Folder names provide official training action labels; no cache metadata is read."""
    qa_paths: set[str] | None = None
    if scope == "qa":
        tr = pd.read_csv(ROOT / "training_qa.csv")
        qa_paths = set(tr.loc[tr.source.eq("HARn"), "path"].unique())

    out: list[Clip] = []
    for video in sorted(RAW_HARN.glob("*/*/*/Depth/Depth.mp4")):
        # .../HARn/<action>/<user>/<trial>/Depth/Depth.mp4
        trial, user, action = video.parents[1].name, video.parents[2].name, video.parents[3].name
        qpath = f"HARn/{action}/{user}/{trial}"
        if qa_paths is not None and qpath not in qa_paths:
            continue
        out.append(Clip(key=f"train::{qpath}", qpath=qpath, user=user,
                        action=action, split="train", video=video))
    if not out:
        raise RuntimeError("no raw HARn Depth videos were found")
    return out


def real_no_lmt_clips() -> tuple[list[Clip], pd.DataFrame]:
    te = pd.read_csv(ROOT / "test_qa.csv")
    meta = pd.read_csv(ROOT / "champ" / "meta.csv")
    meta_paths = set(meta.qa_path)
    h = te[te.source.eq("HARn")].copy()
    h["clip"] = h.path.str.extract(r"LM_test_(\d+)").astype(int)
    h["clip_id"] = h["clip"].map(lambda n: f"LM_test_{n:04d}")
    affected = h[~h.clip_id.isin(meta_paths)].copy().sort_values("qa_id")
    clips: list[Clip] = []
    for clip_id in sorted(affected.clip_id.unique()):
        video = qpath_to_video(clip_id, is_test=True)
        clips.append(Clip(key=f"test::{clip_id}", qpath=clip_id, user="",
                          action=None, split="test", video=video))
    return clips, affected


def manifest(scope: str) -> tuple[list[Clip], pd.DataFrame]:
    train = raw_train_clips(scope)
    test, affected = real_no_lmt_clips()
    rows = []
    for c in [*train, *test]:
        rows.append(
            dict(key=c.key, qpath=c.qpath, split=c.split, user=c.user,
                 action=c.action, video=str(c.video.relative_to(ROOT)),
                 exists=c.video.exists())
        )
    out = pd.DataFrame(rows)
    out.to_csv(EXP / f"manifest_{scope}.csv", index=False)
    affected.to_csv(EXP / "real_no_lmt_questions.csv", index=False)
    availability = []
    for clip_id in sorted(affected.clip_id.unique()):
        root = RAW_TEST / clip_id
        availability.append(
            dict(
                clip_id=clip_id,
                Depth=(root / "Depth" / "Depth.mp4").is_file(),
                Depth_Color=(root / "Depth_Color" / "Depth_Color.mp4").is_file(),
                IR=(root / "IR" / "IR.mp4").is_file(),
                Thermal=(root / "Thermal" / "Thermal.mp4").is_file(),
                lmt_directory=(ROOT / "hf_data_manual" / "LMT_(IMU,Radar,Skeleton)" /
                               "Testing" / "large_model_track_test" / clip_id).is_dir(),
            )
        )
    pd.DataFrame(availability).to_csv(EXP / "real_no_lmt_availability.csv", index=False)
    return [*train, *test], affected


def cache_path(scope: str) -> Path:
    return EXP / f"depth_dino_direct_{scope}.npz"


def read_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    return {key: z[key].astype(np.float32) for key in z.files}


def write_cache(path: Path, store: dict[str, np.ndarray]) -> None:
    # Only our new experiment cache is ever written.  The caller requires --resume if it exists.
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **store)
    os.replace(tmp, path)


def extract(scope: str, *, resume: bool, limit: int | None, device_arg: str) -> None:
    clips, affected = manifest(scope)
    path = cache_path(scope)
    if path.exists() and not resume:
        raise FileExistsError(f"{path} already exists; use --resume to extend it")
    store = read_cache(path) if resume else {}
    # A prior interrupted/older extractor could contain a descriptor with an empty
    # temporal pool.  Re-encode only those local experiment keys on resume.
    invalid = [key for key, value in store.items() if not np.isfinite(value).all()]
    for key in invalid:
        del store[key]
    if invalid:
        print(f"discarding {len(invalid)} non-finite local descriptors for re-encoding")
    missing = [c for c in clips if c.key not in store]
    if limit is not None:
        missing = missing[:limit]
    print(f"scope={scope}; clips={len(clips)}; cached={len(store)}; to encode={len(missing)}")
    print(f"real no-LMT: {affected.clip_id.nunique()} clips / {len(affected)} questions")
    if not missing:
        return

    device = device_arg
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading frozen DINOv2 on {device}", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           pretrained=True, trust_repo=True, verbose=False).eval().to(device)
    failed: list[dict[str, str]] = []
    t0 = time.time()
    for i, clip in enumerate(missing, start=1):
        if not clip.video.exists():
            failed.append(dict(key=clip.key, qpath=clip.qpath, error="missing Depth video"))
            continue
        try:
            desc = encode_one(model, device, clip.video)
            if desc is None:
                failed.append(dict(key=clip.key, qpath=clip.qpath, error="no decoded frames"))
            else:
                store[clip.key] = desc
        except Exception as exc:  # preserve failures instead of silently making a fallback claim
            failed.append(dict(key=clip.key, qpath=clip.qpath, error=repr(exc)))
        if i % 25 == 0 or i == len(missing):
            write_cache(path, store)
            elapsed = time.time() - t0
            print(f"  {i}/{len(missing)} encoded; cache={len(store)}; "
                  f"{i / max(elapsed, 1e-9):.2f} clips/s; elapsed={elapsed / 60:.1f}m", flush=True)
    pd.DataFrame(failed, columns=["key", "qpath", "error"]).to_csv(
        EXP / f"extract_failures_{scope}.csv", index=False
    )
    print(f"wrote {path.name}: {len(store)} descriptors; failures={len(failed)}")


def fold_map_from_v8() -> dict[str, int]:
    oof = pd.read_csv(ROOT / "oof_v8_final.csv")
    # The frozen v8 OOF partition is subject-level across both HAU and HARn.  Some
    # training subjects have no QA-referenced HARn row but do have labelled raw HARn clips;
    # retaining them in every outer fold according to this same assignment is valid extra
    # training data, not a held-out-subject leak.
    check = oof.groupby("subject_id").fold.nunique()
    if not (check == 1).all():
        raise AssertionError(f"v8 fold assignments are inconsistent: {check[check.ne(1)].to_dict()}")
    return oof.groupby("subject_id").fold.first().astype(int).to_dict()


def phrase_map(training_questions: pd.DataFrame) -> dict[str, str]:
    """Action phrase mapping learned only from the outer-fold training subjects."""
    single = training_questions[training_questions.category.eq("single")]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in single.itertuples():
        answer = str(getattr(r, r.answer))
        counts[normalise(answer)][r.action] += 1
    return {phrase: max(c, key=lambda a: (c[a], a)) for phrase, c in counts.items()}


def object_prior(training_questions: pd.DataFrame) -> tuple[dict[str, Counter[str]], Counter[str]]:
    obj = training_questions[training_questions.category.eq("object_interaction")]
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for r in obj.itertuples():
        answer = normalise(getattr(r, r.answer))
        by_action[r.action][answer] += 1
        global_counts[answer] += 1
    return by_action, global_counts


def option_prediction(
    row: pd.Series,
    classes: list[str],
    probability: np.ndarray,
    phrases: dict[str, str],
    obj_by_action: dict[str, Counter[str]],
    obj_global: Counter[str],
) -> tuple[str | None, float, float, list[str | None]]:
    """Map a pure video action posterior to a visible option answer."""
    class_index = {a: i for i, a in enumerate(classes)}
    choices = [str(row[c]) for c in "ABCD" if pd.notna(row[c])]
    letters = list("ABCD")[: len(choices)]
    category = row.category
    mapped: list[str | None] = []
    if category == "single":
        mapped = [phrases.get(normalise(choice)) for choice in choices]
        raw = np.array(
            [probability[class_index[action]] if action in class_index else 0.0 for action in mapped],
            dtype=float,
        )
    elif category == "object_interaction":
        mapped = [None] * len(choices)
        raw = np.zeros(len(choices), dtype=float)
        # The object prior is fit on outer-fold training labels.  The posterior itself is
        # depth-video-only; no LMT/action cache enters this calculation.
        for i, choice in enumerate(choices):
            word = normalise(choice)
            for ai, action in enumerate(classes):
                action_count = obj_by_action.get(action, Counter())
                denom = sum(action_count.values())
                if denom:
                    raw[i] += probability[ai] * action_count[word] / denom
            # preserve a deterministic, weak tie-breaker when no action has this object.
            raw[i] += 1e-8 * obj_global[word]
    else:
        return None, float("nan"), float("nan"), mapped

    if len(raw) == 0 or not np.isfinite(raw).all() or np.all(raw == raw[0]):
        return None, float("nan"), float("nan"), mapped
    order = np.argsort(raw)[::-1]
    top, second = int(order[0]), int(order[1])
    total = float(raw.sum())
    confidence = float(raw[top] / total) if total > 0 else float("nan")
    margin = float((raw[top] - raw[second]) / total) if total > 0 else float("nan")
    return letters[top], confidence, margin, mapped


def metric_row(data: pd.DataFrame, label: str) -> dict[str, Any]:
    d = data[data.video_prediction.notna()].copy()
    changed = d.video_prediction.ne(d.v8_prediction)
    win = changed & d.video_correct.eq(1) & d.v8_correct.eq(0)
    loss = changed & d.video_correct.eq(0) & d.v8_correct.eq(1)
    neutral = changed & ~(win | loss)
    denom = int(win.sum() + loss.sum())
    return dict(
        slice=label,
        n=int(len(d)),
        video_correct=int(d.video_correct.sum()),
        video_accuracy=float(d.video_correct.mean()) if len(d) else float("nan"),
        v8_correct=int(d.v8_correct.sum()),
        v8_accuracy=float(d.v8_correct.mean()) if len(d) else float("nan"),
        disagreements=int(changed.sum()),
        w_to_r=int(win.sum()),
        r_to_w=int(loss.sum()),
        neutral_disagreements=int(neutral.sum()),
        flip_precision=float(win.sum() / denom) if denom else float("nan"),
        net=int(win.sum() - loss.sum()),
        net_per_100_flips=float(100 * (win.sum() - loss.sum()) / changed.sum()) if changed.any() else float("nan"),
    )


def has_no_sensor_option(mapped_actions: str) -> bool:
    try:
        return bool(set(json.loads(mapped_actions)) & NO_SENSOR_ACTIONS)
    except (TypeError, json.JSONDecodeError):
        return False


def evaluate(scope: str) -> None:
    cache = read_cache(cache_path(scope))
    if not cache:
        raise FileNotFoundError(f"No descriptor cache at {cache_path(scope)}; run --extract first")
    clips, affected = manifest(scope)
    train_clips = [c for c in clips if c.split == "train" and c.key in cache]
    test_clips = [c for c in clips if c.split == "test" and c.key in cache]
    if not train_clips:
        raise RuntimeError("no training descriptors available")

    tr = pd.read_csv(ROOT / "training_qa.csv")
    harn = tr[tr.source.eq("HARn")].copy()
    extracted = harn.path.str.extract(r"HARn/([^/]+)/(user\d+)/")
    harn["action"], harn["user"] = extracted[0], extracted[1]
    v8 = pd.read_csv(ROOT / "oof_v8_final.csv")
    v8 = v8[["qa_id", "pred", "correct", "fold"]].rename(
        columns={"pred": "v8_prediction", "correct": "v8_correct"}
    )
    q = harn.merge(v8, on="qa_id", how="inner", validate="one_to_one")
    fold_of = fold_map_from_v8()
    q["fold_from_user"] = q.user.map(fold_of)
    if not q.fold.eq(q.fold_from_user).all():
        raise AssertionError("OOF fold does not match its subject assignment")

    # Every raw descriptor's label comes from its public training folder name, and every
    # outer validation subject is absent from the classifier's fit rows.
    train_df = pd.DataFrame(
        [dict(key=c.key, qpath=c.qpath, user=c.user, action=c.action) for c in train_clips]
    )
    train_df["fold"] = train_df.user.map(fold_of)
    unassigned = train_df.fold.isna()
    if unassigned.any():
        # Do not silently train with users whose held-out assignment is unknown.
        train_df = train_df[~unassigned].copy()
    train_df["fold"] = train_df.fold.astype(int)
    classes = sorted(train_df.action.unique())
    xall = np.stack([cache[k] for k in train_df.key])
    action_by_qpath = dict(zip(train_df.qpath, train_df.action))
    q = q[q.path.isin(action_by_qpath)].copy()
    q["actual_action"] = q.path.map(action_by_qpath)

    print(
        f"scope={scope}; descriptors train={len(train_df)}; classes={len(classes)}; "
        f"question rows={len(q)}; test descriptors={len(test_clips)}",
        flush=True,
    )
    print("held-out users per v8 fold:", pd.Series(fold_of).value_counts().sort_index().to_dict())

    oof_rows: list[dict[str, Any]] = []
    test_prob: dict[str, np.ndarray] = {}
    for fold in sorted(train_df.fold.unique()):
        train_mask = train_df.fold.to_numpy() != fold
        val_questions = q[q.fold.eq(fold)].copy()
        if val_questions.empty:
            continue
        # Each QA clip is also present once in train_df; recover its descriptor without using
        # any LMT-derived parent alignment.
        xq = np.stack([cache[f"train::{p}"] for p in val_questions.path])
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs"),
        )
        model.fit(xall[train_mask], train_df.loc[train_mask, "action"])
        proba = model.predict_proba(xq)
        learned_classes = list(model[-1].classes_)
        if learned_classes != classes:
            # Every action should have examples outside a fold; preserve an explicit failure if
            # the split violates that condition rather than comparing misaligned probabilities.
            raise AssertionError("a fold lacks an action class; cannot make a 43-way comparison")
        train_questions = q[q.fold.ne(fold)]
        phrases = phrase_map(train_questions)
        obj_by_action, obj_global = object_prior(train_questions)
        for (_, row), p in zip(val_questions.iterrows(), proba):
            pred, conf, margin, mapped = option_prediction(
                row, classes, p, phrases, obj_by_action, obj_global
            )
            oof_rows.append(
                dict(
                    qa_id=row.qa_id, fold=fold, user=row.user, path=row.path,
                    category=row.category, actual_action=row.actual_action, answer=row.answer,
                    v8_prediction=row.v8_prediction, v8_correct=int(row.v8_correct),
                    video_prediction=pred, video_correct=int(pred == row.answer) if pred else 0,
                    action_prediction=classes[int(np.argmax(p))],
                    action_correct=int(classes[int(np.argmax(p))] == row.actual_action),
                    action_top_probability=float(p.max()), option_confidence=conf,
                    option_margin=margin, mapped_actions=json.dumps(mapped),
                )
            )
        print(f"fold {fold}: fit={int(train_mask.sum())}, eval_questions={len(val_questions)}", flush=True)

    oof = pd.DataFrame(oof_rows).sort_values("qa_id").reset_index(drop=True)
    oof["structural_no_sensor_option"] = oof.mapped_actions.map(has_no_sensor_option)
    # On the real target, five single questions already contain exactly one action from the
    # no-sensor set and are solved by the visible modality-availability rule.  A visual
    # specialist should only be considered for the remaining five single and five object rows.
    oof["test_style_actionable"] = oof.category.eq("object_interaction") | ~oof.structural_no_sensor_option
    oof.to_csv(EXP / f"oof_depth_only_{scope}.csv", index=False)

    # Fixed confidence gates are reported descriptively.  The script does not build or modify
    # a submission, and a gate is never selected against the hidden test labels.
    gate_rows: list[dict[str, Any]] = []
    for gate in GATES:
        gated = oof.copy()
        keep = gated.option_confidence.ge(gate) & gated.video_prediction.ne(gated.v8_prediction)
        gated.loc[~keep, "video_prediction"] = gated.loc[~keep, "v8_prediction"]
        gated["video_correct"] = (gated.video_prediction == gated.answer).astype(int)
        for selection, selected in (("all", gated),
                                    ("test-style actionable", gated[gated.test_style_actionable])):
            out = metric_row(selected, f"{selection}; confidence>={gate:.2f}")
            out["gate"] = gate
            out["selection"] = selection
            out["proposed_overrides"] = int(
                keep[gated.test_style_actionable].sum() if selection != "all" else keep.sum()
            )
            gate_rows.append(out)
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(EXP / f"gate_summary_{scope}.csv", index=False)

    summary: list[dict[str, Any]] = [metric_row(oof, "all HARn single+object")]
    summary.append(metric_row(oof[oof.test_style_actionable], "test-style actionable"))
    for category, group in oof.groupby("category"):
        summary.append(metric_row(group, category))
    for fold, group in oof.groupby("fold"):
        summary.append(metric_row(group, f"fold_{fold}"))
    summary.append(
        dict(
            slice="action_top1",
            n=int(len(oof)),
            video_correct=int(oof.action_correct.sum()),
            video_accuracy=float(oof.action_correct.mean()),
            v8_correct=np.nan,
            v8_accuracy=np.nan,
            disagreements=np.nan,
            w_to_r=np.nan,
            r_to_w=np.nan,
            neutral_disagreements=np.nan,
            flip_precision=np.nan,
            net=np.nan,
            net_per_100_flips=np.nan,
        )
    )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(EXP / f"summary_{scope}.csv", index=False)

    # Train once on all raw training subjects for a *diagnostic* test-row table.  It contains
    # no submission and is only interpretable if the OOF gate table clears the stated bar.
    all_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs"),
    )
    all_model.fit(xall, train_df.action)
    all_phrases = phrase_map(q)
    all_obj_by_action, all_obj_global = object_prior(q)
    te_v8 = pd.read_csv(ROOT / "submission_v8.csv").set_index("qa_id").prediction
    test_feature = {c.qpath: cache[c.key] for c in test_clips}
    test_rows: list[dict[str, Any]] = []
    for _, row in affected.iterrows():
        p = all_model.predict_proba(test_feature[row.clip_id][None])[0]
        pred, conf, margin, mapped = option_prediction(
            row, classes, p, all_phrases, all_obj_by_action, all_obj_global
        )
        baseline = te_v8.loc[row.qa_id]
        structural = bool(set(mapped) & NO_SENSOR_ACTIONS)
        actionable = row.category == "object_interaction" or not structural
        test_rows.append(
            dict(
                qa_id=row.qa_id, clip_id=row.clip_id, category=row.category,
                v8_fallback=baseline, video_prediction=pred,
                differs_from_fallback=bool(pred and pred != baseline),
                action_prediction=classes[int(np.argmax(p))],
                action_top_probability=float(p.max()), option_confidence=conf,
                option_margin=margin, mapped_actions=json.dumps(mapped),
                structural_no_sensor_option=structural, test_style_actionable=actionable,
                gate_070=bool(actionable and pred and pred != baseline and conf >= 0.70),
                gate_080=bool(actionable and pred and pred != baseline and conf >= 0.80),
                gate_090=bool(actionable and pred and pred != baseline and conf >= 0.90),
            )
        )
    test_out = pd.DataFrame(test_rows).sort_values("qa_id")
    test_out.to_csv(EXP / f"test_depth_only_diagnostic_{scope}.csv", index=False)

    report = EXP / f"REPORT_{scope}.md"
    lines = [
        "# Direct-Depth missing-LMT ablation",
        "",
        "This run uses raw `Depth/Depth.mp4` only. It does not use LMT, skeleton, IMU, radar, "
        "IR, Depth_Color, `champ/meta.csv`, parent-frame alignment, or existing feature caches.",
        "",
        f"- Scope: `{scope}`; train descriptors: {len(train_df)}; action classes: {len(classes)}.",
        f"- Real target: {affected.clip_id.nunique()} no-LMT clips / {len(affected)} questions "
        f"({dict(affected.category.value_counts())}).",
        "- Comparator: held-out `oof_v8_final.csv` predictions, the same v8 vector used by "
        "`champ/final.py` for real no-evidence fallbacks.",
        "- This directory contains diagnostics only; no Kaggle submission was created.",
        "",
        "## Pooled and per-fold result",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
        "## Fixed confidence-gate audit",
        "",
        "```text",
        gates.to_string(index=False),
        "```",
        "",
        "## Limits",
        "",
        "- This is subject-disjoint, but the real 13 missing-LMT clips have unknown action mix; "
        "the training ablation cannot guarantee a matched hidden-label distribution.",
        "- The v8 comparison is exact as a fallback vector, but v8 itself was trained before this "
        "ablation and is not a video-only model.",
        "- Any test diagnostic must remain non-production unless a predeclared gate shows at least "
        "0.80 W-to-R / (W-to-R + R-to-W) precision with enough support and fold consistency.",
    ]
    report.write_text("\n".join(lines) + "\n")

    print("\n", summary_df.to_string(index=False))
    print("\nFixed gates:\n", gates.to_string(index=False))
    print(f"\nwrote {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("qa", "all"), default="all")
    parser.add_argument("--extract", action="store_true", help="encode direct Depth descriptors")
    parser.add_argument("--evaluate", action="store_true", help="run the subject-disjoint ablation")
    parser.add_argument("--resume", action="store_true", help="resume only this experiment cache")
    parser.add_argument("--limit", type=int, help="encode at most N currently missing clips (diagnostic)")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    args = parser.parse_args()
    if not args.extract and not args.evaluate:
        parser.error("choose --extract, --evaluate, or both")
    if args.extract:
        extract(args.scope, resume=args.resume, limit=args.limit, device_arg=args.device)
    if args.evaluate:
        evaluate(args.scope)


if __name__ == "__main__":
    main()
