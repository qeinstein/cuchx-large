#!/usr/bin/env python3
"""EXP-EMO-DINO-TRAJ-001: label-safe, relative temporal DINO style probe.

This is an isolated research script.  It never imports or mutates the champion
pipeline, writes no submission, and treats ``champ/audit_champ.csv`` as a
read-only OOF baseline snapshot.

Question
--------
Can *temporal trajectories* of frozen Depth-DINO frame embeddings distinguish
the manner/protocol of two otherwise comparable trials better than the static
or summary DINO features already in the champion?

Design safeguards
-----------------
* The descriptor receives only an ordered (T, 384) DINO sequence.  It contains
  frame-to-frame deltas, native-frame duration, velocity/acceleration/rhythm,
  and fixed-random-projection delta directions.  It contains no mean frame
  embedding, no option text, no label, and no action ID.
* It preserves the native 10-Hz sequence rather than resampling it.  Short
  contiguous phase summaries are calculated by pooling original frame ranges;
  raw duration/path length remain explicit features.
* The pair head receives only differences/ratios between sibling descriptors
  plus the *candidate* manner groups.  Hence action and subject appearance are
  nuisance variables removed by a within-session comparison.
* Every learned object (the orientation head) is fit only on training subjects
  for its held-out fold.  The fixed random projection is deterministic and has
  no fitted parameters.
* Results include both an oracle orientation diagnostic and the decision-level
  W->R/R->W audit against immutable champion OOF predictions.  The former is
  not promoted as an override result.

Usage
-----
    ./venv/bin/python research/dino_temporal_style_probe_20260904/probe_dino_trajectory.py

The default result directory is created once.  A rerun refuses to overwrite it;
pass a distinct ``--out`` path when running another configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAMP = ROOT / "champ"
sys.path.insert(0, str(CHAMP))

from core import (  # noqa: E402
    GROUPS,
    fit_group_model,
    gt_letters,
    infer_blocks,
    load_all,
    make_pseudo,
    mgroup,
)
from pseudotest import folds, thin_to_pairs  # noqa: E402


EXPERIMENT = "EXP-EMO-DINO-TRAJ-001"
SEED = 20260904
N_PHASE = 8
N_PROJ = 8
PAIR_THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


# This matrix is fixed before inspecting any labels or folds.  Its purpose is
# simply to retain signed delta direction cheaply without fitting PCA on a
# held-out subject.  Static DINO appearance cannot enter because it is applied
# only after frame differencing.
_rng = np.random.default_rng(SEED)
DELTA_PROJECTION = _rng.normal(
    0.0, 1.0 / np.sqrt(384), size=(384, N_PROJ)
).astype(np.float32)


def _safe(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _phase_pool(x: np.ndarray, n: int = N_PHASE) -> np.ndarray:
    """Pool contiguous native samples into ``n`` ordered pieces.

    This deliberately does not interpolate or warp a sequence.  A two-second
    clip and a fifty-second clip retain distinct duration and per-frame motion;
    only their *ordered local averages* are put in comparable descriptor slots.
    """
    x = _safe(x)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) == 0:
        return np.zeros((n, x.shape[1]), dtype=np.float32)
    edges = np.linspace(0, len(x), n + 1).round().astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            # With very short sequences use the nearest actual sample rather
            # than inventing an interpolated frame.
            ix = min(max(lo, 0), len(x) - 1)
            out.append(x[ix])
        else:
            out.append(np.mean(x[lo:hi], axis=0))
    return np.asarray(out, dtype=np.float32)


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = _safe(x).ravel()
    if len(x) <= lag + 2:
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = float(a.std()), float(b.std())
    if sa < 1e-7 or sb < 1e-7:
        return 0.0
    return float(np.clip(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb), -1.0, 1.0))


def _spectrum(x: np.ndarray, n: int = 6) -> np.ndarray:
    """First non-DC normalized spectral bins of a native-frame signal."""
    x = _safe(x).ravel()
    if len(x) < 4:
        return np.zeros(n, dtype=np.float32)
    z = x - x.mean()
    power = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    power = power[1 : n + 1]
    if len(power) < n:
        power = np.pad(power, (0, n - len(power)))
    return (power / (power.sum() + 1e-8)).astype(np.float32)


def dino_trajectory_descriptor(frames: np.ndarray) -> np.ndarray:
    """Make a DINO-only descriptor from an ordered, native-rate trajectory.

    All directional values come from frame differences of L2-normalized DINO
    tokens.  This removes each clip's static scene/subject representation from
    the feature route; the subsequent pair head looks only at sibling
    differences, which conditions movement style on the shared action script.
    """
    z = _safe(frames)
    if z.ndim != 2 or z.shape[1] != 384 or len(z) < 4:
        raise ValueError(f"expected >=4 x 384 DINO frame sequence, got {z.shape}")
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)
    dz = np.diff(z, axis=0)
    speed = np.linalg.norm(dz, axis=1)
    dd = np.diff(dz, axis=0)
    acceleration = np.linalg.norm(dd, axis=1) if len(dd) else np.zeros(1, dtype=np.float32)
    if len(dz) > 1:
        denom = np.linalg.norm(dz[:-1], axis=1) * np.linalg.norm(dz[1:], axis=1) + 1e-7
        curvature = 1.0 - np.clip(np.sum(dz[:-1] * dz[1:], axis=1) / denom, -1.0, 1.0)
    else:
        curvature = np.zeros(1, dtype=np.float32)

    path = float(speed.sum())
    net = float(np.linalg.norm(z[-1] - z[0]))
    speed_q = np.percentile(speed, [10, 50, 90, 99])
    acc_q = np.percentile(acceleration, [50, 90])
    glob = np.asarray(
        [
            np.log1p(len(z)),              # original frame-count/duration
            np.log1p(path),                 # total native-rate trajectory length
            np.log1p(net),
            net / (path + 1e-6),            # path directness
            speed.mean(), speed.std(), *speed_q,
            speed.max(),
            acceleration.mean(), *acc_q,
            curvature.mean(), np.percentile(curvature, 90),
            speed.std() / (speed.mean() + 1e-6),
        ],
        dtype=np.float32,
    )

    # Raw phase means retain which part of a trial has activity; normalized
    # phase means isolate its temporal shape from absolute pace.
    speed_phase = _phase_pool(speed).ravel()
    speed_shape = speed_phase / (float(speed.mean()) + 1e-6)
    acc_phase = _phase_pool(acceleration).ravel()

    # Signed delta directions are projected with a label-free fixed map.
    # Their phase pooling differentiates the order of movement components,
    # unlike prior scalar motion summaries.
    proj_phase = _phase_pool(dz @ DELTA_PROJECTION).ravel()
    proj_std = np.std(dz @ DELTA_PROJECTION, axis=0)

    rhythm = np.concatenate(
        [_spectrum(speed), np.asarray([_autocorr(speed, lag) for lag in (1, 2, 5, 10)], dtype=np.float32)]
    )
    out = np.concatenate([glob, speed_phase, speed_shape, acc_phase, proj_phase, proj_std, rhythm])
    out = _safe(out)
    # Useful as an accidental-change guard: 17 globals + 8 + 8 + 8 + 64 + 8 + 10.
    # (`glob` is 17-wide: len, path, net, directness, speed mean/std/q10/q50/q90/q99/max,
    # accel mean/q50/q90, curvature mean/p90, speed CV -- the original docstring miscounted it as 16.)
    assert len(out) == 123, len(out)
    return out.astype(np.float32, copy=False)


@dataclass(frozen=True)
class DescriptorStore:
    vectors: dict[str, np.ndarray]
    coverage: pd.DataFrame


def build_descriptors(emotion: pd.DataFrame) -> DescriptorStore:
    """Read only the 809 training Emotion trajectories from the frozen cache."""
    cache_path = CHAMP / "dino_frames.npz"
    z = np.load(cache_path, allow_pickle=False)
    vectors: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for path in sorted(emotion.path.unique()):
        if path not in z.files:
            raise RuntimeError(f"Emotion DINO coverage failure: {path!r} absent from {cache_path}")
        f = z[path]
        v = dino_trajectory_descriptor(f)
        vectors[path] = v
        rows.append(
            {
                "path": path,
                "frames_native": int(len(f)),
                "embedding_dim": int(f.shape[1]),
                "descriptor_dim": int(len(v)),
                "finite": int(np.isfinite(v).all()),
            }
        )
    coverage = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    assert len(vectors) == len(emotion.path.unique()) == 809
    assert coverage.finite.eq(1).all()
    return DescriptorStore(vectors=vectors, coverage=coverage)


def _label(row: pd.Series | object, letter: str) -> str:
    return str(row[letter] if isinstance(row, pd.Series) else getattr(row, letter)).strip()


def _letter_for(row: pd.Series | object, manner: str) -> str | None:
    for letter in "ABCD":
        if _label(row, letter) == manner:
            return letter
    return None


def _true_manner(row: pd.Series | object) -> str:
    if not isinstance(row, pd.Series):
        row = pd.Series(row._asdict())
    return _label(row, gt_letters(row)[0])


def _session_key(path: str) -> tuple[str, str, str] | None:
    """Diagnostic-only true grouping for pseudo-test reports; never used in scores."""
    bits = str(path).split("/")
    if len(bits) != 3 or bits[0] != "HAU" or "-" not in bits[2]:
        return None
    a, b, _ = bits[2].split("-")
    return bits[1], a, b


def pair_vector(a: np.ndarray, b: np.ndarray, group_a: str, group_b: str) -> np.ndarray:
    """Candidate-conditioned relative descriptor for one ordered sibling pair.

    There are no absolute clip values or slot/index features here.  The only
    label-bearing inputs are the two candidate groups whose orientation is being
    tested, exactly as they would be available through a question's options.
    """
    a, b = _safe(a), _safe(b)
    diff = a - b
    rel = diff / (np.abs(a) + np.abs(b) + 1e-4)
    groups = np.zeros(2 * len(GROUPS), dtype=np.float32)
    groups[GROUPS.index(group_a)] = 1.0
    groups[len(GROUPS) + GROUPS.index(group_b)] = 1.0
    return np.concatenate([diff, np.clip(rel, -20.0, 20.0), groups]).astype(np.float32)


def build_pair_training(emotion_train: pd.DataFrame, store: DescriptorStore) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Construct balanced correct-vs-swapped pair hypotheses from train subjects only."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    stats: Counter[str] = Counter()
    for _, block in emotion_train.groupby(["user", "aa", "bb"], sort=False):
        rows = list(block.sort_values("cc").itertuples(index=False))
        for i, j in combinations(range(len(rows)), 2):
            left, right = rows[i], rows[j]
            ma, mb = _true_manner(left), _true_manner(right)
            ga, gb = mgroup(ma), mgroup(mb)
            if ga == gb:
                stats["same_group_skipped"] += 1
                continue
            a, b = store.vectors.get(left.path), store.vectors.get(right.path)
            if a is None or b is None:
                stats["missing_descriptor"] += 1
                continue
            # Symmetric construction makes class balance and orientation exact.
            xs += [pair_vector(a, b, ga, gb), pair_vector(a, b, gb, ga)]
            ys += [1, 0]
            stats["valid_pairs"] += 1
    if not xs:
        raise RuntimeError("No cross-group training pairs")
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int8)
    assert len(x) == len(y) and set(y) == {0, 1}
    stats["hypotheses"] = len(y)
    stats["features"] = x.shape[1]
    return x, y, dict(stats)


@dataclass
class PairHead:
    name: str
    model: object

    def probability(self, x: np.ndarray) -> float:
        p = self.model.predict_proba(np.asarray(x, dtype=np.float32)[None])[0]
        classes = list(self.model.classes_)
        return float(p[classes.index(1)])


def fit_heads(x: np.ndarray, y: np.ndarray) -> dict[str, PairHead]:
    """Fit a conservative linear and nonlinear head to cross-check capacity risk."""
    ridge = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.08, max_iter=4000, random_state=SEED)),
        ]
    )
    ridge.fit(x, y)

    # Small leaves and strong L2 intentionally resist a handful of users or
    # one repeated script becoming a memorized signature.
    tree = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.05,
                    max_leaf_nodes=8,
                    min_samples_leaf=20,
                    l2_regularization=16.0,
                    random_state=SEED,
                ),
            ),
        ]
    )
    tree.fit(x, y)
    return {"ridge": PairHead("ridge", ridge), "tree": PairHead("tree", tree)}


def orientation_score(head: PairHead, a: np.ndarray, b: np.ndarray, ma: str, mb: str) -> tuple[float, float, float]:
    """Positive supports current orientation; negative supports its transposition."""
    ga, gb = mgroup(ma), mgroup(mb)
    pc = np.clip(head.probability(pair_vector(a, b, ga, gb)), 1e-5, 1 - 1e-5)
    ps = np.clip(head.probability(pair_vector(a, b, gb, ga)), 1e-5, 1 - 1e-5)
    return float(np.log(pc / (1 - pc)) - np.log(ps / (1 - ps))), float(pc), float(ps)


def visible_emotion_blocks(vis: pd.DataFrame, blocks: Iterable[Iterable[int]]) -> list[tuple[int, list[pd.Series]]]:
    emotion = vis[vis.category.eq("emotion")].set_index("idx")
    out = []
    for block_id, block in enumerate(blocks):
        rows = [emotion.loc[i] for i in block if i in emotion.index]
        if len(rows) >= 2:
            out.append((block_id, rows))
    return out


def oracle_rows(
    head: PairHead,
    vis: pd.DataFrame,
    blocks: Iterable[Iterable[int]],
    answers: dict[str, str],
    store: DescriptorStore,
    fold: int,
    regime: str,
) -> list[dict[str, object]]:
    """Gold-only post-hoc orientation diagnostic; never used by candidate scoring."""
    out: list[dict[str, object]] = []
    for block_id, rows in visible_emotion_blocks(vis, blocks):
        for i, j in combinations(range(len(rows)), 2):
            left, right = rows[i], rows[j]
            ma, mb = _label(left, answers[left.qa_id]), _label(right, answers[right.qa_id])
            if mgroup(ma) == mgroup(mb):
                continue
            a, b = store.vectors[left.true_path], store.vectors[right.true_path]
            score, pc, ps = orientation_score(head, a, b, ma, mb)
            same_session = int(_session_key(left.true_path) == _session_key(right.true_path))
            out.append(
                {
                    "regime": regime,
                    "head": head.name,
                    "fold": fold,
                    "block": block_id,
                    "k": len(rows),
                    "qa_i": left.qa_id,
                    "qa_j": right.qa_id,
                    "true_group_i": mgroup(ma),
                    "true_group_j": mgroup(mb),
                    "score": score,
                    "p_current": pc,
                    "p_swapped": ps,
                    "correct_orientation": int(score > 0),
                    # Diagnostic only: the model sees neither hidden user nor true session.
                    "same_true_session": same_session,
                }
            )
    return out


def candidate_rows(
    head: PairHead,
    vis: pd.DataFrame,
    blocks: Iterable[Iterable[int]],
    baseline: pd.DataFrame,
    store: DescriptorStore,
    fold: int,
    regime: str,
) -> list[dict[str, object]]:
    """Score valid test-visible one-transposition proposals against a fixed baseline."""
    pred = dict(zip(baseline.qa_id, baseline.pred))
    out: list[dict[str, object]] = []
    for block_id, rows in visible_emotion_blocks(vis, blocks):
        for i, j in combinations(range(len(rows)), 2):
            left, right = rows[i], rows[j]
            pi, pj = pred.get(left.qa_id), pred.get(right.qa_id)
            if pi not in "ABCD" or pj not in "ABCD":
                continue
            ma, mb = _label(left, pi), _label(right, pj)
            if ma == mb or mgroup(ma) == mgroup(mb):
                continue
            ni, nj = _letter_for(left, mb), _letter_for(right, ma)
            # A label-level swap is useful only if it produces valid letters in
            # both original question option lists.
            if ni is None or nj is None or ni == pi or nj == pj:
                continue
            a, b = store.vectors[left.true_path], store.vectors[right.true_path]
            score, pc, ps = orientation_score(head, a, b, ma, mb)
            out.append(
                {
                    "regime": regime,
                    "head": head.name,
                    "fold": fold,
                    "block": block_id,
                    "k": len(rows),
                    "qa_i": left.qa_id,
                    "qa_j": right.qa_id,
                    "base_i": pi,
                    "base_j": pj,
                    "new_i": ni,
                    "new_j": nj,
                    "group_i": mgroup(ma),
                    "group_j": mgroup(mb),
                    "score": score,
                    "p_current": pc,
                    "p_swapped": ps,
                    # Post-hoc grouping diagnostic only; excluded from selection.
                    "same_true_session": int(_session_key(left.true_path) == _session_key(right.true_path)),
                }
            )
    return out


def select_nonoverlapping(candidates: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Take strongest disjoint swaps per inferred block at a fixed confidence gate."""
    if candidates.empty:
        return candidates.copy()
    chosen = []
    columns = ["regime", "head", "fold", "block"]
    for _, group in candidates[candidates.score <= -threshold].sort_values("score").groupby(columns, sort=False):
        used: set[str] = set()
        for row in group.itertuples(index=False):
            if row.qa_i in used or row.qa_j in used:
                continue
            chosen.append(row._asdict())
            used.update([row.qa_i, row.qa_j])
    return pd.DataFrame(chosen, columns=candidates.columns)


def flip_audit(candidates: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit direct changes at every predeclared confidence threshold."""
    base = baseline.set_index("qa_id").copy()
    if not np.array_equal((base.pred.astype(str) == base.answer.astype(str)).astype(int), base.correct.astype(int)):
        raise AssertionError("baseline correct column does not match its immutable predictions")
    summaries: list[dict[str, object]] = []
    folds_out: list[dict[str, object]] = []
    for threshold in PAIR_THRESHOLDS:
        selected = select_nonoverlapping(candidates, threshold)
        updates: dict[str, str] = {}
        for r in selected.itertuples(index=False):
            updates[r.qa_i] = r.new_i
            updates[r.qa_j] = r.new_j
        changed = base.loc[list(updates)] if updates else base.iloc[0:0].copy()
        if len(changed):
            changed = changed.assign(new_prediction=[updates[q] for q in changed.index])
            old = changed.correct.astype(int).to_numpy()
            new = (changed.new_prediction.astype(str).to_numpy() == changed.answer.astype(str).to_numpy()).astype(int)
            wr = int(((old == 0) & (new == 1)).sum())
            rw = int(((old == 1) & (new == 0)).sum())
            ww = int(((old == 0) & (new == 0)).sum())
            rr = int(((old == 1) & (new == 1)).sum())
        else:
            wr = rw = ww = rr = 0
        summaries.append(
            {
                "threshold": threshold,
                "selected_pairs": int(len(selected)),
                "flips": int(len(updates)),
                "W_to_R": wr,
                "R_to_W": rw,
                "wrong_to_wrong": ww,
                "right_to_right": rr,
                "flip_precision": (wr / (wr + rw) if wr + rw else np.nan),
                "net": wr - rw,
            }
        )
        for fold, g in changed.groupby("fold"):
            old = g.correct.astype(int).to_numpy()
            new = (g.new_prediction.astype(str).to_numpy() == g.answer.astype(str).to_numpy()).astype(int)
            w = int(((old == 0) & (new == 1)).sum())
            r = int(((old == 1) & (new == 0)).sum())
            folds_out.append(
                {
                    "threshold": threshold,
                    "fold": int(fold),
                    "flips": int(len(g)),
                    "W_to_R": w,
                    "R_to_W": r,
                    "flip_precision": (w / (w + r) if w + r else np.nan),
                    "net": w - r,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(folds_out)


def check_fold_contract(train: pd.DataFrame, audit_e: pd.DataFrame) -> dict[int, list[str]]:
    users = sorted(train.user.dropna().unique())
    held_by_fold = {fi: list(held) for fi, held in enumerate(folds(users, 5))}
    expected = {u: fi for fi, held in held_by_fold.items() for u in held}
    q = train[train.category.eq("emotion")][["qa_id", "user"]].merge(
        audit_e[["qa_id", "fold"]], on="qa_id", validate="one_to_one"
    )
    mismatches = q.fold.astype(int) != q.user.map(expected).astype(int)
    if mismatches.any():
        raise AssertionError(f"audit fold contract mismatch for {int(mismatches.sum())} Emotion rows")
    return held_by_fold


@dataclass
class FoldState:
    held: list[str]
    group_model: dict
    heads: dict[str, PairHead]
    pair_stats: dict[str, int]


def build_fold_states(train: pd.DataFrame, store: DescriptorStore, held_by_fold: dict[int, list[str]]) -> dict[int, FoldState]:
    states: dict[int, FoldState] = {}
    for fold, held in held_by_fold.items():
        trn = train[~train.user.isin(held)]
        x, y, stats = build_pair_training(trn[trn.category.eq("emotion")], store)
        states[fold] = FoldState(
            held=held,
            group_model=fit_group_model(trn),
            heads=fit_heads(x, y),
            pair_stats=stats,
        )
        print(
            f"fold {fold}: held={','.join(held)} hypotheses={len(y)} features={x.shape[1]} "
            f"cross-group-pairs={stats['valid_pairs']}",
            flush=True,
        )
    return states


def run_regime(
    train: pd.DataFrame,
    meta: pd.DataFrame,
    store: DescriptorStore,
    states: dict[int, FoldState],
    baseline: pd.DataFrame,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate ordinary or held-out pair-thinned pseudo-test construction."""
    all_oracle: list[dict[str, object]] = []
    all_candidates: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for fold, state in states.items():
        evaluation = thin_to_pairs(train, state.held, 0.38, seed=fold) if regime == "pair_thinned" else train
        vis, key, _ = make_pseudo(evaluation, meta, state.held, seed=100 + fold)
        blocks = infer_blocks(vis, state.group_model)
        present = set(vis.loc[vis.category.eq("emotion"), "qa_id"])
        base = baseline[baseline.fold.eq(fold)].copy()
        if set(base.qa_id) != present:
            raise AssertionError(
                f"{regime} fold {fold}: baseline/evaluation mismatch "
                f"({len(base)} base vs {len(present)} visible Emotion rows)"
            )
        answer = dict(zip(key.qa_id, key.answer))
        diag_blocks = visible_emotion_blocks(vis, blocks)
        block_rows.extend(
            {
                "regime": regime,
                "fold": fold,
                "inferred_block": block_id,
                "n_emotion_clips": len(rows),
                "all_same_true_session": int(
                    len({_session_key(r.true_path) for r in rows}) == 1
                ),
            }
            for block_id, rows in diag_blocks
        )
        for head in state.heads.values():
            all_oracle.extend(oracle_rows(head, vis, blocks, answer, store, fold, regime))
            all_candidates.extend(candidate_rows(head, vis, blocks, base, store, fold, regime))
        print(
            f"{regime} fold {fold}: visible Emotion={len(present)} inferred blocks={len(diag_blocks)}",
            flush=True,
        )
    return pd.DataFrame(all_oracle), pd.DataFrame(all_candidates), pd.DataFrame(block_rows)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_output(path: Path) -> None:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results", help="new empty output directory")
    args = parser.parse_args()
    prepare_output(args.out)
    out = args.out.resolve()

    train, _, meta = load_all()
    emotion = train[(train.source.eq("HAU")) & (train.category.eq("emotion"))].copy()
    audit_path = CHAMP / "audit_champ.csv"
    audit = pd.read_csv(audit_path)
    audit_e = audit[(audit.source.eq("HAU")) & (audit.category.eq("emotion"))].copy()
    if len(emotion) != 809 or len(audit_e) != 809:
        raise AssertionError(f"expected 809 Emotion rows; found train={len(emotion)} audit={len(audit_e)}")
    if audit_path.is_symlink():
        raise AssertionError("audit snapshot must be a regular immutable file, not a symlink")
    held_by_fold = check_fold_contract(train, audit_e)
    store = build_descriptors(emotion)
    store.coverage.to_csv(out / "descriptor_coverage.csv", index=False)

    states = build_fold_states(train, store, held_by_fold)
    ordinary_o, ordinary_c, ordinary_b = run_regime(
        train, meta, store, states, audit_e, "ordinary"
    )

    pair_base = pd.read_csv(CHAMP / "oof_pairstress.csv")
    pair_base = pair_base[(pair_base.source.eq("HAU")) & (pair_base.category.eq("emotion"))].copy()
    pair_o, pair_c, pair_b = run_regime(
        train, meta, store, states, pair_base, "pair_thinned"
    )

    oracle = pd.concat([ordinary_o, pair_o], ignore_index=True)
    candidates = pd.concat([ordinary_c, pair_c], ignore_index=True)
    inferred = pd.concat([ordinary_b, pair_b], ignore_index=True)
    oracle.to_csv(out / "orientation_oracle_oof.csv", index=False)
    candidates.to_csv(out / "candidate_pair_scores_oof.csv", index=False)
    inferred.to_csv(out / "inferred_block_diagnostics.csv", index=False)

    orientation_summary = (
        oracle.groupby(["regime", "head", "fold", "k", "same_true_session"], dropna=False)
        .correct_orientation.agg(n="size", correct="sum", accuracy="mean")
        .reset_index()
    )
    orientation_summary.to_csv(out / "orientation_oracle_summary.csv", index=False)

    flip_reports: list[pd.DataFrame] = []
    fold_reports: list[pd.DataFrame] = []
    for regime, base in (("ordinary", audit_e), ("pair_thinned", pair_base)):
        for head, cand in candidates[candidates.regime.eq(regime)].groupby("head", sort=False):
            summary, by_fold = flip_audit(cand, base)
            summary.insert(0, "head", head)
            summary.insert(0, "regime", regime)
            by_fold.insert(0, "head", head)
            by_fold.insert(0, "regime", regime)
            flip_reports.append(summary)
            fold_reports.append(by_fold)
    flips = pd.concat(flip_reports, ignore_index=True)
    flips_by_fold = pd.concat(fold_reports, ignore_index=True)
    flips.to_csv(out / "flip_audit_by_threshold.csv", index=False)
    flips_by_fold.to_csv(out / "flip_audit_by_fold.csv", index=False)

    summary = {
        "experiment": EXPERIMENT,
        "descriptor": {
            "source": "champ/dino_frames.npz",
            "cache_sequences": 809,
            "descriptor_dim": int(next(iter(store.vectors.values())).shape[0]),
            "native_order_duration_preserved": True,
            "uses_static_frame_embedding": False,
            "uses_imu_radar_skeleton_dtw": False,
            "projection": f"fixed Gaussian random projection seed={SEED}, dim={N_PROJ}",
        },
        "validation": {
            "subject_disjoint_folds": {str(f): state.held for f, state in states.items()},
            "audit_snapshot": str(audit_path.relative_to(ROOT)),
            "audit_sha256": _hash(audit_path),
            "ordinary_emotion_rows": int(len(audit_e)),
            "pair_thinned_emotion_rows": int(len(pair_base)),
            "pair_thinned_fraction": 0.38,
            "pair_stats_by_fold": {str(f): state.pair_stats for f, state in states.items()},
        },
        "counts": {
            "oracle_rows": int(len(oracle)),
            "candidate_pairs": int(len(candidates)),
            "ordinary_candidate_pairs": int(len(ordinary_c)),
            "pair_thinned_candidate_pairs": int(len(pair_c)),
        },
        "promotion_rule": "No candidate is promotable unless its direct flip audit is high precision and positive in every relevant fold; orientation accuracy alone is not sufficient.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\n=== orientation diagnostic (gold used only after scoring) ===")
    print(orientation_summary.to_string(index=False))
    print("\n=== direct flip audit versus frozen baselines ===")
    print(flips.to_string(index=False))
    print(f"\nWrote isolated artifacts to {out}")


if __name__ == "__main__":
    main()
