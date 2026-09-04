"""EXP-EMO-RELRAW-001: raw, reference-based motion-style probe.

This is deliberately a *specialist* experiment.  It does not refit the championship
pipeline and it never writes a Kaggle submission.  It asks whether raw temporal
skeletal and DINOv2-depth trajectories can decide the orientation of a pair of
otherwise comparable trials in the same inferred session:

    (clip_i, clip_j, current manners) -> keep current orientation / swap them

The descriptor is label-free and is computed before any fold split.  The pair head is
fit only on training subjects in every fold.  Evaluation is against the immutable
``champ/audit_champ.csv`` snapshot, not against a later rerun of mutable code.

The design is intentionally close to reference-based action-quality assessment: it
uses differences between temporal trajectories from sibling trials, so action and
subject identity are nuisance variables rather than the target.  It is distinct from
the previously rejected summary-feature and DTW-only arms: no DTW features, IMU, or
radar values are used here.

Usage:
    ./venv/bin/python emolab/probe_relative_raw.py

Outputs are isolated under research/emotion_relative_raw_20260904/.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "emotion_relative_raw_20260904")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "champ"))

from core import (  # noqa: E402
    GROUPS,
    fit_group_model,
    gt_letters,
    infer_blocks,
    load_all,
    make_pseudo,
    mgroup,
    opts,
    real_test_view,
)
from pseudotest import folds, thin_to_pairs  # noqa: E402


N_PROFILE = 10
N_DINO_BINS = 6
RNG = np.random.default_rng(20260904)
# Fixed, label-free random projections preserve directions in the 384-D frozen DINO
# trajectory without fitting an unsupervised transform on a held-out subject.
DINO_R = RNG.normal(0.0, 1.0 / np.sqrt(384), size=(384, 10)).astype(np.float32)

BODY = {
    "all": list(range(17)),
    "torso": [5, 6, 11, 12],
    "larm": [5, 7, 9],
    "rarm": [6, 8, 10],
    "lleg": [11, 13, 15],
    "rleg": [12, 14, 16],
}


def _interp(x: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a 1-D temporal signal onto phase, not wall-clock time."""
    x = np.asarray(x, dtype=np.float32).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) == 1:
        return np.full(n, x[0], dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def _spectral_shape(x: np.ndarray, n: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 4:
        return np.zeros(n, dtype=np.float32)
    z = x - np.mean(x)
    p = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    p = p[1 : n + 1]
    if len(p) < n:
        p = np.pad(p, (0, n - len(p)))
    return (p / (p.sum() + 1e-8)).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def skeleton_descriptor(k: np.ndarray | None, fr: np.ndarray | None) -> np.ndarray:
    """Body-centred, joint-wise temporal style descriptor from raw 3-D skeletons."""
    if k is None or len(k) < 5:
        # Kept as NaN so the fold-fitted imputer cannot mistake missing skeletons for stillness.
        return np.full(154, np.nan, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if fr is None or len(fr) != len(k):
        fr = np.arange(len(k), dtype=np.float32)
    else:
        fr = np.asarray(fr, dtype=np.float32)
    dt = np.diff(fr)
    dt[~np.isfinite(dt) | (dt <= 0)] = 1.0

    root = k[:, [11, 12]].mean(axis=1)
    torso = (np.linalg.norm(k[:, 5] - k[:, 11], axis=1)
             + np.linalg.norm(k[:, 6] - k[:, 12], axis=1))
    torso = torso[np.isfinite(torso) & (torso > 1e-5)]
    scale = float(np.median(torso)) if len(torso) else 1.0
    p = (k - root[:, None, :]) / max(scale, 1e-4)
    v = np.diff(p, axis=0) / dt[:, None, None]
    vm = np.linalg.norm(v, axis=2)

    out: list[float] = [float(np.log1p(len(k))), float(np.log1p(max(fr[-1] - fr[0], 0.0)))]
    profiles: dict[str, np.ndarray] = {}
    for name, inds in BODY.items():
        s = np.nanmean(vm[:, inds], axis=1)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        profiles[name] = s
        q = np.percentile(s, [10, 50, 90]) if len(s) else np.zeros(3)
        out.extend(np.log1p([float(s.mean()), float(s.std()), *q]).tolist())
        phase = _interp(s, N_PROFILE)
        out.extend((phase / (float(s.mean()) + 1e-5)).tolist())
        out.extend(_spectral_shape(s).tolist())

    # Movement amplitude / body-part involvement, independently of speed.
    for name in ("torso", "larm", "rarm", "lleg", "rleg"):
        q = p[:, BODY[name], :]
        std = np.nanmean(np.nanstd(q, axis=0), axis=0)
        ran = np.nanmean(np.nanpercentile(q, 90, axis=0) - np.nanpercentile(q, 10, axis=0), axis=0)
        out.extend(np.nan_to_num(std).tolist())
        out.extend(np.nan_to_num(ran).tolist())

    # Directly captures synchrony, asymmetric involvement, and pause/rhythm structure.
    out.extend([
        _corr(profiles["larm"], profiles["rarm"]),
        _corr(profiles["lleg"], profiles["rleg"]),
        _corr(profiles["torso"], profiles["all"]),
    ])
    for name in ("all", "larm", "rarm", "lleg", "rleg"):
        s = profiles[name]
        th = float(np.percentile(s, 25)) if len(s) else 0.0
        out.append(float(np.mean(s <= th + 1e-7)))
    x = np.asarray(out, dtype=np.float32)
    assert len(x) == 154, len(x)
    return x


def dino_descriptor(z: np.ndarray | None) -> np.ndarray:
    """Temporal direction/profile descriptor from frozen DINO depth-frame embeddings."""
    if z is None or len(z) < 4:
        return np.full(89, np.nan, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)
    dz = np.diff(z, axis=0)
    speed = np.linalg.norm(dz, axis=1)
    proj = dz @ DINO_R
    out: list[float] = []
    out.extend(np.log1p([float(len(z)), float(speed.mean()), float(speed.std()),
                         float(np.percentile(speed, 50)), float(np.percentile(speed, 90))]))
    phase = _interp(speed, N_PROFILE)
    out.extend((phase / (float(speed.mean()) + 1e-6)).tolist())
    out.extend(_spectral_shape(speed).tolist())
    # Phase-local directional change: static DINO content cancels because only deltas enter.
    for a, b in zip(np.linspace(0, len(proj), N_DINO_BINS, endpoint=False, dtype=int),
                    np.linspace(0, len(proj), N_DINO_BINS + 1, dtype=int)[1:]):
        q = proj[a:max(a + 1, b)]
        out.extend(np.mean(q, axis=0).tolist())
    out.extend(np.std(proj, axis=0).tolist())
    x = np.asarray(out, dtype=np.float32)
    assert len(x) == 89, len(x)
    return x


@dataclass
class RawStore:
    vectors: dict[str, np.ndarray]
    coverage: pd.DataFrame


def build_descriptors(meta: pd.DataFrame) -> RawStore:
    """Build label-free clip vectors keyed by the canonical qa_path/cache key."""
    sk = np.load(os.path.join(ROOT, "champ", "skel_seq.npz"), allow_pickle=True)
    di = np.load(os.path.join(ROOT, "champ", "dino_frames.npz"), allow_pickle=True)
    vectors: dict[str, np.ndarray] = {}
    rows = []
    for r in meta[meta.kind.isin(["train_hau", "test"])].drop_duplicates("qa_path").itertuples():
        kk = sk.get(r.unit_dir + "|K") if r.unit_dir + "|K" in sk.files else None
        ff = sk.get(r.unit_dir + "|F") if r.unit_dir + "|F" in sk.files else None
        zz = di.get(r.qa_path) if r.qa_path in di.files else None
        x = np.concatenate([skeleton_descriptor(kk, ff), dino_descriptor(zz)]).astype(np.float32)
        vectors[r.qa_path] = x
        rows.append(dict(qa_path=r.qa_path, has_skeleton=int(kk is not None),
                         has_dino=int(zz is not None), n_features=len(x)))
    cov = pd.DataFrame(rows).sort_values("qa_path")
    cov.to_csv(os.path.join(OUT, "raw_descriptor_coverage.csv"), index=False)
    return RawStore(vectors=vectors, coverage=cov)


def _label(row: pd.Series | object, letter: str) -> str:
    return str(row[letter] if isinstance(row, pd.Series) else getattr(row, letter)).strip()


def _letter_for(row: pd.Series | object, label: str) -> str | None:
    for letter in "ABCD":
        if _label(row, letter) == label:
            return letter
    return None


def _true_label(row: pd.Series) -> str:
    return _label(row, gt_letters(row)[0])


def pair_vector(va: np.ndarray, vb: np.ndarray, ma: str, mb: str,
                i: int, j: int, k: int) -> np.ndarray:
    """Features for the assignment hypothesis i->ma, j->mb.

    The signed trajectory difference is unchanged by a hypothesis swap; the appended
    one-hot group labels let the fixed classifier learn which relative physical pattern
    supports e.g. FAST on i and SLOW on j versus the reverse.
    """
    va = np.asarray(va, dtype=np.float32)
    vb = np.asarray(vb, dtype=np.float32)
    diff = va - vb
    rel = diff / (np.abs(va) + np.abs(vb) + 1e-4)
    ga, gb = mgroup(ma), mgroup(mb)
    g = np.zeros(2 * len(GROUPS), dtype=np.float32)
    g[GROUPS.index(ga)] = 1.0
    g[len(GROUPS) + GROUPS.index(gb)] = 1.0
    slot = np.array([i / max(k - 1, 1), j / max(k - 1, 1), float(k)], dtype=np.float32)
    return np.concatenate([diff, rel, g, slot]).astype(np.float32)


@dataclass
class PairHead:
    imputer: SimpleImputer
    clf: HistGradientBoostingClassifier
    include_motion: bool

    def probability(self, x: np.ndarray) -> float:
        z = self.imputer.transform(x.reshape(1, -1))
        p = self.clf.predict_proba(z)[0]
        return float(p[list(self.clf.classes_).index(1)])


def fit_pair_head(train: pd.DataFrame, store: RawStore, include_motion: bool = True) -> PairHead:
    """Fit only on known training subjects and true training sessions."""
    emo = train[train.category.eq("emotion")].copy()
    xs: list[np.ndarray] = []
    yy: list[int] = []
    for _, g in emo.groupby(["user", "aa", "bb"], sort=False):
        g = g.sort_values("cc")
        rs = list(g.itertuples(index=False))
        k = len(rs)
        for i, j in combinations(range(k), 2):
            a, b = rs[i], rs[j]
            ma, mb = _true_label(pd.Series(a._asdict())), _true_label(pd.Series(b._asdict()))
            # This mechanism is deliberately about physical manner groups.  Same-group
            # word swaps require language/slot evidence, not raw movement evidence.
            if ma == mb or mgroup(ma) == mgroup(mb):
                continue
            va = store.vectors.get(a.path)
            vb = store.vectors.get(b.path)
            if va is None or vb is None:
                continue
            if include_motion:
                x_good = pair_vector(va, vb, ma, mb, i, j, k)
                x_swap = pair_vector(va, vb, mb, ma, i, j, k)
            else:
                # Matched nuisance control: label-group and position priors only.  It
                # establishes whether seemingly high pair orientation is really motion.
                z = np.zeros_like(va)
                x_good = pair_vector(z, z, ma, mb, i, j, k)
                x_swap = pair_vector(z, z, mb, ma, i, j, k)
            xs.extend([x_good, x_swap])
            yy.extend([1, 0])
    if not xs:
        raise RuntimeError("no training pair rows")
    x = np.vstack(xs)
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(x)
    # Conservative capacity is intentional: there are only ~1.6k orientation examples.
    clf = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05,
                                         max_leaf_nodes=12, l2_regularization=8.0,
                                         random_state=20260904)
    clf.fit(x, yy)
    return PairHead(imp, clf, include_motion=include_motion)


def orientation_score(head: PairHead, va: np.ndarray, vb: np.ndarray,
                      ma: str, mb: str, i: int, j: int, k: int) -> tuple[float, float, float]:
    """Positive means current assignment is more plausible; negative favors swapping."""
    if not head.include_motion:
        va, vb = np.zeros_like(va), np.zeros_like(vb)
    pc = np.clip(head.probability(pair_vector(va, vb, ma, mb, i, j, k)), 1e-5, 1 - 1e-5)
    ps = np.clip(head.probability(pair_vector(va, vb, mb, ma, i, j, k)), 1e-5, 1 - 1e-5)
    return float(np.log(pc / (1 - pc)) - np.log(ps / (1 - ps))), float(pc), float(ps)


def _block_rows(vis: pd.DataFrame, blocks: list[list[int]]) -> list[list[pd.Series]]:
    emotion = vis[vis.category.eq("emotion")].set_index("idx")
    out = []
    for block in blocks:
        rows = [emotion.loc[i] for i in block if i in emotion.index]
        if len(rows) >= 2:
            out.append(rows)
    return out


def oracle_orientation_rows(head: PairHead, vis: pd.DataFrame, blocks: list[list[int]],
                            store: RawStore, answers: dict[str, str], fold: int,
                            regime: str) -> list[dict]:
    """Held-out, label-diagnostic orientation score (not used at inference)."""
    rows = []
    for block in _block_rows(vis, blocks):
        k = len(block)
        for i, j in combinations(range(k), 2):
            a, b = block[i], block[j]
            # ``vis`` deliberately has no answer column.  This mapping is consulted only
            # after scoring, for the diagnostic; it never enters the candidate head.
            ma = _label(a, answers[a.qa_id][0])
            mb = _label(b, answers[b.qa_id][0])
            if ma == mb or mgroup(ma) == mgroup(mb):
                continue
            va, vb = store.vectors.get(a.true_path), store.vectors.get(b.true_path)
            if va is None or vb is None:
                continue
            score, pc, ps = orientation_score(head, va, vb, ma, mb, i, j, k)
            rows.append(dict(regime=regime, fold=fold, qa_i=a.qa_id, qa_j=b.qa_id,
                             k=k, true_group_i=mgroup(ma), true_group_j=mgroup(mb),
                             score=score, p_current=pc, p_swapped=ps,
                             correct_orientation=int(score > 0)))
    return rows


def candidate_rows(head: PairHead, vis: pd.DataFrame, blocks: list[list[int]],
                   base: pd.DataFrame, store: RawStore, fold: int, regime: str) -> list[dict]:
    """Generate test-visible one-transposition proposals on top of a fixed baseline."""
    bp = dict(zip(base.qa_id, base.pred))
    rows = []
    for bi, block in enumerate(_block_rows(vis, blocks)):
        k = len(block)
        for i, j in combinations(range(k), 2):
            a, b = block[i], block[j]
            pa, pb = bp.get(a.qa_id), bp.get(b.qa_id)
            if pa not in "ABCD" or pb not in "ABCD":
                continue
            ma, mb = _label(a, pa), _label(b, pb)
            if ma == mb or mgroup(ma) == mgroup(mb):
                continue
            # A permutation must remain an option-wise valid prediction, not merely a
            # label-level suggestion.
            na, nb = _letter_for(a, mb), _letter_for(b, ma)
            if na is None or nb is None:
                continue
            va, vb = store.vectors.get(a.true_path), store.vectors.get(b.true_path)
            if va is None or vb is None:
                continue
            score, pc, ps = orientation_score(head, va, vb, ma, mb, i, j, k)
            rows.append(dict(regime=regime, fold=fold, block=bi, k=k,
                             qa_i=a.qa_id, qa_j=b.qa_id,
                             base_i=pa, base_j=pb, new_i=na, new_j=nb,
                             label_i=ma, label_j=mb,
                             group_i=mgroup(ma), group_j=mgroup(mb),
                             score=score, p_current=pc, p_swapped=ps))
    return rows


def select_swaps(cand: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pick non-overlapping pairs per inferred block, strongest swap evidence first."""
    if cand.empty:
        return cand.copy()
    keep = []
    for _, g in cand[cand.score <= -threshold].sort_values("score").groupby(["regime", "fold", "block"], sort=False):
        used: set[str] = set()
        for r in g.itertuples(index=False):
            if r.qa_i not in used and r.qa_j not in used:
                keep.append(r._asdict())
                used.update([r.qa_i, r.qa_j])
    return pd.DataFrame(keep, columns=cand.columns)


def flip_report(cand: pd.DataFrame, base: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Audit every threshold against the fixed baseline; no threshold is silently chosen."""
    b = base.set_index("qa_id")
    out = []
    for t in thresholds:
        s = select_swaps(cand, t)
        upd = {}
        for r in s.itertuples(index=False):
            upd[r.qa_i] = r.new_i
            upd[r.qa_j] = r.new_j
        changed = b.loc[list(upd)] if upd else b.iloc[0:0]
        if len(changed):
            new = np.array([upd[q] for q in changed.index])
            old_ok = changed.correct.to_numpy(int)
            new_ok = (new == changed.answer.to_numpy(str)).astype(int)
            wr = int(((old_ok == 0) & (new_ok == 1)).sum())
            rw = int(((old_ok == 1) & (new_ok == 0)).sum())
            unchanged_wrong = int(((old_ok == 0) & (new_ok == 0)).sum())
            unchanged_right = int(((old_ok == 1) & (new_ok == 1)).sum())
        else:
            wr = rw = unchanged_wrong = unchanged_right = 0
        folds_seen = []
        for f, g in (changed.assign(new=[upd[q] for q in changed.index]).groupby("fold") if len(changed) else []):
            oo = g.correct.to_numpy(int)
            nn = (g.new.to_numpy(str) == g.answer.to_numpy(str)).astype(int)
            folds_seen.append(f"{f}:{int(((oo==0)&(nn==1)).sum())}-{int(((oo==1)&(nn==0)).sum())}")
        out.append(dict(threshold=t, pairs=len(s), flips=len(upd), W_to_R=wr, R_to_W=rw,
                        precision=(wr / (wr + rw) if wr + rw else np.nan), net=wr - rw,
                        still_wrong=unchanged_wrong, still_right=unchanged_right,
                        per_fold=";".join(folds_seen)))
    return pd.DataFrame(out)


def run_regime(tr: pd.DataFrame, meta: pd.DataFrame, store: RawStore,
               base: pd.DataFrame, regime: str, include_motion: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF candidate / oracle rows in ordinary and pair-thinned held-out protocols."""
    users = sorted(tr.user.dropna().unique())
    all_oracle, all_cand = [], []
    for fi, hold in enumerate(folds(users, 5)):
        train = tr[~tr.user.isin(hold)]
        head = fit_pair_head(train, store, include_motion=include_motion)
        evaluation = thin_to_pairs(tr, hold, 0.38, seed=fi) if regime == "pair_thinned" else tr
        vis, key, _ = make_pseudo(evaluation, meta, hold, seed=100 + fi)
        blocks = infer_blocks(vis, fit_group_model(train))
        b = base[base.fold.eq(fi)].copy()
        answers = dict(zip(key.qa_id, key.answer))
        all_oracle.extend(oracle_orientation_rows(head, vis, blocks, store, answers, fi, regime))
        all_cand.extend(candidate_rows(head, vis, blocks, b, store, fi, regime))
        print(f"  {regime} fold {fi}: {len(blocks)} inferred blocks", flush=True)
    return pd.DataFrame(all_oracle), pd.DataFrame(all_cand)


def test_candidates(tr: pd.DataFrame, te: pd.DataFrame, meta: pd.DataFrame,
                    store: RawStore, head: PairHead) -> pd.DataFrame:
    """Score only test-visible repaired blocks; produces a ledger, never a submission."""
    vis = real_test_view(te)
    by_idx = vis[vis.source.eq("HAU")].groupby("idx").first()
    with open(os.path.join(ROOT, "champ", "test_blocks_repaired.json")) as f:
        ids = json.load(f)
    blocks = [[int(i) for i in b if int(i) in by_idx.index] for b in ids]
    # Candidate strength is assessed against both the submitted champion and the staged
    # S/T/W/X correction layer, because T can legitimately alter an emotion baseline.
    out = []
    for name in ("submission_092105_SUBMITTED.csv", "submission_corrlayer_S_W_T.csv"):
        p = os.path.join(ROOT, name)
        if not os.path.exists(p):
            continue
        base = pd.read_csv(p).rename(columns={"prediction": "pred"})
        out.append(pd.DataFrame(candidate_rows(head, vis, blocks, base, store,
                                                fold=-1, regime=name)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    tr, te, meta = load_all()
    audit = pd.read_csv(os.path.join(ROOT, "champ", "audit_champ.csv"))
    audit_e = audit[(audit.source.eq("HAU")) & (audit.category.eq("emotion"))].copy()
    assert len(audit_e) == 809, len(audit_e)
    store = build_descriptors(meta)
    print("raw coverage", store.coverage[["has_skeleton", "has_dino"]].sum().to_dict(), flush=True)

    ordinary_o, ordinary_c = run_regime(tr, meta, store, audit_e, "ordinary")
    stress_base = pd.read_csv(os.path.join(ROOT, "champ", "oof_pairstress.csv"))
    stress_e = stress_base[(stress_base.source.eq("HAU")) & (stress_base.category.eq("emotion"))].copy()
    stress_o, stress_c = run_regime(tr, meta, store, stress_e, "pair_thinned")
    # Matched prior-only control.  It is a key falsification check because group/slot
    # priors alone can make pair orientation look deceptively easy.
    prior_ordinary_o, prior_ordinary_c = run_regime(
        tr, meta, store, audit_e, "ordinary", include_motion=False)
    prior_stress_o, prior_stress_c = run_regime(
        tr, meta, store, stress_e, "pair_thinned", include_motion=False)
    oracle = pd.concat([ordinary_o, stress_o], ignore_index=True)
    cand = pd.concat([ordinary_c, stress_c], ignore_index=True)
    oracle.to_csv(os.path.join(OUT, "orientation_oracle_oof.csv"), index=False)
    cand.to_csv(os.path.join(OUT, "orientation_candidates_oof.csv"), index=False)

    o_summary = (oracle.groupby(["regime", "fold", "k"], dropna=False)
                 .correct_orientation.agg(n="size", correct="sum", accuracy="mean").reset_index())
    o_summary.to_csv(os.path.join(OUT, "orientation_oracle_summary.csv"), index=False)
    reports = []
    for regime, c in cand.groupby("regime", sort=False):
        baseline = audit_e if regime == "ordinary" else stress_e
        r = flip_report(c, baseline, thresholds=[0.0, 0.5, 1.0, 1.5, 2.0])
        r.insert(0, "regime", regime)
        reports.append(r)
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(os.path.join(OUT, "flip_audit_by_threshold.csv"), index=False)

    prior_oracle = pd.concat([prior_ordinary_o, prior_stress_o], ignore_index=True)
    prior_cand = pd.concat([prior_ordinary_c, prior_stress_c], ignore_index=True)
    prior_oracle.to_csv(os.path.join(OUT, "orientation_prior_control_oof.csv"), index=False)
    prior_cand.to_csv(os.path.join(OUT, "orientation_prior_control_candidates_oof.csv"), index=False)
    prior_summary = (prior_oracle.groupby(["regime", "fold", "k"], dropna=False)
                     .correct_orientation.agg(n="size", correct="sum", accuracy="mean").reset_index())
    prior_summary.to_csv(os.path.join(OUT, "orientation_prior_control_summary.csv"), index=False)

    head_all = fit_pair_head(tr, store)
    tc = test_candidates(tr, te, meta, store, head_all)
    if not tc.empty:
        tc.to_csv(os.path.join(OUT, "test_pair_score_ledger.csv"), index=False)
    summary = {
        "experiment": "EXP-EMO-RELRAW-001",
        "descriptor": "raw body-centred skeleton temporal profiles + DINO depth-embedding delta trajectories",
        "no_modalities": ["IMU", "radar", "DTW"],
        "baseline": "champ/audit_champ.csv (immutable submitted-champion OOF snapshot)",
        "ordinary_orientation_accuracy": float(ordinary_o.correct_orientation.mean()) if len(ordinary_o) else None,
        "pair_thinned_orientation_accuracy": float(stress_o.correct_orientation.mean()) if len(stress_o) else None,
        "ordinary_prior_only_orientation_accuracy": float(prior_ordinary_o.correct_orientation.mean()) if len(prior_ordinary_o) else None,
        "pair_thinned_prior_only_orientation_accuracy": float(prior_stress_o.correct_orientation.mean()) if len(prior_stress_o) else None,
        "ordinary_candidate_pairs": int(len(ordinary_c)),
        "pair_thinned_candidate_pairs": int(len(stress_c)),
        "test_scored_pairs": int(len(tc)),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("\n=== raw relative orientation summary ===")
    print(o_summary.to_string(index=False))
    print("\n=== direct flip audit versus fixed baselines ===")
    print(report.to_string(index=False))
    if not tc.empty:
        print("\n=== test candidates are scored only; not applied ===")
        print(tc.groupby("regime").score.agg(["count", "min", "median", "max"]).to_string())


if __name__ == "__main__":
    main()
