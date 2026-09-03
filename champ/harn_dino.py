"""HARn clip action identity from frozen DINOv2 depth features.

Training clips need no extra encoding: every HARn segment is an exact sub-interval of a
parent HAU clip on a shared global frame axis, so its features are a slice of the parent's
(index = global_frame - parent_f0).  Test HARn clips are encoded directly.

Baseline to beat: the skeleton+IMU sequence classifier at 52.5% top-1 (harn_clf.py), which
in turn replaced 21.9% from clip-level aggregate features.
"""
import os, sys, json
import numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D, harn as H
from core import load_all
from pseudotest import folds
import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DN = None


def dn():
    global _DN
    if _DN is None:
        _DN = np.load(os.path.join(ROOT, 'champ', 'dino_frames.npz'))
    return _DN


def pool(A):
    """Temporal pooling of a (T, 384) feature sequence -> a fixed descriptor.
    Thirds preserve coarse temporal structure, which matters for actions defined by a
    beginning/end (sit down vs stand up)."""
    if len(A) == 0:
        return None
    A = A.astype(np.float32)
    n = len(A)
    th = [A[:max(1, n // 3)], A[max(1, n // 3):max(2, 2 * n // 3)], A[max(2, 2 * n // 3):]]
    dv = np.abs(np.diff(A, axis=0)) if n > 1 else np.zeros((1, A.shape[1]), np.float32)
    return np.concatenate([A.mean(0), A.max(0), A.std(0),
                           th[0].mean(0), th[1].mean(0), th[2].mean(0),
                           dv.mean(0), dv.max(0)])


def clip_descriptor(qa_path, meta_idx, nest, kind):
    """DINOv2 descriptor for a HARn clip, sliced from its parent when it is a train clip."""
    Z = dn()
    if qa_path in Z:                                   # test HARn clips are encoded directly
        return pool(Z[qa_path])
    par = nest.get(qa_path)
    if par is None or par not in Z:
        return None
    try:
        cf0, cf1 = meta_idx.loc[qa_path, 'f0'], meta_idx.loc[qa_path, 'f1']
        pf0 = meta_idx.loc[par, 'f0']
    except KeyError:
        return None
    if not np.isfinite(cf0) or not np.isfinite(pf0):
        return None
    A = Z[par]
    i0 = int(max(0, cf0 - pf0)); i1 = int(min(len(A), cf1 - pf0 + 1))
    if i1 - i0 < 2:
        return None
    return pool(A[i0:i1])


def build(meta, tr):
    P.caches(meta)
    mi = meta.set_index('qa_path')
    nest = P._C['nest']
    rows, y, users, paths = [], [], [], []
    for r in meta[meta.kind == 'train_harn'].itertuples():
        d = clip_descriptor(r.qa_path, mi, nest, 'train_harn')
        if d is None:
            continue
        rows.append(d); y.append(r.action); users.append(r.user); paths.append(r.qa_path)
    return np.stack(rows), np.array(y), np.array(users), np.array(paths)


def main():
    tr, te, meta = load_all()
    X, y, u, paths = build(meta, tr)
    print('HARn train descriptors:', X.shape, 'classes', len(set(y)))
    classes = sorted(set(y))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    allu = sorted(set(u))
    store = {}
    c = n = 0
    for fi, hold in enumerate(folds(allu, 5)):
        m = ~np.isin(u, hold)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1))
        clf.fit(X[m], y[m])
        lp = clf.predict_log_proba(X[~m])
        cl = list(clf.classes_)
        for p, row, t in zip(paths[~m], lp, y[~m]):
            full = np.full(len(classes), -30.0, np.float32)
            for j, cn in enumerate(cl):
                full[classes.index(cn)] = row[j]
            store['oof|' + p] = full
            c += int(classes[int(np.argmax(full))] == t); n += 1
        print(f'  fold {fi} cumulative top-1 {c/n:.4f}', flush=True)
    print(f'\n  DINOv2 HARn action top-1: {c}/{n} = {c/n:.4f}')
    print('  baseline skeleton+IMU sequence classifier: 1530/2916 = 0.5247')
    # full model -> test HARn clips
    mi = meta.set_index('qa_path'); nest = P._C['nest']
    tp = [r.qa_path for r in meta[meta.kind == 'test'].itertuples()
          if int(r.qa_path.split('_')[-1]) < 65]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1))
    clf.fit(X, y)
    cl = list(clf.classes_)
    ok = 0
    for p in tp:
        d = clip_descriptor(p, mi, nest, 'test')
        if d is None:
            continue
        row = clf.predict_log_proba(d[None])[0]
        full = np.full(len(classes), -30.0, np.float32)
        for j, cn in enumerate(cl):
            full[classes.index(cn)] = row[j]
        store['test|' + p] = full; ok += 1
    np.savez_compressed(os.path.join(ROOT, 'champ', 'harn_dino.npz'),
                        classes=np.array(classes), **store)
    print(f'  test HARn clips encoded: {ok} of {len(tp)}')
    print('  cached', len(store))


if __name__ == '__main__':
    main()
