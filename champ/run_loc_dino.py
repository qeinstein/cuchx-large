"""Candidate-conditioned localizer on frozen DINOv2 depth features, alone and fused with
skeleton+IMU.  Reports sequence accuracy AND complementarity against the champion decoder.

MODE=dino   -> DINOv2 depth only
MODE=fused  -> DINOv2 depth + skeleton + IMU  (default)
"""
import os, sys, json, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D, loc as L, seqpair as SP
from core import load_all, opts
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get('MODE', 'fused')
_DN = None


def dino(qa_path, T):
    global _DN
    if _DN is None:
        _DN = np.load(os.path.join(ROOT, 'champ', 'dino_frames.npz'))
    if qa_path not in _DN:
        return None
    A = _DN[qa_path].astype(np.float32)
    if len(A) != T:
        idx = np.clip((np.arange(T) * len(A)) // max(1, T), 0, len(A) - 1)
        A = A[idx]
    return A


def build(meta, tr):
    items = L.build(meta, tr)
    keep = []
    for it in items:
        T = it['M'].shape[1]
        A = dino(it['qa_path'], T)
        if A is None:
            continue
        w = min(9, max(1, (T // 2) * 2 - 1))
        k = np.ones(w) / w
        sm = np.stack([np.convolve(A[:, j], k, mode='same')[:T] for j in range(A.shape[1])], 1)
        dv = np.abs(np.diff(A, axis=0, prepend=A[:1]))
        parts = [A, sm, dv]
        if MODE == 'fused':
            parts.append(it['X'])           # existing skeleton + IMU frame features
        it['X'] = np.concatenate(parts, 1).astype(np.float32)
        keep.append(it)
    return keep


def main(epochs=60, nfold=5):
    tr, te, meta = load_all()
    items = build(meta, tr)
    print(f'MODE={MODE}  items {len(items)}  dim {items[0]["X"].shape[1]}', flush=True)
    users = sorted({it['user'] for it in items})
    seq = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
    seqby = {r.path: r for _, r in seq.iterrows()}
    store = {}
    ex = [0, 0]; pw = [0, 0]
    for fi, hold in enumerate(folds(users, nfold)):
        trn = [it for it in items if it['user'] not in hold]
        tst = [it for it in items if it['user'] in hold]
        m = L.train(trn, epochs=epochs, seed=fi)
        pr = L.predict(m, [dict(qa_path=i['qa_path'], X=i['X'], F=i['F']) for i in tst])
        for it in tst:
            S = pr[it['qa_path']]
            store['oof|' + it['qa_path']] = S
            r = seqby.get(it['qa_path'])
            if r is None:
                continue
            oo = opts(r)
            if not all(o in L.OPT2CLS for o in oo):
                continue
            cls4 = [L.OPT2CLS[o] for o in oo]
            T = S.shape[1]
            cen = [float((S[c] / (S[c].sum() + 1e-9) * np.arange(T)).sum()) for c in cls4]
            order = sorted(range(4), key=lambda i: cen[i])
            ex[0] += int(''.join('ABCD'[i] for i in order) == str(r['answer'])); ex[1] += 1
            tpos = {Lb: i for i, Lb in enumerate(str(r['answer']))}
            for a, b in itertools.combinations(range(4), 2):
                truth = tpos['ABCD'[a]] < tpos['ABCD'[b]]
                pw[0] += int((cen[a] < cen[b]) == truth); pw[1] += 1
        print(f'  fold {fi} done', flush=True)
    print(f'\n  {MODE} sequence exact (centroid): {ex[0]}/{ex[1]} = {ex[0]/max(1,ex[1]):.4f}')
    print(f'  {MODE} pairwise                 : {pw[0]}/{pw[1]} = {pw[0]/max(1,pw[1]):.4f}')
    print('  reference: skel+IMU loc 156/305 0.8464 | depth-stats 102/305 0.7667 | '
          'champion decoder 176/308 | true-onset oracle 0.9883 pairwise')
    out = os.path.join(ROOT, 'champ', f'loc_{MODE}_maps.npz')
    np.savez_compressed(out, **store)
    print('cached', len(store), 'OOF maps ->', out)


if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
