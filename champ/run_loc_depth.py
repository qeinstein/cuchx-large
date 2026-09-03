"""Information-audit probe: train the SAME localizer on Depth+IR frame features instead of
skeleton+IMU, and measure whether it carries temporal signal that skeleton+IMU does not.

Deliberately cheap: 34 hand-made statistics per frame, no backbone, no pretrained weights.
If this shows incremental signal on the sequence residual pairs, a real video model is worth
building.  If it shows none, the depth stream is not the missing information source.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D, loc as L
from core import load_all, opts
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DF = np.load(os.path.join(ROOT, 'champ', 'depth_frames.npz'))


def depth_feats(qa_path, T):
    """Depth/IR per-frame statistics, resampled to the skeleton frame count."""
    if qa_path not in _DF:
        return np.zeros((T, 34), np.float32)
    A = _DF[qa_path]
    if len(A) == T:
        return A
    idx = np.clip((np.arange(T) * len(A)) // max(1, T), 0, len(A) - 1)
    return A[idx]


def build_depth(meta, tr):
    """Same targets as loc.build, but the inputs are Depth/IR statistics only."""
    items = L.build(meta, tr)
    keep = []
    for it in items:
        T = it['M'].shape[1]
        X = depth_feats(it['qa_path'], T)
        if np.abs(X).sum() == 0:
            continue
        # add short-window temporal context so a 34-dim frame is not seen in isolation
        k = np.ones(9) / 9.0
        sm = np.stack([np.convolve(X[:, j], k, mode='same') for j in range(X.shape[1])], 1)
        dv = np.abs(np.diff(X, axis=0, prepend=X[:1]))
        it['X'] = np.concatenate([X, sm, dv], 1).astype(np.float32)
        keep.append(it)
    return keep


def main(epochs=60, nfold=5):
    tr, te, meta = load_all()
    items = build_depth(meta, tr)
    print('depth-localizer items', len(items), 'dim', items[0]['X'].shape[1])
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
            cen = []
            for c in cls4:
                w = S[c] / (S[c].sum() + 1e-9)
                cen.append(float((w * np.arange(T)).sum()))
            order = sorted(range(4), key=lambda i: cen[i])
            ex[0] += int(''.join('ABCD'[i] for i in order) == str(r['answer'])); ex[1] += 1
            tpos = {Lb: i for i, Lb in enumerate(str(r['answer']))}
            for a, b in itertools.combinations(range(4), 2):
                truth = tpos['ABCD'[a]] < tpos['ABCD'[b]]
                pw[0] += int((cen[a] < cen[b]) == truth); pw[1] += 1
        print(f'  fold {fi} done', flush=True)
    print(f'\n  DEPTH-only sequence exact (centroid): {ex[0]}/{ex[1]} = {ex[0]/max(1,ex[1]):.4f}')
    print(f'  DEPTH-only pairwise                : {pw[0]}/{pw[1]} = {pw[0]/max(1,pw[1]):.4f}')
    print(f'  skeleton+IMU localizer  : 156/305 exact, 0.8464 pairwise')
    print(f'  checkpoint decoder      : 176/308 exact, 0.8770 pairwise (dense logits)')
    np.savez_compressed(os.path.join(ROOT, 'champ', 'loc_depth_maps.npz'), **store)
    print('cached', len(store), 'OOF depth maps')
    if os.environ.get('CHAMP_SKIP_TEST'):
        return
    # test-side maps for later fusion, if the probe justifies it
    ti = [dict(qa_path=r.qa_path,
               X=None, F=np.arange(len(_DF[r.qa_path])) if r.qa_path in _DF else None)
          for r in meta[meta.kind == 'test'].itertuples() if r.qa_path in _DF]
    ti = [it for it in ti if len(_DF[it['qa_path']]) >= 3]
    for it in ti:
        A = _DF[it['qa_path']]
        w = min(9, max(1, len(A) // 2 * 2 - 1))
        k = np.ones(w) / w
        sm = np.stack([np.convolve(A[:, j], k, mode='same')[:len(A)]
                       for j in range(A.shape[1])], 1)
        dv = np.abs(np.diff(A, axis=0, prepend=A[:1]))
        it['X'] = np.concatenate([A, sm, dv], 1).astype(np.float32)
        it['F'] = np.arange(len(A))
    m = L.train(items, epochs=epochs, seed=99)
    pr = L.predict(m, ti)
    for k_, v in pr.items():
        store['test|' + k_] = v
    np.savez_compressed(os.path.join(ROOT, 'champ', 'loc_depth_maps.npz'), **store)
    print('cached', len(store), 'depth maps')


if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
