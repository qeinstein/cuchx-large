"""Pairwise DTW between the trials of one HAU session, for manner assignment.

Why this and not more per-clip statistics: the three clips of a session are the SAME subject
performing the SAME script three times, differing only in manner.  The difference between two
of them is therefore almost exactly a time warp, and the shape of the optimal warping path is
a direct, content-free measurement of relative tempo -- including where one trial pauses and
the other does not.  Global summary statistics (what the champion's pairwise model consumes)
throw that structure away; a measured 269-column radar/IMU expansion did not improve manner-
group accuracy at all (0.5958 -> 0.6007), which says the missing information is relational.
"""
import os, sys, json, itertools, collections
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_Z = None


def skel(unit):
    global _Z
    if _Z is None:
        _Z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    k = unit + '|K'
    return _Z[k] if k in _Z else None


def pose_seq(K, maxT=420):
    """Translation/scale-invariant pose descriptor per frame, plus speed."""
    hip = K[:, [11, 12]].mean(1, keepdims=True)
    torso = (np.linalg.norm(K[:, 5] - K[:, 11], axis=-1)
             + np.linalg.norm(K[:, 6] - K[:, 12], axis=-1))
    s = np.median(torso[torso > 1e-3]) if np.any(torso > 1e-3) else 1.0
    P = (K - hip) / max(s, 1e-3)
    X = P.reshape(len(K), -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ratio = 1.0
    if len(X) > maxT:                     # subsample but record the ratio so tempo survives
        idx = np.linspace(0, len(X) - 1, maxT).astype(int)
        ratio = len(X) / maxT
        X = X[idx]
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n, ratio


def dtw_path(A, B, band=0.35):
    """Banded DTW; returns the warping path as arrays (ia, ib) and the normalised cost."""
    n, m = len(A), len(B)
    w = max(int(band * max(n, m)), abs(n - m) + 2)
    C = 1.0 - A @ B.T                                   # cosine distance
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    ptr = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        lo = max(1, int(i * m / n) - w); hi = min(m, int(i * m / n) + w)
        c = C[i - 1, lo - 1:hi]
        d0 = D[i - 1, lo - 1:hi]                        # diagonal
        d1 = D[i - 1, lo:hi + 1]                        # up
        prev = D[i, lo - 1:hi]                          # left (filled progressively)
        row = np.full(hi - lo + 1, INF)
        pt = np.zeros(hi - lo + 1, np.int8)
        run = INF
        for t in range(hi - lo + 1):
            cand = [d0[t] if t < len(d0) else INF,
                    d1[t] if t < len(d1) else INF,
                    run]
            k = int(np.argmin(cand))
            row[t] = cand[k] + c[t]
            pt[t] = k
            run = row[t]
        D[i, lo:hi + 1] = row
        ptr[i, lo:hi + 1] = pt
    # backtrack
    i, j = n, m
    ia = []; ib = []
    while i > 0 and j > 0:
        ia.append(i - 1); ib.append(j - 1)
        k = ptr[i, j]
        if k == 0: i -= 1; j -= 1
        elif k == 1: j -= 1
        else: i -= 1
    ia = np.array(ia[::-1]); ib = np.array(ib[::-1])
    cost = float(D[n, m] / max(len(ia), 1))
    return ia, ib, cost


def path_feats(A, B, ra, rb):
    ia, ib, cost = dtw_path(A, B)
    n, m = len(A), len(B)
    f = {}
    f['dtw_cost'] = cost
    f['dtw_lenratio'] = (n * ra) / max(m * rb, 1e-9)
    f['dtw_logratio'] = float(np.log(max((n * ra) / max(m * rb, 1e-9), 1e-9)))
    di = np.diff(ia); dj = np.diff(ib)
    if len(di) == 0:
        return f
    f['dtw_horiz'] = float((di == 0).mean())            # B advances, A waits -> A slower here
    f['dtw_vert'] = float((dj == 0).mean())
    f['dtw_diag'] = float(((di > 0) & (dj > 0)).mean())
    f['dtw_hv'] = f['dtw_horiz'] - f['dtw_vert']
    # deviation of the path from the straight diagonal
    dev = ib / max(m - 1, 1) - ia / max(n - 1, 1)
    f['dtw_dev_mean'] = float(dev.mean()); f['dtw_dev_std'] = float(dev.std())
    f['dtw_dev_max'] = float(dev.max()); f['dtw_dev_min'] = float(dev.min())
    f['dtw_dev_absmean'] = float(np.abs(dev).mean())
    # local slope in normalised time, and its dispersion (irregular tempo = hesitation/jitter)
    sl = (dj / max(m - 1, 1)) / (di / max(n - 1, 1) + 1e-6)
    sl = sl[np.isfinite(sl)]
    if len(sl):
        f['dtw_slope_med'] = float(np.median(sl)); f['dtw_slope_iqr'] = float(
            np.percentile(sl, 75) - np.percentile(sl, 25))
        f['dtw_slope_std'] = float(np.std(np.log1p(sl)))
    # longest wait runs = pauses
    def maxrun(x):
        best = c = 0
        for v in x:
            c = c + 1 if v == 0 else 0
            best = max(best, c)
        return best
    f['dtw_run_h'] = maxrun(di) / max(len(di), 1)
    f['dtw_run_v'] = maxrun(dj) / max(len(dj), 1)
    f['dtw_run_hv'] = f['dtw_run_h'] - f['dtw_run_v']
    # residual pose disagreement after warping: same script executed differently
    r = 1.0 - (A[ia] * B[ib]).sum(1)
    f['dtw_res_mean'] = float(r.mean()); f['dtw_res_p90'] = float(np.percentile(r, 90))
    f['dtw_res_std'] = float(r.std())
    return f


def block_pairs(units):
    seqs = {}
    for u in units:
        K = skel(u)
        seqs[u] = pose_seq(K) if K is not None and len(K) > 8 else None
    out = {}
    for a, b in itertools.permutations(range(len(units)), 2):
        ua, ub = units[a], units[b]
        if seqs[ua] is None or seqs[ub] is None:
            out[(a, b)] = {}
            continue
        A, ra = seqs[ua]; B, rb = seqs[ub]
        try:
            out[(a, b)] = path_feats(A, B, ra, rb)
        except Exception:
            out[(a, b)] = {}
    return out


def _job(arg):
    key, units = arg
    return key, block_pairs(units)


if __name__ == '__main__':
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    sys.path.insert(0, os.path.join(ROOT, 'champ'))
    from core import load_all, real_test_view, fit_group_model, infer_blocks
    tr, te, _ = load_all()
    p2u = dict(zip(meta.qa_path, meta.unit_dir))
    jobs = []
    hau = tr[tr.source == 'HAU']
    for (u, a, b), g in hau.groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        units = [p2u.get(p) for p in paths]
        if any(x is None for x in units): continue
        jobs.append((f'train|{u}|{a}|{b}|' + ','.join(paths), units))
    vis = real_test_view(te)
    blocks = infer_blocks(vis, fit_group_model(tr))
    i2c = dict(zip(vis.idx, vis.true_path))
    for blk in blocks:
        clips = [i2c[i] for i in blk]
        units = [p2u.get(c) for c in clips]
        if any(x is None for x in units): continue
        jobs.append((f'test|' + ','.join(str(i) for i in blk), units))
    print('%d blocks (%d train, %d test)' % (len(jobs), sum(1 for j in jobs if j[0].startswith('train')),
                                             sum(1 for j in jobs if j[0].startswith('test'))), flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n, (k, v) in enumerate(ex.map(_job, jobs, chunksize=2)):
            res[k] = v
            if (n + 1) % 40 == 0: print('  %d/%d' % (n + 1, len(jobs)), flush=True)
    rows = []
    for k, v in res.items():
        for (a, b), f in v.items():
            rows.append(dict(key=k, i=a, j=b, **f))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'emolab', 'dtw_pairs.csv'), index=False)
    print('wrote', df.shape)
