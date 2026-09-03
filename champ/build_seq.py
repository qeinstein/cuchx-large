"""Cache per-unit skeleton sequences (T x 17 x 3) keyed by global frame index, and build
dense frame-level action labels for HAU sessions from the HARn segment frame ranges.
"""
import os, re, json, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FN = re.compile(r'Color_.*_(\d{8})\.json$')


def seq(d):
    p = os.path.join(d, 'Skeleton', 'predictions')
    if not os.path.isdir(p):
        return None, None
    items = sorted((int(m.group(1)), f) for f in os.listdir(p) for m in [FN.match(f)] if m)
    K, F = [], []
    for fr, f in items:
        try:
            j = json.load(open(os.path.join(p, f)))
        except Exception:
            continue
        if isinstance(j, list):
            if not j:
                continue
            j = max(j, key=lambda o: float(np.mean(o.get('keypoint_scores') or [0])))
        kp = j.get('keypoints')
        if kp is None:
            continue
        a = np.asarray(kp, float).reshape(-1, 3)
        if a.shape[0] < 17:
            continue
        K.append(a[:17]); F.append(fr)
    if len(K) < 4:
        return None, None
    return np.stack(K).astype(np.float32), np.array(F, np.int32)


def one(d):
    K, F = seq(d)
    return d, K, F


if __name__ == '__main__':
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    dirs = meta.unit_dir.tolist()
    store = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (d, K, F) in enumerate(ex.map(one, dirs, chunksize=8)):
            if K is not None:
                store[d + '|K'] = K
                store[d + '|F'] = F
            if (i + 1) % 500 == 0:
                print(' ', i + 1, flush=True)
    out = os.path.join(ROOT, 'champ', 'skel_seq.npz')
    np.savez_compressed(out, **store)
    print('units cached:', len(store) // 2, '->', out)
