"""Path-keyed cache of pairwise DTW warping-path features, for champ/emopair.py.

Covers every ordered pair of clips inside a true training session and inside every inferred
(and repaired) test block, so the same cache serves fitting, the pseudo-test and the real test.
"""
import os, sys, itertools, collections
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
sys.path.insert(0, os.path.join(ROOT, 'emolab'))
from dtw_feats import skel, pose_seq, path_feats


def _job(pair):
    pa, pb, ua, ub = pair
    Ka, Kb = skel(ua), skel(ub)
    if Ka is None or Kb is None or len(Ka) < 9 or len(Kb) < 9:
        return (pa, pb, {})
    A, ra = pose_seq(Ka); B, rb = pose_seq(Kb)
    try:
        return (pa, pb, path_feats(A, B, ra, rb))
    except Exception:
        return (pa, pb, {})


def main():
    from core import load_all, real_test_view, fit_group_model, infer_blocks
    import repair as RP, re
    tr, te, meta = load_all()
    p2u = dict(zip(meta.qa_path, meta.unit_dir))
    want = set()
    hau = tr[tr.source == 'HAU']
    for (u, a, b), g in hau.groupby(['user', 'aa', 'bb']):
        paths = sorted(g.path.unique())
        for x, y in itertools.permutations(paths, 2):
            want.add((x, y))
    gm = fit_group_model(tr)
    vis = real_test_view(te)
    blocks = infer_blocks(vis, gm)
    gmod = RP.fit_gap_model(tr, meta)
    m = meta[meta.kind == 'test'].copy()
    m['idx'] = [int(re.search(r'LM_test_(\d+)', p).group(1)) for p in m.qa_path]
    t0 = dict(zip(m.idx, m.t0))
    rep, _ = RP.repair(blocks, vis, gm, gmod, t0)
    i2c = dict(zip(vis.idx, vis.true_path))
    for bl in list(blocks) + list(rep):
        cs = [i2c[int(i)] for i in bl]
        for x, y in itertools.permutations(cs, 2):
            want.add((x, y))
    want = [(x, y, p2u.get(x), p2u.get(y)) for x, y in sorted(want)]
    want = [w for w in want if w[2] and w[3]]
    print('%d ordered clip pairs' % len(want), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for n, (pa, pb, f) in enumerate(ex.map(_job, want, chunksize=4)):
            rows.append(dict(pa=pa, pb=pb, **f))
            if (n + 1) % 200 == 0: print('  %d/%d' % (n + 1, len(want)), flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, 'champ', 'dtw_pairs_bypath.csv')
    df.to_csv(out, index=False)
    print('wrote', out, df.shape)


if __name__ == '__main__':
    main()
