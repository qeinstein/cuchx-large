"""Build the clip metadata substrate: skeleton frame ids + absolute timestamps for every
train HAU session, train HARn segment, and test clip.  Cached to champ/meta.parquet.

Everything here is derivable from files on disk with no reference to QA answers.
"""
import os, re, json, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMT = os.path.join(ROOT, 'hf_data_manual', 'LMT_(IMU,Radar,Skeleton)')
VIS = os.path.join(ROOT, 'hf_data_manual')

FN = re.compile(r'Color_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.(\d{3})_(\d{8})\.json$')


def skel_span(unit_dir):
    p = os.path.join(unit_dir, 'Skeleton', 'predictions')
    if not os.path.isdir(p):
        return None
    fr, ts = [], []
    for f in os.listdir(p):
        m = FN.match(f)
        if not m:
            continue
        g = m.groups()
        day = int(g[2]) + 31 * int(g[1])          # monotone within the release
        sec = day * 86400 + int(g[3]) * 3600 + int(g[4]) * 60 + int(g[5]) + int(g[6]) / 1000
        fr.append(int(g[7])); ts.append(sec)
    if not fr:
        return None
    fr = np.array(fr); ts = np.array(ts)
    o = np.argsort(fr)
    return dict(f0=int(fr[o][0]), f1=int(fr[o][-1]), nf=len(fr),
                t0=float(ts.min()), t1=float(ts.max()),
                fps=float((len(fr) - 1) / max(1e-6, ts.max() - ts.min())))


def scan():
    rows = []
    # ---- train HAU sessions
    base = os.path.join(LMT, 'Training', 'HAU')
    for u in sorted(os.listdir(base)):
        if not u.startswith('user'):
            continue
        for t in sorted(os.listdir(os.path.join(base, u))):
            d = os.path.join(base, u, t)
            s = skel_span(d)
            rows.append(dict(kind='train_hau', qa_path=f'HAU/{u}/{t}', user=u, trial=t,
                             action=None, unit_dir=d, **(s or {})))
    # ---- train HARn segments
    base = os.path.join(LMT, 'Training', 'HARn')
    for a in sorted(os.listdir(base)):
        pa = os.path.join(base, a)
        if not os.path.isdir(pa):
            continue
        for u in sorted(os.listdir(pa)):
            for t in sorted(os.listdir(os.path.join(pa, u))):
                d = os.path.join(pa, u, t)
                s = skel_span(d)
                rows.append(dict(kind='train_harn', qa_path=f'HARn/{a}/{u}/{t}', user=u, trial=t,
                                 action=a, unit_dir=d, **(s or {})))
    # ---- test clips
    base = os.path.join(LMT, 'Testing', 'large_model_track_test')
    for c in sorted(os.listdir(base)):
        d = os.path.join(base, c)
        if not os.path.isdir(d):
            continue
        s = skel_span(d)
        rows.append(dict(kind='test', qa_path=c, user=None, trial=None, action=None,
                         unit_dir=d, **(s or {})))
    df = pd.DataFrame(rows)
    for c in ['f0', 'f1', 'nf', 't0', 't1', 'fps']:
        if c not in df:
            df[c] = np.nan
    return df


if __name__ == '__main__':
    df = scan()
    out = os.path.join(ROOT, 'champ', 'meta.csv')
    df.to_csv(out, index=False)
    print(df.kind.value_counts().to_string())
    print('missing skeleton:', df.f0.isna().sum(), 'of', len(df))
    print('median fps: %.2f' % df.fps.median())
    print('written', out)
