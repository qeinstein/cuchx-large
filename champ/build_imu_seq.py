"""Resample the 5 wearable IMUs onto each unit's skeleton frame timestamps.

Output champ/imu_seq.npz: per unit a T x 30 array (5 devices x [|acc|, ax,ay,az -> |gyr|, gx..]).
"""
import os, re, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FN = re.compile(r'Color_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.(\d{3})_(\d{8})\.json$')
FN_SHORT = re.compile(r'Color_(\d{8})\.json$')
DEVS = ['WTC', 'WTLA', 'WTRA', 'WTLL', 'WTRL']


def frame_times(d):
    p = os.path.join(d, 'Skeleton', 'predictions')
    if not os.path.isdir(p):
        return None, None
    fr, ts = [], []
    for f in os.listdir(p):
        m = FN.match(f)
        if m:
            g = m.groups()
            day = int(g[2]) + 31 * int(g[1])
            ts.append(day * 86400 + int(g[3]) * 3600 + int(g[4]) * 60 + int(g[5]) + int(g[6]) / 1000)
            fr.append(int(g[7]))
            continue
        m = FN_SHORT.match(f)
        if m:
            fr.append(int(m.group(1))); ts.append(np.nan)
    if not fr:
        return None, None
    o = np.argsort(fr)
    return np.array(fr)[o], np.array(ts, float)[o]


def one(d):
    fr, ts = frame_times(d)
    if fr is None:
        return d, None
    p = os.path.join(d, 'IMU')
    out = np.zeros((len(fr), len(DEVS) * 6), np.float32)
    if not os.path.isdir(p):
        return d, out
    if ts is None or not np.isfinite(np.asarray(ts, float)).any():
        # short-filename units have frame ids but no wall-clock: IMU cannot be
        # resampled onto the skeleton axis -> honest zeros (documented)
        return d, out
    for f in sorted(os.listdir(p)):
        if not f.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(os.path.join(p, f), low_memory=False)
        except Exception:
            continue
        cols = list(df.columns)
        tc = cols[0]; dc = cols[1]
        acc = [c for c in cols if '加速度' in c or c.lower().startswith('acc')][:3]
        gyr = [c for c in cols if '角速度' in c or c.lower().startswith('as') or c.lower().startswith('gyr')][:3]
        if len(acc) < 3 or len(gyr) < 3:
            continue
        t = pd.to_datetime(df[tc], errors='coerce')
        if t.isna().all():
            continue
        # seconds on the same synthetic clock as frame_times
        tsec = (t.dt.day + 31 * t.dt.month) * 86400 + t.dt.hour * 3600 \
               + t.dt.minute * 60 + t.dt.second + t.dt.microsecond / 1e6
        tsec = tsec.to_numpy(float)
        name = df[dc].astype(str).str.split('(').str[0].str.strip()
        A = df[acc].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        G = df[gyr].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        for di, dv in enumerate(DEVS):
            m = (name == dv).to_numpy() & np.isfinite(tsec)
            if m.sum() < 3:
                continue
            o = np.argsort(tsec[m])
            tt = tsec[m][o]
            for j in range(3):
                out[:, di * 6 + j] = np.interp(ts, tt, np.nan_to_num(A[m][o][:, j]))
                out[:, di * 6 + 3 + j] = np.interp(ts, tt, np.nan_to_num(G[m][o][:, j]))
    return d, np.nan_to_num(out)


if __name__ == '__main__':
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    dirs = meta.unit_dir.tolist()
    store = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (d, X) in enumerate(ex.map(one, dirs, chunksize=8)):
            if X is not None:
                store[d] = X
            if (i + 1) % 500 == 0:
                print(' ', i + 1, flush=True)
    np.savez_compressed(os.path.join(ROOT, 'champ', 'imu_seq.npz'), **store)
    print('units:', len(store))
    nz = sum(1 for v in store.values() if np.abs(v).sum() > 0)
    print('units with non-zero IMU:', nz)
