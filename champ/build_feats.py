"""Leakage-safe per-unit physical features from skeleton + IMU.

No QA answer, user id or trial id is used.  Output: champ/feats.csv keyed by unit_dir.
"""
import os, re, json, sys, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FN = re.compile(r'Color_.*_(\d{8})\.json$')


def _stats(x, pre):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f'{pre}_{k}': np.nan for k in ('mean', 'std', 'p50', 'p90', 'max')}
    return {f'{pre}_mean': x.mean(), f'{pre}_std': x.std(), f'{pre}_p50': np.median(x),
            f'{pre}_p90': np.percentile(x, 90), f'{pre}_max': x.max()}


def skel_feats(d):
    p = os.path.join(d, 'Skeleton', 'predictions')
    out = {}
    if not os.path.isdir(p):
        return out
    items = []
    for f in os.listdir(p):
        m = FN.match(f)
        if m:
            items.append((int(m.group(1)), f))
    if len(items) < 4:
        return out
    items.sort()
    K, S = [], []
    for _, f in items:
        try:
            j = json.load(open(os.path.join(p, f)))
        except Exception:
            continue
        if isinstance(j, list):
            if not j:
                continue
            j = max(j, key=lambda o: float(np.mean(o.get('keypoint_scores') or [0])))
        kp = j.get('keypoints')
        sc = j.get('keypoint_scores')
        if kp is None:
            continue
        a = np.asarray(kp, float).reshape(-1, 3)
        if a.shape[0] < 17:
            continue
        K.append(a[:17])
        S.append(np.asarray(sc, float).ravel()[:17] if sc is not None else np.ones(17))
    if len(K) < 4:
        return out
    K = np.stack(K)                     # T,17,3
    S = np.stack(S)
    fr = np.array([i for i, _ in items[:len(K)]], float)
    dt = np.diff(fr)
    dt[dt <= 0] = 1.0
    # normalise scale by torso length so subject size / camera distance cancels
    torso = np.linalg.norm(K[:, 5] - K[:, 11], axis=-1) + np.linalg.norm(K[:, 6] - K[:, 12], axis=-1)
    scale = np.median(torso[torso > 1e-3]) if np.any(torso > 1e-3) else 1.0
    Kn = K / max(scale, 1e-3)
    v = np.linalg.norm(np.diff(Kn, axis=0), axis=-1) / dt[:, None]       # T-1,17
    a = np.linalg.norm(np.diff(np.diff(Kn, axis=0), axis=0), axis=-1)
    jk = np.abs(np.diff(np.linalg.norm(np.diff(np.diff(Kn, axis=0), axis=0), axis=-1), axis=0))
    out.update(_stats(v.mean(1), 'sk_v'))
    out.update(_stats(v.max(1), 'sk_vmax'))
    out.update(_stats(a.mean(1), 'sk_a'))
    out.update(_stats(jk.mean(1), 'sk_jerk'))
    out.update(_stats(v[:, [9, 10]].mean(1), 'sk_vwrist'))
    out.update(_stats(v[:, [15, 16]].mean(1), 'sk_vankle'))
    out['sk_conf'] = float(np.nanmean(S))
    out['sk_T'] = float(len(K))
    out['sk_scale'] = float(scale)
    # path length and net displacement of the hip centre -> locomotion vs in-place
    hip = Kn[:, [11, 12]].mean(1)
    out['sk_hip_path'] = float(np.linalg.norm(np.diff(hip, axis=0), axis=-1).sum())
    out['sk_hip_net'] = float(np.linalg.norm(hip[-1] - hip[0]))
    # cadence: dominant spectral frequency of mean joint speed
    sig = v.mean(1) - v.mean()
    if sig.size >= 16:
        F = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
        fq = np.fft.rfftfreq(sig.size, d=1.0 / 10.0)
        F[0] = 0
        out['sk_cad_hz'] = float(fq[np.argmax(F)])
        out['sk_cad_pow'] = float(F.max() / (F.sum() + 1e-9))
        out['sk_spec_cent'] = float((F * fq).sum() / (F.sum() + 1e-9))
    # fraction of frames that are "moving"
    thr = np.percentile(v.mean(1), 60)
    out['sk_active_frac'] = float((v.mean(1) > max(thr, 1e-6)).mean())
    out['sk_v_iqr'] = float(np.percentile(v.mean(1), 75) - np.percentile(v.mean(1), 25))
    return out


def imu_feats(d):
    out = {}
    p = os.path.join(d, 'IMU')
    if not os.path.isdir(p):
        return out
    accs, gyrs = [], []
    for f in sorted(os.listdir(p)):
        if not f.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(os.path.join(p, f), low_memory=False)
        except Exception:
            continue
        if df.shape[1] < 8:
            continue
        cols = list(df.columns)
        acc = [c for c in cols if '加速度' in c or c.lower().startswith('acc')]
        gyr = [c for c in cols if '角速度' in c or 'AsX' in c or c.lower().startswith('gyr')]
        if len(acc) >= 3:
            A = df[acc[:3]].apply(pd.to_numeric, errors='coerce').to_numpy(float)
            accs.append(np.linalg.norm(A, axis=1))
        if len(gyr) >= 3:
            G = df[gyr[:3]].apply(pd.to_numeric, errors='coerce').to_numpy(float)
            gyrs.append(np.linalg.norm(G, axis=1))
    if accs:
        a = np.concatenate(accs)
        out.update(_stats(a, 'imu_acc'))
        out.update(_stats(np.abs(np.diff(a)), 'imu_dacc'))
        out['imu_acc_energy'] = float(np.nanmean((a - np.nanmean(a)) ** 2))
    if gyrs:
        g = np.concatenate(gyrs)
        out.update(_stats(g, 'imu_gyr'))
        out['imu_gyr_energy'] = float(np.nanmean((g - np.nanmean(g)) ** 2))
    return out


def radar_feats(d):
    out = {}
    f = os.path.join(d, 'Radar', 'Radar.csv')
    if not os.path.isfile(f):
        return out
    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception:
        return out
    if 'v' in df.columns and len(df):
        v = pd.to_numeric(df['v'], errors='coerce').to_numpy(float)
        out.update(_stats(np.abs(v), 'rad_v'))
        out['rad_n'] = float(len(df))
        if 'frame' in df.columns:
            out['rad_frames'] = float(pd.to_numeric(df['frame'], errors='coerce').nunique())
    return out


def one(d):
    r = {'unit_dir': d}
    try:
        r.update(skel_feats(d))
    except Exception:
        pass
    try:
        r.update(imu_feats(d))
    except Exception:
        pass
    try:
        r.update(radar_feats(d))
    except Exception:
        pass
    return r


if __name__ == '__main__':
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    dirs = meta.unit_dir.tolist()
    out_p = os.path.join(ROOT, 'champ', 'feats.csv')
    done = set()
    if os.path.exists(out_p) and '--resume' in sys.argv:
        done = set(pd.read_csv(out_p).unit_dir)
    todo = [d for d in dirs if d not in done]
    print('units to process:', len(todo))
    res = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, todo, chunksize=8)):
            res.append(r)
            if (i + 1) % 400 == 0:
                print(' ', i + 1, flush=True)
    df = pd.DataFrame(res)
    if done:
        df = pd.concat([pd.read_csv(out_p), df], ignore_index=True)
    df.to_csv(out_p, index=False)
    print('written', out_p, df.shape)
    print('non-null skeleton feats:', df.get('sk_v_mean', pd.Series(dtype=float)).notna().sum(), '/', len(df))
