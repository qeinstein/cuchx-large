"""Rich radar micro-Doppler and IMU spectral features for HAU units.

Motivation, measured: the champion's manner-group classifier reaches only 0.603 5-way
accuracy, and a PERFECT group classifier would take emotion from 0.907 to 0.989 OOF.  Its
physical feature set contains exactly one radar column (rad_v_p90) and five IMU summary
columns, for a task that is entirely about how fast and how jittery the motion was.
Radar carries per-detection radial velocity at ~20 Hz and the five IMUs carry ~10 Hz
acceleration and angular rate: both are natural manner sensors and both are unexploited.
"""
import os, sys, glob, math, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _spec(x, fs, bands):
    """Normalised band powers, spectral centroid, entropy, dominant frequency."""
    out = {}
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        for lo, hi in bands: out[f'bp{lo}_{hi}'] = np.nan
        out.update(cent=np.nan, ent=np.nan, dom=np.nan, tot=np.nan)
        return out
    x = x - x.mean()
    w = np.hanning(n)
    F = np.fft.rfft(x * w)
    P = (np.abs(F) ** 2)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    tot = P.sum() + 1e-12
    for lo, hi in bands:
        out[f'bp{lo}_{hi}'] = float(P[(f >= lo) & (f < hi)].sum() / tot)
    out['cent'] = float((P * f).sum() / tot)
    p = P / tot
    out['ent'] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(len(p)))
    out['dom'] = float(f[int(P[1:].argmax()) + 1]) if len(P) > 1 else np.nan
    out['tot'] = float(np.log1p(tot))
    return out


def _ts(s):
    return pd.to_datetime(s, errors='coerce')


def radar_feats(unit):
    p = os.path.join(unit, 'Radar', 'Radar.csv')
    if not os.path.exists(p): return {}
    try:
        d = pd.read_csv(p)
    except Exception:
        return {}
    if len(d) == 0 or 'v' not in d.columns: return {}
    d = d.dropna(subset=['frame', 'v'])
    if len(d) == 0: return {}
    av = d.v.abs().to_numpy(float)
    f = {}
    f['rr_ndet'] = len(d)
    f['rr_nframe'] = int(d.frame.nunique())
    f['rr_det_per_frame'] = len(d) / max(d.frame.nunique(), 1)
    for q in [50, 75, 90, 95, 99]:
        f[f'rr_av_p{q}'] = float(np.percentile(av, q))
    f['rr_av_mean'] = float(av.mean()); f['rr_av_std'] = float(av.std())
    f['rr_av_max'] = float(av.max())
    f['rr_nz_frac'] = float((av > 1e-9).mean())
    for t in [0.1, 0.3, 0.5]:
        f[f'rr_frac_gt{t}'] = float((av > t).mean())
    nz = av[av > 1e-9]
    f['rr_nzmean'] = float(nz.mean()) if len(nz) else 0.0
    f['rr_nzp90'] = float(np.percentile(nz, 90)) if len(nz) else 0.0
    f['rr_energy'] = float(np.log1p((d.v.to_numpy(float) ** 2).sum()))
    f['rr_signed'] = float(d.v.mean())
    f['rr_approach'] = float((d.v > 1e-9).sum() / max((av > 1e-9).sum(), 1))
    # velocity histogram entropy
    h, _ = np.histogram(av, bins=20, range=(0, max(av.max(), 1e-6)))
    h = h / (h.sum() + 1e-12)
    f['rr_hist_ent'] = float(-(h[h > 0] * np.log(h[h > 0])).sum() / math.log(20))
    if 'snr' in d:
        s = d.snr.to_numpy(float)
        f['rr_snr_mean'] = float(np.nanmean(s))
        f['rr_wv'] = float((av * s).sum() / (s.sum() + 1e-9))
    for c in ['x', 'y', 'z']:
        if c in d:
            f[f'rr_{c}_std'] = float(d[c].std()); f[f'rr_{c}_mean'] = float(d[c].mean())
    # per-frame time series -> temporal spectrum of the Doppler envelope
    g = d.assign(av=av).groupby('frame')
    ser = g.av.max().sort_index()
    nd = g.size().sort_index()
    t = _ts(d.timestamp)
    dur = float((t.max() - t.min()).total_seconds()) if t.notna().any() else np.nan
    fs = (len(ser) / dur) if dur and dur > 0 else 20.0
    f['rr_dur'] = dur; f['rr_fs'] = fs
    for k, v in _spec(ser.to_numpy(), fs, [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 8)]).items():
        f[f'rr_env_{k}'] = v
    for k, v in _spec(nd.to_numpy(), fs, [(0, .5), (.5, 1), (1, 2), (2, 4)]).items():
        f[f'rr_nd_{k}'] = v
    a = ser.to_numpy()
    if len(a) > 3:
        f['rr_env_ac1'] = float(np.corrcoef(a[:-1], a[1:])[0, 1])
        f['rr_env_std'] = float(a.std()); f['rr_env_mean'] = float(a.mean())
        f['rr_env_jerk'] = float(np.abs(np.diff(a)).mean())
        f['rr_env_cv'] = float(a.std() / (a.mean() + 1e-9))
    return f


DEV = {'WTC': 'C', 'WTLA': 'LA', 'WTRA': 'RA', 'WTLL': 'LL', 'WTRL': 'RL'}
BANDS = [(0, .5), (.5, 1), (1, 2), (2, 3), (3, 5)]


def imu_feats(unit):
    out = {}
    frames = []
    for fn in ['up(LA+RA+C).csv', 'down(LL+RL).csv']:
        p = os.path.join(unit, 'IMU', fn)
        if not os.path.exists(p): continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if len(d) == 0 or d.shape[1] < 8: continue
        d = d.rename(columns={d.columns[0]: 'ts', d.columns[1]: 'dev'})
        cols = list(d.columns)
        d['acc'] = np.linalg.norm(d[cols[2:5]].to_numpy(float), axis=1)
        d['gyr'] = np.linalg.norm(d[cols[5:8]].to_numpy(float), axis=1)
        d['tt'] = _ts(d.ts)
        frames.append(d[['dev', 'acc', 'gyr', 'tt']])
    if not frames: return {}
    D = pd.concat(frames, ignore_index=True)
    D['tag'] = [next((v for k, v in DEV.items() if isinstance(x, str) and x.startswith(k)), None)
                for x in D.dev]
    agg = {}
    for tag, g in D.dropna(subset=['tag']).groupby('tag'):
        g = g.sort_values('tt')
        dur = float((g.tt.max() - g.tt.min()).total_seconds()) if g.tt.notna().any() else np.nan
        fs = len(g) / dur if dur and dur > 0 else 10.0
        for sig in ['acc', 'gyr']:
            x = g[sig].to_numpy(float)
            x = x[np.isfinite(x)]
            if len(x) < 8: continue
            pre = f'iu_{tag}_{sig}'
            out[f'{pre}_mean'] = float(x.mean()); out[f'{pre}_std'] = float(x.std())
            out[f'{pre}_p90'] = float(np.percentile(x, 90))
            out[f'{pre}_p99'] = float(np.percentile(x, 99))
            out[f'{pre}_iqr'] = float(np.percentile(x, 75) - np.percentile(x, 25))
            dx = np.diff(x)
            out[f'{pre}_jerk'] = float(np.abs(dx).mean())
            out[f'{pre}_jerkstd'] = float(dx.std())
            xd = x - x.mean()
            out[f'{pre}_zcr'] = float((np.diff(np.sign(xd)) != 0).mean())
            out[f'{pre}_cv'] = float(x.std() / (abs(x.mean()) + 1e-9))
            for k, v in _spec(x, fs, BANDS).items():
                out[f'{pre}_{k}'] = v
            agg[(tag, sig)] = dict(mean=x.mean(), std=x.std(), jerk=np.abs(dx).mean(),
                                   hi=out.get(f'{pre}_bp3_5', np.nan),
                                   cent=out.get(f'{pre}_cent', np.nan))
        out[f'iu_{tag}_fs'] = fs; out[f'iu_{tag}_n'] = len(g); out[f'iu_{tag}_dur'] = dur
    # cross-device summaries
    for sig in ['acc', 'gyr']:
        vals = {t: agg[(t, sig)] for t in ['C', 'LA', 'RA', 'LL', 'RL'] if (t, sig) in agg}
        if not vals: continue
        arms = [vals[t]['std'] for t in ['LA', 'RA'] if t in vals]
        legs = [vals[t]['std'] for t in ['LL', 'RL'] if t in vals]
        out[f'iu_{sig}_ndev'] = len(vals)
        out[f'iu_{sig}_stdmean'] = float(np.mean([v['std'] for v in vals.values()]))
        out[f'iu_{sig}_stdmax'] = float(np.max([v['std'] for v in vals.values()]))
        out[f'iu_{sig}_jerkmean'] = float(np.mean([v['jerk'] for v in vals.values()]))
        out[f'iu_{sig}_himean'] = float(np.nanmean([v['hi'] for v in vals.values()]))
        out[f'iu_{sig}_centmean'] = float(np.nanmean([v['cent'] for v in vals.values()]))
        if arms and legs:
            out[f'iu_{sig}_armleg'] = float(np.mean(arms) / (np.mean(legs) + 1e-9))
        if 'LA' in vals and 'RA' in vals:
            out[f'iu_{sig}_lr_arm'] = float(abs(vals['LA']['std'] - vals['RA']['std'])
                                            / (vals['LA']['std'] + vals['RA']['std'] + 1e-9))
        if 'LL' in vals and 'RL' in vals:
            out[f'iu_{sig}_lr_leg'] = float(abs(vals['LL']['std'] - vals['RL']['std'])
                                            / (vals['LL']['std'] + vals['RL']['std'] + 1e-9))
        if 'C' in vals and arms:
            out[f'iu_{sig}_arm_chest'] = float(np.mean(arms) / (vals['C']['std'] + 1e-9))
    return out


def one(unit):
    d = {'unit_dir': unit}
    try: d.update(radar_feats(unit))
    except Exception as e: d['rr_err'] = 1
    try: d.update(imu_feats(unit))
    except Exception as e: d['iu_err'] = 1
    return d


if __name__ == '__main__':
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    which = sys.argv[1] if len(sys.argv) > 1 else 'hau'
    if which == 'hau':
        units = meta[meta.kind.isin(['train_hau', 'test'])].unit_dir.unique().tolist()
        outp = os.path.join(ROOT, 'emolab', 'rich_hau.csv')
    else:
        units = meta[meta.kind == 'train_harn'].unit_dir.unique().tolist()
        outp = os.path.join(ROOT, 'emolab', 'rich_harn.csv')
    print(f'{len(units)} units -> {outp}', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, units, chunksize=4)):
            rows.append(r)
            if (i + 1) % 100 == 0: print(f'  {i+1}/{len(units)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(outp, index=False)
    print('wrote', df.shape)
