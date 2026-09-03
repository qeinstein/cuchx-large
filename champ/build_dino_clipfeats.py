"""Clip-level manner features derived from the DINOv2 frame sequence, appended to feats.csv.

Rationale for emotion: manner is *how* an action is performed - speed, smoothness,
hesitation.  Feature-space velocity ||delta f_t|| of a perceptual embedding is an
appearance-grounded motion magnitude that is implicitly normalised by what the person is
doing, which is exactly the action-conditioning the residual emotion errors need
(72% of them cross manner groups, and 79% are within-session swaps).
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stats(A):
    A = A.astype(np.float32)
    T = len(A)
    if T < 4:
        return {}
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)
    d = np.linalg.norm(np.diff(An, axis=0), axis=1)          # feature-space velocity
    dd = np.abs(np.diff(d)) if T > 2 else np.zeros(1)
    cos = (An[1:] * An[:-1]).sum(1)                          # frame-to-frame similarity
    out = {
        'dino_v_mean': d.mean(), 'dino_v_std': d.std(), 'dino_v_p90': np.percentile(d, 90),
        'dino_v_p50': np.median(d), 'dino_v_max': d.max(),
        'dino_acc_mean': dd.mean(), 'dino_acc_p90': np.percentile(dd, 90),
        'dino_cos_mean': cos.mean(), 'dino_cos_min': cos.min(),
        'dino_v_iqr': np.percentile(d, 75) - np.percentile(d, 25),
        'dino_T': float(T),
        # fraction of the clip that is perceptually 'moving'
        'dino_active': float((d > np.percentile(d, 60)).mean()),
        # total perceptual path length, and how directly the clip travels
        'dino_path': float(d.sum()),
        'dino_net': float(np.linalg.norm(An[-1] - An[0])),
    }
    out['dino_straight'] = out['dino_net'] / (out['dino_path'] + 1e-6)
    sig = d - d.mean()
    if len(sig) >= 16:
        F = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        fq = np.fft.rfftfreq(len(sig), d=1.0 / 10.0)
        F[0] = 0
        out['dino_cad_hz'] = float(fq[np.argmax(F)])
        out['dino_cad_pow'] = float(F.max() / (F.sum() + 1e-9))
        out['dino_spec_cent'] = float((F * fq).sum() / (F.sum() + 1e-9))
    return out


def main():
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    Z = np.load(os.path.join(ROOT, 'champ', 'dino_frames.npz'))
    fe = pd.read_csv(os.path.join(ROOT, 'champ', 'feats.csv'))
    fe = fe[[c for c in fe.columns if not c.startswith('dino_')]]
    rows = []
    for r in meta.itertuples():
        if r.qa_path not in Z.files:
            continue
        s = stats(Z[r.qa_path])
        if s:
            rows.append(dict(unit_dir=r.unit_dir, **s))
    add = pd.DataFrame(rows).drop_duplicates('unit_dir')
    out = fe.merge(add, on='unit_dir', how='left')
    out.to_csv(os.path.join(ROOT, 'champ', 'feats.csv'), index=False)
    print('added %d dino columns for %d units (of %d rows)'
          % (add.shape[1] - 1, len(add), len(out)))
    print('columns:', [c for c in out.columns if c.startswith('dino_')])


if __name__ == '__main__':
    main()
