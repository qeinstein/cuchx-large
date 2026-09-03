"""Cheapest possible per-frame Depth/IR features, for the information-audit probe.

Frame alignment is exact: the Depth/IR videos have one frame per skeleton frame, and
video frame i corresponds to global frame f0 + i (verified: depth_n == nskel for every
clip checked).  So HARn segment intervals land directly on video frames with no offset.

Per frame we keep only cheap statistics - no backbone, no pretrained weights:
  * mean / std of the depth image
  * frame-difference motion energy
  * row and column centroid of the motion
  * a 4x3 grid of motion energy (where in the scene the movement is)
This is deliberately weak.  Its only job is to answer whether the depth stream carries
temporal signal that skeleton+IMU does not, before committing to a real video model.
"""
import os, sys, re
import numpy as np, pandas as pd
import cv2
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 32, 24
GX, GY = 4, 3


def video_path(qa_path, kind, mod='Depth'):
    if kind == 'train_hau':
        u, t = qa_path.split('/')[1], qa_path.split('/')[2]
        return os.path.join(ROOT, 'hf_data_manual', 'HAU', u, t, mod, f'{mod}.mp4')
    if kind == 'train_harn':
        _, a, u, t = qa_path.split('/')
        return os.path.join(ROOT, 'hf_data_manual', 'HARn', a, u, t, mod, f'{mod}.mp4')
    return os.path.join(ROOT, 'hf_data_manual', 'large_model_track_test', qa_path,
                        mod, f'{mod}.mp4')


def extract(args):
    qa_path, kind = args
    out = {}
    for mod in ('Depth', 'IR'):
        p = video_path(qa_path, kind, mod)
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p)
        rows, prev = [], None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            f = [g.mean(), g.std()]
            if prev is None:
                d = np.zeros_like(g)
            else:
                d = np.abs(g - prev)
            prev = g
            e = d.sum() + 1e-9
            ys, xs = np.mgrid[0:H, 0:W]
            f += [d.mean(), float((d * ys).sum() / e / H), float((d * xs).sum() / e / W)]
            cell = d.reshape(GY, H // GY, GX, W // GX).mean(axis=(1, 3)).ravel()
            f += list(cell)
            rows.append(f)
        cap.release()
        if rows:
            out[mod] = np.asarray(rows, np.float32)
    if not out:
        return qa_path, None
    n = min(v.shape[0] for v in out.values())
    return qa_path, np.concatenate([out[m][:n] for m in sorted(out)], axis=1)


if __name__ == '__main__':
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    kinds = sys.argv[1].split(',') if len(sys.argv) > 1 else ['train_hau', 'test']
    todo = [(r.qa_path, r.kind) for r in meta.itertuples() if r.kind in kinds]
    print('clips to decode:', len(todo))
    store, miss = {}, 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (p, X) in enumerate(ex.map(extract, todo, chunksize=4)):
            if X is None:
                miss += 1
            else:
                store[p] = X
            if (i + 1) % 100 == 0:
                print(' ', i + 1, flush=True)
    np.savez_compressed(os.path.join(ROOT, 'champ', 'depth_frames.npz'), **store)
    print('cached', len(store), 'missing', miss)
    ex0 = next(iter(store.values()))
    print('per-frame feature dim:', ex0.shape[1])
