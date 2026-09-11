"""Clip-level skeleton+IMU descriptors (final video campaign).

Maps cache keys to QA clip ids:
  Training/HAU/<user>/<s>-<clip>  -> HAU/<user>/<s>-<clip>
  Training/HARn/<action>/<user>/... -> HARn/<action>/<user>/...
  Testing/large_model_track_test/LM_test_N -> LM_test_N
Features: per-joint mean/std, velocity energy, hip_z travel (skeleton);
per-channel mean/std/min/max (IMU, when present). Deterministic, no labels.
"""
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKEL = os.path.join(ROOT, 'champ', 'skel_seq.npz')
IMU = os.path.join(ROOT, 'champ', 'imu_seq.npz')


def clip_id(key):
    """Cache key -> (kind, id) with kind in HAU/HARn/LM."""
    k = key.split('|')[0]
    m = re.search(r'/Training/HAU/([^/]+)/([^/]+)$', k)
    if m:
        return ('HAU', 'HAU/%s/%s' % (m.group(1), m.group(2)))
    m = re.search(r'/Training/HARn/([^/]+)/([^/]+)/([^/]+)$', k)
    if m:
        return ('HARn', 'HARn/%s/%s/%s' % (m.group(1), m.group(2), m.group(3)))
    m = re.search(r'/Testing/large_model_track_test/(LM_test_\d+)$', k)
    if m:
        return ('LM', m.group(1))
    return (None, None)


def skel_feats(K):
    K = np.asarray(K, dtype=np.float64)
    T = K.shape[0]
    out = []
    out.append(K.mean(axis=0).ravel())
    out.append(K.std(axis=0).ravel())
    V = np.diff(K, axis=0)
    out.append(np.abs(V).mean(axis=0).ravel())
    out.append(np.abs(V).std(axis=0).ravel())
    out.append(np.array([np.abs(V).mean(), np.abs(V).max(),
                         K[:, :, 2].std(), float(T)]))
    hipz = K[:, 0, 2] if K.shape[1] > 0 else np.zeros(T)
    out.append(np.array([hipz[-1] - hipz[0], np.abs(np.diff(hipz)).sum()]))
    return np.concatenate(out)


def imu_feats(X):
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X.mean(axis=0), X.std(axis=0),
                           X.min(axis=0), X.max(axis=0)])


def build_matrix():
    sk = np.load(SKEL, allow_pickle=True)
    try:
        im = np.load(IMU, allow_pickle=True)
        ikeys = set(im.files)
    except Exception:
        im, ikeys = None, set()
    ids, vecs = [], []
    for key in sk.files:
        if not key.endswith('|K'):
            continue
        kind, cid = clip_id(key)
        if cid is None:
            continue
        v = skel_feats(sk[key])
        ibase = key.rsplit('|', 1)[0]
        if ibase in ikeys:
            v = np.concatenate([v, imu_feats(im[ibase])])
        else:
            v = np.concatenate([v, np.zeros(120)])
        ids.append((kind, cid))
        vecs.append(v)
    X = np.stack(vecs)
    return ids, X


if __name__ == '__main__':
    ids, X = build_matrix()
    print('clips', len(ids), 'dim', X.shape[1])
    np.savez_compressed(os.path.join(ROOT, 'research', 'final_video_20260910',
                                     'clip_features.npz'),
                        ids=np.array(ids, dtype=object), X=X)
    print('wrote clip_features.npz')
