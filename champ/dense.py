"""Dense frame-level action segmentation trained on HARn-derived labels.

Labels: for every train HAU session, each HARn segment sharing its (user, trial) key marks
[f0, f1] on the session's global frame axis with that segment's action class; the rest is
background.  Verified earlier: 779/779 sessions have all segments strictly inside the parent.
"""
import os, sys, json, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'

VOCAB = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))
ACTIONS = sorted(VOCAB['HARN2HAU'])            # 40 HARn folders
A2I = {a: i for i, a in enumerate(ACTIONS)}
BG = len(ACTIONS)
NCLS = len(ACTIONS) + 1
HAU_OPT = [VOCAB['HARN2HAU'][a] for a in ACTIONS]
OPT2I = {o: i for i, o in enumerate(HAU_OPT)}


# ------------------------------------------------------------------ per-frame features
def frame_feats(K):
    """K: T x 17 x 3 -> T x D, translation/scale invariant plus derivatives."""
    hip = K[:, [11, 12]].mean(1, keepdims=True)
    torso = (np.linalg.norm(K[:, 5] - K[:, 11], axis=-1)
             + np.linalg.norm(K[:, 6] - K[:, 12], axis=-1))
    s = np.median(torso[torso > 1e-3]) if np.any(torso > 1e-3) else 1.0
    P = (K - hip) / max(s, 1e-3)
    V = np.zeros_like(P); V[1:] = np.diff(P, axis=0)
    A = np.zeros_like(P); A[1:] = np.diff(V, axis=0)
    hipn = (hip[:, 0] - hip[0, 0]) / max(s, 1e-3)
    hv = np.zeros_like(hipn); hv[1:] = np.diff(hipn, axis=0)
    sp = np.linalg.norm(V, axis=-1)
    X = np.concatenate([P.reshape(len(K), -1), V.reshape(len(K), -1),
                        A.reshape(len(K), -1), hipn, hv, sp], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ------------------------------------------------------------------ dataset
_IMU = None


def imu_of(unit, T):
    global _IMU
    if _IMU is None:
        _IMU = np.load(os.path.join(ROOT, 'champ', 'imu_seq.npz'))
    if unit not in _IMU:
        return np.zeros((T, 30), np.float32)
    A = _IMU[unit].astype(np.float32)
    if len(A) == T:
        return A
    if len(A) == 0:
        return np.zeros((T, 30), np.float32)
    idx = np.clip((np.arange(T) * len(A) // max(1, T)), 0, len(A) - 1)
    return A[idx]


def imu_feats(unit, T):
    """Raw + short-window energy of the resampled IMU, as per-frame channels."""
    A = imu_of(unit, T)
    acc = np.stack([np.linalg.norm(A[:, d * 6:d * 6 + 3], axis=1) for d in range(5)], 1)
    gyr = np.stack([np.linalg.norm(A[:, d * 6 + 3:d * 6 + 6], axis=1) for d in range(5)], 1)
    k = np.ones(9) / 9.0
    def sm(M):
        if M.shape[0] < 9:
            return M
        return np.stack([np.convolve(M[:, j], k, mode='same') for j in range(M.shape[1])], 1)
    dacc = np.abs(np.diff(acc, axis=0, prepend=acc[:1]))
    X = np.concatenate([A, acc, gyr, sm(acc), sm(gyr), sm(dacc)], 1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_dense(meta):
    """Returns list of (unit_dir, user, trial, X, y) for train HAU sessions with labels."""
    Z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    segs = {}
    for r in meta[meta.kind == 'train_harn'].itertuples():
        segs.setdefault((r.user, r.trial), []).append((r.action, r.f0, r.f1))
    items = []
    for r in meta[meta.kind == 'train_hau'].itertuples():
        k = r.unit_dir + '|K'
        if k not in Z:
            continue
        K = Z[k]; Fr = Z[r.unit_dir + '|F']
        X = np.concatenate([frame_feats(K), imu_feats(r.unit_dir, len(K))], 1)
        y = np.full(len(Fr), BG, np.int64)
        sv = segs.get((r.user, r.trial), [])
        for a, f0, f1 in sv:
            if a not in A2I or not np.isfinite(f0):
                continue
            m = (Fr >= f0) & (Fr <= f1)
            y[m] = A2I[a]
        items.append(dict(unit=r.unit_dir, user=r.user, trial=r.trial,
                          qa_path=r.qa_path, X=X, y=y, F=Fr,
                          pool=sorted({a for a, _, _ in sv if a in A2I}))) 
    return items


def infer_items(meta, kinds=('test',)):
    Z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    out = []
    for r in meta[meta.kind.isin(kinds)].itertuples():
        k = r.unit_dir + '|K'
        if k not in Z:
            continue
        K = Z[k]
        out.append(dict(unit=r.unit_dir, qa_path=r.qa_path,
                        X=np.concatenate([frame_feats(K), imu_feats(r.unit_dir, len(K))], 1),
                        F=Z[r.unit_dir + '|F']))
    return out


# ------------------------------------------------------------------ model
class TCN(nn.Module):
    def __init__(self, din, ch=160, nl=7, ncls=NCLS, p=0.15):
        super().__init__()
        self.inp = nn.Conv1d(din, ch, 1)
        self.blocks = nn.ModuleList()
        for i in range(nl):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=d, dilation=d), nn.GroupNorm(8, ch),
                nn.GELU(), nn.Dropout(p), nn.Conv1d(ch, ch, 1)))
        self.head = nn.Conv1d(ch, ncls, 1)

    def forward(self, x):                      # x: B,D,T
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h)


def train_dense(items, epochs=40, seed=0, verbose=True):
    torch.manual_seed(seed)
    din = items[0]['X'].shape[1]
    mu = np.concatenate([it['X'] for it in items]).mean(0)
    sd = np.concatenate([it['X'] for it in items]).std(0) + 1e-6
    net = TCN(din).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2.5e-3, epochs * len(items) // 8 + 1)
    cnt = np.bincount(np.concatenate([it['y'] for it in items]), minlength=NCLS).astype(float)
    w = (cnt.sum() / (cnt + 50.0)) ** 0.5
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEV)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train(); tl = 0.0; nb = 0
        order = rng.permutation(len(items))
        for bi in range(0, len(order), 8):
            batch = [items[i] for i in order[bi:bi + 8]]
            T = max(len(b['X']) for b in batch)
            X = np.zeros((len(batch), T, din), np.float32)
            Y = np.full((len(batch), T), -100, np.int64)
            for i, b in enumerate(batch):
                n = len(b['X'])
                X[i, :n] = (b['X'] - mu) / sd
                Y[i, :n] = b['y']
            x = torch.tensor(X, device=DEV).transpose(1, 2)
            y = torch.tensor(Y, device=DEV)
            lo = net(x)
            loss = F.cross_entropy(lo, y, weight=w, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            try: sch.step()
            except Exception: pass
            tl += float(loss); nb += 1
        if verbose and (ep + 1) % 10 == 0:
            print(f'    dense ep{ep+1} loss {tl/max(1,nb):.4f}', flush=True)
    return dict(net=net, mu=mu, sd=sd)


@torch.no_grad()
def predict(model, items, bs=8):
    net, mu, sd = model['net'], model['mu'], model['sd']
    net.eval()
    out = {}
    for bi in range(0, len(items), bs):
        batch = items[bi:bi + bs]
        T = max(len(b['X']) for b in batch)
        X = np.zeros((len(batch), T, len(mu)), np.float32)
        for i, b in enumerate(batch):
            X[i, :len(b['X'])] = (b['X'] - mu) / sd
        x = torch.tensor(X, device=DEV).transpose(1, 2)
        lp = torch.log_softmax(net(x), 1).cpu().numpy()
        for i, b in enumerate(batch):
            out[b['qa_path']] = lp[i, :, :len(b['X'])].astype(np.float32)
    return out
