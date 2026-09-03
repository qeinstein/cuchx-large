"""Candidate-conditioned temporal action localization.

Motivation (measured): ordering by TRUE segment onsets gives 98.8% pairwise / 93.5% exact
sequence accuracy, while the 41-way frame classifier yields 87.7% pairwise / 57.8% exact.
The ordering logic is not the problem; per-action localization is.

So drop the 41-way softmax and score each action independently:
    s_t(a) = <h_t, e_a> / sqrt(C) + b_a           (h_t = frame embedding, e_a = action embedding)
trained with a per-frame BINARY objective.  Two consequences:
  * actions no longer compete in a softmax, so a weak action is not suppressed by a strong one
  * the incomplete-label problem is handled exactly, per action:
        segmented                  -> supervise every frame against the true interval
        in the QA pool, no segment -> IGNORE (present, but timing unknown)
        absent from the QA pool    -> supervise as all-negative
    The old model had to call these last two cases 'background' alike, which taught it to
    suppress ~13% of the actions that are genuinely present.
"""
import os, sys, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NA = len(D.ACTIONS)
V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}

IGNORE = -1.0     # sentinel in the target mask


def build(meta, tr):
    """Per-session frame features plus a per-action target mask in {0, 1, IGNORE}."""
    Z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    segs = {}
    for r in meta[meta.kind == 'train_harn'].itertuples():
        if np.isfinite(r.f0):
            segs.setdefault((r.user, r.trial), []).append((r.action, r.f0, r.f1))
    h = tr[(tr.source == 'HAU') & tr.category.isin(
        ['single', 'multi', 'combination', 'sequence'])]
    pool = {}
    for _, r in h.iterrows():
        s = pool.setdefault(r.path, set())
        for L in str(r['answer']):
            if L in 'ABCD':
                for p in str(r[L]).split(','):
                    c = OPT2CLS.get(p.strip())
                    if c is not None:
                        s.add(c)
    items = []
    for r in meta[meta.kind == 'train_hau'].itertuples():
        k = r.unit_dir + '|K'
        if k not in Z:
            continue
        K = Z[k]
        Fr = Z[r.unit_dir + '|F']
        X = np.concatenate([D.frame_feats(K), D.imu_feats(r.unit_dir, len(K))], 1)
        T = len(Fr)
        M = np.zeros((NA, T), np.float32)                 # default: negative
        segmented = set()
        for a, f0, f1 in segs.get((r.user, r.trial), []):
            c = D.A2I.get(a)
            if c is None:
                continue
            segmented.add(c)
            M[c] = 0.0
            M[c][(Fr >= f0) & (Fr <= f1)] = 1.0
        for c in pool.get(r.qa_path, set()) - segmented:
            M[c] = IGNORE                                  # present, timing unknown
        items.append(dict(unit=r.unit_dir, user=r.user, trial=r.trial, qa_path=r.qa_path,
                          X=X, M=M, F=Fr,
                          pool=sorted(pool.get(r.qa_path, set())),
                          segmented=sorted(segmented)))
    return items


def infer_items(meta, kinds=('test',)):
    Z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    out = []
    for r in meta[meta.kind.isin(kinds)].itertuples():
        k = r.unit_dir + '|K'
        if k not in Z:
            continue
        K = Z[k]
        out.append(dict(qa_path=r.qa_path,
                        X=np.concatenate([D.frame_feats(K), D.imu_feats(r.unit_dir, len(K))], 1),
                        F=Z[r.unit_dir + '|F']))
    return out


class Loc(nn.Module):
    def __init__(self, din, ch=224, nl=8, emb=128, p=0.15):
        super().__init__()
        self.inp = nn.Conv1d(din, ch, 1)
        self.blocks = nn.ModuleList()
        for i in range(nl):
            d = 2 ** (i % 6)
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GroupNorm(8, ch),
                nn.GELU(), nn.Dropout(p), nn.Conv1d(ch, ch, 1)))
        self.proj = nn.Conv1d(ch, emb, 1)
        self.act_emb = nn.Parameter(torch.randn(NA, emb) * 0.05)
        self.act_bias = nn.Parameter(torch.zeros(NA))
        self.emb = emb

    def forward(self, x):                       # x: B,din,T -> B,NA,T
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        z = self.proj(h)                        # B,emb,T
        s = torch.einsum('bet,ae->bat', z, self.act_emb) / (self.emb ** 0.5)
        return s + self.act_bias[None, :, None]


def train(items, epochs=70, seed=0, bs=8, ch=224, verbose=False):
    torch.manual_seed(seed)
    din = items[0]['X'].shape[1]
    cat = np.concatenate([it['X'] for it in items])
    mu, sd = cat.mean(0), cat.std(0) + 1e-6
    del cat
    net = Loc(din, ch=ch).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, epochs * (len(items) // bs + 1) + 5)
    # positives are ~1.5% of the (action, frame) cells, so weight them up
    npos = sum(float((it['M'] == 1).sum()) for it in items)
    nneg = sum(float((it['M'] == 0).sum()) for it in items)
    pw = torch.tensor(min(20.0, max(1.0, nneg / max(1.0, npos))), device=DEV)
    if verbose:
        print(f'    pos frames {npos:.0f}  neg {nneg:.0f}  pos_weight {float(pw):.2f}')
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train(); tl = 0.0; nb = 0
        order = rng.permutation(len(items))
        for bi in range(0, len(order), bs):
            batch = [items[i] for i in order[bi:bi + bs]]
            T = max(len(b['X']) for b in batch)
            B = len(batch)
            X = np.zeros((B, T, din), np.float32)
            M = np.full((B, NA, T), IGNORE, np.float32)
            for i, b in enumerate(batch):
                n = len(b['X'])
                X[i, :n] = (b['X'] - mu) / sd
                M[i, :, :n] = b['M']
            x = torch.tensor(X, device=DEV).transpose(1, 2)
            m = torch.tensor(M, device=DEV)
            s = net(x)
            valid = m >= 0
            loss = F.binary_cross_entropy_with_logits(
                s[valid], m[valid], pos_weight=pw)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            try: sch.step()
            except Exception: pass
            tl += float(loss); nb += 1
        if verbose and (ep + 1) % 20 == 0:
            print(f'    loc ep{ep+1} loss {tl/max(1,nb):.4f}', flush=True)
    return dict(net=net, mu=mu, sd=sd)


@torch.no_grad()
def predict(model, items, bs=8):
    net, mu, sd = model['net'], model['mu'], model['sd']
    net.eval(); out = {}
    for bi in range(0, len(items), bs):
        batch = items[bi:bi + bs]
        T = max(len(b['X']) for b in batch)
        X = np.zeros((len(batch), T, len(mu)), np.float32)
        for i, b in enumerate(batch):
            X[i, :len(b['X'])] = (b['X'] - mu) / sd
        x = torch.tensor(X, device=DEV).transpose(1, 2)
        s = torch.sigmoid(net(x)).cpu().numpy()
        for i, b in enumerate(batch):
            out[b['qa_path']] = s[i, :, :len(b['X'])].astype(np.float32)
    return out
