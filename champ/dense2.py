"""Dense temporal model v2: three supervision sources, all from training labels only.

  (a) frame CE on HARn-segment-covered frames (background kept only where trustworthy)
  (b) clip-level MIL BCE on the QA clip action pool  -> fixes the ~13% unsegmented actions
  (c) pairwise ranking loss on soft onsets from the sequence questions' answer order
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NCLS = D.NCLS
BG = D.BG
V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}


def build(meta, tr):
    """Adds QA-derived clip pool and sequence order to the dense items."""
    items = D.build_dense(meta)
    h = tr[(tr.source == 'HAU') & tr.category.isin(['single', 'multi', 'combination', 'sequence'])]
    pool, order = {}, {}
    for _, r in h.iterrows():
        s = pool.setdefault(r.path, set())
        for L in str(r['answer']):
            if L in 'ABCD':
                for p in str(r[L]).split(','):
                    s.add(p.strip())
        if r.category == 'sequence':
            oo = [str(r[L]).strip() for L in 'ABCD']
            seqcls = [OPT2CLS.get(oo['ABCD'.index(L)]) for L in str(r['answer']) if L in 'ABCD']
            if all(c is not None for c in seqcls) and len(seqcls) == 4:
                order[r.path] = seqcls
    for it in items:
        p = it['qa_path']
        it['pool_cls'] = sorted({OPT2CLS[o] for o in pool.get(p, set()) if o in OPT2CLS})
        it['seg_cls'] = sorted({D.A2I[a] for a in it['pool']})
        it['order'] = order.get(p)
        # frames not covered by a segment are only trustworthy background when every
        # QA-pool action of this clip was actually segmented
        it['bg_trust'] = set(it['pool_cls']) <= set(it['seg_cls'])
    return items


class Net(nn.Module):
    def __init__(self, din, ch=192, nl=7, p=0.15):
        super().__init__()
        self.inp = nn.Conv1d(din, ch, 1)
        self.blocks = nn.ModuleList()
        for i in range(nl):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GroupNorm(8, ch),
                nn.GELU(), nn.Dropout(p), nn.Conv1d(ch, ch, 1)))
        self.head = nn.Conv1d(ch, NCLS, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h)


def soft_onset(lp, mask):
    """lp: B,NCLS,T log-probs; returns B,NCLS soft centroid in [0,1]."""
    T = lp.shape[-1]
    w = torch.softmax(lp.masked_fill(~mask[:, None, :], -1e9) * 2.0, dim=-1)
    t = torch.arange(T, device=lp.device, dtype=lp.dtype) / max(1, T - 1)
    return (w * t).sum(-1)


def train(items, epochs=60, seed=0, ch=192, w_mil=1.0, w_rank=1.0, verbose=False):
    torch.manual_seed(seed)
    din = items[0]['X'].shape[1]
    cat = np.concatenate([it['X'] for it in items])
    mu, sd = cat.mean(0), cat.std(0) + 1e-6
    del cat
    net = Net(din, ch=ch).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-4)
    steps = epochs * (len(items) // 8 + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, steps + 5)
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
            B = len(batch)
            X = np.zeros((B, T, din), np.float32)
            Y = np.full((B, T), -100, np.int64)
            M = np.zeros((B, T), bool)
            for i, b in enumerate(batch):
                n = len(b['X'])
                X[i, :n] = (b['X'] - mu) / sd
                yy = b['y'].copy()
                if not b['bg_trust']:
                    yy = np.where(yy == BG, -100, yy)     # ambiguous background -> ignore
                Y[i, :n] = yy
                M[i, :n] = True
            x = torch.tensor(X, device=DEV).transpose(1, 2)
            y = torch.tensor(Y, device=DEV)
            m = torch.tensor(M, device=DEV)
            lo = net(x)
            lp = torch.log_softmax(lo, 1)
            loss = F.cross_entropy(lo, y, weight=w, ignore_index=-100)
            # (b) clip-level MIL over the QA pool
            pooled = (lp.masked_fill(~m[:, None, :], -1e9)).logsumexp(-1) \
                - torch.log(m.sum(-1).clamp(min=1).to(lp.dtype))[:, None] * 0.0
            mx = lp.masked_fill(~m[:, None, :], -1e9).max(-1).values      # B,NCLS
            tgt = torch.zeros((B, NCLS), device=DEV)
            for i, b in enumerate(batch):
                for c in b['pool_cls']:
                    tgt[i, c] = 1.0
                tgt[i, BG] = 1.0
            mil = F.binary_cross_entropy_with_logits(mx[:, :BG] * 1.0, tgt[:, :BG])
            loss = loss + w_mil * mil
            # (c) order ranking from the sequence answers
            rk = torch.zeros((), device=DEV)
            nrk = 0
            on = soft_onset(lp, m)
            for i, b in enumerate(batch):
                if not b['order']:
                    continue
                o = b['order']
                for a in range(len(o) - 1):
                    for bb in range(a + 1, len(o)):
                        rk = rk + F.relu(on[i, o[a]] - on[i, o[bb]] + 0.05)
                        nrk += 1
            if nrk:
                loss = loss + w_rank * rk / nrk
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            try: sch.step()
            except Exception: pass
            tl += float(loss); nb += 1
        if verbose and (ep + 1) % 15 == 0:
            print(f'    ep{ep+1} loss {tl/max(1,nb):.4f}', flush=True)
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
        lp = torch.log_softmax(net(x), 1).cpu().numpy()
        for i, b in enumerate(batch):
            out[b['qa_path']] = lp[i, :, :len(b['X'])].astype(np.float32)
    return out
