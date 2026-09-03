"""Sequence-level action classifier for HARn clips (skeleton + IMU), replacing the
aggregate-feature model.  Trained on the 2927 HARn segments, labels = folder name."""
import os, sys, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


def build_items(meta):
    Z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    out = []
    for r in meta[meta.kind == 'train_harn'].itertuples():
        k = r.unit_dir + '|K'
        if k not in Z:
            continue
        K = Z[k]
        out.append(dict(qa_path=r.qa_path, user=r.user, action=r.action,
                        X=np.concatenate([D.frame_feats(K), D.imu_feats(r.unit_dir, len(K))], 1)))
    return out


def infer_set(meta, paths):
    Z = np.load(os.path.join(ROOT, 'champ', 'skel_seq.npz'))
    mi = meta.set_index('qa_path')
    out = []
    for p in paths:
        if p not in mi.index:
            continue
        u = mi.loc[p, 'unit_dir']
        if u + '|K' not in Z:
            continue
        K = Z[u + '|K']
        out.append(dict(qa_path=p, X=np.concatenate([D.frame_feats(K), D.imu_feats(u, len(K))], 1)))
    return out


class Clf(nn.Module):
    def __init__(self, din, ncls, ch=192, nl=5, p=0.2):
        super().__init__()
        self.inp = nn.Conv1d(din, ch, 1)
        self.blocks = nn.ModuleList()
        for i in range(nl):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GroupNorm(8, ch),
                nn.GELU(), nn.Dropout(p), nn.Conv1d(ch, ch, 1)))
        self.head = nn.Sequential(nn.Linear(2 * ch, 256), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(256, ncls))

    def forward(self, x, m):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        h = h.masked_fill(~m[:, None, :], -1e9)
        mx = h.max(-1).values
        av = h.masked_fill(~m[:, None, :], 0).sum(-1) / m.sum(-1).clamp(min=1)[:, None]
        return self.head(torch.cat([mx, av], 1))


def train(items, classes, epochs=45, seed=0, bs=32):
    torch.manual_seed(seed)
    din = items[0]['X'].shape[1]
    cat = np.concatenate([it['X'] for it in items])
    mu, sd = cat.mean(0), cat.std(0) + 1e-6
    del cat
    c2i = {c: i for i, c in enumerate(classes)}
    net = Clf(din, len(classes)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2.5e-3, epochs * (len(items) // bs + 1) + 5)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(items))
        for bi in range(0, len(order), bs):
            batch = [items[i] for i in order[bi:bi + bs]]
            T = max(len(b['X']) for b in batch)
            X = np.zeros((len(batch), T, din), np.float32)
            M = np.zeros((len(batch), T), bool)
            for i, b in enumerate(batch):
                n = len(b['X']); X[i, :n] = (b['X'] - mu) / sd; M[i, :n] = True
            x = torch.tensor(X, device=DEV).transpose(1, 2)
            m = torch.tensor(M, device=DEV)
            y = torch.tensor([c2i[b['action']] for b in batch], device=DEV)
            loss = F.cross_entropy(net(x, m), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
            try: sch.step()
            except Exception: pass
    return dict(net=net, mu=mu, sd=sd, classes=classes)


@torch.no_grad()
def predict(model, items, bs=32):
    net, mu, sd = model['net'], model['mu'], model['sd']
    net.eval(); out = {}
    for bi in range(0, len(items), bs):
        batch = items[bi:bi + bs]
        T = max(len(b['X']) for b in batch)
        X = np.zeros((len(batch), T, len(mu)), np.float32)
        M = np.zeros((len(batch), T), bool)
        for i, b in enumerate(batch):
            n = len(b['X']); X[i, :n] = (b['X'] - mu) / sd; M[i, :n] = True
        x = torch.tensor(X, device=DEV).transpose(1, 2)
        m = torch.tensor(M, device=DEV)
        lp = torch.log_softmax(net(x, m), -1).cpu().numpy()
        for i, b in enumerate(batch):
            out[b['qa_path']] = lp[i]
    return out


def main(epochs=45):
    from core import load_all
    tr, te, meta = load_all()
    items = build_items(meta)
    classes = sorted({it['action'] for it in items})
    print('HARn segments', len(items), 'classes', len(classes))
    users = sorted({it['user'] for it in items})
    store = {}
    c = n = 0
    for fi, hold in enumerate(folds(users, 5)):
        trn = [it for it in items if it['user'] not in hold]
        tst = [it for it in items if it['user'] in hold]
        m = train(trn, classes, epochs=epochs, seed=fi)
        pr = predict(m, tst)
        for it in tst:
            store['oof|' + it['qa_path']] = pr[it['qa_path']]
            c += int(classes[int(np.argmax(pr[it['qa_path']]))] == it['action']); n += 1
        print(f'  fold {fi} top1 {c/n:.4f}', flush=True)
    print(f'  OOF HARn clip action top-1 {c}/{n} = {c/n:.4f}  (aggregate-feature model: 0.2185)')
    # full model -> test HARn clips
    tp = [p for p in te[te.source == 'HARn'].path.unique()]
    mi = meta.set_index('qa_path')
    tpaths = [p.split('/')[-2] if False else p for p in tp]
    clips = [p for p in mi.index if str(p).startswith('LM_test_')
             and int(str(p).split('_')[-1]) < 65]
    m = train(items, classes, epochs=epochs, seed=99)
    pr = predict(m, infer_set(meta, clips))
    for k, v in pr.items():
        store['test|' + k] = v
    np.savez_compressed(os.path.join(ROOT, 'champ', 'harn_clf.npz'),
                        classes=np.array(classes), **store)
    print('cached', len(store))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 45)
