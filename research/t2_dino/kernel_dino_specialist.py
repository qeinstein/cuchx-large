"""DINO-family action-perception specialist: Kaggle GPU experiment.

Stage 1 (extraction, GPU): frozen DINOv2 (ViT-S/14 primary, ViT-B/14 ablation)
  frame embeddings on Depth_Color + Thermal, full-frame + 1.6x actor crop
  (research/t2_crop box method), train-HAU + test clips.
Stage 2 (heads, GPU/CPU): lightweight temporal heads trained SUBJECT-DISJOINT
  for (a) HARn action identity (b) HAU clip action-set presence, plus a
  late-fusion test vs the skeleton+IMU path (harn_clf.npz OOF).

Inputs (attached datasets, no upload needed):
  /kaggle/input/cuchx-videos-20260914       HAU/* + large_model_track_test/*
  /kaggle/input/cuchx-dense-inputs-20260914 meta.csv, training_qa.csv, vocab.json,
                                            skel_seq.npz, imu_seq.npz,
                                            harn_clf.npz, make_crops.py
Outputs (/kaggle/working):
  dino_s_depth.npz    keys <qa_path>|full, <qa_path>|crop1.6  (T,384) float16
  dino_s_thermal.npz  same for Thermal (25 fps; per-clip n/fps in manifest)
  dino_b_depth.npz    ViT-B/14 Depth full-stream only (T,768) float16
  oof_harn_action.npz per-stream OOF log-probs + per-fold top-1 + per-action table
  oof_hau_pool.npz    HAU clip action-set presence OOF (AP per action, mAP)
  fusion_curve.csv    DINO-vs-skeleton+IMU late-fusion weight sweep (HARn top-1)
  manifest.json       backbone availability, fps, coverage, alignment check, timings

Usage:
  STAGE=pilot python3 kernel_dino_specialist.py   # ~15 min smoke + throughput
  STAGE=full  python3 kernel_dino_specialist.py   # full experiment, budget <3.5h

Requires enable_internet=true (torch.hub model download).
"""
import os
import sys
import time
import json
import traceback

import numpy as np
import pandas as pd

def _resolve_root(marker):
    import pathlib
    hits = sorted(str(p.parent) for p in
                  pathlib.Path('/kaggle/input').rglob(marker))
    if not hits:
        raise FileNotFoundError(f'marker {marker} not mounted')
    # prefer the shallowest mount (dataset root, not nested copy)
    return sorted(hits, key=len)[0]


def _resolve_vid():
    import pathlib
    for p in sorted(pathlib.Path('/kaggle/input').rglob('HAU')):
        if p.is_dir() and (p.parent / 'large_model_track_test').exists():
            return str(p.parent)
    for p in sorted(pathlib.Path('/kaggle/input').rglob('HAU')):
        if p.is_dir():
            return str(p.parent)
    raise FileNotFoundError('video tree (HAU/) not mounted')


VID = _resolve_vid()
INP = _resolve_root('meta.csv')
OUT = '/kaggle/working'
STAGE = os.environ.get('STAGE', 'pilot').strip().lower()
BUDGET_MIN = float(os.environ.get('BUDGET_MIN', '210'))
BATCH = 64
RES = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

sys.path.insert(0, INP)
try:
    from make_crops import (sample_frames, compute_box, boxes_for_margins,
                            crop_frame)
    HAVE_BOX = True
except Exception as e:
    print('WARN: make_crops import failed (%s); crops fall back to center'
          % e, flush=True)
    HAVE_BOX = False

MANIFEST = {'stage': STAGE, 'backbones': {}, 'streams': {},
            'alignment': {}, 'timings': {}, 'fallbacks': [], 'errors': []}


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ backbones
def load_backbone(entry, dev):
    """Load one torch.hub DINOv2 entrypoint; record outcome in manifest."""
    import torch
    t0 = time.time()
    try:
        m = torch.hub.load('facebookresearch/dinov2', entry, pretrained=True,
                           trust_repo=True, verbose=False)
        m = m.eval().to(dev)
        dim = {'dinov2_vits14': 384, 'dinov2_vitb14': 768,
               'dinov2_vitl14': 1024, 'dinov2_vitg14': 1536}[entry]
        MANIFEST['backbones'][entry] = {'ok': True,
                                        'secs': round(time.time() - t0, 1),
                                        'dim': dim}
        log('  backbone %s OK (dim %d, %.0fs)' % (entry, dim, time.time() - t0))
        return m, dim
    except Exception as e:
        MANIFEST['backbones'][entry] = {'ok': False, 'err': str(e)[:300]}
        log('  backbone %s FAILED: %s' % (entry, str(e)[:200]))
        return None, 0


def probe_dinov3():
    """Read-only availability probe for DINOv3 (no weights fetched on failure)."""
    import torch
    info = {'torch_hub_list': None, 'err': None}
    try:
        entries = torch.hub.list('facebookresearch/dinov3', trust_repo=True)
        info['torch_hub_list'] = [e for e in entries if 'dino' in e.lower()]
        log('  dinov3 hub entries: %s' % info['torch_hub_list'])
    except Exception as e:
        info['err'] = str(e)[:300]
        log('  dinov3 hub probe failed (expected if offline/unsupported): %s'
            % str(e)[:150])
    MANIFEST['backbones']['dinov3_probe'] = info
    return info


# ------------------------------------------------------------------ video paths
def build_todo(meta):
    """(qa_path, kind, mod, mp4) rows for every clip x modality present."""
    todo = []
    for r in meta.itertuples():
        if r.kind == 'train_hau':
            base = os.path.join(VID, r.qa_path)
        elif r.kind == 'test':
            base = os.path.join(VID, 'large_model_track_test', r.qa_path)
        else:
            continue
        for mod in ('Depth_Color', 'Thermal'):
            p = os.path.join(base, mod, mod + '.mp4')
            if os.path.exists(p):
                todo.append((r.qa_path, r.kind, mod, p))
    # test dirs on disk but absent from meta (11 known): encode under bare dir id
    have = {t[0] for t in todo if t[1] == 'test'}
    tdir = os.path.join(VID, 'large_model_track_test')
    if os.path.isdir(tdir):
        for d in sorted(os.listdir(tdir)):
            if d in have or not os.path.isdir(os.path.join(tdir, d)):
                continue
            for mod in ('Depth_Color', 'Thermal'):
                p = os.path.join(tdir, d, mod, mod + '.mp4')
                if os.path.exists(p):
                    todo.append((d, 'test', mod, p))
    return todo


def clip_box(mp4, mod, W, H):
    """1.6x actor box via t2_crop, else center-square fallback."""
    if HAVE_BOX:
        try:
            frames = sample_frames(mp4, 12)
            info = compute_box(frames, mod)
            return boxes_for_margins(info['box'], (1.6,), W, H)[1.6], 'fused'
        except Exception as e:
            MANIFEST['errors'].append('box %s: %s' % (mp4, str(e)[:120]))
    s = int(0.7 * min(W, H))
    return [ (W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2 ], 'center'


# ------------------------------------------------------------------ extraction
def encode_clip(model, mp4, box, dev, mean, std, streams=('full', 'crop')):
    """Stride-1 COLOR embedding of one clip. Returns {stream: (T,D) f16}."""
    import cv2
    import torch
    cap = cv2.VideoCapture(mp4)
    bufs = {s: [] for s in streams}
    feats = {s: [] for s in streams}
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if 'full' in bufs:
            bufs['full'].append(cv2.resize(fr, (RES, RES),
                                           interpolation=cv2.INTER_AREA))
        if 'crop' in bufs:
            bufs['crop'].append(cv2.resize(crop_frame(fr, box), (RES, RES),
                                           interpolation=cv2.INTER_AREA))
        if len(bufs[streams[0]]) == BATCH:
            for s in streams:
                x = torch.from_numpy(np.stack(bufs[s])).to(dev).float()
                x = x.div_(255.0)
                x = torch.flip(x, dims=[3]).permute(0, 3, 1, 2).contiguous()
                x = (x - mean) / std
                with torch.no_grad():
                    feats[s].append(model(x).float().cpu().numpy())
                bufs[s] = []
    cap.release()
    out = {}
    for s in streams:
        if bufs[s]:
            import torch as _t
            x = _t.from_numpy(np.stack(bufs[s])).to(dev).float().div_(255.0)
            x = _t.flip(x, dims=[3]).permute(0, 3, 1, 2).contiguous()
            x = (x - mean) / std
            with _t.no_grad():
                feats[s].append(model(x).float().cpu().numpy())
        out[s] = (np.concatenate(feats[s], 0).astype(np.float16)
                  if feats[s] else None)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return out, n, float(fps)


def run_extraction(todo, model, dev, tag, streams, store, t_start):
    import torch
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    nfr = 0
    for i, (qp, kind, mod, mp4) in enumerate(todo):
        if (time.time() - t_start) / 60 > BUDGET_MIN:
            MANIFEST['fallbacks'].append('time budget hit during %s at %d/%d'
                                         % (tag, i, len(todo)))
            break
        need = ['%s|%s' % (qp, s if s == 'full' else 'crop1.6') for s in streams]
        if all(k in store and mod in store[k + '|mod'] for k in need):
            continue
        import cv2
        cap = cv2.VideoCapture(mp4)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        box, src = clip_box(mp4, mod, W, H)
        try:
            out, n, fps = encode_clip(model, mp4, box, dev, mean, std, streams)
        except Exception as e:
            MANIFEST['errors'].append('encode %s %s: %s'
                                      % (qp, mod, str(e)[:150]))
            continue
        for s in streams:
            if out[s] is not None:
                key = '%s|%s' % (qp, s if s == 'full' else 'crop1.6')
                store[key] = out[s]
                store[key + '|mod'] = np.array([mod])
                store[key + '|boxsrc'] = np.array([src])
        MANIFEST['streams'].setdefault(tag, {}).setdefault(mod, []).append(
            {'clip': qp, 'n': n, 'fps': round(fps or -1, 2), 'box': src})
        nfr += n
        if (i + 1) % 25 == 0:
            el = (time.time() - t_start) / 60
            log('  [%s] %d/%d clips %d frames %.1f fps elapsed %.1fm'
                % (tag, i + 1, len(todo), nfr, nfr / max(el * 60, 1), el))
    return nfr


# ------------------------------------------------- alignment check + HARn nest
def check_alignment(meta):
    """Verify Depth frame count == skeleton nf per clip (the prior claim)."""
    import cv2
    ok = mismatch = missing = 0
    ex = []
    for r in meta[meta.kind.isin(('train_hau', 'test'))].itertuples():
        base = (os.path.join(VID, r.qa_path) if r.kind == 'train_hau'
                else os.path.join(VID, 'large_model_track_test', r.qa_path))
        p = os.path.join(base, 'Depth_Color', 'Depth_Color.mp4')
        if not os.path.exists(p):
            missing += 1
            continue
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if np.isfinite(r.nf) and n == int(r.nf):
            ok += 1
        else:
            mismatch += 1
            if len(ex) < 10:
                ex.append({'clip': r.qa_path, 'mp4': n, 'nf': r.nf})
    MANIFEST['alignment'] = {'depth_eq_nf': ok, 'mismatch': mismatch,
                             'missing': missing, 'examples': ex}
    log('  alignment Depth frames==nf: ok=%d mismatch=%d missing=%d'
        % (ok, mismatch, missing))


def build_nest(meta):
    """HARn clip -> parent HAU clip, from timestamps + global frames only."""
    nest = {}
    hl = [(r.qa_path, r.t0, r.t1, r.f0, r.f1)
          for r in meta[meta.kind == 'train_hau'].itertuples()
          if np.isfinite(r.t0)]
    for r in meta[meta.kind == 'train_harn'].itertuples():
        if not np.isfinite(r.t0):
            continue
        c = [h for h in hl if h[1] <= r.t0 + 0.5 and h[2] >= r.t1 - 0.5
             and h[3] <= r.f0 and h[4] >= r.f1 and h[0] != r.qa_path]
        if len(c) == 1:
            nest[r.qa_path] = c[0][0]
    log('  nest: %d/%d train HARn clips uniquely nested'
        % (len(nest), (meta.kind == 'train_harn').sum()))
    return nest


def pool(A):
    """harn_dino.py pooling, verbatim (mean/max/std + thirds + |d| stats)."""
    if A is None or len(A) == 0:
        return None
    A = np.asarray(A, np.float32)
    n = len(A)
    th = [A[:max(1, n // 3)], A[max(1, n // 3):max(2, 2 * n // 3)],
          A[max(2, 2 * n // 3):]]
    dv = (np.abs(np.diff(A, axis=0)) if n > 1
          else np.zeros((1, A.shape[1]), np.float32))
    return np.concatenate([A.mean(0), A.max(0), A.std(0),
                           th[0].mean(0), th[1].mean(0), th[2].mean(0),
                           dv.mean(0), dv.max(0)])


def harn_descriptors(Z, meta, nest, stream):
    """Slice parent HAU features onto each train HARn interval, then pool."""
    mi = meta.set_index('qa_path')
    rows, y, users, paths = [], [], [], []
    for r in meta[meta.kind == 'train_harn'].itertuples():
        par = nest.get(r.qa_path)
        key = (par + '|' + stream) if par else None
        if not key or key not in Z:
            continue
        A = np.asarray(Z[key], np.float32)
        try:
            cf0, cf1 = mi.loc[r.qa_path, 'f0'], mi.loc[r.qa_path, 'f1']
            pf0 = mi.loc[par, 'f0']
        except KeyError:
            continue
        if not (np.isfinite(cf0) and np.isfinite(pf0)):
            continue
        # Depth stride-1 == global frames; other rates map fractionally
        depth_n = None
        for cand in (par + '|full', par + '|crop1.6'):
            if cand in Z and len(np.asarray(Z[cand])) > 0:
                depth_n = len(np.asarray(Z[cand]))
                break
        if depth_n and len(A) != depth_n:
            sc = len(A) / depth_n
            i0 = int(max(0, (cf0 - pf0) * sc))
            i1 = int(min(len(A), (cf1 - pf0 + 1) * sc))
        else:
            i0 = int(max(0, cf0 - pf0))
            i1 = int(min(len(A), cf1 - pf0 + 1))
        if i1 - i0 < 2:
            continue
        d = pool(A[i0:i1])
        if d is None:
            continue
        rows.append(d)
        y.append(r.action)
        users.append(r.user)
        paths.append(r.qa_path)
    if not rows:
        return None, None, None, None
    return np.stack(rows), np.array(y), np.array(users), np.array(paths)


def folds(users, n=5, seed=7):
    u = sorted(users, key=lambda s: int(s[4:]))
    rng = np.random.default_rng(seed)
    u = list(rng.permutation(u))
    return [u[i::n] for i in range(n)]


# ------------------------------------------------------------------ heads
def oof_logreg(X, y, users, paths, classes, C=1.0):
    """5-fold subject-disjoint LogReg OOF (harn_dino.py protocol)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    allu = sorted(set(users))
    store, c, n = {}, 0, 0
    per_class = {cl: [0, 0] for cl in classes}
    for fi, hold in enumerate(folds(allu, 5)):
        m = ~np.isin(users, hold)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=C, n_jobs=-1))
        clf.fit(X[m], y[m])
        lp = clf.predict_log_proba(X[~m])
        cl = list(clf.classes_)
        for p, row, t in zip(paths[~m], lp, y[~m]):
            full = np.full(len(classes), -30.0, np.float32)
            for j, cn in enumerate(cl):
                full[classes.index(cn)] = row[j]
            store['oof|' + p] = full
            pred = classes[int(np.argmax(full))]
            c += int(pred == t)
            n += 1
            per_class[t][0] += int(pred == t)
            per_class[t][1] += 1
    acc = c / max(n, 1)
    table = {k: {'correct': v[0], 'n': v[1],
                 'acc': round(v[0] / max(v[1], 1), 4)}
             for k, v in per_class.items()}
    return store, acc, (c, n), table


def attn_head_oof(seqs, y, users, paths, classes, dev, epochs=25):
    """Torch attention-pooling MLP, 5-fold subject-disjoint. seqs: list (T,D)."""
    import torch
    import torch.nn as nn
    c2i = {c: i for i, c in enumerate(classes)}
    yi = np.array([c2i[v] for v in y])
    D = seqs[0].shape[1]
    store, c, n = {}, 0, 0
    for fi, hold in enumerate(folds(sorted(set(users)), 5)):
        m = ~np.isin(users, hold)
        tr_idx = np.where(m)[0]
        te_idx = np.where(~m)[0]

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(D, 256)
                self.att = nn.Linear(256, 1)
                self.cls = nn.Linear(256, len(classes))

            def forward(self, x, mask):
                h = torch.relu(self.proj(x))
                a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
                w = torch.softmax(a, 1).unsqueeze(-1)
                return self.cls((h * w).sum(1))

        net = Attn().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()

        def batch(idxs):
            T = max(len(seqs[i]) for i in idxs)
            xb = torch.zeros(len(idxs), T, D)
            mb = torch.zeros(len(idxs), T, bool)
            for bi, i in enumerate(idxs):
                t = len(seqs[i])
                xb[bi, :t] = torch.from_numpy(np.asarray(seqs[i], np.float32))
                mb[bi, :t] = True
            return xb.to(dev), mb.to(dev)

        rng = np.random.default_rng(fi)
        net.train()
        for ep in range(epochs):
            order = rng.permutation(tr_idx)
            for s in range(0, len(order), 64):
                idx = order[s:s + 64]
                xb, mb = batch(idx)
                yb = torch.from_numpy(yi[idx]).to(dev)
                opt.zero_grad()
                loss = ce(net(xb, mb), yb)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            for s in range(0, len(te_idx), 128):
                idx = te_idx[s:s + 128]
                xb, mb = batch(idx)
                lp = torch.log_softmax(net(xb, mb), 1).cpu().numpy()
                for p, row, t in zip(paths[idx], lp, y[idx]):
                    store['oof|' + p] = row.astype(np.float32)
                    c += int(classes[int(np.argmax(row))] == t)
                    n += 1
        log('    attn fold %d done' % fi, )
    return store, c / max(n, 1), (c, n)


def hau_clip_action_sets(tr):
    """clip qa_path -> set of HAU answer texts (single/multi/combination)."""
    sets = {}
    sub = tr[(tr.source == 'HAU')
             & tr.category.isin(('single', 'multi', 'combination'))]
    for r in sub.itertuples():
        s = sets.setdefault(r.path, set())
        for L in str(r.answer):
            if L in 'ABCD':
                for p in str(getattr(r, L)).split(','):
                    s.add(p.strip())
    return sets


def oof_multilabel(X, Y, users, paths, actions):
    """One-vs-rest LogReg OOF; per-action average precision + mAP."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    allu = sorted(set(users))
    P = np.full_like(Y, np.nan, dtype=float)
    for fi, hold in enumerate(folds(allu, 5)):
        m = ~np.isin(users, hold)
        sc = StandardScaler().fit(X[m])
        Xt, Xh = sc.transform(X[m]), sc.transform(X[~m])
        for j in range(Y.shape[1]):
            if Y[m, j].sum() < 3 or Y[m, j].sum() == m.sum():
                P[~m, j] = Y[m, j].mean()
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xt, Y[m, j])
            P[~m, j] = clf.predict_proba(Xh)[:, 1]
    aps = {}
    for j, a in enumerate(actions):
        try:
            aps[a] = float(average_precision_score(Y[:, j], P[:, j]))
        except Exception:
            aps[a] = float('nan')
    valid = [v for v in aps.values() if np.isfinite(v)]
    micro = float(average_precision_score(Y.ravel(), np.nan_to_num(P).ravel()))
    return P, {'macro_mAP': float(np.mean(valid)) if valid else float('nan'),
               'micro_AP': micro, 'per_action_AP': aps}


# ------------------------------------------------------------------ main
def main():
    import torch
    t_all = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    MANIFEST['device'] = dev
    log('STAGE=%s device=%s budget=%.0fmin' % (STAGE, dev, BUDGET_MIN))
    if dev == 'cpu' and STAGE == 'full':
        log('WARN: no GPU; full stage on CPU will be very slow (still runs)')

    meta = pd.read_csv(os.path.join(INP, 'meta.csv'))
    tr = pd.read_csv(os.path.join(INP, 'training_qa.csv'))
    todo_all = build_todo(meta)
    log('clips x modalities present: %d' % len(todo_all))

    log('[1/5] backbone loads')
    probe_dinov3()
    model_s, dim_s = load_backbone('dinov2_vits14', dev)

    if STAGE == 'pilot':
        log('[2/5] PILOT extraction (2 users + 2 test clips, ViT-S)')
        pu = sorted(meta[meta.kind == 'train_hau'].user.unique())[:2]
        pq = set(meta[meta.kind == 'train_hau']
                 .groupby('user').head(4).query('user in @pu').qa_path)
        pt = sorted(meta[meta.kind == 'test'].qa_path)[:2]
        todo = [t for t in todo_all if t[0] in pq or t[0] in pt]
        store = {}
        t0 = time.time()
        nfr = run_extraction(todo, model_s, dev, 'pilot_s', ('full', 'crop'),
                             store, t0)
        fps = nfr / max(time.time() - t0, 1)
        MANIFEST['timings']['pilot_fps_2stream'] = round(fps, 1)
        proj_h = (len(todo_all) * 2 * 200 / max(fps, 1)) / 3600
        MANIFEST['timings']['projected_full_hours'] = round(proj_h, 2)
        log('  pilot: %d forwards %.0f fps -> full projection ~%.1fh'
            % (nfr, fps, proj_h))
        check_alignment(meta)
        np.savez(os.path.join(OUT, 'dino_pilot.npz'), **store)
    else:
        log('[2/5] FULL extraction ViT-S (Depth + Thermal, full + crop1.6)')
        t_enc = time.time()
        store_s = {'Depth_Color': {}, 'Thermal': {}}
        for mod in ('Depth_Color', 'Thermal'):
            todo = [t for t in todo_all if t[2] == mod]
            nfr = run_extraction(todo, model_s, dev, 'full_s_' + mod,
                                 ('full', 'crop'), store_s[mod], t_enc)
            log('  %s: %d frames' % (mod, nfr))
        np.savez_compressed(os.path.join(OUT, 'dino_s_depth.npz'), **store_s['Depth_Color'])
        np.savez_compressed(os.path.join(OUT, 'dino_s_thermal.npz'), **store_s['Thermal'])
        MANIFEST['timings']['encode_s_min'] = round((time.time() - t_enc) / 60,
                                                   1)
        check_alignment(meta)
        # ViT-B ablation on Depth full-stream (guarded by remaining budget)
        elapsed = (time.time() - t_all) / 60
        if elapsed < BUDGET_MIN - 60:
            log('[3/5] ViT-B/14 Depth full-stream ablation')
            model_b, dim_b = load_backbone('dinov2_vitb14', dev)
            if model_b is not None:
                store_b = {}
                todo = [t for t in todo_all if t[2] == 'Depth_Color']
                run_extraction(todo, model_b, dev, 'full_b_depth', ('full',),
                               store_b, t_all)
                np.savez_compressed(os.path.join(OUT, 'dino_b_depth.npz'), **store_b)
        else:
            MANIFEST['fallbacks'].append('skipped ViT-B: %.0fmin elapsed' %
                                         elapsed)

    if STAGE == 'pilot':
        json.dump(MANIFEST, open(os.path.join(OUT, 'manifest.json'), 'w'),
                  indent=1, default=str)
        log('PILOT done in %.1fmin. Inspect manifest.json, then run STAGE=full.'
            % ((time.time() - t_all) / 60))
        return

    # ---------------- stage 2: temporal heads (subject-disjoint OOF)
    log('[4/5] HARn action-identity OOF heads')
    nest = build_nest(meta)
    Zd = dict(np.load(os.path.join(OUT, 'dino_s_depth.npz')))
    Zt = dict(np.load(os.path.join(OUT, 'dino_s_thermal.npz')))
    classes = sorted(meta[meta.kind == 'train_harn'].action.unique())
    oof_store, report = {}, {}
    heads = [('depth_full', Zd, 'full', None),
             ('depth_crop', Zd, 'crop1.6', None),
             ('thermal_full', Zt, 'full', None),
             ('thermal_crop', Zt, 'crop1.6', None)]

    def get_desc(Z, stream):
        return harn_descriptors(Z, meta, nest, stream)

    descs = {}
    for name, Z, stream, _ in heads:
        X, y, u, p = get_desc(Z, stream)
        if X is None:
            log('  %s: no descriptors' % name)
            continue
        descs[name] = (X, y, u, p)
        store, acc, (c, n), table = oof_logreg(X, y, u, p, classes)
        for k, v in store.items():
            oof_store['%s|%s' % (name, k)] = v
        report[name] = {'top1': round(acc, 4), 'correct': c, 'n': n,
                        'per_action': table}
        log('  LogReg %-12s top-1 %d/%d = %.4f' % (name, c, n, acc))
    # concat full+crop per modality (matched rows only)
    for mod, Z, zname in (('depth', Zd, 'depth'), ('thermal', Zt, 'thermal')):
        a, b = zname + '_full', zname + '_crop'
        if a in descs and b in descs:
            Xa, ya, ua, pa = descs[a]
            Xb, yb, ub, pb = descs[b]
            common = sorted(set(pa) & set(pb))
            ia = {p: i for i, p in enumerate(pa)}
            ib = {p: i for i, p in enumerate(pb)}
            Xc = np.stack([np.concatenate([Xa[ia[p]], Xb[ib[p]]])
                             for p in common])  # (n, 2D); v3/v4 crashed here
            yc = np.array([ya[ia[p]] for p in common])
            uc = np.array([ua[ia[p]] for p in common])
            store, acc, (c, n), table = oof_logreg(Xc, yc, uc,
                                                  np.array(common), classes)
            for k, v in store.items():
                oof_store['%s_concat|%s' % (mod, k)] = v
            report[mod + '_concat'] = {'top1': round(acc, 4), 'correct': c,
                                      'n': n, 'per_action': table}
            log('  LogReg %-12s top-1 %d/%d = %.4f'
                % (mod + '_concat', c, n, acc))
    # attention head on best single stream (frame sequences, not pooled)
    if descs:
        best = max(descs, key=lambda k: report[k]['top1'])
        Z = Zd if best.startswith('depth') else Zt
        stream = 'full' if 'full' in best else 'crop1.6'
        mi = meta.set_index('qa_path')
        seqs, y, u, p = [], [], [], []
        for r in meta[meta.kind == 'train_harn'].itertuples():
            par = nest.get(r.qa_path)
            key = (par + '|' + stream) if par else None
            if not key or key not in Z:
                continue
            A = np.asarray(Z[key], np.float32)
            cf0, cf1, pf0 = (mi.loc[r.qa_path, 'f0'], mi.loc[r.qa_path, 'f1'],
                             mi.loc[par, 'f0'])
            if not (np.isfinite(cf0) and np.isfinite(pf0)):
                continue
            i0 = int(max(0, cf0 - pf0))
            i1 = int(min(len(A), cf1 - pf0 + 1))
            if i1 - i0 < 2:
                continue
            seqs.append(A[i0:i1])
            y.append(r.action)
            u.append(r.user)
            p.append(r.qa_path)
        if seqs:
            store, acc, (c, n) = attn_head_oof(seqs, np.array(y), np.array(u),
                                              np.array(p), classes, dev)
            for k, v in store.items():
                oof_store['attn_%s|%s' % (best, k)] = v
            report['attn_' + best] = {'top1': round(acc, 4), 'correct': c,
                                     'n': n}
            log('  Attn   %-12s top-1 %d/%d = %.4f' % (best, c, n, acc))
    np.savez_compressed(os.path.join(OUT, 'oof_harn_action.npz'),
                        classes=np.array(classes), **oof_store)
    json.dump(report, open(os.path.join(OUT, 'harn_action_report.json'), 'w'),
              indent=1, default=str)

    log('[5/5] HAU clip action-set presence + skeleton fusion test')
    vocab = json.load(open(os.path.join(INP, 'vocab.json')))
    hau2idx = {v: i for i, v in
               enumerate(sorted(set(vocab['HARN2HAU'].values())))}
    actions = sorted(hau2idx)
    clipsets = hau_clip_action_sets(tr)
    pool_store, pool_report = {}, {}
    for mod, Z in (('depth', Zd), ('thermal', Zt)):
        rows, Y, u, p = [], [], [], []
        for r in meta[meta.kind == 'train_hau'].itertuples():
            kf, kc = r.qa_path + '|full', r.qa_path + '|crop1.6'
            if kf not in Z or kc not in Z:
                continue
            d = np.concatenate([pool(np.asarray(Z[kf], np.float32)),
                                pool(np.asarray(Z[kc], np.float32))])
            s = clipsets.get(r.qa_path, set())
            rows.append(d)
            Y.append([1 if a in s else 0 for a in actions])
            u.append(r.user)
            p.append(r.qa_path)
        if not rows:
            continue
        P, rep = oof_multilabel(np.stack(rows), np.array(Y), np.array(u),
                                np.array(p), actions)
        pool_store[mod + '|P'] = P
        pool_store[mod + '|Y'] = np.array(Y)
        pool_store[mod + '|paths'] = np.array(p)
        pool_report[mod] = {k: v for k, v in rep.items()
                            if k != 'per_action_AP'}
        pool_report[mod]['per_action_AP'] = rep['per_action_AP']
        log('  pool %-7s macro-mAP %.4f micro-AP %.4f'
            % (mod, rep['macro_mAP'], rep['micro_AP']))
    np.savez_compressed(os.path.join(OUT, 'oof_hau_pool.npz'),
                        actions=np.array(actions), **pool_store)
    json.dump(pool_report, open(os.path.join(OUT, 'hau_pool_report.json'), 'w'),
              indent=1, default=str)

    # fusion: best DINO HARn OOF vs skeleton+IMU harn_clf OOF (same clips)
    try:
        hz = np.load(os.path.join(INP, 'harn_clf.npz'), allow_pickle=True)
        hcls = [str(x) for x in hz['classes']]
        hlog = {k: hz[k] for k in hz.files if '|' in k}
        best = max([k for k in report if 'correct' in report[k]],
                   key=lambda k: report[k]['top1'])
        pref = best + '|oof|'
        dlog = {k[len(pref):]: oof_store[k] for k in oof_store
                if k.startswith(pref)}
        common = [p for p in dlog if 'oof|' + p in hlog]
        if common:
            di = [classes.index(c) for c in hcls]
            rows = []
            for w in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]:
                c = n = 0
                cd = cn = 0
                for p in common:
                    dl = dlog[p][di]
                    hl = np.asarray(hlog['oof|' + p], np.float32)
                    pred = hcls[int(np.argmax((1 - w) * hl + w * dl))]
                    d_pred = hcls[int(np.argmax(dl))]
                    truth = p.split('/')[1]
                    c += int(pred == truth)
                    n += 1
                    cd += int(d_pred == truth)
                    cn += 1
                rows.append({'w_dino': w, 'fused_top1': round(c / n, 4),
                             'correct': c, 'n': n,
                             'dino_only': round(cd / cn, 4)})
            pd.DataFrame(rows).to_csv(os.path.join(OUT, 'fusion_curve.csv'),
                                     index=False)
            log('  fusion curve (best stream %s, %d clips):' % (best, len(common)))
            for r_ in rows:
                log('    w=%.1f fused=%.4f' % (r_['w_dino'], r_['fused_top1']))
    except Exception:
        MANIFEST['errors'].append('fusion:\n' + traceback.format_exc()[-800:])
        log('  fusion test failed (see manifest)')

    MANIFEST['timings']['total_min'] = round((time.time() - t_all) / 60, 1)
    json.dump(MANIFEST, open(os.path.join(OUT, 'manifest.json'), 'w'),
              indent=1, default=str)
    log('FULL done in %.1fmin' % ((time.time() - t_all) / 60))


if __name__ == '__main__':
    main()


