"""DINO test inference: HAU action-presence head -> train-validated test flips.

Protocol (all subject-disjoint, vectorized sklearn, CPU minutes):
  1. Train-HAU whole-clip descriptors: concat(pool(full), pool(crop1.6)).
  2. OOF multilabel presence via kernel's oof_multilabel (5-fold by user).
  3. VALIDATE on train HAU single/multi/combination questions: DINO guess from
     OOF presence vs champion v8 OOF on identical rows. Gate: DINO must beat
     v8 on a category before any test flip in that category ships.
  4. TEST: refit OvR on all train-HAU, presence for test clips, per-question
     DINO guess + margin, disagreements vs 334 champion ranked by margin.

Usage:
  python3 research/t2_dino/test_infer.py --dir /tmp/dino_out --mod depth
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/fluxx/Workspace/cuchx-large/research/t2_dino')
from run_stage2_local import load_kernel_lib, load_memmap  # noqa: E402

ROOT = '/home/fluxx/Workspace/cuchx-large'
K = None


def clip_descriptors(Z, clips):
    """Vectorized whole-clip concat pools. Returns (X, ok_mask)."""
    rows, ok = [], []
    for c in clips:
        kf, kc = c + '|full', c + '|crop1.6'
        if kf not in Z or kc not in Z:
            ok.append(False)
            continue
        rows.append(np.concatenate([K.pool(np.asarray(Z[kf], np.float32)),
                                   K.pool(np.asarray(Z[kc], np.float32))]))
        ok.append(True)
    return (np.stack(rows) if rows else None), np.array(ok)


def guess_single(row, pres, actions):
    oo = [str(getattr(row, L)).strip() for L in 'ABCD']
    try:
        sc = [pres[actions.index(o)] if o in actions else -1 for o in oo]
    except ValueError:
        return None, 0.0
    bi = int(np.argmax(sc))
    srt = sorted(sc, reverse=True)
    return 'ABCD'[bi], float(srt[0] - srt[1]) if len(srt) > 1 else 0.0


def guess_multi(row, pres, actions, t=0.5):
    oo = [str(getattr(row, L)).strip() for L in 'ABCD']
    out = ''.join(L for L, o in zip('ABCD', oo)
                  if o in actions and pres[actions.index(o)] >= t)
    return out if out else None


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--mod', default='depth', choices=['depth', 'thermal'])
    ap.add_argument('--t', type=float, default=0.5)
    a = ap.parse_args()
    K = load_kernel_lib()

    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    tr = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
    te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
    v8 = pd.read_csv(os.path.join(ROOT, 'research', 'oof_v8repair.csv'))
    voc = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))
    actions = sorted(set(voc['HARN2HAU'].values()))
    clipsets = K.hau_clip_action_sets(tr)
    champ = pd.read_csv(os.path.join(
        ROOT, 'submissions', 'submission_097660_334of342_CHAMPION.csv')
        ).set_index('qa_id').prediction.astype(str)

    fn = 'dino_s_%s.npz' % a.mod
    Z = load_memmap(a.dir, fn)
    print('%s: %d keys' % (fn, len(Z)), flush=True)

    # ---- train descriptors + labels
    hau = meta[meta.kind == 'train_hau'].reset_index(drop=True)
    Xtr, ok = clip_descriptors(Z, hau.qa_path.tolist())
    hau = hau[ok].reset_index(drop=True)
    Y = np.array([[1 if x in clipsets.get(c, set()) else 0 for x in actions]
                  for c in hau.qa_path])
    print('train-HAU with embeddings: %d/%d' % (len(hau), ok.size), flush=True)

    # ---- OOF presence (subject-disjoint)
    P, rep = K.oof_multilabel(Xtr, hau.user.to_numpy(), hau.qa_path.to_numpy(), actions)
    print('pool OOF macro-mAP %.4f micro-AP %.4f'
          % (rep['macro_mAP'], rep['micro_AP']), flush=True)
    pmap = dict(zip(hau.qa_path, P))

    # ---- train-side validation vs v8 on identical rows
    v8m = v8.set_index('qa_id')
   Repo = tr.copy()
    Repo['clip'] = Repo['path']
    val = []
    for cat in ('single', 'multi', 'combination'):
        sub = Repo[(Repo.source == 'HAU') & (Repo.category == cat)]
        rows = [r for r in sub.itertuples() if r.clip in pmap and r.qa_id in v8m.index]
        if not rows:
            print('%s: no validatable rows' % cat, flush=True)
            continue
        if cat == 'single':
            dg = [guess_single(r, pmap[r.clip], actions)[0] == str(r.answer) for r in rows]
        else:
            dg = [guess_multi(r, pmap[r.clip], actions, a.t) == str(r.answer) for r in rows]
        vg = [bool(v8m.loc[r.qa_id, 'correct']) for r in rows]
        dino_c, v8_c, n = sum(dg), sum(vg), len(rows)
        print('%s: DINO %d/%d=%.4f  v8 %d/%d=%.4f  delta=%+d'
              % (cat, dino_c, n, dino_c / n, v8_c, n, v8_c / n, dino_c - v8_c), flush=True)
        # DINO-wins / v8-wins breakdown
        dw = sum(1 for d, v in zip(dg, vg) if d and not v)
        vw = sum(1 for d, v in zip(dg, vg) if v and not d)
        print('   DINO-only-wins=%d v8-only-wins=%d both-wrong=%d'
              % (dw, vw, sum(1 for d, v in zip(dg, vg) if not d and not v)), flush=True)
        val.append({'cat': cat, 'dino': dino_c, 'v8': v8_c, 'n': n,
                    'dino_only': dw, 'v8_only': vw})

    # ---- TEST: refit on all train, predict test clips
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xt = sc.transform(Xtr)
    tclips = sorted({c for c in (k.split('|')[0] for k in Z if k.endswith('|full'))})
    # keep only test clips (LM_test_* present in test_qa paths)
    te_clips = sorted({p.split('/')[1] for p in te.path if '/LM_test_' in p})
    tclips = [c for c in tclips if c in set(te_clips)]
    Xte, okte = clip_descriptors(Z, tclips)
    tclips = [c for c, o in zip(tclips, okte) if o]
    Xteh = sc.transform(Xte)
    Pte = np.zeros((len(tclips), len(actions)))
    for j in range(len(actions)):
        if Y[:, j].sum() < 3 or Y[:, j].sum() == len(Y):
            Pte[:, j] = Y[:, j].mean()
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xt, Y[:, j])
        Pte[:, j] = clf.predict_proba(Xteh)[:, 1]
    tmap = dict(zip(tclips, Pte))
    print('test clips with presence: %d/%d' % (len(tclips), len(te_clips)), flush=True)
    del Z
    gc.collect()

    # ---- per-question DINO guesses + disagreements vs 334
    out = []
    for r in te.itertuples():
        if r.source != 'HAU' or r.category not in ('single', 'multi', 'combination'):
            continue
        parts = str(r.path).split('/')
        clip = parts[1] if len(parts) > 1 else None
        if clip not in tmap:
            continue
        pres = tmap[clip]
        if r.category == 'single':
            g, m = guess_single(r, pres, actions)
        else:
            g, m = guess_multi(r, pres, actions, a.t), 0.0
            if g is not None:
                oo = [str(getattr(r, L)).strip() for L in 'ABCD']
                incl = [pres[actions.index(o)] for L, o in zip('ABCD', oo)
                        if L in g and o in actions]
                excl = [pres[actions.index(o)] for L, o in zip('ABCD', oo)
                        if L not in g and o in actions]
                m = (min(incl) - a.t) if incl else 0.0
                if excl:
                    m = min(m, a.t - max(excl))
        if g is None:
            continue
        c = champ.get(r.qa_id, '')
        out.append({'qa_id': r.qa_id, 'cat': r.category, 'clip': clip,
                    'champ': c, 'dino': g, 'margin': round(float(m), 4),
                    'disagree': int(g != c)})
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(ROOT, 'research', 't2_dino', 'test_dino_guesses.csv'), index=False)
    print('wrote test_dino_guesses.csv (%d rows)' % len(d), flush=True)
    if len(d):
        print(d.groupby(['cat', 'disagree']).size().to_string(), flush=True)
        dis = d[d.disagree == 1].sort_values('margin', ascending=False)
        print('--- top disagreements by margin ---', flush=True)
        print(dis.head(25).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
