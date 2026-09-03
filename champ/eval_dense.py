"""Subject-disjoint evaluation of the dense temporal model: frame accuracy, action presence,
and sequence-question exact-order accuracy from predicted onsets."""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all, opts, gt_letters
import dense as D
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def onsets(lp, names, smooth=5):
    """lp: NCLS x T log-probs -> per-action (presence score, onset frame, centroid)."""
    P = np.exp(lp)
    if smooth > 1 and P.shape[1] > smooth:
        k = np.ones(smooth) / smooth
        P = np.stack([np.convolve(P[c], k, mode='same') for c in range(P.shape[0])])
    T = P.shape[1]
    out = {}
    for i, a in enumerate(names):
        p = P[i]
        sc = float(p.max())
        w = p / (p.sum() + 1e-9)
        out[a] = dict(score=sc, mean=float(p.mean()),
                      onset=float(np.argmax(p >= 0.5 * p.max())) / max(1, T - 1),
                      centroid=float((w * np.arange(T)).sum()) / max(1, T - 1),
                      peak=float(np.argmax(p)) / max(1, T - 1))
    return out


def run(epochs=40, nfold=5):
    tr, te, meta = load_all()
    items = D.build_dense(meta)
    byuser = {}
    for it in items:
        byuser.setdefault(it['user'], []).append(it)
    users = sorted(byuser)
    seq = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
    seqby = {r.path: r for _, r in seq.iterrows()}
    V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
    optnames = [V[a] for a in D.ACTIONS]

    fa_n = fa_c = 0
    pres_tp = pres_fp = pres_fn = 0
    sq_n = sq_c = 0
    per = []
    allpred = {}
    for fi, hold in enumerate(folds(users, nfold)):
        trn = [it for it in items if it['user'] not in hold]
        tst = [it for it in items if it['user'] in hold]
        model = D.train_dense(trn, epochs=epochs, seed=fi, verbose=False)
        pr = D.predict(model, [dict(qa_path=it['qa_path'], X=it['X'], F=it['F']) for it in tst])
        fc = fn = 0
        for it in tst:
            lp = pr[it['qa_path']]
            yh = lp.argmax(0)
            fc += int((yh == it['y']).sum()); fn += len(it['y'])
            o = onsets(lp, optnames)
            allpred[it['qa_path']] = o
            truth = {V[a] for a in it['pool']}
            hit = {a for a in optnames if o[a]['score'] >= 0.5}
            pres_tp += len(hit & truth); pres_fp += len(hit - truth); pres_fn += len(truth - hit)
            r = seqby.get(it['qa_path'])
            if r is not None:
                oo = opts(r)
                order = sorted(range(4), key=lambda i: o[oo[i]]['centroid']
                               if oo[i] in o else 9.0)
                pred = ''.join('ABCD'[i] for i in order)
                sq_n += 1; sq_c += int(pred == str(r['answer']))
        fa_n += fn; fa_c += fc
        per.append(dict(fold=fi, frame_acc=fc / fn))
        print(f'  fold {fi}: frame acc {fc/fn:.4f}', flush=True)
    print(f'  DENSE frame accuracy {fa_c}/{fa_n} = {fa_c/fa_n:.4f}')
    pp = pres_tp / max(1, pres_tp + pres_fp); rr = pres_tp / max(1, pres_tp + pres_fn)
    print(f'  segment presence  P={pp:.3f} R={rr:.3f} F1={2*pp*rr/max(1e-9,pp+rr):.3f}')
    print(f'  SEQUENCE exact order (onset centroid, no learning) {sq_c}/{sq_n} = {sq_c/max(1,sq_n):.4f}')
    np.save(os.path.join(ROOT, 'champ', 'dense_oof.npy'), allpred, allow_pickle=True)
    return allpred


if __name__ == '__main__':
    run(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
