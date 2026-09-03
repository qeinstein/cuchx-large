"""Train the candidate-conditioned localizer 5-fold subject-disjoint, cache OOF + test
per-action activation maps, and immediately report sequence accuracy from onsets."""
import os, sys, json, itertools
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D, loc as L
from core import load_all, opts
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def onsets_from_map(S, cls4, thr=0.5):
    """S: NA x T activations in [0,1].  Returns onset / centroid / peak per candidate."""
    T = S.shape[1]
    o = {}
    for c in cls4:
        p = S[c]
        mx = max(p.max(), 1e-6)
        w = p / (p.sum() + 1e-9)
        o[c] = dict(onset=float(np.argmax(p >= thr * mx)) / max(1, T - 1),
                    centroid=float((w * np.arange(T)).sum()) / max(1, T - 1),
                    peak=float(np.argmax(p)) / max(1, T - 1),
                    mx=float(mx))
    return o


def main(epochs=70, nfold=5, out='loc_maps.npz'):
    tr, te, meta = load_all()
    items = L.build(meta, tr)
    print('localizer items', len(items), 'dim', items[0]['X'].shape[1])
    ign = np.mean([(it['M'] == L.IGNORE).any(1).sum() for it in items])
    print('mean #actions ignored per clip (present but unsegmented): %.2f' % ign)
    users = sorted({it['user'] for it in items})
    seq = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
    seqby = {r.path: r for _, r in seq.iterrows()}
    store = {}
    ex = {k: [0, 0] for k in ('onset', 'centroid', 'peak')}
    pw = [0, 0]
    for fi, hold in enumerate(folds(users, nfold)):
        trn = [it for it in items if it['user'] not in hold]
        tst = [it for it in items if it['user'] in hold]
        m = L.train(trn, epochs=epochs, seed=fi)
        pr = L.predict(m, [dict(qa_path=i['qa_path'], X=i['X'], F=i['F']) for i in tst])
        for it in tst:
            S = pr[it['qa_path']]
            store['oof|' + it['qa_path']] = S
            r = seqby.get(it['qa_path'])
            if r is None:
                continue
            oo = opts(r)
            if not all(o in L.OPT2CLS for o in oo):
                continue
            cls4 = [L.OPT2CLS[o] for o in oo]
            o = onsets_from_map(S, cls4)
            for key in ex:
                order = sorted(range(4), key=lambda i: o[cls4[i]][key])
                ex[key][0] += int(''.join('ABCD'[i] for i in order) == str(r['answer']))
                ex[key][1] += 1
            tpos = {Lb: i for i, Lb in enumerate(str(r['answer']))}
            for a, b in itertools.combinations(range(4), 2):
                truth = tpos['ABCD'[a]] < tpos['ABCD'[b]]
                pw[0] += int((o[cls4[a]]['onset'] < o[cls4[b]]['onset']) == truth); pw[1] += 1
        print(f'  fold {fi} done', flush=True)
    for key, (c, n) in ex.items():
        print(f'  SEQUENCE exact by {key:9s}: {c}/{n} = {c/max(1,n):.4f}')
    print(f'  pairwise (raw onset order)   : {pw[0]}/{pw[1]} = {pw[0]/max(1,pw[1]):.4f}')
    print('  checkpoint: 176/308 = 0.5714 | ceiling from true onsets 173/185 = 0.9351')
    ti = L.infer_items(meta, kinds=('test',))
    m = L.train(items, epochs=epochs, seed=99)
    pr = L.predict(m, ti)
    for k, v in pr.items():
        store['test|' + k] = v
    np.savez_compressed(os.path.join(ROOT, 'champ', out), **store)
    print('cached', len(store), 'maps ->', out)


if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 70)
