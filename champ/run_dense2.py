import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all
import dense as D, dense2 as D2
from pseudotest import folds
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(epochs=60, nfold=5, nseed=1, out='dense2_logits.npz'):
    tr, te, meta = load_all()
    items = D2.build(meta, tr)
    stride = int(os.environ.get('CHAMP_DENSE2_STRIDE', '1'))
    if stride > 1:
        # Temporal downsampling is an inference-equivalent research speed arm:
        # D2's losses are clip/presence based, and decode.presence_stats accepts
        # variable-length logits.  The default path remains frame-exact.
        for it in items:
            it['X'] = it['X'][::stride]
            it['F'] = it['F'][::stride]
            it['y'] = it['y'][::stride]
    print('items', len(items), 'dim', items[0]['X'].shape[1],
          'bg_trust frac %.3f' % np.mean([it['bg_trust'] for it in items]),
          'with order', sum(1 for it in items if it['order']))
    users = sorted({it['user'] for it in items})
    store = {}; fc = fn = 0
    for fi, hold in enumerate(folds(users, nfold)):
        trn = [it for it in items if it['user'] not in hold]
        tst = [it for it in items if it['user'] in hold]
        acc = None
        for s in range(nseed):
            m = D2.train(trn, epochs=epochs, seed=100 * s + fi,
                         ch=int(os.environ.get('CHAMP_DENSE2_CH', '192')),
                         w_mil=float(os.environ.get('CHAMP_DENSE2_W_MIL', '1.0')),
                         w_rank=float(os.environ.get('CHAMP_DENSE2_W_RANK', '1.0')))
            pr = D2.predict(m, [dict(qa_path=i['qa_path'], X=i['X'], F=i['F']) for i in tst])
            acc = pr if acc is None else {k: np.logaddexp(acc[k], pr[k]) - np.log(2) for k in pr}
        c = n = 0
        for it in tst:
            store['oof|' + it['qa_path']] = acc[it['qa_path']]
            msk = it['y'] != BG if False else np.ones(len(it['y']), bool)
            c += int((acc[it['qa_path']].argmax(0) == it['y']).sum()); n += len(it['y'])
        fc += c; fn += n
        print(f'  fold {fi} frame acc {c/n:.4f}', flush=True)
    print(f'  OOF frame acc {fc}/{fn} = {fc/fn:.4f}')
    ti = D.infer_items(meta, kinds=('test',))
    acc = None
    for s in range(nseed):
        m = D2.train(items, epochs=epochs, seed=999 + s,
                     ch=int(os.environ.get('CHAMP_DENSE2_CH', '192')),
                     w_mil=float(os.environ.get('CHAMP_DENSE2_W_MIL', '1.0')),
                     w_rank=float(os.environ.get('CHAMP_DENSE2_W_RANK', '1.0')))
        pr = D2.predict(m, ti)
        acc = pr if acc is None else {k: np.logaddexp(acc[k], pr[k]) - np.log(2) for k in pr}
    for k, v in acc.items():
        store['test|' + k] = v
    np.savez_compressed(os.path.join(ROOT, 'champ', out), **store)
    print('cached', len(store))


BG = D.BG
if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 60,
         nseed=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
