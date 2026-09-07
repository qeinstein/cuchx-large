"""Train the dense model 5-fold subject-disjoint, cache OOF frame log-probs, then
also train on all subjects and cache test log-probs."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_all
import dense as D
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(epochs=45, nfold=5, seeds=(0,)):
    tr, te, meta = load_all()
    items = D.build_dense(meta)
    print('dense items', len(items), 'feat dim', items[0]['X'].shape[1])
    users = sorted({it['user'] for it in items})
    store = {}
    fa_c = fa_n = 0
    for fi, hold in enumerate(folds(users, nfold)):
        trn = [it for it in items if it['user'] not in hold]
        tst = [it for it in items if it['user'] in hold]
        acc = None
        for sd in seeds:
            m = D.train_dense(trn, epochs=epochs, seed=100 * sd + fi, verbose=False)
            pr = D.predict(m, [dict(qa_path=i['qa_path'], X=i['X'], F=i['F']) for i in tst])
            acc = pr if acc is None else {k: np.logaddexp(acc[k], pr[k]) - np.log(2) for k in pr}
        c = n = 0
        for it in tst:
            store['oof|' + it['qa_path']] = acc[it['qa_path']]
            c += int((acc[it['qa_path']].argmax(0) == it['y']).sum()); n += len(it['y'])
        fa_c += c; fa_n += n
        print(f'  fold {fi} frame acc {c/n:.4f}', flush=True)
    print(f'  OOF frame accuracy {fa_c}/{fa_n} = {fa_c/fa_n:.4f}')
    # full-data model -> test
    ti = D.infer_items(meta, kinds=('test',))
    acc = None
    for sd in seeds:
        m = D.train_dense(items, epochs=epochs, seed=999 + sd, verbose=False)
        pr = D.predict(m, ti)
        acc = pr if acc is None else {k: np.logaddexp(acc[k], pr[k]) - np.log(2) for k in pr}
    for k, v in acc.items():
        store['test|' + k] = v
    out_name = os.environ.get('CHAMP_DENSE_OUT', 'dense_logits.npz')
    np.savez_compressed(os.path.join(ROOT, 'champ', out_name), **store)
    print('cached', len(store), 'logit arrays')


if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 45,
         seeds=tuple(range(int(sys.argv[2]) if len(sys.argv) > 2 else 1)))
