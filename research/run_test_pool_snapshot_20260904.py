import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'champ'))
import pipeline as P

tr, te, meta = P.load_all()
P.caches(meta)
ctx = P.fit_all(tr, meta, [], split_train='oof')
vis = P.real_test_view(te)
pred, blocks, pool_of, diag = P.solve(vis, ctx, 'test')

def plain(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x

out = {
    'pred': pred,
    'blocks': blocks,
    'pool_of': {int(k): sorted(v) for k, v in pool_of.items()},
    'diag': plain(dict(diag)),
}
with open(os.path.join(os.path.dirname(__file__), 'test_pool_snapshot.pkl'), 'wb') as f:
    pickle.dump(out, f, protocol=4)
print('blocks', len(blocks), 'pools', len(pool_of))
print('diag', plain(dict(diag)))
print('snapshot written')
