"""Which protocol slots does a two-clip session block actually contain?

Measured residual cluster (5-fold pseudo-test with 38% of held-out sessions thinned to two
trials, matching the real test mix of 34 triples + 21 pairs):

    emotion accuracy, inferred block size 2 : 143/196 = 0.7296
    emotion accuracy, inferred block size 3 : 468/522 = 0.8966

The candidate manner set is still right in a pair block (|I| = 3 for 21 of the 23 test pairs,
and P(|I|>=3 | same session) = 0.990 on training data), so the extra 16.7pp of error is not
about recognising manners.  It comes from an extra latent the triple case does not have:
WHICH two of the three protocol slots (trial 1 = slow, 2 = neutral, 3 = fast) are present.
solve_emotion marginalises over that latent with a flat prior.  This module supplies a
learned one, from the two clips' physical features -- chiefly duration, since the training
frame counts fall 222 / 178 / 140 with trial index.

Fitted subject-disjoint; features are exactly the ones solve_emotion already has in hand.
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import PHYS, block_features, gt_letters
from sklearn.ensemble import HistGradientBoostingClassifier

NCLS = [(0, 1), (0, 2), (1, 2)]

# Which slot-set model backs the CHAMP_W_SLOT hook.
#   'gbm'   the HistGradientBoosting map over all of block_features + clock columns (below).
#           Measured: emotion at inferred block size 2 0.7296 -> 0.7551, OOF total +5.
#   'clock' champ/clockslot.py, an explicit generative gap / duration-ratio likelihood.
# Either way CHAMP_W_SLOT=0 leaves solve_emotion on the champion path untouched.
MODEL = os.environ.get('CHAMP_SLOT_MODEL', 'gbm')


def _row(bf, k, aux=None, paths=None):
    """Feature row from the two clips' block_features, as solve_emotion computes them,
    plus the two test-visible clock/length columns that carry most of the signal:
    the recording gap (a skipped trial doubles it) and the raw frame counts."""
    f = {}
    for c in PHYS:
        a0, a1 = bf[0].get(f'a_{c}', np.nan), bf[1].get(f'a_{c}', np.nan)
        f[f'a0_{c}'] = a0; f[f'a1_{c}'] = a1
        f[f'd_{c}'] = a0 - a1
        f[f'r_{c}'] = (a0 - a1) / (abs(a0) + abs(a1) + 1e-9)
    f['k'] = k
    if aux is not None and paths is not None and len(paths) == 2:
        t0 = [aux.get(p, (np.nan, np.nan))[0] for p in paths]
        nf = [aux.get(p, (np.nan, np.nan))[1] for p in paths]
        f['gap'] = t0[1] - t0[0]
        f['loggap'] = np.log(abs(t0[1] - t0[0]) + 1.0)
        f['nf0'] = nf[0]; f['nf1'] = nf[1]
        f['nf_ratio'] = nf[0] / max(nf[1], 1.0)
        f['nf_sum'] = nf[0] + nf[1]
    return f


def make_aux(meta):
    return {r.qa_path: (r.t0, r.nf) for r in meta.itertuples()}


def fit(tr, meta, hold_users=()):
    if MODEL == 'clock':
        import clockslot as CS
        return CS.fit(tr, meta, hold_users)
    return fit_gbm(tr, meta, hold_users)


def predict(sp, bf, k, slots, paths=None, pool=None):
    if MODEL == 'clock':
        import clockslot as CS
        return CS.predict(sp, bf, k, slots, paths, pool)
    return predict_gbm(sp, bf, k, slots, paths)


def fit_gbm(tr, meta, hold_users=()):
    mfeat = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)}
             for r in meta.itertuples()}
    aux = make_aux(meta)
    e = tr[(tr.category == 'emotion') & (~tr.user.isin(list(hold_users)))]
    X, y = [], []
    for (u, a, b), g in e.groupby(['user', 'aa', 'bb']):
        g = g.sort_values('cc')
        rows = [r for _, r in g.iterrows()]
        if len(rows) < 3:
            continue
        for i, j in itertools.combinations(range(3), 2):
            blk = [rows[i].path, rows[j].path]
            bf = block_features(blk, mfeat, 3)      # slots = 3, exactly as at inference
            X.append(_row(bf, 3, aux, blk)); y.append(NCLS.index((i, j)))
    Xd = pd.DataFrame(X)
    Xd = Xd.loc[:, Xd.notna().any()]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xd.to_numpy(float), np.array(y))
    return dict(clf=clf, cols=list(Xd.columns), classes=list(clf.classes_), aux=aux)


def predict_gbm(sp, bf, k, slots, paths=None):
    """-> {present_tuple: log P}.  Only defined for the k=2, slots=3 case."""
    if sp is None or k != 2 or slots != 3:
        return None
    X = pd.DataFrame([_row(bf, slots, sp.get('aux'), paths)])\
        .reindex(columns=sp['cols']).to_numpy(float)
    p = sp['clf'].predict_proba(X)[0]
    out = {}
    for ci, c in enumerate(sp['classes']):
        out[NCLS[c]] = float(np.log(max(p[ci], 1e-6)))
    for t in NCLS:
        out.setdefault(t, float(np.log(1e-6)))
    return out
