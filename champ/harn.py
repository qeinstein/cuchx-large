"""HARn-clip action recognition -> `single` and `object_interaction` answers.

Two evidence sources, both test-visible:
  * a segment-level action classifier over the clip's own features
  * temporal nesting: every HARn clip is a sub-interval of one HAU clip, so its action must
    lie in that session's recovered action pool, and the dense model can be read on the
    clip's exact frame interval of the parent
"""
import os, sys, json
import numpy as np, pandas as pd
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D
from build_feats import _stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOC = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))
H2S = VOC['HARN2SELF']
S2A = {v: k for k, v in H2S.items()}
H2H = VOC['HARN2HAU']

# Measured biconditional on all 562 HARn QA clips: a clip has skeleton/IMU data in the
# LMT package IFF its action is NOT one of these four (50/50 both directions, no exceptions).
# The four classes were simply never recorded with the wearable sensors, so modality
# availability alone identifies them - and that is visible at test time.
NO_SENSOR_ACTIONS = {'40_Stand_on_one_leg_(balance)', '41_Peel_fruits_with_a_knife',
                     '42_Throw_away_(rubbish)', '43_Open_the_cabinet_(to_get_things)'}

FCOLS = None


def feat_cols(meta):
    global FCOLS
    if FCOLS is None:
        FCOLS = [c for c in meta.columns
                 if c.startswith(('sk_', 'imu_', 'rad_'))] + ['nf', 'f0']
    return FCOLS


def fit_action_clf(meta, hold_users):
    """43-class action classifier on HARn segments, excluding the held-out users."""
    d = meta[(meta.kind == 'train_harn') & (~meta.user.isin(hold_users))]
    cols = feat_cols(meta)
    X = d[cols].to_numpy(float)
    y = d.action.to_numpy()
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.07, max_depth=6,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(X, y)
    return clf, cols


def fit_object_prior(tr, hold_users):
    ob = tr[(tr.source == 'HARn') & (tr.category == 'object_interaction')
            & (~tr.user.isin(hold_users))]
    pri = defaultdict(Counter); glob = Counter()
    for _, r in ob.iterrows():
        a = str(r[str(r['answer'])]).strip()
        pri[r.harn_action][a] += 1; glob[a] += 1
    return pri, glob


def nest_parent(harn_meta_row, hau_rows):
    """Unique HAU clip whose [t0,t1] and frame range contain this HARn clip."""
    t0, t1, f0, f1 = harn_meta_row
    if not np.isfinite(t0):
        return None
    c = [h for h in hau_rows
         if np.isfinite(h[1]) and h[1] <= t0 + 0.5 and h[2] >= t1 - 0.5
         and h[3] <= f0 and h[4] >= f1]
    return c[0][0] if len(c) == 1 else None
