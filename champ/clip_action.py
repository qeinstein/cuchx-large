"""Clip-level action-presence head on top of frozen dense logits.

The session pool decoder is useful for structure, but it can fail when a block is
short or grouping is uncertain.  This head learns the simpler question-level
task directly: given a clip and one candidate action, estimate whether that
action is present.  It uses only dense presence statistics plus visible option
metadata at inference time.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D
import decode as DC

ACTCATS = ('single', 'multi', 'combination', 'sequence')


def _features(action, category, option_index, option_actions, stats):
    c = D.OPT2I.get(action)
    f = dict(option_index=option_index, option_n=len(option_actions),
             option_frac=len(option_actions) / 4.0)
    for i, cat in enumerate(ACTCATS):
        f[f'cat_{cat}'] = int(category == cat)
    for i in range(len(D.ACTIONS)):
        f[f'act_{i}'] = int(c == i)
    def sval(name, fallback=None):
        if stats is None or c is None:
            return fallback
        if name in stats:
            return stats[name][c]
        # eval_dense.py's compact audit cache stores stats by action name.
        d = stats.get(action, {}) if isinstance(stats, dict) else {}
        if name in d:
            return d[name]
        aliases = {'mx': 'score', 'topk': 'score'}
        if aliases.get(name) in d:
            return d[aliases[name]]
        return fallback

    if c is None or stats is None or sval('mx') is None:
        for name in ('mx', 'mean', 'topk', 'frac', 'lmx', 'lmean',
                     'centroid', 'peak', 'T'):
            f[name] = np.nan
    else:
        for name in ('mx', 'mean', 'topk', 'frac', 'lmx', 'lmean',
                     'centroid', 'peak', 'T'):
            value = sval(name)
            f[name] = float(value) if value is not None else np.nan
        if np.isfinite(f['lmx']) and np.isfinite(f['lmean']):
            f['logit_gap'] = float(f['lmx'] - f['lmean'])
        else:
            # Compact caches contain probability max/mean, which is enough for
            # a monotone confidence-gap feature after logit transformation.
            f['logit_gap'] = float(
                np.log(np.clip(f['mx'], 1e-5, 1 - 1e-5) /
                       np.clip(1 - f['mx'], 1e-5, 1)) -
                np.log(np.clip(f['mean'], 1e-5, 1 - 1e-5) /
                       np.clip(1 - f['mean'], 1e-5, 1))) if np.isfinite(f['mean']) else np.nan
    return f


def _selected_actions(r):
    out = set()
    for letter in str(r.answer):
        if letter in 'ABCD':
            out.update(a.strip() for a in str(getattr(r, letter)).split(',') if a.strip())
    return out


def build_training_rows(tr, statcache):
    """Build candidate action labels, skipping unknown combination negatives."""
    X, y, w = [], [], []
    q = tr[(tr.source == 'HAU') & tr.category.isin(ACTCATS)]
    for r in q.itertuples():
        selected = _selected_actions(r)
        stats = statcache.get(r.path)
        for oi, letter in enumerate('ABCD'):
            actions = [a.strip() for a in str(getattr(r, letter)).split(',') if a.strip()]
            if not actions:
                continue
            # Single/multi questions give reliable positive/negative action
            # labels. Sequence gives positives but no negatives. In combination,
            # an unselected option may still share an action with the selected
            # pair, so retain positives and mark other candidates unknown.
            for action in actions:
                if r.category == 'combination' and action not in selected:
                    continue
                label = int(action in selected)
                weight = 0.25 if r.category == 'sequence' else 1.0
                X.append(_features(action, r.category, oi, actions, stats))
                y.append(label); w.append(weight)
    return pd.DataFrame(X), np.asarray(y, int), np.asarray(w, float)


def fit(tr, statcache):
    X, y, w = build_training_rows(tr, statcache)
    X = X.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in X.columns if np.isfinite(
        pd.to_numeric(X[c], errors='coerce').to_numpy(float)).any()]
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=int(os.environ.get('CHAMP_CLIP_ACTION_MAX_ITER', '220')),
        learning_rate=0.055, max_depth=3, min_samples_leaf=25,
        l2_regularization=2.0, random_state=0)
    clf.fit(X[cols].to_numpy(float), y, sample_weight=w)
    return clf, cols


def score_question(r, stats, clf, cols):
    rows, labels = [], []
    for oi, letter in enumerate('ABCD'):
        actions = [a.strip() for a in str(getattr(r, letter)).split(',') if a.strip()]
        # Combination options are scored as the joint presence of all atoms.
        vals = []
        for action in actions:
            f = _features(action, r.category, oi, actions, stats)
            rows.append(f); labels.append((oi, action))
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).reindex(columns=cols)
    p = np.clip(clf.predict_proba(X.to_numpy(float))[:, 1], 1e-5, 1 - 1e-5)
    logit = np.log(p / (1 - p))
    byopt = defaultdict(float)
    for (oi, _action), s in zip(labels, logit):
        byopt[oi] += float(s)
    return byopt, p


from collections import defaultdict


def predict_question(r, stats, clf, cols, multi_threshold=0.5):
    """Return a valid A-D answer using calibrated action probabilities."""
    # Recompute per-action probabilities so multi can threshold individual atoms.
    rows, labels = [], []
    for oi, letter in enumerate('ABCD'):
        actions = [a.strip() for a in str(getattr(r, letter)).split(',') if a.strip()]
        for action in actions:
            rows.append(_features(action, r.category, oi, actions, stats))
            labels.append((oi, action))
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).reindex(columns=cols)
    p = np.clip(clf.predict_proba(X.to_numpy(float))[:, 1], 1e-5, 1 - 1e-5)
    op_score = defaultdict(float)
    op_hit = defaultdict(bool)
    for (oi, _action), prob in zip(labels, p):
        op_score[oi] += float(np.log(prob / (1 - prob)))
        op_hit[oi] = op_hit[oi] or (prob >= multi_threshold)
    if r.category in ('single', 'combination'):
        oi = max(range(4), key=lambda x: op_score[x])
        return 'ABCD'[oi]
    if r.category == 'multi':
        hit = [oi for oi in range(4) if op_hit[oi]]
        if not hit:
            hit = [max(range(4), key=lambda x: op_score[x])]
        return ''.join('ABCD'[oi] for oi in hit)
    return None
