"""MECHANISM W - the object question already knows its action.

`Which object is the person interacting with?` is asked about a HARn clip.  Given the clip's
action, a per-action object prior is essentially deterministic: for 25_Watch_TV the answer is
'a remote' in 10/10 training questions, for 22_Turn_pages 'a documents' in 10/10, and so on.
The champion's object head nevertheless re-estimates the action from its own action classifier,
pool and DINOv2 fusion (champ/pipeline.py, the `cand_a` block) instead of reading it off the
`single` question asked about THE SAME CLIP -- a question the champion answers at 0.951.

Measured 5-fold subject-disjoint, on the 34 of 133 training object questions whose clip also
carries a single question with a usable prediction:

    champion                                27 / 34 = 0.794
    action from the single answer + prior    34 / 34 = 1.000
    flips 7, W->R 7, R->W 0, precision 1.000, net +7

All seven gains are 25_Watch_TV, where the champion scores 3/10 and predicts 'a phone' five
times while 'a remote' has prior count 10 and 'a phone' count 0 for that action.

A second, purely logical detector covers clips with no sibling single question: an option that
is NEVER the correct object for any action in the 133 training questions cannot be the answer.
On OOF the champion never picks such an option (0 rows), and where exactly one option is
ever-correct it already scores 44/44 -- so this rule only ever fires on the real test set,
where it fires twice.
"""
import os, sys, collections
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all


def fit(tr, hold_users=()):
    """Per-action object prior, the global object prior, and phrase->HARn-folder map."""
    hn = tr[(tr.source == 'HARn') & (~tr.user.isin(list(hold_users)))]
    obj, sing = hn[hn.category == 'object_interaction'], hn[hn.category == 'single']
    pri = collections.defaultdict(collections.Counter)
    glob = collections.Counter()
    for _, r in obj.iterrows():
        L = [x for x in str(r['answer']) if x in 'ABCD']
        if L:
            t = str(r[L[0]]).strip(); pri[r.harn_action][t] += 1; glob[t] += 1
    p2f = {}
    for _, r in sing.iterrows():
        L = [x for x in str(r['answer']) if x in 'ABCD']
        if L:
            p2f.setdefault(str(r[L[0]]).strip(), collections.Counter())[r.harn_action] += 1
    return dict(pri=pri, glob=glob, p2f={k: v.most_common(1)[0][0] for k, v in p2f.items()})


def joint_resolve(model, sr, r):
    """Best (single letter, object letter) pair by per-action object prior count."""
    pri, p2f = model['pri'], model['p2f']
    best = (0, None, None)
    for L in 'ABCD':
        fo = p2f.get(str(sr[L]).strip())
        if not fo:
            continue
        for i in range(4):
            n = pri[fo][str(r['ABCD'[i]]).strip()]
            if n > best[0]:
                best = (n, L, 'ABCD'[i])
    return best


def apply_test(model, te, champ_pred):
    """-> DataFrame of proposed overrides with an audit reason."""
    pri, glob, p2f = model['pri'], model['glob'], model['p2f']
    h = te[te.source == 'HARn'].copy()
    h['clip'] = [p.split('/')[1] for p in h.path]
    sing = h[h.category == 'single'].set_index('clip')
    obj = h[h.category == 'object_interaction']
    out = []
    for _, r in obj.iterrows():
        oo = [str(r[L]).strip() for L in 'ABCD']
        cur = champ_pred[r.qa_id]
        chosen = oo[ord(cur) - 65]
        rule = None; new = None; why = None
        # W1: action pinned by the single question on the same clip
        ck = r['clip']
        if ck in sing.index:
            sr = sing.loc[ck]
            if isinstance(sr, pd.DataFrame):
                sr = sr.iloc[0]
            sp = champ_pred[sr.qa_id]
            if sp in 'ABCD':
                folder = p2f.get(str(sr[sp]).strip())
                if folder and pri[folder]:
                    cnt = [pri[folder][o] for o in oo]
                    if max(cnt) > 0:
                        cand = 'ABCD'[int(np.argmax(cnt))]
                        if cand != cur and cnt[ord(cur) - 65] == 0:
                            rule = 'W1'; new = cand
                            why = (f"single {sr.qa_id}='{str(sr[sp]).strip()}' -> {folder}; "
                                   f"per-action object prior {dict(zip(oo, cnt))}; "
                                   f"champion option has prior 0")
        # W2: the champion picked an object that is never a correct answer for any action
        if rule is None and glob[chosen] == 0:
            cnt = [glob[o] for o in oo]
            if max(cnt) > 0:
                cand = 'ABCD'[int(np.argmax(cnt))]
                if cand != cur:
                    rule = 'W2'; new = cand
                    why = (f"champion chose '{chosen}', which is never the correct object in "
                           f"133 training questions; global prior {dict(zip(oo, cnt))}")
        # X: no object option is compatible with the single answer -> re-solve the pair.
        # Measured: when this fires, the champion's SINGLE accuracy is 0.375 (n=8) against
        # 1.000 where an object option does fit, so the single answer is the suspect one.
        if rule is None and ck in sing.index:
            sr = sing.loc[ck]
            if isinstance(sr, pd.DataFrame):
                sr = sr.iloc[0]
            sp = champ_pred[sr.qa_id]
            folder = p2f.get(str(sr[sp]).strip()) if sp in 'ABCD' else None
            cnt = [pri[folder][o] for o in oo] if folder else [0, 0, 0, 0]
            if max(cnt) == 0:
                n, bl, bo = joint_resolve(model, sr, r)
                if n > 0:
                    if bl != sp:
                        out.append(dict(qa_id=sr.qa_id, clip=ck, champ=sp, prediction=bl,
                                        rule='X', why=(
                            f"no object option fits single='{str(sr[sp]).strip()}'; objects "
                            f"{oo} pin the action to '{str(sr[bl]).strip()}' "
                            f"(per-action prior count {n})")))
                    if bo != cur:
                        rule = 'X'; new = bo
                        why = (f"joint re-solve with single {sr.qa_id} -> "
                               f"'{str(sr[bl]).strip()}'; per-action prior count {n}")
        if rule:
            out.append(dict(qa_id=r.qa_id, clip=r['clip'], champ=cur, prediction=new,
                            rule=rule, why=why))
    return pd.DataFrame(out)


if __name__ == '__main__':
    tr, te, meta = load_all()
    sub = pd.read_csv(os.path.join(ROOT, 'submissions/submission_092105_SUBMITTED.csv'))
    cp = dict(zip(sub.qa_id, sub.prediction))
    D = apply_test(fit(tr), te, cp)
    D.to_csv(os.path.join(ROOT, 'objlab', 'test_object_overrides.csv'), index=False)
    print(D[['qa_id', 'clip', 'champ', 'prediction', 'rule']].to_string(index=False))
    print('\n' + '\n'.join(f'{r.qa_id}: {r.why}' for r in D.itertuples()))
