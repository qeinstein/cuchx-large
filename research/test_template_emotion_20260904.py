import os
import pickle
import re
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
TR = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
TE = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
SUB = pd.read_csv(os.path.join(ROOT, 'submissions/submission_093859_SUBMITTED.csv')).set_index('qa_id')
SNAP = pickle.load(open(os.path.join(ROOT, 'research', 'test_pool_snapshot.pkl'), 'rb'))
TR['user'] = TR.path.str.extract(r'(user\d+)')
TR['aa'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[0]
TR['bb'] = TR.path.str.extract(r'/(\d+)-(\d+)-(\d+)')[1]
TR['blk'] = list(zip(TR.user, TR.aa, TR.bb))
ACTC = ['single', 'multi', 'combination', 'sequence']
E = TR[(TR.source == 'HAU') & (TR.category == 'emotion')]

def letters(x): return [c for c in str(x) if c in 'ABCD']
def opts(r): return [str(r[c]).strip() for c in 'ABCD']
def pool(b):
    q = TR[(TR.source == 'HAU') & TR.blk.map(lambda x: x == b) & TR.category.isin(ACTC)]
    out = set()
    for _, r in q.iterrows():
        for c in letters(r.answer): out.update(x.strip() for x in str(r[c]).split(','))
    return frozenset(out)

blocks = {b: g.sort_values('path') for b, g in E.groupby(['user', 'aa', 'bb'])}
pools = {b: pool(b) for b in blocks}
labels = {}
for _, r in E.iterrows():
    c = letters(r.answer)[0]
    labels[r.qa_id] = str(r[c]).strip()

def source_labels(b):
    return [labels[r.qa_id] for _, r in blocks[b].sort_values('path').iterrows()]

def exact_prediction(block, target_rows):
    src = [b for b in blocks if pools[b] == block['pool']]
    if len(target_rows) != 3 or not src:
        return {}, len(src)
    votes = [Counter(source_labels(b)[i] for b in src if len(source_labels(b)) == 3)
             for i in range(3)]
    out = {}
    for i, (_, r) in enumerate(target_rows.iterrows()):
        if not votes[i]: continue
        lab = votes[i].most_common(1)[0][0]
        oo = opts(r)
        if lab in oo:
            out[r.qa_id] = 'ABCD'[oo.index(lab)]
    return out, len(src)

te = TE[TE.category == 'emotion'].copy()
te['idx'] = te.path.str.extract(r'LM_test_(\d+)').astype(int)
block_for = {}
for b in SNAP['blocks']:
    for i in b: block_for[int(i)] = tuple(int(x) for x in b)
rows = []
for b in SNAP['blocks']:
    rr = te[te.idx.map(lambda x: block_for.get(int(x)) == tuple(b))].sort_values('idx')
    if not len(rr): continue
    p = frozenset(SNAP['pool_of'].get(int(b[0]), []))
    pred, nsrc = exact_prediction({'pool': p}, rr)
    for _, r in rr.iterrows():
        if r.qa_id not in pred: continue
        old = SUB.loc[r.qa_id, 'prediction']
        rows.append(dict(qa_id=r.qa_id, idx=int(r.idx), block=str(tuple(b)), nsrc=nsrc,
                         old=old, new=pred[r.qa_id], flip=old != pred[r.qa_id],
                         label=pred[r.qa_id] and str(r[pred[r.qa_id]])))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, 'research', 'test_template_emotion_audit.csv'), index=False)
print('full-triple exact-label coverage', len(out), 'source blocks', out.nsrc.nunique() if len(out) else 0,
      'flips', int(out.flip.sum()) if len(out) else 0)
print(out[out.flip].to_string(index=False))
