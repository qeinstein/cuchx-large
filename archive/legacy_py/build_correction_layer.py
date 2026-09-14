"""Assemble a selective correction layer over submission_092105_SUBMITTED.csv.

The champion stays the default.  A row changes only when a mechanism that was validated
subject-disjoint says so, and every change carries an audit line saying which mechanism and
with what OOF flip precision.

Precedence rule: the 10 rows that the 0.92105 submission already changed relative to the
0.90643 base were themselves validated overrides (mechanism H at 0.778 flip precision,
mechanism E at 0.800).  A new mechanism does not overwrite them.

Inputs
  seqlab/test_sequence_audit.csv    mechanism S (sequence joint order decoding)
  champ/diff_<name>.csv             pipeline-level mechanisms, produced by champ/make_candidate.py
                                    against a matched baseline run
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CHAMP = os.path.join(ROOT, 'submission_092105_SUBMITTED.csv')
BASE_090643 = os.path.join(ROOT, 'submission_090643_regen.csv')

# mechanism registry: name -> (source file, OOF flip precision, OOF net, one-line description)
MECH = {
    'S': dict(prec=0.853, net=+53, n_oof=112,
              desc='sequence joint total-order decoding over the session block'),
    'S_pair': dict(prec=0.782, net=+70, n_oof=203,
                   desc='  (same mechanism, two-question-block regime, measured separately)'),
    'T': dict(prec=0.821, net=+36, n_oof=68,
              desc='block conformance repair: split inferred blocks that cannot be one session'),
    'U': dict(prec=None, net=None, n_oof=None,
              desc='learned prior over which protocol slots a two-clip block contains'),
    'V': dict(prec=None, net=None, n_oof=None,
              desc='DTW warping-path features in the pairwise manner discriminator'),
    'W': dict(prec=1.000, net=+7, n_oof=7,
              desc='object question reads its action off the single answer for the same clip'),
}


def load_S():
    p = os.path.join(ROOT, 'seqlab', 'test_sequence_audit.csv')
    if not os.path.exists(p):
        return pd.DataFrame(columns=['qa_id', 'prediction', 'mech', 'why'])
    a = pd.read_csv(p)
    a = a[a.flip]
    return pd.DataFrame(dict(
        qa_id=a.qa_id, prediction=a.pred, mech='S',
        why=['margin %.2f | block %s | champ_block_self_consistent=%s | order: %s'
             % (m, b, c, o) for m, b, c, o in
             zip(a.margin, a.block, a.champ_block_consistent, a.latent_order)]))


def load_W(include_w2=True):
    p = os.path.join(ROOT, 'objlab', 'test_object_overrides.csv')
    if not os.path.exists(p):
        return pd.DataFrame(columns=['qa_id', 'prediction', 'mech', 'why'])
    d = pd.read_csv(p)
    if not include_w2:
        d = d[d.rule == 'W1']
    return pd.DataFrame(dict(qa_id=d.qa_id, prediction=d.prediction,
                             mech='W:' + d.rule, why=d.why))


def load_T():
    """Repair-only delta, from a PAIRED pair of real-test runs that differ solely in
    CHAMP_REPAIR.  Verified: on all 7 changed rows the paired baseline agrees with both the
    0.92105 champion and the 0.90643 base, so the change is cleanly attributable."""
    p = os.path.join(ROOT, 'champ', 'diff_T_repair_cf.csv')
    if not os.path.exists(p):
        return pd.DataFrame(columns=['qa_id', 'prediction', 'mech', 'why'])
    d = pd.read_csv(p)
    return pd.DataFrame(dict(qa_id=d.qa_id, prediction=d.prediction_rep, mech='T',
                             why=[f'clip {c}: block repair changed this session '
                                  f'({a} -> {b})' for c, a, b in
                                  zip(d['clip'], d.prediction_b0, d.prediction_rep)]))


def load_pipeline_mech(name, tag):
    p = os.path.join(ROOT, 'champ', f'diff_{tag}.csv')
    if not os.path.exists(p):
        return pd.DataFrame(columns=['qa_id', 'prediction', 'mech', 'why'])
    d = pd.read_csv(p)
    return pd.DataFrame(dict(qa_id=d.qa_id, prediction=d.prediction_new, mech=name,
                             why=[f'{tag}: {a} -> {b}' for a, b in
                                  zip(d.prediction_ch, d.prediction_new)]))


def build(out_name, parts, protect_existing=True):
    ch = pd.read_csv(CHAMP)
    base = pd.read_csv(BASE_090643)
    protected = set(ch.qa_id[ch.prediction.values != base.prediction.values]) if protect_existing else set()
    sub = ch.copy().set_index('qa_id')
    audit = []
    applied = set()
    for p in parts:
        for r in p.itertuples():
            if r.qa_id in protected:
                audit.append(dict(qa_id=r.qa_id, mech=r.mech, action='skipped',
                                  why='row already carries a validated 0.92105 override'))
                continue
            if r.qa_id in applied:
                audit.append(dict(qa_id=r.qa_id, mech=r.mech, action='skipped',
                                  why='earlier mechanism already changed this row'))
                continue
            old = sub.loc[r.qa_id, 'prediction']
            if str(old) == str(r.prediction):
                continue
            sub.loc[r.qa_id, 'prediction'] = r.prediction
            applied.add(r.qa_id)
            audit.append(dict(qa_id=r.qa_id, mech=r.mech, action='applied',
                              champion=old, new=r.prediction, why=r.why))
    sub = sub.reset_index()
    assert len(sub) == 682 and sub.prediction.notna().all()
    assert (sub.qa_id.values == ch.qa_id.values).all()
    te = pd.read_csv(os.path.join(ROOT, 'test_qa.csv'))
    chg = sub.prediction.values != ch.prediction.values
    lens = te.set_index('qa_id').loc[sub.qa_id[chg], 'category']
    out = os.path.join(ROOT, f'{out_name}.csv')
    sub.to_csv(out, index=False)
    A = pd.DataFrame(audit)
    A.to_csv(os.path.join(ROOT, f'{out_name}_audit.csv'), index=False)
    print(f'{out_name}: {int(chg.sum())} changes vs the 0.92105 champion')
    print('  by category:', lens.value_counts().to_dict())
    if len(A):
        print('  by mechanism/action:')
        print(A.groupby(['mech', 'action']).size().to_string())
    print('  wrote', out, 'and', f'{out_name}_audit.csv')
    return sub


if __name__ == '__main__':
    which = sys.argv[1:] or ['S']
    parts = []
    for w in which:
        if w == 'S':
            parts.append(load_S())
        elif w == 'W':
            parts.append(load_W(True))
        elif w == 'W1':
            parts.append(load_W(False))
        elif w == 'T':
            parts.append(load_T())
        else:
            name, tag = w.split(':', 1)
            parts.append(load_pipeline_mech(name, tag))
    build('submission_corrlayer_' + '_'.join(w.split(':')[0] for w in which), parts)
