"""Subject-disjoint evaluation of the pairwise precedence sequence decoder.
Reports integer counts, pairwise accuracy, and wins/losses vs the checkpoint decoder."""
import os, sys, json, itertools
import numpy as np, pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense as D, seqpair as SP
from core import load_all, opts
from pseudotest import folds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(nfold=5):
    tr, te, meta = load_all()
    segs = SP.build_segment_data(meta)
    users = sorted(tr.user.dropna().unique())
    seq = tr[(tr.source == 'HAU') & (tr.category == 'sequence')]
    # The older champ/oof_final.csv is not part of the current checkout.  Prefer it
    # when present for historical compatibility, otherwise use the committed OOF
    # produced by the current final-video base architecture.
    base_candidates = [
        os.environ.get('CHAMP_OOF_BASE'),
        os.path.join(ROOT, 'champ', 'oof_final.csv'),
        os.path.join(ROOT, 'research', 'final_video_20260910', 'oof_pipeline_base.csv'),
        os.path.join(ROOT, 'champ', 'oof_e2e_baseline_20260909.csv'),
    ]
    base_path = next((p for p in base_candidates if p and os.path.exists(p)), None)
    if base_path is None:
        raise FileNotFoundError('No OOF baseline found; set CHAMP_OOF_BASE to a valid CSV')
    base = pd.read_csv(base_path)
    base = dict(zip(base.qa_id, base.pred))

    out = []
    pw_ok = pw_n = 0
    for fi, hold in enumerate(folds(users, nfold)):
        trn_users = [u for u in users if u not in hold]
        Xd, y, src, _, _ = SP.build_training_pairs(tr, meta, segs, trn_users)
        clf, cols = SP.fit_pair_model(Xd, y)
        # priors for the held-out fold must also come from training subjects only
        dpri, apri = SP.fit_priors(segs, [k for k in segs if k[0] in trn_users])
        for r in seq[seq.user.isin(hold)].itertuples():
            oo = opts(r)
            prof = SP.clip_profiles('oof', r.path)
            if prof is None or not all(o in SP.OPT2CLS for o in oo):
                out.append(dict(qa_id=r.qa_id, fold=fi, pred=None, answer=r.answer,
                                correct=0, nopred=1))
                continue
            cls4 = [SP.OPT2CLS[o] for o in oo]
            perm, Pm, idx = SP.decode(prof, cls4, clf, cols, dpri, apri)
            pred = ''.join('ABCD'[p] for p in perm)
            # pairwise accuracy against the true order
            tpos = {L: i for i, L in enumerate(str(r.answer))}
            for a, b in itertools.combinations(range(4), 2):
                truth = tpos['ABCD'[a]] < tpos['ABCD'[b]]
                pw_ok += int((Pm[idx[(a, b)]] > 0.5) == truth); pw_n += 1
            out.append(dict(qa_id=r.qa_id, fold=fi, pred=pred, answer=r.answer,
                            correct=int(pred == str(r.answer)), nopred=0))
        print(f'  fold {fi}: {len(Xd)} training pairs '
              f'({(src=="seg").sum()} segment / {(src=="qa").sum()} answer-derived)',
              flush=True)
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(ROOT, 'champ', 'oof_seqpair.csv'), index=False)
    d['base'] = [str(base.get(q)) for q in d.qa_id]
    d['base_correct'] = (d.base == d.answer.astype(str)).astype(int)
    print(f'\n  pairwise precedence accuracy : {pw_ok}/{pw_n} = {pw_ok/pw_n:.4f}')
    print(f'  SEQUENCE exact (pairwise decoder): {d.correct.sum()}/{len(d)} '
          f'= {d.correct.mean():.4f}')
    print(f'  SEQUENCE exact (checkpoint)      : {d.base_correct.sum()}/{len(d)} '
          f'= {d.base_correct.mean():.4f}')
    w = ((d.correct == 1) & (d.base_correct == 0)).sum()
    l = ((d.correct == 0) & (d.base_correct == 1)).sum()
    print(f'  wins {w}  losses {l}  net {w-l:+d}')
    print('  per-fold:', '  '.join(
        f'f{f}={g.correct.sum()}/{len(g)}(base {g.base_correct.sum()})'
        for f, g in d.groupby('fold')))
    print(f'\n  GATE: minimum 215/308, preferred 240/308, championship 260/308')
    return d


if __name__ == '__main__':
    main()
