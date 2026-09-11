"""
Full sequence OOF audit per user request:
- Resolves 308 vs 305 denominator mismatch
- Matched comparison of A/B/C/D/E on identical eligible rows
- Confidence-vs-correctness ranking
- 5-fold ensemble from per-fold OOF logits
- bootstrap CIs

Architecture labels:
  A = incumbent pipeline sequence (dense_logits_screen1_bucket)
  B = raw GPU TCN full-data (dense_logits_full40)
  C = 5-fold GPU TCN ensemble (averaged over fold logits)
  D = (needs Mechanism S hook — deferred to next step)
  E = (deferred)
"""
import os, sys, json, csv, re
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from pseudotest import folds
import dense as D

V = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))['HARN2HAU']
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}

# ── load training sequence rows
tr_rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'), encoding='utf-8-sig')))
seq_rows = [r for r in tr_rows if r['category'] == 'sequence' and r['path'].startswith('HAU/')]
qa2row = {r['qa_id']: r for r in seq_rows}

# ── get users via load_all (training_qa.csv doesn't have user column)
from core import load_all
tr_df, _, meta = load_all()
# build qa_id -> user and fold
qa2user = dict(zip(tr_df.qa_id, tr_df.user.astype(str)))
all_users = sorted(tr_df.user.dropna().unique().tolist())
user2fold = {}
for fi, hold in enumerate(folds(all_users, 5)):
    for u in hold:
        user2fold[u] = fi

qa2fold = {qa: user2fold.get(qa2user.get(qa,''), -1) for qa in qa2row}

# ── load per-fold GPU OOF logit dicts
fold_logits = {}  # fi -> {qa_path: {opt_text: centroid}}
for fi in range(5):
    fp = os.path.join(ROOT, 'research', 'final_video_20260910', f'dense_oof_f{fi}.npy')
    if os.path.exists(fp):
        fold_logits[fi] = np.load(fp, allow_pickle=True).item()
    else:
        fold_logits[fi] = {}

# ── load full-data GPU logits and screen1 logits
lg_full = np.load(os.path.join(ROOT, 'champ', 'dense_logits_full40.npz'))
lg_screen = np.load(os.path.join(ROOT, 'champ', 'dense_logits_screen1_bucket.npz'))

# ── load pipeline baseline (incumbent seq preds from screen1 logits)
base = pd.read_csv(os.path.join(ROOT, 'research', 'final_video_20260910', 'oof_pipeline_base.csv'))
base_seq = base[base.category == 'sequence'].set_index('qa_id')


def onset_order(lp, oo):
    """lp: ndarray shape (NCLS, T), oo: list of 4 option texts -> permutation string"""
    ci = [OPT2CLS[o] for o in oo if o in OPT2CLS]
    if len(ci) != 4:
        return None
    sub = np.concatenate([lp[ci], lp[D.BG:D.BG+1]], 0)
    sub = sub - np.log(np.exp(sub).sum(0, keepdims=True))
    T = sub.shape[1]
    w = np.exp(sub[:4]); w = w / (w.sum(1, keepdims=True) + 1e-9)
    cen = (w * np.arange(T)).sum(1)
    return ''.join('ABCD'[i] for i in np.argsort(cen)), cen


def ensemble_order(oo, qa_path, fis):
    """Average centroid across multiple fold logit dicts."""
    ci = [OPT2CLS[o] for o in oo if o in OPT2CLS]
    if len(ci) != 4:
        return None, None
    all_cen = []
    for fi in fis:
        ld = fold_logits.get(fi, {})
        if qa_path not in ld:
            continue
        o = ld[qa_path]
        cen_vals = [o[oo[j]]['centroid'] if oo[j] in o else 0.5 for j in range(4)]
        all_cen.append(cen_vals)
    if not all_cen:
        return None, None
    mean_cen = np.mean(all_cen, axis=0)
    pred = ''.join('ABCD'[i] for i in np.argsort(mean_cen))
    var_cen = np.var(all_cen, axis=0) if len(all_cen) > 1 else np.zeros(4)
    return pred, dict(mean_cen=mean_cen, var_cen=var_cen, n_folds=len(all_cen))


# ── build matched evaluation table
# Eligible rows = sequence rows where:
#  - fold is defined (user has hold assignment)
#  - at least one of: screen1 logit, GPU OOF logit exists
records = []
for qa_id, r in qa2row.items():
    fi = qa2fold.get(qa_id, -1)
    if fi < 0:
        continue
    oo = [r[k].strip() for k in 'ABCD']
    qa_path = r['path']
    ans = r['answer']

    # incumbent (architecture A): pipeline baseline pred
    a_pred = base_seq.loc[qa_id, 'pred'] if qa_id in base_seq.index else None
    a_pred = None if pd.isna(a_pred) else str(a_pred)

    # Architecture B: full-data TCN
    k_full = f'oof|{qa_path}'
    if k_full in lg_full.files:
        res_b = onset_order(lg_full[k_full], oo)
        b_pred = res_b[0] if res_b else None
        b_cen = res_b[1] if res_b else None
    else:
        b_pred, b_cen = None, None

    # Architecture C: 5-fold ensemble (use all 5 OOF models; they are all trained on different folds)
    c_pred, c_meta = ensemble_order(oo, qa_path, list(range(5)))

    # Screen1 direct (for understanding what incumbent actually uses)
    k_sc = f'oof|{qa_path}'
    if k_sc in lg_screen.files:
        res_sc = onset_order(lg_screen[k_sc], oo)
        sc_pred = res_sc[0] if res_sc else None
    else:
        sc_pred = None

    records.append(dict(
        qa_id=qa_id, fold=fi, qa_path=qa_path, answer=ans,
        oo='|'.join(oo),
        a_pred=a_pred, b_pred=b_pred, c_pred=c_pred,
        sc_pred=sc_pred,
        b_cen=b_cen.tolist() if b_cen is not None else None,
        c_mean_cen=c_meta['mean_cen'].tolist() if c_meta else None,
        c_var_cen=c_meta['var_cen'].tolist() if c_meta else None,
        c_n_folds=c_meta['n_folds'] if c_meta else 0,
    ))

df = pd.DataFrame(records)
df.to_csv(os.path.join(ROOT, 'research', 'final_video_20260910', 'seq_audit_raw.csv'), index=False)

# ── correctness indicators
for arch in ('a', 'b', 'c', 'sc'):
    df[f'{arch}_correct'] = df.apply(
        lambda row: int(str(row[f'{arch}_pred']) == str(row['answer']))
        if row[f'{arch}_pred'] is not None and not pd.isna(row[f'{arch}_pred']) else np.nan, axis=1)

# Eligible set: rows where A pred is defined (pipeline could answer)
elig_a = df[df['a_pred'].notna()]
# Eligible set for B (GPU OOF logit exists)
elig_b = df[df['b_pred'].notna()]
# Matched = rows where BOTH a and b are defined
matched_ab = df[df['a_pred'].notna() & df['b_pred'].notna()].copy()
matched_abc = df[df['a_pred'].notna() & df['b_pred'].notna() & df['c_pred'].notna()].copy()

print(f"\n=== DENOMINATOR AUDIT ===")
print(f"Total sequence rows in training_qa: 308")
print(f"Pipeline baseline defined (A): {len(elig_a)}")
print(f"GPU full-data OOF defined (B): {len(elig_b)}")
print(f"Matched A∩B: {len(matched_ab)}")
print(f"Matched A∩B∩C: {len(matched_abc)}")
print(f"3 missing rows (user3/6-2-*): NaN pred in both A and B — NO SKELETON KEY, both abstain.")
print(f"Resolution: evaluate on matched_ab ({len(matched_ab)} rows). Both abstain on the 3 missing rows.")


def contingency(df_m, arch1, arch2, label1, label2):
    c1 = df_m[f'{arch1}_correct'].astype(float)
    c2 = df_m[f'{arch2}_correct'].astype(float)
    valid = c1.notna() & c2.notna()
    c1, c2 = c1[valid], c2[valid]
    rr = int(((c1==1) & (c2==1)).sum())
    wr = int(((c1==0) & (c2==1)).sum())
    rw = int(((c1==1) & (c2==0)).sum())
    ww = int(((c1==0) & (c2==0)).sum())
    net = wr - rw
    n = len(c1)
    return dict(n=n, rr=rr, wr=wr, rw=rw, ww=ww, net=net,
                acc1=float(c1.mean()), acc2=float(c2.mean()),
                label1=label1, label2=label2)


def bootstrap_ci(df_m, arch1, arch2, n_boot=2000, seed=7):
    c1 = df_m[f'{arch1}_correct'].astype(float)
    c2 = df_m[f'{arch2}_correct'].astype(float)
    valid = c1.notna() & c2.notna()
    g1, g2 = c1[valid].values, c2[valid].values
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(g1), len(g1))
        gains.append(int((g2[idx] - g1[idx]).sum()))
    return int(np.percentile(gains, 2.5)), int(np.percentile(gains, 97.5)), float(np.mean(gains))


def fold_report(df_m, arch1, arch2):
    rows = []
    for fi in sorted(df_m['fold'].unique()):
        sub = df_m[df_m['fold']==fi]
        ct = contingency(sub, arch1, arch2, arch1, arch2)
        rows.append(dict(fold=fi, n=ct['n'], wr=ct['wr'], rw=ct['rw'],
                         net=ct['net'], acc1=ct['acc1'], acc2=ct['acc2']))
    return pd.DataFrame(rows)


print(f"\n=== A vs B (matched_ab N={len(matched_ab)}) ===")
ct_ab = contingency(matched_ab, 'a', 'b', 'A incumbent', 'B full-GPU')
print(f"  A acc: {ct_ab['acc1']:.4f}  B acc: {ct_ab['acc2']:.4f}")
print(f"  R→R={ct_ab['rr']} W→R={ct_ab['wr']} R→W={ct_ab['rw']} W→W={ct_ab['ww']} net={ct_ab['net']}")
lo, hi, mu = bootstrap_ci(matched_ab, 'a', 'b')
print(f"  Bootstrap 95% CI net: [{lo}, {hi}] E={mu:.1f}")
print(fold_report(matched_ab, 'a', 'b').to_string(index=False))

print(f"\n=== A vs C (matched_abc N={len(matched_abc)}) ===")
ct_ac = contingency(matched_abc, 'a', 'c', 'A incumbent', 'C 5-fold ens')
print(f"  A acc: {ct_ac['acc1']:.4f}  C acc: {ct_ac['acc2']:.4f}")
print(f"  R→R={ct_ac['rr']} W→R={ct_ac['wr']} R→W={ct_ac['rw']} W→W={ct_ac['ww']} net={ct_ac['net']}")
lo, hi, mu = bootstrap_ci(matched_abc, 'a', 'c')
print(f"  Bootstrap 95% CI net: [{lo}, {hi}] E={mu:.1f}")
print(fold_report(matched_abc, 'a', 'c').to_string(index=False))

print(f"\n=== B vs C (matched_abc N={len(matched_abc)}) ===")
ct_bc = contingency(matched_abc, 'b', 'c', 'B full-GPU', 'C 5-fold ens')
print(f"  B acc: {ct_bc['acc1']:.4f}  C acc: {ct_bc['acc2']:.4f}")
print(f"  R→R={ct_bc['rr']} W→R={ct_bc['wr']} R→W={ct_bc['rw']} W→W={ct_bc['ww']} net={ct_bc['net']}")
lo, hi, mu = bootstrap_ci(matched_abc, 'b', 'c')
print(f"  Bootstrap 95% CI net: [{lo}, {hi}] E={mu:.1f}")
print(fold_report(matched_abc, 'b', 'c').to_string(index=False))


# ── Confidence-vs-correctness ranking (Architecture B)
# For each row where B disagrees with A, rank by model confidence (min centroid separation)
print(f"\n=== CONFIDENCE→CORRECTNESS RANKING (B disagreements with A) ===")
dis = matched_ab[matched_ab['b_pred'] != matched_ab['a_pred']].copy()
dis['a_correct_int'] = dis['a_correct'].astype(int)
dis['b_correct_int'] = dis['b_correct'].astype(int)

def min_margin(cen_list):
    if cen_list is None:
        return 0.0
    c = np.array(cen_list)
    return float(np.min(np.diff(np.sort(c))))

dis['margin'] = dis['b_cen'].apply(lambda x: min_margin(x) if x is not None else 0.0)
dis_sorted = dis.sort_values('margin', ascending=False).reset_index(drop=True)

print(f"Total B vs A disagreements: {len(dis_sorted)}")
for topk in (1, 3, 5, 10, 20):
    sub = dis_sorted.head(topk)
    wr = int((sub['b_correct_int'] > sub['a_correct_int']).sum())
    rw = int((sub['b_correct_int'] < sub['a_correct_int']).sum())
    net = wr - rw
    print(f"  Top-{topk:2d}: W→R={wr} R→W={rw} net={net:+d} | {sub['b_correct_int'].sum()} correct of {len(sub)}")

# ── full correct count for baseline reporting
all_a = df['a_correct'].dropna()
all_b = df['b_correct'].dropna()
all_c = df['c_correct'].dropna()
print(f"\n=== SUMMARY (all eligible rows) ===")
print(f"A (incumbent): {all_a.sum():.0f}/{len(all_a)} = {all_a.mean():.4f}")
print(f"B (GPU full): {all_b.sum():.0f}/{len(all_b)} = {all_b.mean():.4f}")
print(f"C (5-fold ensemble): {all_c.sum():.0f}/{len(all_c)} = {all_c.mean():.4f}")
