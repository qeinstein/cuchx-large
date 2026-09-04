#!/usr/bin/env python3
"""PCRME Stage 7 -- decision-level audit: does phase-relative evidence, fed into the SAME
joint bijective session assignment the champion uses, change the actual predicted LETTER,
and if so, is it right more often than it is wrong against the frozen champion snapshot?

This is deliberately NOT a full re-implementation of `champ/core.solve_emotion` (which also
carries the emopair pairwise term, pool-context, and slot marginalisation edge cases). It
isolates exactly the new evidence's contribution by reusing the SAME two terms solve_emotion
scores unary evidence with -- the physical-group log-likelihood-ratio and the position prior --
and swapping only the physical-group classifier's input features between:

  BASELINE : champ/core.block_features() PHYS features (what solve_emotion uses today)
  EXTENDED : BASELINE + phase-relative aggregate (Stage 1/2's new evidence)

Both arms use IDENTICAL position priors, IDENTICAL candidate-set/slot logic, and IDENTICAL
assignment search (permutation search over candidates, exactly mirroring solve_emotion's own
`k <= len(cand)` injective-assignment enumeration and its `slots > k` marginalisation for
2-clip blocks). The comparison isolates the ONE thing this session is testing.

Coverage note: only 776/809 clips have a Stage-1 phase-relative vector. For clips without one,
the EXTENDED arm falls back to zero-padding (equivalent to "no phase evidence available"); this
is recorded per-row so its effect is not hidden inside an aggregate number.
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CHAMP = ROOT / "champ"
sys.path.insert(0, str(CHAMP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import load_all, gt_letters, mgroup, GROUPS, PHYS, opts, fit_manner  # noqa: E402
from probe_falsification import build_phase_features  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent


def block_features_ext(blk, mfeat, k, cols):
    """Exact clone of champ.core.block_features, parameterised over an arbitrary column list
    instead of the module-global PHYS, so it can be evaluated on the extended feature space
    without touching global state used elsewhere in the pipeline."""
    rows = []
    M = np.array([[mfeat.get(b, {}).get(c, np.nan) for c in cols] for b in blk], float)
    with np.errstate(all="ignore"):
        mu = np.nanmean(M, 0)
        sd = np.nanstd(M, 0) + 1e-9
        Z = (M - mu) / sd
        R = np.argsort(np.argsort(np.where(np.isnan(M), -np.inf, M), 0), 0) / max(1, k - 1)
    for i, b in enumerate(blk):
        d = {f"a_{c}": M[i, j] for j, c in enumerate(cols)}
        d.update({f"z_{c}": Z[i, j] for j, c in enumerate(cols)})
        d.update({f"r_{c}": R[i, j] for j, c in enumerate(cols)})
        d["pos"] = i
        d["k"] = k
        d["posfrac"] = i / max(1, k - 1)
        rows.append(d)
    return rows


def assign_block(rows, cand_groups_by_manner, P, cls, mm, k):
    """Minimal re-implementation of solve_emotion's unary assignment for one block, using the
    SAME two terms (physical group evidence + position prior), no pairwise/pool term. Returns
    the predicted letter per row and a margin (best minus 2nd best total score of the winning
    assignment's own row) for later confidence gating.
    """
    O = [set(opts(r)) for r in rows]
    I = set.intersection(*O) if len(O) > 1 else set(O[0])
    cand = sorted(I) if len(I) >= k else sorted(set().union(*O))
    slots = len(cand) if (k < len(cand) <= 4) else k
    C = np.full((k, len(cand)), 60.0)
    own = [set(opts(r)) for r in rows]
    for i in range(k):
        pg_phys = {g: P[i, cls.index(g)] if g in cls else 1e-6 for g in GROUPS}
        for j, m in enumerate(cand):
            if m not in own[i]:
                continue
            g = mgroup(m)
            lp = (np.log(max(pg_phys[g], 1e-9)) - np.log(max(mm["gprior"][g], 1e-9))
                  + np.log(max(mm["ppm"](m, i, k), 1e-9))
                  + np.log(max(mm["pmg"].get(m, 1e-4), 1e-6)))
            C[i, j] = -lp
    if slots > k:
        best = None
        for perm in itertools.permutations(range(slots)):
            for present in itertools.combinations(range(slots), k):
                lab = [cand[perm[s]] for s in present]
                if any(lab[i] not in own[i] for i in range(k)):
                    continue
                tot = 0.0
                for s in range(slots):
                    m = cand[perm[s]]
                    tot += (np.log(max(mm["ppm"](m, s, slots), 1e-9))
                            + np.log(max(mm["pmg"].get(m, 1e-4), 1e-6)))
                for i, s in enumerate(present):
                    m = cand[perm[s]]
                    g = mgroup(m)
                    pg = P[i, cls.index(g)] if g in cls else 1e-6
                    tot += np.log(max(pg, 1e-9)) - np.log(max(mm["gprior"][g], 1e-9))
                if best is None or tot > best[0]:
                    best = (tot, lab)
        lab = best[1] if best else [None] * k
    else:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(C)
        lab = [None] * k
        for i, j in zip(ri, ci):
            lab[i] = cand[j]
    out = []
    for i, r in enumerate(rows):
        m = lab[i]
        L = next((L for L in "ABCD" if str(r[L]).strip() == m), None) if m else None
        if L is None:
            sc = []
            pg_phys = {g: P[i, cls.index(g)] if g in cls else 1e-6 for g in GROUPS}
            for LL in "ABCD":
                mo = str(r[LL]).strip()
                g = mgroup(mo)
                sc.append((np.log(max(pg_phys[g], 1e-9)) - np.log(max(mm["gprior"][g], 1e-9))
                           + np.log(max(mm["ppm"](mo, i, k), 1e-9))
                           + np.log(max(mm["pmg"].get(mo, 1e-4), 1e-6)), LL))
            L = max(sc)[1]
        out.append(L)
    return out


def main():
    tr, te, meta = load_all()
    mfeat_base = {r.qa_path: {c: getattr(r, c) for c in PHYS if hasattr(r, c)} for r in meta.itertuples()}
    em = tr[tr.category == "emotion"].copy()
    _letters = [gt_letters(r)[0] for _, r in em.iterrows()]
    em["truth_letter"] = _letters
    em["label"] = [str(r[L]).strip() for (_, r), L in zip(em.iterrows(), _letters)]
    em["blk"] = list(zip(em.user, em.aa, em.bb))

    aligned = pd.read_csv(HERE / "aligned_phases.csv")
    feat_by_idx = build_phase_features(aligned)
    raw = aligned.loc[list(feat_by_idx.keys())].copy()
    Xb = np.stack([feat_by_idx[i] for i in raw.index])
    raw = raw.reset_index(drop=True)
    Xc = Xb.copy()
    for (sess, act), g in raw.groupby(["session", "action"]):
        idx = g.index.to_numpy()
        Xc[idx] = Xb[idx] - Xb[idx].mean(0)
    raw["clip_key"] = raw.parent_qa_path
    rel_by_clip = {k: Xc[g.index.to_numpy()].mean(0) for k, g in raw.groupby("clip_key")}
    phase_dim = Xb.shape[1]
    phase_cols = [f"ph_{i}" for i in range(phase_dim)]

    mfeat_ext = {p: dict(v) for p, v in mfeat_base.items()}
    has_phase = set()
    for path, vec in rel_by_clip.items():
        mfeat_ext.setdefault(path, {}).update({c: float(v) for c, v in zip(phase_cols, vec)})
        has_phase.add(path)
    for path in mfeat_ext:
        if path not in has_phase:
            mfeat_ext[path].update({c: 0.0 for c in phase_cols})

    ext_cols = PHYS + phase_cols

    audit = pd.read_csv(CHAMP / "audit_champ.csv")
    audit = audit[audit.category == "emotion"].set_index("qa_id")

    ufold = {}
    for f in range(5):
        for u in pd.read_csv(ROOT / f"splits/fold_{f}_val.csv").subject_id.unique():
            ufold[u] = f

    rows_out = []
    for f in range(5):
        hold = [u for u, ff in ufold.items() if ff == f]
        trn = tr[~tr.user.isin(hold)]
        mm = fit_manner(trn, meta)  # position priors + gprior, identical for both arms

        # baseline physical classifier: PHYS block_features, training subjects only
        Xtr_b, ytr = [], []
        for b, g in em[~em.user.isin(hold)].groupby("blk"):
            g = g.sort_values("cc")
            paths = list(g.path)
            bf = block_features_ext(paths, mfeat_base, len(paths), PHYS)
            for d, lab in zip(bf, g.label):
                Xtr_b.append(d)
                ytr.append(mgroup(lab))
        Xtr_b = pd.DataFrame(Xtr_b)
        base_cols = [c for c in Xtr_b.columns if c not in ("pos", "k", "posfrac")]
        clf_base = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5,
                                                   l2_regularization=1.0, random_state=0)
        clf_base.fit(Xtr_b[base_cols].to_numpy(float), ytr)

        Xtr_e = []
        for b, g in em[~em.user.isin(hold)].groupby("blk"):
            g = g.sort_values("cc")
            paths = list(g.path)
            bf = block_features_ext(paths, mfeat_ext, len(paths), ext_cols)
            Xtr_e += bf
        Xtr_e = pd.DataFrame(Xtr_e)
        ext_cols_full = [c for c in Xtr_e.columns if c not in ("pos", "k", "posfrac")]
        clf_ext = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5,
                                                  l2_regularization=1.0, random_state=0)
        clf_ext.fit(Xtr_e[ext_cols_full].to_numpy(float), ytr)

        for b, g in em[em.user.isin(hold)].groupby("blk"):
            g = g.sort_values("cc")
            rows = [r for _, r in g.iterrows()]
            k = len(rows)
            paths = [r.path for r in rows]

            bf_b = block_features_ext(paths, mfeat_base, k, PHYS)
            Xb_ = pd.DataFrame(bf_b).reindex(columns=base_cols).to_numpy(float)
            Pb = clf_base.predict_proba(Xb_)

            bf_e = block_features_ext(paths, mfeat_ext, k, ext_cols)
            Xe_ = pd.DataFrame(bf_e).reindex(columns=ext_cols_full).to_numpy(float)
            Pe = clf_ext.predict_proba(Xe_)

            lab_base = assign_block(rows, None, Pb, list(clf_base.classes_), mm, k)
            lab_ext = assign_block(rows, None, Pe, list(clf_ext.classes_), mm, k)

            for i, r in enumerate(rows):
                qid = r.qa_id
                if qid not in audit.index:
                    continue
                rows_out.append(dict(
                    qa_id=qid, fold=f, k=k, truth=r.truth_letter,
                    base_pred=lab_base[i], ext_pred=lab_ext[i],
                    champ_pred=audit.loc[qid, "pred"], champ_ok=audit.loc[qid, "correct"],
                    has_phase=r.path in has_phase,
                ))

    R = pd.DataFrame(rows_out)
    R["base_ok"] = R.base_pred == R.truth
    R["ext_ok"] = R.ext_pred == R.truth
    R["champ_ok"] = R.champ_ok.astype(bool)
    R.to_csv(HERE / "stage7_decision_audit.csv", index=False)

    print(f"n rows: {len(R)}")
    print(f"unary-only baseline (position+phys, no pairwise/pool): {R.base_ok.mean():.4f}")
    print(f"unary-only extended (+ phase-relative):                {R.ext_ok.mean():.4f}")
    print(f"[reference] full champion pipeline (audit_champ.csv):  {R.champ_ok.mean():.4f}")

    print("\n=== decision-level: EXTENDED vs actual CHAMPION prediction ===")
    fl = R[R.ext_pred != R.champ_pred]
    wr = int(((~R.champ_ok) & R.ext_ok & (R.ext_pred != R.champ_pred)).sum())
    rw = int((R.champ_ok & (~R.ext_ok) & (R.ext_pred != R.champ_pred)).sum())
    ww = len(fl) - wr - rw
    print(f"flips {len(fl)}  W->R {wr}  R->W {rw}  W->W {ww}  "
          f"precision {wr/max(wr+rw,1):.3f}  net {wr-rw:+d}  net/100 {100*(wr-rw)/max(len(fl),1):.1f}")
    print("\nper fold:")
    for f in sorted(R.fold.unique()):
        d = R[R.fold == f]
        fd = d[d.ext_pred != d.champ_pred]
        w = int(((~d.champ_ok) & d.ext_ok & (d.ext_pred != d.champ_pred)).sum())
        l = int((d.champ_ok & (~d.ext_ok) & (d.ext_pred != d.champ_pred)).sum())
        print(f"  fold {f}: n={len(d):3d} flips={len(fd):3d} W->R={w:3d} R->W={l:3d} "
              f"prec={w/max(w+l,1):.3f} net={w-l:+3d}")

    print("\nby phase coverage:")
    for hp, d in R.groupby("has_phase"):
        fd = d[d.ext_pred != d.champ_pred]
        w = int(((~d.champ_ok) & d.ext_ok & (d.ext_pred != d.champ_pred)).sum())
        l = int((d.champ_ok & (~d.ext_ok) & (d.ext_pred != d.champ_pred)).sum())
        print(f"  has_phase={hp}: n={len(d):3d} flips={len(fd):3d} W->R={w:3d} R->W={l:3d} "
              f"prec={w/max(w+l,1):.3f}")

    print("\nby block size k:")
    for kk, d in R.groupby("k"):
        fd = d[d.ext_pred != d.champ_pred]
        w = int(((~d.champ_ok) & d.ext_ok & (d.ext_pred != d.champ_pred)).sum())
        l = int((d.champ_ok & (~d.ext_ok) & (d.ext_pred != d.champ_pred)).sum())
        print(f"  k={kk}: n={len(d):3d} flips={len(fd):3d} W->R={w:3d} R->W={l:3d} "
              f"prec={w/max(w+l,1):.3f}")

    print(f"\nwrote {HERE / 'stage7_decision_audit.csv'}")


if __name__ == "__main__":
    main()
