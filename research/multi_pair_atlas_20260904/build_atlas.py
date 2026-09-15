"""Build an action-level atlas for the current CUHK-X Multi champion.

This is an audit/research path.  It does not modify ``champ/`` or any submission.  The
production pool solver is called first; its candidate-pool enumeration is then repeated
here only to retain every satisfying pool and its score.  The OOF run is subject-disjoint
and uses the real-test-like pair-thinning policy (adjacent trial withholding).

Usage::

    PYTHONHASHSEED=0 venv/bin/python research/multi_pair_atlas_20260904/build_atlas.py

Outputs are written below this directory:

    action_atlas_oof_ends.csv
    block_atlas_oof_ends.csv
    candidate_pools_oof_ends.csv
    action_atlas_test.csv
    block_atlas_test.csv
    candidate_pools_test.csv
    summary.json

The OOF action labels are split into observable decisions (the candidate/action and pool
selection) and audit-only labels (session-pool truth, Multi answer truth).  The latter are
never passed back into the solver or selector.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# These are the validated current-champion settings.  Set them before importing champ so
# repair.py captures CONFORM_FIRST at import time.  Pool bias stays empty by design.
os.environ.setdefault("CHAMP_REPAIR", "1")
os.environ.setdefault("CHAMP_CONFORM_FIRST", "1")
os.environ.setdefault("CHAMP_POOL_BIAS", "{}")
os.environ.setdefault("CHAMP_W_SLOT", "0")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAMP_DIR = ROOT / "champ"
sys.path.insert(0, str(CHAMP_DIR))

from core import load_all, make_pseudo, opts, real_test_view  # noqa: E402
import dense as D  # noqa: E402
import decode as DC  # noqa: E402
import pipeline as P  # noqa: E402
import pool as PL  # noqa: E402
from pseudotest import folds, thin_to_pairs  # noqa: E402


ACTCATS = ["single", "multi", "combination", "sequence"]
REGIME = "ends"
PAIR_FRAC = 0.38
SPLIT_NAME = "oof_ends"
CHAMPION_SUBMISSION = ROOT / "submissions/submission_093859_SUBMITTED.csv"

V = json.loads((CHAMP_DIR / "vocab.json").read_text())["HARN2HAU"]
OPT_TO_HARN = {v: k for k, v in V.items()}
OPT2CLS = {V[a]: D.A2I[a] for a in D.ACTIONS}


def action_tokens(value: object) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def letters_to_actions(row, letters: object) -> set[str]:
    out: set[str] = set()
    for letter in str(letters):
        if letter in "ABCD":
            out.update(action_tokens(getattr(row, letter)))
    return out


def actions_key(values: set[str] | list[str] | tuple[str, ...]) -> str:
    return "||".join(sorted(values))


def safe_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sigmoid(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def enumerate_component_all(atoms, cq, forced, lo):
    """Exact all-candidate counterpart of pool._enum_component.

    The production solver maximises the same score over the same ``ok`` mask.  Keeping this
    routine local makes the atlas additive and prevents diagnostics from changing solver
    behavior.
    """
    atoms = list(atoms)
    n = len(atoms)
    if n > 22:
        return [], False
    X = np.arange(1 << n, dtype=np.int64)
    ok = np.ones(1 << n, dtype=bool)
    idx = {a: i for i, a in enumerate(atoms)}
    for cat, ops in cq:
        tcount = np.zeros(1 << n, dtype=np.int8)
        for op in ops:
            if any(a not in forced and a not in idx for a in op):
                continue
            mask = 0
            bad = False
            for a in op:
                if a in idx:
                    mask |= 1 << idx[a]
                elif a not in forced:
                    bad = True
            if bad:
                continue
            tcount += ((X & mask) == mask)
        if cat in ("single", "combination"):
            ok &= tcount == 1
        elif cat == "multi":
            ok &= tcount >= 1
        if not ok.any():
            return [], False
    w = np.array([lo.get(a, 0.0) for a in atoms], dtype=float)
    bits = ((X[:, None] >> np.arange(n)) & 1).astype(np.int8)
    scores = bits @ w - (1 - bits) @ w
    out = []
    for i in np.flatnonzero(ok):
        out.append((
            {atoms[j] for j in range(n) if int(i) >> j & 1},
            float(scores[i]),
        ))
    return out, True


def all_candidate_pools(vis, blk, statcache, scorer, split):
    """Return all full-block candidates and the exact production selected pool."""
    uni, rows, qs, forced = PL.candidate_evidence(vis, blk, split, statcache)
    if not qs:
        return dict(uni=[], rows={}, qs=[], forced=set(), candidates=[], selected=set(),
                    selected_score=np.nan, status="no_questions", lo={})
    lo = scorer(rows, uni)
    bias = PL.POOL_BIAS.get(str(len(blk)), 0.0)
    if bias:
        lo = {a: v + bias for a, v in lo.items()}
    free, comps, _fixed_q = PL._components(qs, uni, forced)
    component_candidates = []
    status = "enumerated"
    for atoms, cq in comps:
        cc, sat = enumerate_component_all(atoms, cq, forced, lo)
        if not sat or not cc:
            status = "unsat_fallback"
            cc = [({a for a in atoms if lo.get(a, -9.0) > 0},
                   float(sum(lo.get(a, 0.0) if lo.get(a, 0.0) > 0 else -lo.get(a, 0.0)
                             for a in atoms)))]
        component_candidates.append(cc)
    if not component_candidates:
        candidates = [(set(forced), 0.0)]
    else:
        candidates = []
        for pieces in itertools.product(*component_candidates):
            pool = set(forced)
            score = 0.0
            for piece, part_score in pieces:
                pool.update(piece)
                score += part_score
            candidates.append((pool, float(score)))
    candidates.sort(key=lambda x: (-x[1], actions_key(x[0])))
    selected = set(candidates[0][0]) if candidates else set(forced)
    selected_score = float(candidates[0][1]) if candidates else np.nan
    return dict(uni=uni, rows=rows, qs=qs, forced=set(forced), candidates=candidates,
                selected=selected, selected_score=selected_score, status=status, lo=lo,
                free=free, comps=comps)


def candidate_rank_view(details):
    by_action = {}
    candidates = details["candidates"]
    for rank, (pool, score) in enumerate(candidates, 1):
        for action in pool:
            if action not in by_action:
                by_action[action] = dict(rank=rank, score=float(score), pool=set(pool))
    return by_action


def multi_prediction_from_pool(row, pool: set[str]) -> str:
    hits = [i for i, letter in enumerate("ABCD")
            if all(a in pool for a in action_tokens(getattr(row, letter)))]
    if not hits:
        return ""
    return "".join("ABCD"[i] for i in sorted(hits))


def load_optional_presence(path: Path):
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def presence_for(z, key, cls):
    if z is None or key not in z:
        return np.nan
    try:
        return float(DC.presence_stats(z[key])["mx"][cls])
    except Exception:
        return np.nan


def make_truth_maps(tr):
    tpool = P.true_pool(tr)
    path_session = {}
    clip_truth = defaultdict(set)
    for r in tr[tr.source == "HAU"].itertuples():
        path_session[r.path] = (r.user, r.aa, r.bb)
        if r.category in ACTCATS:
            clip_truth[r.path].update(letters_to_actions(r, r.answer))
    return tpool, path_session, clip_truth


def train_action_rates(trn, tpool):
    sessions = sorted(tpool)
    counts = Counter()
    for key in sessions:
        counts.update(tpool[key])
    n = max(1, len(sessions))
    return dict(
        session_count=len(sessions),
        action_session_count=dict(counts),
        action_session_rate={a: c / n for a, c in counts.items()},
    )


def build_child_support(vis, pred, answer_by_q=None):
    """Return observable and audit-only HARn child support per HAU parent."""
    pred_support = defaultdict(Counter)
    gt_support = defaultdict(Counter)
    idx_by_path = dict(zip(vis.true_path, vis.idx))
    for r in vis[vis.source == "HARn"].itertuples():
        parent = P._C["nest"].get(r.true_path)
        if parent is None or r.category != "single":
            continue
        letter = pred.get(r.qa_id)
        if letter and len(str(letter)) == 1 and str(letter) in "ABCD":
            pred_support[parent][str(getattr(r, str(letter)))] += 1
        if answer_by_q is not None:
            gt = answer_by_q.get(r.qa_id)
            if gt and len(str(gt)) == 1 and str(gt) in "ABCD":
                gt_support[parent][str(getattr(r, str(gt)))] += 1
    return pred_support, gt_support


def modality_fields(path, split, meta_idx, skel, imu_keys, dino_keys, dense_dino):
    mr = meta_idx.loc[path] if path in meta_idx.index else None
    cls = dict(
        skeleton_available=int(path + "|K" in skel),
        physical_features_available=int(mr is not None and np.isfinite(mr.get("sk_v_mean", np.nan))),
        imu_available=int(path in imu_keys),
        dense_logits_available=int(f"{split}|{path}" in P._C["lg"]),
        dino_video_available=int(path in dino_keys),
    )
    cls["dino_dense_available"] = int(dense_dino is not None and f"{split}|{path}" in dense_dino)
    return cls


def fit_action_reliability(atlas):
    """OOF-only descriptive reliability of the pool classifier by fold/action."""
    if atlas.empty or "ground_truth_pool_member" not in atlas:
        return {}
    d = atlas[atlas.split == "oof"].copy()
    d = d[d.ground_truth_pool_member.notna() & d.pool_probability.notna()]
    rel = {}
    for (fold, action), g in d.groupby(["fold", "action"]):
        y = g.ground_truth_pool_member.astype(int).to_numpy()
        p = g.pool_probability.to_numpy(float)
        pred = (p >= 0.5).astype(int)
        rel[(int(fold), str(action))] = dict(
            n=int(len(g)),
            positive_rate=float(y.mean()),
            precision_at_0p5=float(y[pred == 1].mean()) if (pred == 1).any() else np.nan,
            recall_at_0p5=float(y[pred == 1].sum() / max(1, y.sum())),
            brier=float(np.mean((p - y) ** 2)),
        )
    return rel


def atlas_block(
    *, split, fold, block_id, vis, blk, details, pred, answer_by_q, aux, path_session,
    tpool, clip_truth, clip_multi_pred, clip_pred_by_cat, child_pred, child_gt, train_rates, meta_idx,
    skel, imu_keys, dino_keys, dense_dino, candidate_rows, block_rows, atlas_rows,
):
    pathof = dict(zip(vis.idx, vis.true_path))
    paths = [pathof[i] for i in blk]
    selected = details["selected"]
    # Use the production pool keyed by the first clip as the authoritative selected pool.
    # The reconstruction is checked below and a mismatch is retained as a diagnostic.
    # True session/block diagnostics are OOF-only.  They never enter the candidate scoring.
    true_ids = []
    pure = np.nan
    true_size = np.nan
    if aux is not None:
        true_map = aux.get("true_block", {})
        true_ids = sorted({true_map.get(p) for p in paths if p in true_map})
        true_ids = [x for x in true_ids if x is not None]
        pure = int(len(true_ids) == 1)
        if pure:
            true_size = len(aux["blocks"][true_ids[0]])
    qs = details["qs"]
    emotion_rows = vis[(vis.idx.isin(blk)) & (vis.category == "emotion")]
    eo = [set(opts(r)) for r in emotion_rows.itertuples()]
    inter = set.intersection(*eo) if eo else set()
    block_conform = int(len(inter) >= len(blk)) if eo else np.nan
    block_row = dict(
        split=split,
        fold=int(fold),
        block_id=block_id,
        block_indices=safe_json([int(x) for x in blk]),
        block_paths=safe_json(paths),
        block_size=int(len(blk)),
        block_regime="pair" if len(blk) == 2 else "triple" if len(blk) == 3 else f"{len(blk)}-clip",
        block_n_questions=int(len(qs)),
        block_universe_size=int(len(details["uni"])),
        block_free_size=int(len(details.get("free", []))),
        block_forced_actions=actions_key(details["forced"]),
        block_intersection_size=int(len(inter)),
        block_conformance=block_conform,
        true_session_ids=safe_json(true_ids),
        true_block_pure=pure,
        true_block_size=true_size,
        candidate_count=int(len(details["candidates"])),
        enumeration_status=details["status"],
        selected_pool=actions_key(selected),
        selected_pool_cardinality=int(len(selected)),
        selected_pool_score=float(details["selected_score"]),
        production_selected_pool=actions_key(details.get("pipeline_selected", selected)),
        production_pool_match=int(details.get("pipeline_selected", selected) == selected),
    )
    block_rows.append(block_row)

    for rank, (pool, score) in enumerate(details["candidates"], 1):
        candidate_rows.append(dict(
            split=split,
            fold=int(fold),
            block_id=block_id,
            block_size=int(len(blk)),
            candidate_rank=int(rank),
            candidate_score=float(score),
            candidate_pool=actions_key(pool),
            candidate_cardinality=int(len(pool)),
            is_selected=int(pool == selected),
        ))

    by_action = candidate_rank_view(details)
    idx_to_path = pathof
    # Sibling support is based on trial-level action evidence, not the selected session pool.
    for r in [x for x in qs if x.category == "multi"]:
        qid = r.qa_id
        gt_answer = answer_by_q.get(qid) if answer_by_q is not None else np.nan
        champ_answer = pred.get(qid) if split == "oof" else answer_by_q.get(qid)
        # test answer_by_q is supplied from the frozen submission in the test path
        if split == "oof":
            gt_actions = letters_to_actions(r, gt_answer)
            champ_actions = letters_to_actions(r, champ_answer)
        else:
            gt_actions = set()
            champ_actions = letters_to_actions(r, champ_answer)
        question_occ = defaultdict(list)
        for letter in "ABCD":
            for a in action_tokens(getattr(r, letter)):
                question_occ[a].append(letter)
        for action in details["uni"]:
            row = details["rows"].get(action, {})
            opt_letters = "".join(question_occ.get(action, []))
            in_q = int(bool(opt_letters))
            bxa = by_action.get(action)
            best_rank = bxa["rank"] if bxa else np.nan
            best_score = bxa["score"] if bxa else np.nan
            best_pool = bxa["pool"] if bxa else set()
            champ_pool_member = int(action in selected)
            best_pred = multi_prediction_from_pool(r, best_pool) if bxa else ""
            sibling_paths = [p for p in paths if p != r.true_path]
            gt_sib = [int(action in clip_truth.get(p, set())) for p in sibling_paths]
            pred_sib = [int(action in clip_multi_pred.get(p, set())) for p in sibling_paths]
            pred_sib_all = [int(any(action in clip_pred_by_cat.get(p, {}).get(cat, set())
                                    for cat in ACTCATS)) for p in sibling_paths]
            pred_sib_single = [int(action in clip_pred_by_cat.get(p, {}).get("single", set()))
                               for p in sibling_paths]
            pred_sib_comb = [int(action in clip_pred_by_cat.get(p, {}).get("combination", set()))
                             for p in sibling_paths]
            pred_sib_seq = [int(action in clip_pred_by_cat.get(p, {}).get("sequence", set()))
                            for p in sibling_paths]
            harn_pred = sum(child_pred.get(p, {}).get(action, 0) for p in paths)
            harn_gt = sum(child_gt.get(p, {}).get(action, 0) for p in paths)
            mf = modality_fields(r.true_path, split, meta_idx, skel, imu_keys, dino_keys, dense_dino)
            cls = OPT2CLS.get(action)
            stat = P.stat_of(split, r.true_path) if cls is not None else None
            training_count = train_rates["action_session_count"].get(action, 0)
            training_rate = train_rates["action_session_rate"].get(action, 0.0)
            gt_pool = np.nan
            gt_multi = np.nan
            if split == "oof":
                sess = path_session.get(r.true_path)
                gt_pool = int(sess is not None and action in tpool.get(sess, set()))
                gt_multi = int(action in gt_actions)
            champ_multi = int(action in champ_actions)
            pool_omission = int(gt_pool == 1 and champ_pool_member == 0) if split == "oof" else np.nan
            fn = int(gt_multi == 1 and champ_multi == 0) if split == "oof" else np.nan
            fp = int(gt_multi == 0 and champ_multi == 1) if split == "oof" else np.nan
            atlas_rows.append(dict(
                split=split,
                fold=int(fold),
                block_id=block_id,
                qa_id=qid,
                user=path_session.get(r.true_path, (np.nan, np.nan, np.nan))[0]
                    if split == "oof" else np.nan,
                trial_key="-".join(map(str, path_session.get(r.true_path, ("", "", ""))[1:]))
                    if split == "oof" else np.nan,
                source=r.source,
                category=r.category,
                block_size=int(len(blk)),
                block_regime="pair" if len(blk) == 2 else "triple" if len(blk) == 3 else f"{len(blk)}-clip",
                action=action,
                action_identity=OPT_TO_HARN.get(action, ""),
                option_letters=opt_letters,
                action_in_this_multi_options=in_q,
                ground_truth_answer=gt_answer if split == "oof" else np.nan,
                champion_answer=champ_answer,
                best_x_answer=best_pred,
                champion_multi_correct=(int(champ_answer == gt_answer) if split == "oof" else np.nan),
                ground_truth_pool_member=gt_pool,
                ground_truth_multi_member=gt_multi,
                champion_pool_member=champ_pool_member,
                champion_multi_member=champ_multi,
                pool_omission=pool_omission,
                false_negative_action=fn,
                spurious_action=fp,
                selected_session_pool=actions_key(selected),
                selected_pool_cardinality=int(len(selected)),
                selected_pool_score=float(details["selected_score"]),
                best_pool_containing_action=actions_key(best_pool) if bxa else "",
                best_pool_containing_action_rank=best_rank,
                best_pool_containing_action_score=best_score,
                selected_minus_best_x_score=(float(details["selected_score"] - best_score)
                                             if bxa else np.nan),
                best_x_pool_cardinality=int(len(best_pool)) if bxa else np.nan,
                best_x_added_actions=actions_key(best_pool - selected) if bxa else "",
                best_x_removed_actions=actions_key(selected - best_pool) if bxa else "",
                displaced_action=actions_key(selected - best_pool) if bxa else "",
                best_x_multi_correct=(int(best_pred == gt_answer) if split == "oof" else np.nan),
                best_x_would_change_question=int(best_pred != champ_answer) if bxa else 0,
                napp=float(row.get("napp", np.nan)),
                napp_frac=float(row.get("napp_frac", np.nan)),
                n_single=int(row.get("n_single", 0)),
                n_multi=int(row.get("n_multi", 0)),
                n_comb=int(row.get("n_comb", 0)),
                n_seq=int(row.get("n_seq", 0)),
                forced=int(row.get("forced", 0)),
                candidate_universe_size=int(row.get("uni", len(details["uni"]))),
                block_question_count=int(row.get("nq", len(details["qs"]))),
                co_max=float(row.get("co_max", np.nan)),
                co_mean=float(row.get("co_mean", np.nan)),
                co_n=int(row.get("co_n", 0)),
                dense_skel_imu_probability_max=float(row.get("mx_max", np.nan)),
                dense_skel_imu_probability_mean=float(row.get("mean_mean", np.nan)),
                dense_skel_imu_probability_topk=float(row.get("topk_max", np.nan)),
                dense_skel_imu_probability_fraction=float(row.get("frac_max", np.nan)),
                dense_skel_imu_logit_max=float(row.get("lmx_max", np.nan)),
                dense_dino_probability_max=presence_for(dense_dino, f"{split}|{r.true_path}", cls)
                    if cls is not None else np.nan,
                pool_logodds=float(details["lo"].get(action, np.nan)),
                pool_probability=sigmoid(float(details["lo"].get(action, np.nan))),
                train_action_session_count=int(training_count),
                train_action_session_rate=float(training_rate),
                sibling_count=int(len(sibling_paths)),
                sibling_gt_action_count=int(sum(gt_sib)),
                sibling_pred_multi_action_count=int(sum(pred_sib)),
                sibling_pred_action_count=int(sum(pred_sib_all)),
                sibling_pred_single_action_count=int(sum(pred_sib_single)),
                sibling_pred_combination_action_count=int(sum(pred_sib_comb)),
                sibling_pred_sequence_action_count=int(sum(pred_sib_seq)),
                harn_child_pred_support_count=int(harn_pred),
                harn_child_gt_support_count=int(harn_gt) if split == "oof" else np.nan,
                parent_session_pool_support=int(champ_pool_member),
                block_conformance=block_conform,
                block_intersection_size=int(len(inter)),
                candidate_pool_count=int(len(details["candidates"])),
                enumeration_status=details["status"],
                **mf,
            ))


def run_oof(tr, te, meta, tpool, path_session, clip_truth, dense_dino):
    P._C.clear()
    P.caches(meta)
    users = sorted(tr.user.dropna().unique())
    action_rows, block_rows, candidate_rows = [], [], []
    for fi, hold in enumerate(folds(users, 5)):
        ctx = P.fit_all(tr, meta, hold)
        tr_eval = thin_to_pairs(tr, hold, PAIR_FRAC, seed=fi, policy=REGIME)
        vis, key, aux = make_pseudo(tr_eval, meta, hold, seed=100 + fi)
        pred, blocks, _pool_of, _diag = P.solve(vis, ctx, "oof")
        answers = dict(zip(key.qa_id, key.answer))
        child_pred, child_gt = build_child_support(vis, pred, answers)
        train_rates = train_action_rates(tr[~tr.user.isin(hold)], P.true_pool(tr[~tr.user.isin(hold)]))
        statcache = P.statcache_view("oof")
        pathof = dict(zip(vis.idx, vis.true_path))
        idx_by_path = {p: i for i, p in pathof.items()}
        clip_multi_pred = defaultdict(set)
        clip_pred_by_cat = defaultdict(lambda: defaultdict(set))
        for r in vis[vis.category == "multi"].itertuples():
            clip_multi_pred[r.true_path].update(letters_to_actions(r, pred.get(r.qa_id, "")))
        for r in vis[vis.category.isin(ACTCATS)].itertuples():
            clip_pred_by_cat[r.true_path][r.category].update(letters_to_actions(r, pred.get(r.qa_id, "")))
        # The current validated T layer is part of the live Multi champion.  This diagnostic
        # reconstructs its selected pool from the actual returned block, while preserving a
        # mismatch if any implementation/environment detail changes.
        for bi, blk in enumerate(blocks):
            details = all_candidate_pools(vis, blk, statcache, ctx["scorer"], "oof")
            selected_from_pipeline = set(_pool_of.get(int(blk[0]), details["selected"]))
            details["pipeline_selected"] = selected_from_pipeline
            if selected_from_pipeline != details["selected"]:
                details["selected"] = selected_from_pipeline
                details["candidates"].sort(key=lambda x: (-x[1], actions_key(x[0])))
                for i, (pool, score) in enumerate(details["candidates"]):
                    if pool == selected_from_pipeline:
                        details["selected_score"] = float(score)
                        break
            block_id = f"f{fi}b{bi:03d}"
            atlas_block(
                split="oof", fold=fi, block_id=block_id, vis=vis, blk=blk,
                details=details, pred=pred, answer_by_q=answers, aux=aux,
                path_session=path_session, tpool=tpool, clip_truth=clip_truth,
                clip_multi_pred=clip_multi_pred, clip_pred_by_cat=clip_pred_by_cat,
                child_pred=child_pred, child_gt=child_gt,
                train_rates=train_rates, meta_idx=P._C["meta_idx"], skel=P._C["skel"],
                imu_keys=RUN_STATE["imu_keys"], dino_keys=RUN_STATE["dino_keys"],
                dense_dino=dense_dino, candidate_rows=candidate_rows,
                block_rows=block_rows, atlas_rows=action_rows,
            )
        print(f"OOF fold {fi}: {len(hold)} users, {len(blocks)} blocks, "
              f"{sum(1 for r in action_rows if r['fold'] == fi)} action rows", flush=True)
    return pd.DataFrame(action_rows), pd.DataFrame(block_rows), pd.DataFrame(candidate_rows)


def run_test(tr, te, meta, tpool, path_session, clip_truth, dense_dino):
    P._C.clear()
    P.caches(meta)
    ctx = P.fit_all(tr, meta, hold_users=[])
    vis = real_test_view(te)
    pred, blocks, _pool_of, _diag = P.solve(vis, ctx, "test")
    frozen = pd.read_csv(CHAMPION_SUBMISSION).set_index("qa_id")["prediction"].to_dict()
    child_pred, child_gt = build_child_support(vis, pred, None)
    train_rates = train_action_rates(tr, tpool)
    statcache = P.statcache_view("test")
    action_rows, block_rows, candidate_rows = [], [], []
    clip_multi_pred = defaultdict(set)
    clip_pred_by_cat = defaultdict(lambda: defaultdict(set))
    for r in vis[vis.category == "multi"].itertuples():
        clip_multi_pred[r.true_path].update(letters_to_actions(r, frozen.get(r.qa_id, "")))
    for r in vis[vis.category.isin(ACTCATS)].itertuples():
        clip_pred_by_cat[r.true_path][r.category].update(letters_to_actions(r, frozen.get(r.qa_id, "")))
    for bi, blk in enumerate(blocks):
        details = all_candidate_pools(vis, blk, statcache, ctx["scorer"], "test")
        details["pipeline_selected"] = set(_pool_of.get(int(blk[0]), details["selected"]))
        if details["pipeline_selected"] != details["selected"]:
            details["selected"] = details["pipeline_selected"]
            for pool, score in details["candidates"]:
                if pool == details["selected"]:
                    details["selected_score"] = float(score)
                    break
        block_id = f"testb{bi:03d}"
        test_answers = {q: frozen.get(q, "") for q in te.qa_id}
        atlas_block(
            split="test", fold=-1, block_id=block_id, vis=vis, blk=blk,
            details=details, pred=pred, answer_by_q=test_answers, aux=None,
            path_session=path_session, tpool=tpool, clip_truth=clip_truth,
            clip_multi_pred=clip_multi_pred, clip_pred_by_cat=clip_pred_by_cat,
            child_pred=child_pred, child_gt=child_gt,
            train_rates=train_rates, meta_idx=P._C["meta_idx"], skel=P._C["skel"],
            imu_keys=RUN_STATE["imu_keys"], dino_keys=RUN_STATE["dino_keys"],
            dense_dino=dense_dino, candidate_rows=candidate_rows,
            block_rows=block_rows, atlas_rows=action_rows,
        )
    return pd.DataFrame(action_rows), pd.DataFrame(block_rows), pd.DataFrame(candidate_rows)


def summary(oof, oof_blocks, test, test_blocks):
    out = dict(
        champion_artifact=str(CHAMPION_SUBMISSION),
        champion_sha256=hashlib.sha256(CHAMPION_SUBMISSION.read_bytes()).hexdigest(),
        oof_protocol=dict(subject_disjoint=True, folds=5, pair_frac=PAIR_FRAC,
                          withheld_trial_policy=REGIME, repair=True, conform_first=True),
        oof_action_rows=int(len(oof)),
        oof_multi_questions=int(oof.qa_id.nunique()) if len(oof) else 0,
        oof_multi_exact_correct=int(oof.drop_duplicates("qa_id").champion_multi_correct.sum())
            if len(oof) else 0,
        oof_multi_exact_total=int(oof.qa_id.nunique()) if len(oof) else 0,
        oof_false_negative_actions=int(oof.false_negative_action.sum()) if len(oof) else 0,
        oof_spurious_actions=int(oof.spurious_action.sum()) if len(oof) else 0,
        oof_pool_omissions=int(oof.pool_omission.sum()) if len(oof) else 0,
        oof_blocks=int(len(oof_blocks)),
        oof_candidate_pool_rows=int(len(CANDIDATE_STATE["oof"])),
        test_multi_questions=int(test.qa_id.nunique()) if len(test) else 0,
        test_action_rows=int(len(test)),
        test_blocks=int(len(test_blocks)),
        test_candidate_pool_rows=int(len(CANDIDATE_STATE["test"])),
    )
    for regime, d in [("oof", oof), ("test", test)]:
        if len(d):
            out[f"{regime}_by_block_regime"] = {
                str(k): dict(n=int(len(g)), questions=int(g.qa_id.nunique()),
                             fn=int(g.false_negative_action.sum()) if regime == "oof" else None,
                             fp=int(g.spurious_action.sum()) if regime == "oof" else None)
                for k, g in d.groupby("block_regime")
            }
    return out


RUN_STATE = {}
CANDIDATE_STATE = {}


def main():
    tr, te, meta = load_all()
    P.caches(meta)
    tpool, path_session, clip_truth = make_truth_maps(tr)
    # Presence caches are read-only diagnostic inputs.  dense_dino is never fed into the
    # champion solver; it is recorded only if a same-key cached alternative exists.
    imu = np.load(CHAMP_DIR / "imu_seq.npz", allow_pickle=True) if (CHAMP_DIR / "imu_seq.npz").exists() else None
    dino = np.load(CHAMP_DIR / "dino_frames.npz", allow_pickle=True) if (CHAMP_DIR / "dino_frames.npz").exists() else None
    dense_dino = load_optional_presence(CHAMP_DIR / "dense_dino_logits.npz")
    RUN_STATE.update(
        imu_keys=set(imu.files) if imu is not None else set(),
        dino_keys=set(dino.files) if dino is not None else set(),
    )
    oof, oof_blocks, oof_candidates = run_oof(tr, te, meta, tpool, path_session, clip_truth, dense_dino)
    CANDIDATE_STATE["oof"] = oof_candidates
    test, test_blocks, test_candidates = run_test(tr, te, meta, tpool, path_session, clip_truth, dense_dino)
    CANDIDATE_STATE["test"] = test_candidates

    rel = fit_action_reliability(oof)
    for d in (oof,):
        if len(d):
            d["action_oof_reliability_n"] = [rel.get((int(f), a), {}).get("n", np.nan)
                                               for f, a in zip(d.fold, d.action)]
            d["action_oof_precision_at_0p5"] = [rel.get((int(f), a), {}).get("precision_at_0p5", np.nan)
                                                  for f, a in zip(d.fold, d.action)]
            d["action_oof_brier"] = [rel.get((int(f), a), {}).get("brier", np.nan)
                                      for f, a in zip(d.fold, d.action)]
    out_map = {
        "action_atlas_oof_ends.csv": oof,
        "block_atlas_oof_ends.csv": oof_blocks,
        "candidate_pools_oof_ends.csv": oof_candidates,
        "action_atlas_test.csv": test,
        "block_atlas_test.csv": test_blocks,
        "candidate_pools_test.csv": test_candidates,
    }
    for name, frame in out_map.items():
        frame.to_csv(HERE / name, index=False)
    s = summary(oof, oof_blocks, test, test_blocks)
    s["action_reliability"] = {f"fold{f}|{a}": v for (f, a), v in rel.items()}
    (HERE / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True, default=str) + "\n")

    print("\n=== ACTION-LEVEL MULTI ATLAS ===")
    print(f"champion: {CHAMPION_SUBMISSION} sha256={s['champion_sha256']}")
    print(f"OOF action rows: {len(oof)} across {oof.qa_id.nunique()} Multi questions")
    print(f"OOF false-negative actions: {int(oof.false_negative_action.sum())}; "
          f"spurious actions: {int(oof.spurious_action.sum())}; "
          f"pool omissions: {int(oof.pool_omission.sum())}")
    print(f"OOF Multi exact: {int(oof.drop_duplicates('qa_id').champion_multi_correct.sum())}/"
          f"{oof.qa_id.nunique()} = {oof.drop_duplicates('qa_id').champion_multi_correct.mean():.4f}")
    print("\nfalse negatives by regime/action:")
    fn = oof[oof.false_negative_action == 1]
    if len(fn):
        print(fn.groupby(["block_regime", "action"]).size().sort_values(ascending=False).head(40).to_string())
    else:
        print("  none")
    print("\nOOF exact Multi by regime:")
    q = oof.drop_duplicates("qa_id")
    print(q.groupby("block_regime").champion_multi_correct.agg(n="size", ok="sum", acc="mean").round(4).to_string())
    print("\nTest action rows/candidate pools:", len(test), len(test_candidates))
    print("Outputs:", HERE)


if __name__ == "__main__":
    main()
