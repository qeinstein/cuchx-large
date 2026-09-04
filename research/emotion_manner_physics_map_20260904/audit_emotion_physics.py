#!/usr/bin/env python3
"""Reproducible residual-confusion and physical-signal audit for HAU emotion.

This is an analysis-only artifact.  It reads the preserved champion OOF snapshot
(`champ/audit_champ.csv`), labels, and feature caches; it neither trains a model
nor writes a submission.  The output directory is intentionally isolated from
`champ/` so research cannot overwrite a production artifact.

Run from the repository root:

    ./venv/bin/python research/emotion_manner_physics_map_20260904/audit_emotion_physics.py

The report distinguishes the frozen audit snapshot from other OOF CSVs because
their aggregate scores differ.  Do not substitute a different OOF file without
recording the change in the report provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLES = HERE / "tables"

# This is deliberately copied from champ/core.py rather than imported from it:
# core.py is active shared pipeline code and can be edited while this audit runs.
FAST = {
    "Quickly", "quickly", "Rapidly", "Hastily", "Hasitly", "Hurriedly", "Swiftly",
    "Briskly", "Urgently", "Frantically", "Impatiently", "Eagerly", "Forcefully",
}
SLOW = {
    "Slowly", "Leisurely", "Unhurriedly", "Calmly", "Peacefully", "Relaxedly",
    "Lazily", "Gently", "Softly", "Quietly", "Comfortably", "Soothingly",
    "Patiently", "Lightly", "Contently", "Absentmindedly",
}
CARE = {
    "Carefully", "Cautiously", "Meticulously", "Precisely", "Thoroughly",
    "Deliberately", "Methodically", "Attentively", "Intently", "Diligently",
    "Earnestly", "Neatly", "Orderly", "Seriously", "Serioiusly",
}
NERV = {"Nervously", "Anxiously", "Tensely", "Tensly", "Restlessly"}
GROUPS = ["SLOW", "CARE", "NEUT", "NERV", "FAST"]

# The same coarse context split used by champ/emopair.py.  An emotion observation
# inherits the whole session script, so this is a script-context annotation, not a
# frame-level action label.
LOCOMOTION = {
    "Walking", "Running", "Squats", "Lunges", "Jumping jacks", "Stretching",
    "Sitting down", "Standing up", "Lying down", "Mopping", "Sweeping",
}
ACT_CATS = ["single", "multi", "combination", "sequence"]
HAU_RE = re.compile(r"^HAU/(user\d+)/(\d+)-(\d+)-(\d+)$")


def mgroup(label: str) -> str:
    if label in FAST:
        return "FAST"
    if label in SLOW:
        return "SLOW"
    if label in CARE:
        return "CARE"
    if label in NERV:
        return "NERV"
    return "NEUT"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qlabel(row: pd.Series, letter: object) -> str | None:
    """Return the option text for a one-letter emotion decision, if valid."""
    text = str(letter).strip()
    if len(text) == 1 and text in "ABCD":
        return str(row[text]).strip()
    return None


def hau_parts(path: str) -> tuple[str, str, str, str]:
    m = HAU_RE.match(str(path))
    if not m:
        raise ValueError(f"not an HAU path: {path!r}")
    return m.groups()


def session_key(row: pd.Series) -> tuple[str, str, str]:
    return (str(row.user), str(row.aa), str(row.bb))


def markdown_table(frame: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    """Small, dependency-free Markdown formatter for the report's headline tables."""
    d = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    if n is not None:
        d = d.head(n)
    if d.empty:
        return "_(none)_"
    for col in d.columns:
        d[col] = d[col].map(lambda x: "" if pd.isna(x) else str(x))
    head = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in d.astype(str).itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def fmt_pct(x: float) -> str:
    return "" if not np.isfinite(x) else f"{100 * x:.1f}%"


def finite(x: object) -> bool:
    return pd.notna(x) and np.isfinite(float(x))


def raw_skeleton_dynamics(rows: pd.DataFrame) -> pd.DataFrame:
    """Derive transparent temporal measures from the existing skeleton cache.

    The cache was built from raw Skeleton/predictions files.  We use the stored frame
    indices to express velocity/acceleration in seconds, centre on the hip, and scale
    by torso length.  `low_motion_*` uses 10% of that clip's p90 speed; it is a
    relative low-motion indicator, not a human-annotated pause label.
    """
    cache_path = ROOT / "champ" / "skel_seq.npz"
    z = np.load(cache_path, allow_pickle=False)
    out: list[dict[str, object]] = []
    for row in rows[["path", "unit_dir", "fps"]].drop_duplicates("path").itertuples(index=False):
        key_k = f"{row.unit_dir}|K"
        key_f = f"{row.unit_dir}|F"
        rec: dict[str, object] = {"path": row.path}
        if key_k not in z or key_f not in z:
            out.append(rec)
            continue
        K = np.asarray(z[key_k], dtype=float)
        F = np.asarray(z[key_f], dtype=float)
        if K.ndim != 3 or K.shape[0] < 8 or K.shape[1] < 17 or len(F) != len(K):
            out.append(rec)
            continue
        fps = float(row.fps) if finite(row.fps) and float(row.fps) > 0 else 10.0
        hip = K[:, [11, 12]].mean(axis=1, keepdims=True)
        torso = (np.linalg.norm(K[:, 5] - K[:, 11], axis=-1)
                 + np.linalg.norm(K[:, 6] - K[:, 12], axis=-1))
        torso = torso[np.isfinite(torso) & (torso > 1e-6)]
        scale = float(np.median(torso)) if len(torso) else 1.0
        P = (K - hip) / max(scale, 1e-6)
        dt = np.diff(F) / fps
        dt[~np.isfinite(dt) | (dt <= 0)] = 1.0 / fps
        V = np.diff(P, axis=0) / dt[:, None, None]
        speed = np.linalg.norm(V, axis=-1).mean(axis=1)
        wrist = np.linalg.norm(V[:, [9, 10]], axis=-1).mean(axis=1)
        ankle = np.linalg.norm(V[:, [15, 16]], axis=-1).mean(axis=1)
        # Hip motion was factored out of P; preserve it separately for locomotion.
        H = hip[:, 0] / max(scale, 1e-6)
        H_v = np.diff(H, axis=0) / dt[:, None]
        hip_speed = np.linalg.norm(H_v, axis=-1)
        dt2 = (dt[1:] + dt[:-1]) / 2.0
        A = np.diff(V, axis=0) / dt2[:, None, None]
        acc = np.linalg.norm(A, axis=-1).mean(axis=1) if len(A) else np.array([])
        if len(A) >= 2:
            dt3 = (dt2[1:] + dt2[:-1]) / 2.0
            J = np.diff(A, axis=0) / dt3[:, None, None]
            jerk = np.linalg.norm(J, axis=-1).mean(axis=1)
        else:
            jerk = np.array([])
        p90 = float(np.nanpercentile(speed, 90)) if len(speed) else np.nan
        threshold = max(1e-6, 0.10 * p90) if finite(p90) else np.nan
        low = speed <= threshold if finite(threshold) else np.zeros(len(speed), dtype=bool)
        run = longest = 0
        for value, dti in zip(low, dt):
            if value:
                run += float(dti)
                longest = max(longest, run)
            else:
                run = 0.0
        # Mean joint 5th--95th percentile range is a scale-normalised pose amplitude.
        amp = np.linalg.norm(np.nanpercentile(P, 95, axis=0) - np.nanpercentile(P, 5, axis=0), axis=-1)
        hp = np.linalg.norm(np.diff(H, axis=0), axis=-1)
        hip_path = float(np.nansum(hp))
        hip_net = float(np.linalg.norm(H[-1] - H[0]))
        # The dominant speed frequency is only meaningful if the clip has enough frames.
        cad_hz = cad_peak = np.nan
        if len(speed) >= 32:
            med_dt = float(np.median(dt))
            sig = speed - np.nanmean(speed)
            power = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
            freq = np.fft.rfftfreq(len(sig), d=med_dt)
            band = (freq >= 0.10) & (freq <= min(4.0, 0.5 / med_dt))
            if np.any(band) and np.nansum(power[band]) > 0:
                j = np.flatnonzero(band)[np.argmax(power[band])]
                cad_hz = float(freq[j])
                cad_peak = float(power[j] / np.nansum(power[band]))
        def avg(a: np.ndarray) -> float:
            return float(np.nanmean(a)) if len(a) else np.nan
        def pctl(a: np.ndarray, q: float) -> float:
            return float(np.nanpercentile(a, q)) if len(a) else np.nan
        rec.update({
            "raw_duration_s": float(np.nansum(dt)),
            "raw_speed_mean": avg(speed), "raw_speed_p90": pctl(speed, 90),
            "raw_speed_cv": float(np.nanstd(speed) / (np.nanmean(speed) + 1e-9)) if len(speed) else np.nan,
            "raw_acc_mean": avg(acc), "raw_jerk_mean": avg(jerk),
            "raw_wrist_speed_mean": avg(wrist), "raw_ankle_speed_mean": avg(ankle),
            "raw_wrist_ankle_ratio": avg(wrist) / (avg(ankle) + 1e-9) if len(speed) else np.nan,
            "raw_hip_speed_mean": avg(hip_speed), "raw_hip_path": hip_path,
            "raw_hip_efficiency": hip_net / (hip_path + 1e-9),
            "raw_pose_amplitude": avg(amp),
            "raw_low_motion_frac": float(np.mean(low)) if len(low) else np.nan,
            "raw_longest_low_motion_s": float(longest),
            "raw_cad_hz": cad_hz, "raw_cad_peak_frac": cad_peak,
        })
        out.append(rec)
    z.close()
    return pd.DataFrame(out)


def add_within_session_features(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Add action-/subject-normalised values by centring/ranking inside a session."""
    d = d.copy()
    additions: dict[str, pd.Series] = {}
    for feature in features:
        if feature not in d:
            continue
        x = pd.to_numeric(d[feature], errors="coerce")
        d[feature] = x
        zcol, rcol = f"z__{feature}", f"rank__{feature}"
        z = pd.Series(np.nan, index=d.index, dtype=float)
        rank = pd.Series(np.nan, index=d.index, dtype=float)
        for _, idx in d.groupby(["user", "aa", "bb"], sort=False).groups.items():
            v = x.loc[idx]
            valid = v.dropna()
            if len(valid) < 2:
                continue
            sd = float(valid.std(ddof=0))
            z.loc[valid.index] = ((valid - valid.mean()) / sd) if sd > 1e-12 else 0.0
            rank.loc[valid.index] = (valid.rank(method="average") - 1) / max(1, len(valid) - 1)
        additions[zcol] = z
        additions[rcol] = rank
    return pd.concat([d, pd.DataFrame(additions, index=d.index)], axis=1)


def effect_screen(d: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive group signal screen on session-normalised ranks.

    eta2 is a one-way between-group share of variation in ranks; it is *not* a
    held-out model metric.  The screen is for choosing probes, not deciding an
    override rule.
    """
    rows, group_rows = [], []
    for feature in features:
        col = f"rank__{feature}"
        if col not in d:
            continue
        sub = d[["group", col]].dropna()
        if sub.empty:
            continue
        values = [sub.loc[sub.group == g, col].to_numpy(float) for g in GROUPS]
        values = [v for v in values if len(v)]
        if len(values) >= 2:
            _, p = stats.f_oneway(*values)
        else:
            p = np.nan
        grand = float(sub[col].mean())
        total = float(((sub[col] - grand) ** 2).sum())
        between = sum(len(v) * (float(np.mean(v)) - grand) ** 2 for v in values)
        eta2 = between / total if total > 1e-12 else np.nan
        mean_rank = {}
        for g in GROUPS:
            x = sub.loc[sub.group == g, col]
            mean_rank[g] = float(x.mean()) if len(x) else np.nan
            zcol = f"z__{feature}"
            zx = d.loc[d.group == g, zcol].dropna() if zcol in d else pd.Series(dtype=float)
            group_rows.append({
                "feature": feature, "group": g, "n": int(len(x)),
                "median_within_session_rank": float(x.median()) if len(x) else np.nan,
                "mean_within_session_rank": mean_rank[g],
                "mean_within_session_z": float(zx.mean()) if len(zx) else np.nan,
            })
        rows.append({
            "feature": feature, "n": int(len(sub)), "eta2_group_rank": eta2,
            "anova_p_exploratory": p,
            "fast_minus_slow_mean_rank": mean_rank["FAST"] - mean_rank["SLOW"],
            "care_minus_neut_mean_rank": mean_rank["CARE"] - mean_rank["NEUT"],
            "nerv_minus_neut_mean_rank": mean_rank["NERV"] - mean_rank["NEUT"],
        })
    screen = pd.DataFrame(rows).sort_values(["eta2_group_rank", "feature"], ascending=[False, True])
    return screen, pd.DataFrame(group_rows)


def label_stats(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for label, g in d.groupby("true_label", sort=True):
        wrong = g[~g.correct_bool]
        pred_counts = wrong.pred_label.dropna().value_counts()
        top = "; ".join(f"{p} ({n})" for p, n in pred_counts.head(4).items())
        rec = {
            "label": label, "group": mgroup(label), "n": len(g),
            "correct": int(g.correct_bool.sum()), "errors": int((~g.correct_bool).sum()),
            "accuracy": g.correct_bool.mean(), "top_wrong_predictions": top,
            "median_duration_s": g.duration_s.median(),
            "median_raw_speed": g.raw_speed_mean.median(),
            "median_raw_low_motion_frac": g.raw_low_motion_frac.median(),
        }
        for feature in ["duration_s", "raw_speed_mean", "raw_acc_mean", "raw_jerk_mean",
                        "raw_pose_amplitude", "raw_low_motion_frac", "raw_cad_hz"]:
            z = f"z__{feature}"
            r = f"rank__{feature}"
            if z in g:
                rec[f"mean_z_{feature}"] = g[z].mean()
            if r in g:
                rec[f"mean_rank_{feature}"] = g[r].mean()
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["errors", "n", "label"], ascending=[False, False, True])


def confusion_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = d[d.pred_label.notna()].copy()
    labels = sorted(set(valid.true_label) | set(valid.pred_label))
    wide = pd.crosstab(valid.true_label, valid.pred_label).reindex(index=labels, columns=labels, fill_value=0)
    long = (wide.rename_axis(index="true_label", columns="pred_label").stack().rename("n").reset_index())
    long["is_diagonal"] = long.true_label == long.pred_label
    err = long[~long.is_diagonal & (long.n > 0)].copy()
    true_counts = valid.true_label.value_counts()
    err["true_label_n"] = err.true_label.map(true_counts)
    err["share_of_true_label"] = err.n / err.true_label_n
    examples = (valid[~valid.correct_bool].groupby(["true_label", "pred_label"]).qa_id
                .agg(lambda s: ", ".join(s.head(8))).rename("example_qa_ids").reset_index())
    err = err.merge(examples, on=["true_label", "pred_label"], how="left")
    err["true_group"] = err.true_label.map(mgroup)
    err["pred_group"] = err.pred_label.map(mgroup)
    err = err.sort_values(["n", "true_label", "pred_label"], ascending=[False, True, True])

    sym_rows = []
    for (a, b), g in err.assign(lo=lambda x: x[["true_label", "pred_label"]].min(axis=1),
                                hi=lambda x: x[["true_label", "pred_label"]].max(axis=1)).groupby(["lo", "hi"]):
        ab = int(g.loc[(g.true_label == a) & (g.pred_label == b), "n"].sum())
        ba = int(g.loc[(g.true_label == b) & (g.pred_label == a), "n"].sum())
        sym_rows.append({"label_a": a, "label_b": b, "a_to_b": ab, "b_to_a": ba,
                         "symmetric_error_count": ab + ba,
                         "balanced_min_direction": min(ab, ba),
                         "group_a": mgroup(a), "group_b": mgroup(b)})
    sym = pd.DataFrame(sym_rows).sort_values(
        ["symmetric_error_count", "balanced_min_direction", "label_a", "label_b"],
        ascending=[False, False, True, True],
    )

    group_wide = pd.crosstab(valid.group, valid.pred_group).reindex(index=GROUPS, columns=GROUPS, fill_value=0)
    group_long = group_wide.rename_axis(index="true_group", columns="pred_group").stack().rename("n").reset_index()
    return long, err, sym, group_long


def residual_clusters(sym: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High-confidence components: at least two OOF errors in *both* directions.

    A single A->B/B->A pair is a natural result of one bad assignment block and
    creates misleading giant graph components.  The stricter criterion isolates only
    repeatedly observed, balanced confusion hypotheses; the complete weak-edge map
    remains available in residual_confusion_symmetric.csv.
    """
    g = nx.Graph()
    for r in sym.itertuples(index=False):
        if r.symmetric_error_count >= 4 and r.balanced_min_direction >= 2:
            g.add_edge(r.label_a, r.label_b, weight=int(r.symmetric_error_count))
    edges = pd.DataFrame([
        {"label_a": a, "label_b": b, "symmetric_error_count": attrs["weight"]}
        for a, b, attrs in g.edges(data=True)
    ])
    rows = []
    for i, nodes in enumerate(sorted(nx.connected_components(g), key=lambda x: (-len(x), sorted(x))), start=1):
        sub = g.subgraph(nodes)
        rows.append({
            "cluster_id": i, "n_labels": len(nodes), "labels": " | ".join(sorted(nodes)),
            "edge_weight": sum(d["weight"] for _, _, d in sub.edges(data=True)),
            "n_repeated_links": sub.number_of_edges(),
        })
    return pd.DataFrame(rows), edges


def swap_summary(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate genuine assignment swaps from independent wrong labels."""
    rows, pair_rows = [], []
    for session, g in d.groupby(["user", "aa", "bb"], sort=False):
        g = g.sort_values("cc")
        bad = g[~g.correct_bool & g.pred_label.notna()]
        if not len(bad):
            continue
        truth = list(g.true_label)
        pred = list(g.pred_label)
        permutation_only = Counter(truth) == Counter(pred)
        pairs = []
        for i, j in itertools.combinations(range(len(g)), 2):
            if truth[i] == pred[j] and truth[j] == pred[i] and truth[i] != truth[j]:
                pairs.append((truth[i], truth[j]))
                pair_rows.append({
                    "user": session[0], "aa": session[1], "bb": session[2],
                    "label_a": min(truth[i], truth[j]), "label_b": max(truth[i], truth[j]),
                    "qa_a": g.iloc[i].qa_id, "qa_b": g.iloc[j].qa_id,
                })
        rows.append({
            "user": session[0], "aa": session[1], "bb": session[2], "block_size": len(g),
            "errors": len(bad), "permutation_only": permutation_only,
            "exact_two_cycle_count": len(pairs),
            "qa_ids": ", ".join(bad.qa_id),
        })
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def action_context_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exploded = d[["qa_id", "true_label", "group", "correct_bool", "action_pool",
                  "duration_s", "raw_speed_mean"]].copy()
    exploded["action"] = exploded.action_pool.str.split(" | ", regex=False)
    exploded = exploded.explode("action").dropna(subset=["action"])
    action = (exploded.groupby("action").agg(
        emotion_rows=("qa_id", "size"), errors=("correct_bool", lambda x: int((~x).sum())),
        accuracy=("correct_bool", "mean"), n_labels=("true_label", "nunique"),
        n_groups=("group", "nunique"),
    ).reset_index())
    action["common_true_labels"] = (exploded.groupby("action").true_label
                                     .agg(lambda x: "; ".join(f"{a} ({n})" for a, n in x.value_counts().head(5).items()))
                                     .reindex(action.action).to_numpy())
    action = action.sort_values(["errors", "emotion_rows", "action"], ascending=[False, False, True])

    ag = (exploded.groupby(["action", "group"]).agg(
        emotion_rows=("qa_id", "size"), errors=("correct_bool", lambda x: int((~x).sum())),
        accuracy=("correct_bool", "mean"), median_duration_s=("duration_s", "median"),
        median_raw_speed=("raw_speed_mean", "median"),
    ).reset_index().sort_values(["action", "group"]))
    script = (d.groupby("action_pool").agg(
        emotion_rows=("qa_id", "size"), sessions=("session_id", "nunique"),
        errors=("correct_bool", lambda x: int((~x).sum())), accuracy=("correct_bool", "mean"),
        n_actions=("n_actions", "first"), any_locomotion=("any_locomotion", "first"),
    ).reset_index().sort_values(["errors", "emotion_rows", "action_pool"], ascending=[False, False, True]))
    return action, ag, script


def pair_physical_map(d: pd.DataFrame, sym: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Action-normalised physical contrasts for labels linked by residual confusion.

    Every comparison is within a session containing both true labels, so script and
    subject are held constant.  It remains descriptive: some pairs have only a few
    matched sessions.
    """
    rows = []
    for pair in sym.itertuples(index=False):
        a, b = pair.label_a, pair.label_b
        deltas: dict[str, list[float]] = defaultdict(list)
        sessions = 0
        for _, g in d.groupby("session_id"):
            ga = g[g.true_label == a]
            gb = g[g.true_label == b]
            if not len(ga) or not len(gb):
                continue
            sessions += 1
            # A protocol normally appears once, but mean protects against malformed input.
            for feat in features:
                col = f"rank__{feat}"
                if col not in g:
                    continue
                va, vb = ga[col].mean(), gb[col].mean()
                if finite(va) and finite(vb):
                    deltas[feat].append(float(va - vb))
        for feat, xlist in deltas.items():
            x = np.asarray(xlist, dtype=float)
            if not len(x):
                continue
            sd = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
            rows.append({
                "label_a": a, "label_b": b,
                "symmetric_oof_errors": pair.symmetric_error_count,
                "matched_sessions": sessions, "feature": feat, "n_pairs": len(x),
                "mean_a_minus_b_rank": float(np.mean(x)),
                "median_a_minus_b_rank": float(np.median(x)),
                "sign_consistency": float(max(np.mean(x > 0), np.mean(x < 0))),
                "paired_standardised_mean": float(np.mean(x) / sd) if sd and np.isfinite(sd) and sd > 1e-12 else np.nan,
            })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["abs_mean_rank_delta"] = result.mean_a_minus_b_rank.abs()
    result = (result.sort_values(["label_a", "label_b", "abs_mean_rank_delta"], ascending=[True, True, False])
              .groupby(["label_a", "label_b"], as_index=False).head(8)
              .sort_values(["symmetric_oof_errors", "abs_mean_rank_delta"], ascending=[False, False]))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="champ/audit_champ.csv",
                    help="frozen OOF snapshot relative to the repository root")
    args = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)

    paths = {
        "training_qa": ROOT / "training_qa.csv",
        "oof": ROOT / args.oof,
        "meta": ROOT / "champ/meta.csv",
        "feats": ROOT / "champ/feats.csv",
        "skeleton": ROOT / "champ/skel_seq.npz",
    }
    missing = [str(v) for v in paths.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))

    tr = pd.read_csv(paths["training_qa"])
    oof_all = pd.read_csv(paths["oof"])
    meta = pd.read_csv(paths["meta"])
    feats = pd.read_csv(paths["feats"])
    oof = oof_all[oof_all.category.eq("emotion")].copy()
    emo = tr[tr.category.eq("emotion")].copy()
    if len(emo) != 809 or len(oof) != len(emo):
        raise ValueError(f"expected 809 emotion rows in training and OOF, got {len(emo)} and {len(oof)}")
    if oof.qa_id.nunique() != len(oof):
        raise ValueError("OOF qa_id is not unique")
    d = emo.merge(oof[["qa_id", "fold", "pred", "answer", "correct"]], on="qa_id", suffixes=("", "_oof"), validate="one_to_one")
    if not (d.answer == d.answer_oof).all():
        raise ValueError("training and OOF answer keys disagree")
    parts = d.path.map(hau_parts)
    d[["user", "aa", "bb", "cc"]] = pd.DataFrame(parts.tolist(), index=d.index)
    d["session_id"] = d.user + "/" + d.aa + "/" + d.bb
    d["true_label"] = [qlabel(r, r.answer) for _, r in d.iterrows()]
    d["pred_label"] = [qlabel(r, r.pred) for _, r in d.iterrows()]
    d["group"] = d.true_label.map(mgroup)
    d["pred_group"] = d.pred_label.map(lambda x: mgroup(x) if isinstance(x, str) else None)
    d["correct_bool"] = d.pred.eq(d.answer)
    if not d.correct_bool.astype(int).eq(d.correct.astype(int)).all():
        raise ValueError("stored OOF correct flag does not equal pred == answer")

    # Session action pools use the truth only for descriptive action conditioning; no
    # test prediction or model decision is derived from this table.
    action_rows = tr[(tr.source.eq("HAU")) & tr.category.isin(ACT_CATS)].copy()
    aparts = action_rows.path.map(hau_parts)
    action_rows[["user", "aa", "bb", "cc"]] = pd.DataFrame(aparts.tolist(), index=action_rows.index)
    pools: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for r in action_rows.itertuples(index=False):
        for letter in str(r.answer):
            if letter in "ABCD":
                pools[(r.user, r.aa, r.bb)].update(x.strip() for x in str(getattr(r, letter)).split(",") if x.strip())
    d["action_pool"] = [" | ".join(sorted(pools[session_key(r)])) for _, r in d.iterrows()]
    d["n_actions"] = d.action_pool.str.count(" \\| ") + 1
    d["any_locomotion"] = [bool(pools[session_key(r)] & LOCOMOTION) for _, r in d.iterrows()]
    d["block_size"] = d.groupby("session_id").qa_id.transform("size")

    # Merge direct cache summaries.  `t1-t0` is the timestamp-derived duration; the
    # raw cache duration is independently recomputed below as a cross-check.
    meta_hau = meta[meta.kind.eq("train_hau")][["qa_path", "unit_dir", "nf", "t0", "t1", "fps"]]
    d = d.merge(meta_hau, left_on="path", right_on="qa_path", how="left", validate="one_to_one")
    d = d.merge(feats, on="unit_dir", how="left", validate="many_to_one")
    d["duration_s"] = pd.to_numeric(d.t1, errors="coerce") - pd.to_numeric(d.t0, errors="coerce")
    raw = raw_skeleton_dynamics(d)
    d = d.merge(raw, on="path", how="left", validate="one_to_one")

    existing_features = [
        "duration_s", "sk_v_mean", "sk_v_p90", "sk_vmax_mean", "sk_a_mean", "sk_jerk_mean",
        "sk_vwrist_mean", "sk_vankle_mean", "sk_hip_path", "sk_cad_hz", "sk_cad_pow",
        "sk_spec_cent", "sk_v_iqr", "imu_acc_std", "imu_dacc_mean", "imu_acc_energy",
        "imu_gyr_mean", "imu_gyr_energy", "rad_v_mean", "rad_v_p90", "dino_v_mean",
        "dino_v_p90", "dino_acc_mean", "dino_cos_mean", "dino_v_iqr", "dino_path",
        "dino_straight", "dino_cad_hz", "dino_cad_pow", "dino_spec_cent",
    ]
    raw_features = [
        "raw_duration_s", "raw_speed_mean", "raw_speed_p90", "raw_speed_cv", "raw_acc_mean",
        "raw_jerk_mean", "raw_wrist_speed_mean", "raw_ankle_speed_mean", "raw_wrist_ankle_ratio",
        "raw_hip_speed_mean", "raw_hip_path", "raw_hip_efficiency", "raw_pose_amplitude",
        "raw_low_motion_frac", "raw_longest_low_motion_s", "raw_cad_hz", "raw_cad_peak_frac",
    ]
    physical = [c for c in existing_features + raw_features if c in d.columns]
    d = add_within_session_features(d, physical)

    # Verify that the newly exposed temporal streams agree with the original cache
    # summaries where they intentionally measure the same quantity.  Units differ
    # because build_feats.py uses per-frame differences and this audit uses seconds.
    agreements = []
    for cached, derived in [
        ("duration_s", "raw_duration_s"), ("sk_v_mean", "raw_speed_mean"),
        ("sk_a_mean", "raw_acc_mean"), ("sk_jerk_mean", "raw_jerk_mean"),
        ("sk_vwrist_mean", "raw_wrist_speed_mean"), ("sk_vankle_mean", "raw_ankle_speed_mean"),
        ("sk_hip_path", "raw_hip_path"),
    ]:
        x = d[[cached, derived]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(x) < 3:
            continue
        agreements.append({
            "cached_feature": cached, "derived_feature": derived, "n": len(x),
            "spearman_r": stats.spearmanr(x[cached], x[derived]).statistic,
            "pearson_r": stats.pearsonr(x[cached], x[derived]).statistic,
            "median_derived_to_cached_ratio": (x[derived] / x[cached]).replace([np.inf, -np.inf], np.nan).median(),
        })

    # --- tables: data provenance / ontology / raw calculations
    input_rows = []
    for name, path in paths.items():
        input_rows.append({"input": name, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                           "sha256": sha256(path)})
    pd.DataFrame(input_rows).to_csv(TABLES / "input_provenance.csv", index=False)
    ontology = (d.groupby(["true_label", "group"]).size().rename("n").reset_index()
                .rename(columns={"true_label": "label"}).sort_values(["group", "label"]))
    ontology.to_csv(TABLES / "label_ontology.csv", index=False)
    coverage = pd.DataFrame([{
        "feature": c, "n_finite": int(pd.to_numeric(d[c], errors="coerce").notna().sum()),
        "coverage": pd.to_numeric(d[c], errors="coerce").notna().mean(),
        "median": pd.to_numeric(d[c], errors="coerce").median(),
        "p10": pd.to_numeric(d[c], errors="coerce").quantile(.10),
        "p90": pd.to_numeric(d[c], errors="coerce").quantile(.90),
    } for c in physical])
    coverage.to_csv(TABLES / "physical_feature_coverage.csv", index=False)
    raw.to_csv(TABLES / "raw_skeleton_dynamics.csv", index=False)
    agreement = pd.DataFrame(agreements)
    agreement.to_csv(TABLES / "raw_temporal_cache_crosscheck.csv", index=False)

    # --- residual / confusion tables
    label = label_stats(d, physical)
    label.to_csv(TABLES / "per_label_oof_and_physics.csv", index=False)
    confusion, directional, symmetric, group_confusion = confusion_tables(d)
    confusion.to_csv(TABLES / "label_confusion_long.csv", index=False)
    directional.to_csv(TABLES / "residual_confusion_directional.csv", index=False)
    symmetric.to_csv(TABLES / "residual_confusion_symmetric.csv", index=False)
    group_confusion.to_csv(TABLES / "group_confusion_long.csv", index=False)
    clusters, cluster_edges = residual_clusters(symmetric)
    clusters.to_csv(TABLES / "residual_clusters_repeated_links.csv", index=False)
    cluster_edges.to_csv(TABLES / "residual_cluster_edges_repeated_links.csv", index=False)
    swaps, swap_pairs = swap_summary(d)
    swaps.to_csv(TABLES / "residual_session_swap_summary.csv", index=False)
    swap_pairs.to_csv(TABLES / "residual_exact_two_cycles.csv", index=False)

    # --- physical signal and action-context tables
    screen, group_phys = effect_screen(d, physical)
    screen.to_csv(TABLES / "physical_signal_screen_within_session.csv", index=False)
    group_phys.to_csv(TABLES / "group_physics_within_session.csv", index=False)
    pair_phys = pair_physical_map(d, symmetric.head(30), physical)
    pair_phys.to_csv(TABLES / "residual_pair_physical_map.csv", index=False)
    strong_pairs = symmetric[(symmetric.symmetric_error_count >= 4)
                             & (symmetric.balanced_min_direction >= 2)].copy()
    recurrent_rows = []
    for pair in strong_pairs.itertuples(index=False):
        pp = pair_phys[(pair_phys.label_a == pair.label_a) & (pair_phys.label_b == pair.label_b)].copy()
        pp = pp[pp.n_pairs >= 2].head(4)
        contrast = "; ".join(
            f"{r.feature}: {r.mean_a_minus_b_rank:+.2f} rank (n={r.n_pairs})"
            for r in pp.itertuples(index=False)
        )
        recurrent_rows.append({
            "label_a": pair.label_a, "label_b": pair.label_b,
            "symmetric_oof_errors": pair.symmetric_error_count,
            "matched_sessions": int(pp.matched_sessions.max()) if len(pp) else 0,
            "usable_paired_observations": int(pp.n_pairs.max()) if len(pp) else 0,
            "largest_within_session_rank_contrasts": contrast,
        })
    recurrent = pd.DataFrame(recurrent_rows)
    recurrent.to_csv(TABLES / "recurrent_confusion_pair_summary.csv", index=False)
    action, action_group, script = action_context_tables(d)
    action.to_csv(TABLES / "action_context_residuals.csv", index=False)
    action_group.to_csv(TABLES / "action_context_by_manner_group.csv", index=False)
    script.to_csv(TABLES / "script_context_residuals.csv", index=False)
    folds = (
        d.groupby("fold").agg(
            rows=("qa_id", "size"), correct=("correct_bool", "sum"),
            accuracy=("correct_bool", "mean"), errors=("correct_bool", lambda x: int((~x).sum())),
        ).reset_index()
    )
    folds.to_csv(TABLES / "emotion_oof_by_fold.csv", index=False)
    group_perf = (
        d.groupby("group").agg(
            rows=("qa_id", "size"), correct=("correct_bool", "sum"),
            accuracy=("correct_bool", "mean"), errors=("correct_bool", lambda x: int((~x).sum())),
        ).reindex(GROUPS).reset_index()
    )
    group_perf.to_csv(TABLES / "emotion_oof_by_group.csv", index=False)
    by_block = (
        d.groupby("block_size").agg(
            rows=("qa_id", "size"), correct=("correct_bool", "sum"),
            accuracy=("correct_bool", "mean"), errors=("correct_bool", lambda x: int((~x).sum())),
        ).reset_index()
    )
    by_block.to_csv(TABLES / "emotion_oof_by_block_size.csv", index=False)

    # A separate historical group-classifier artifact is useful context, but it must
    # never be confused with the end-to-end champion answer audit.
    meta_base = ROOT / "emolab/meta_base.csv"
    base_summary = None
    if meta_base.exists():
        base = pd.read_csv(meta_base)
        base_summary = (base.assign(correct=base.y.eq(base.oof)).groupby("y").agg(
            rows=("y", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
            .reset_index().rename(columns={"y": "group"}))
        base_summary.loc[len(base_summary)] = {
            "group": "TOTAL", "rows": len(base), "correct": int(base.y.eq(base.oof).sum()),
            "accuracy": base.y.eq(base.oof).mean(),
        }
        base_summary.to_csv(TABLES / "historical_physical_group_classifier_oof.csv", index=False)

    # Full row ledger supports any proposed specialist's later W->R/R->W audit.
    ledger_cols = [
        "qa_id", "fold", "path", "session_id", "cc", "block_size", "action_pool", "n_actions",
        "any_locomotion", "true_label", "pred_label", "group", "pred_group", "correct_bool",
        "duration_s", *raw_features,
    ] + [f"rank__{c}" for c in physical] + [f"z__{c}" for c in physical]
    d[[c for c in ledger_cols if c in d.columns]].to_csv(TABLES / "emotion_residual_ledger.csv", index=False)

    # --- report
    n = len(d)
    correct = int(d.correct_bool.sum())
    err_n = n - correct
    invalid = int(d.pred_label.isna().sum())
    same_group_err = int(((~d.correct_bool) & d.group.eq(d.pred_group)).sum())
    cross_group_err = int(((~d.correct_bool) & ~d.group.eq(d.pred_group)).sum())
    swap_error_rows = int(swaps.loc[swaps.permutation_only, "errors"].sum()) if len(swaps) else 0
    exact_two_cycle_rows = int(2 * len(swap_pairs))
    top_pairs = symmetric.head(12).copy()
    top_pairs["direction"] = top_pairs.apply(lambda r: f"{r.label_a}→{r.label_b}: {r.a_to_b}; {r.label_b}→{r.label_a}: {r.b_to_a}", axis=1)
    top_labels = label.copy()
    top_labels["accuracy"] = top_labels.accuracy.map(fmt_pct)
    top_actions = action.copy()
    top_actions["accuracy"] = top_actions.accuracy.map(fmt_pct)
    top_signals = screen.copy()
    for col in ["eta2_group_rank", "fast_minus_slow_mean_rank", "care_minus_neut_mean_rank", "nerv_minus_neut_mean_rank"]:
        if col in top_signals:
            top_signals[col] = top_signals[col].map(lambda x: "" if pd.isna(x) else f"{x:+.3f}")
    all_session_count = d.session_id.nunique()
    repeated_link_count = int(len(cluster_edges))
    base_note = "The historical group-classifier artifact was unavailable."
    if base_summary is not None:
        total = base_summary[base_summary.group.eq("TOTAL")].iloc[0]
        by_group = "; ".join(
            f"{r.group} {r.accuracy:.3f}" for r in base_summary[base_summary.group.ne("TOTAL")].itertuples(index=False)
        )
        base_note = (f"The separately saved historical physical-only five-group classifier is "
                     f"{int(total.correct)}/{int(total.rows)} = {total.accuracy:.4f}; by group: {by_group}. "
                     f"It is context for the signal screen, not the end-to-end champion.")
    agreement_note = ""
    if len(agreement):
        speed_row = agreement[agreement.derived_feature.eq("raw_speed_mean")]
        duration_row = agreement[agreement.derived_feature.eq("raw_duration_s")]
        if len(speed_row) and len(duration_row):
            agreement_note = (
                f"As a cache check, raw duration agrees exactly in rank with metadata duration "
                f"(Spearman {duration_row.iloc[0].spearman_r:.3f}), and raw skeleton speed agrees "
                f"with the existing `sk_v_mean` summary (Spearman {speed_row.iloc[0].spearman_r:.3f}); "
                f"see `tables/raw_temporal_cache_crosscheck.csv`."
            )
    report = f"""# Emotion/manner residual confusion and physics map

Generated by `research/emotion_manner_physics_map_20260904/audit_emotion_physics.py`.
This is a **training-only, analysis-only** artifact: no model was trained and no Kaggle submission was created.

## Snapshot contract

The map is pinned to `champ/audit_champ.csv`, not to the similarly named `champ/oof_final.csv`.
The former is the preserved champion audit used by the prior residual report; it yields **{correct}/{n} = {correct/n:.4f}** on emotion.  Other OOF CSVs in this checkout have different aggregate scores and must not be mixed into these counts.

Input hashes, paths, and sizes are in [tables/input_provenance.csv](tables/input_provenance.csv).  All results can be regenerated with:

```sh
./venv/bin/python research/emotion_manner_physics_map_20260904/audit_emotion_physics.py
```

## Residual map

There are **{err_n} residual emotion decisions** across {n} subject-disjoint OOF rows and {all_session_count} HAU sessions.  {same_group_err} ({same_group_err/max(err_n, 1):.1%}) keep the prediction in the same coarse manner group; {cross_group_err} cross groups.  There were {invalid} invalid/non-single-letter emotion predictions.

The session assignment signature matters: {swap_error_rows} error rows sit in sessions where the predicted labels are merely a permutation of the true labels; {exact_two_cycle_rows} rows participate in an exact two-cycle.  That is evidence to evaluate *relative, within-session* methods, but it is not evidence that every residual is a swap.

### Most common directed residuals

{markdown_table(directional.assign(share_of_true_label=lambda x: x.share_of_true_label.map(fmt_pct)), ["true_label", "pred_label", "n", "share_of_true_label", "true_group", "pred_group", "example_qa_ids"], 15)}

### Strongest undirected confusion links

Each row sums both directions.  Clusters in `residual_clusters_repeated_links.csv` require at least **two OOF errors in each direction** (at least four total); there are {repeated_link_count} such repeated, balanced links.  The full weak-edge map remains in `residual_confusion_symmetric.csv`.

{markdown_table(top_pairs, ["label_a", "label_b", "symmetric_error_count", "balanced_min_direction", "direction", "group_a", "group_b"], 12)}

### Labels with the most residual errors

{markdown_table(top_labels, ["label", "group", "n", "errors", "accuracy", "top_wrong_predictions", "median_duration_s", "median_raw_speed"], 15)}

The full label confusion matrix is sparse long-form data in [tables/label_confusion_long.csv](tables/label_confusion_long.csv), with a direct residual ledger in [tables/emotion_residual_ledger.csv](tables/emotion_residual_ledger.csv).

## Physical signal map

`champ/feats.csv` supplies the existing leakage-safe skeleton, IMU, radar, and DINO trajectory summaries.  This audit additionally recomputes transparent skeleton temporal measures from `champ/skel_seq.npz`: hip/torso normalisation, velocity, acceleration, jerk, wrist/ankle involvement, pose amplitude, cadence, and low-motion spans.  Coverage and distribution are in [tables/physical_feature_coverage.csv](tables/physical_feature_coverage.csv); per-clip derived values are in [tables/raw_skeleton_dynamics.csv](tables/raw_skeleton_dynamics.csv).

For action-normalised comparisons, every feature is centred and ranked **inside its own `(user, aa, bb)` session** before label/group analysis.  This holds the subject and script fixed.  The signal screen below ranks an exploratory one-way group eta² over those ranks.  It is descriptive only—not held-out accuracy, and not an override gate.

{base_note}

{markdown_table(top_signals, ["feature", "n", "eta2_group_rank", "fast_minus_slow_mean_rank", "care_minus_neut_mean_rank", "nerv_minus_neut_mean_rank"], 15)}

The broad intensity axis is real but already largely exploited: FAST has substantially higher within-session speed/IMU ranks and shorter duration than SLOW, while the strongest CARE–NEUT contrasts remain near zero.  The historical 0.603 group classifier confirms that aggregate temporal dynamics distinguish speed-like groups far better than the residual fine manner groups.  {agreement_note}

`raw_low_motion_frac` is an operational proxy: frames below 10% of that clip's p90 skeleton speed.  It is not a ground-truth pause annotation.  Conversely, existing `sk_active_frac` is omitted from the screen because `champ/build_feats.py` defines it using each clip's own 60th percentile, which makes it approximately 0.40 by construction and therefore unsuitable for comparing pauses across clips.

For the confusing pairs, [tables/residual_pair_physical_map.csv](tables/residual_pair_physical_map.csv) lists the eight largest **within-session** rank differences for each pair.  Positive `mean_a_minus_b_rank` means the first alphabetical label tends to be higher on that feature in sessions containing both labels.  Read pairs with low `matched_sessions` as hypotheses only.

### Recurrent-pair physical hypotheses

Only the following pairs repeated in both directions at least twice.  These contrasts are action- and subject-normalised, but their small paired samples rule out deployment claims without a new subject-disjoint specialist evaluation.

{markdown_table(recurrent, ["label_a", "label_b", "symmetric_oof_errors", "matched_sessions", "usable_paired_observations", "largest_within_session_rank_contrasts"], 10)}

## Action/script context

The training action answers reconstruct the true session action pool solely for descriptive conditioning.  An emotion clip inherits its session's whole script, so these are not frame-level action claims.  The action membership residual map is:

{markdown_table(top_actions, ["action", "emotion_rows", "errors", "accuracy", "n_labels", "n_groups", "common_true_labels"], 15)}

All action-context and script-context values are preserved in [tables/action_context_residuals.csv](tables/action_context_residuals.csv), [tables/action_context_by_manner_group.csv](tables/action_context_by_manner_group.csv), and [tables/script_context_residuals.csv](tables/script_context_residuals.csv).

## Relevant data and implementation paths

* Labels/options: `training_qa.csv`, emotion rows 2724 onward in this checkout.
* Frozen end-to-end champion OOF: `champ/audit_champ.csv`; the OOF contract and end-to-end solver are in `champ/eval_full.py` and `champ/pipeline.py`.
* Manner ontology, block-relative features, and assignment: `champ/core.py` (`FAST/SLOW/CARE/NERV`, `PHYS`, `block_features`, `solve_emotion`).
* Existing physical extraction: `champ/build_feats.py`; metadata/timing provenance: `champ/build_meta.py`.
* Raw temporal cache: `champ/skel_seq.npz`; IMU sequence cache: `champ/imu_seq.npz`; DINO depth-frame trajectories: `champ/dino_frames.npz`.
* Existing pairwise-relative attempt: `champ/emopair.py`; prior failed/rich probes remain under `emolab/` and must not be mistaken for this audit's new results.

## Interpretation boundary

This map establishes where residual errors and measured physical contrasts are concentrated.  It does **not** establish that a feature will transfer to hidden test clips.  A next model must use subject-disjoint folds, compare directly against this champion at the decision level, report W→R/R→W plus per-fold behavior, and be evaluated separately for triple, pair, and singleton test-like blocks.
"""
    (HERE / "RESULTS.md").write_text(report)
    print(f"wrote {HERE / 'RESULTS.md'}")
    print(f"emotion OOF: {correct}/{n} = {correct/n:.4f}; residuals={err_n}")
    print(f"same-group residuals={same_group_err}; cross-group residuals={cross_group_err}")
    print(f"tables: {TABLES}")


if __name__ == "__main__":
    main()
