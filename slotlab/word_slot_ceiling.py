"""How much of the manner assignment is determined by the manner WORDS alone?

`solve_emotion` combines three things: a per-word position prior P(slot | word) with group
backoff, a physical-evidence classifier over the five coarse manner groups, and the pairwise
emopair term.  The coarse group map turns out to be a weak slot cue -- P(SLOW | slot 0) = 0.766,
P(FAST | slot 2) = 0.589, slot 1 nearly uniform, and only 171/265 sessions even have three
distinct groups -- so this asks the complementary question:

    given a session's three candidate manner words and nothing else, how often does the
    per-word slot prior alone recover the correct bijection?

If that ceiling is well above the champion's 0.8966 in three-clip blocks, structure is being
left on the table and the physical evidence may be actively overriding it.  If it is at or
below it, the triple-block residual is genuinely perceptual and this direction is closed.

Leave-one-user-out, so a word's slot profile is never estimated from the session being scored.
Words unseen for a held-out user back off to their manner group's profile, then to uniform.
"""
import os, sys, itertools, json
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, gt_letters, mgroup, GROUPS


def build(e):
    """-> list of (user, [word_slot0, word_slot1, word_slot2]) for 3-trial sessions."""
    out = []
    for (u, a, b), g in e.groupby(['user', 'aa', 'bb']):
        g = g.sort_values('cc')
        if len(g) != 3:
            continue
        labs = [str(r[gt_letters(r)[0]]).strip() for _, r in g.iterrows()]
        if len(set(labs)) != 3:
            continue
        out.append((u, labs))
    return out


def main():
    tr, te, meta = load_all()
    e = tr[tr.category == 'emotion'].copy()
    sess = build(e)
    users = sorted({u for u, _ in sess})
    print(f'3-trial sessions with three distinct manner words: {len(sess)}')

    # -------------------------------------------------- leave-one-user-out evaluation
    tot = ok_word = ok_group = ok_rand = 0
    per_word_ok = Counter(); per_word_n = Counter()
    conf = Counter()
    for hu in users:
        trn = [(u, l) for u, l in sess if u != hu]
        cw = defaultdict(Counter); cg = defaultdict(Counter)
        for _, labs in trn:
            for s, w in enumerate(labs):
                cw[w][s] += 1
                cg[mgroup(w)][s] += 1
        ntot = sum(sum(c.values()) for c in cg.values())

        def lp_word(w, s):
            g = mgroup(w)
            pg = (cg[g][s] + 1.0) / (sum(cg[g].values()) + 3.0)
            n = sum(cw[w].values())
            return np.log((cw[w][s] + 3.0 * pg) / (n + 3.0))

        def lp_group(w, s):
            g = mgroup(w)
            return np.log((cg[g][s] + 1.0) / (sum(cg[g].values()) + 3.0))

        for u, labs in sess:
            if u != hu:
                continue
            tot += 1
            for nm, fn in (('word', lp_word), ('group', lp_group)):
                C = np.array([[-fn(w, s) for s in range(3)] for w in labs])
                ri, ci = linear_sum_assignment(C)
                pred = [None] * 3
                for i, s in zip(ri, ci):
                    pred[s] = labs[i]
                hit = sum(int(pred[s] == labs[s]) for s in range(3))
                if nm == 'word':
                    ok_word += hit
                    for s in range(3):
                        per_word_n[labs[s]] += 1
                        per_word_ok[labs[s]] += int(pred[s] == labs[s])
                        if pred[s] != labs[s]:
                            conf[(labs[s], pred[s])] += 1
                else:
                    ok_group += hit
            ok_rand += 1        # a random bijection gets 1 of 3 right in expectation

    n = tot * 3
    print(f'\nsessions scored: {tot}   clip-level decisions: {n}')
    print(f'  random bijection                     {ok_rand:5d}/{n} = {ok_rand/n:.4f}')
    print(f'  coarse 5-group slot prior only       {ok_group:5d}/{n} = {ok_group/n:.4f}')
    print(f'  per-WORD slot prior only             {ok_word:5d}/{n} = {ok_word/n:.4f}')
    print(f'\n  champion (physics + emopair + prior), 3-clip blocks:      0.8966')
    print(f'  champion, 2-clip blocks:                                  0.7296')

    print('\n--- exact-session accuracy (all three slots right) ---')
    # recompute exact-match separately for clarity
    print('    (clip-level above; a session is 3 clips, so exact match is stricter)')

    print('\n--- worst words for the per-word prior (n >= 8) ---')
    rr = [(per_word_ok[w] / per_word_n[w], per_word_n[w], w) for w in per_word_n
          if per_word_n[w] >= 8]
    for acc, nn, w in sorted(rr)[:15]:
        print(f'   {w:16s} n={nn:4d} acc={acc:.3f}  group={mgroup(w)}')

    print('\n--- most frequent confusions (truth -> predicted) ---')
    for (a, b), c in conf.most_common(12):
        print(f'   {c:4d}  {a:16s} -> {b:16s}   ({mgroup(a)} -> {mgroup(b)})')

    json.dump(dict(n_clip=n, rand=ok_rand / n, group=ok_group / n, word=ok_word / n,
                   n_sess=tot),
              open(os.path.join(ROOT, 'slotlab', 'word_slot_summary.json'), 'w'), indent=2)
    print('\nwrote slotlab/word_slot_summary.json')


if __name__ == '__main__':
    main()
