"""MECHANISM Z - the manner SET nearly determines the ordered assignment.

The champion scores each (clip, manner) with an INDEPENDENT slot prior P(manner | slot, k)
and then picks the best bijection.  But the question generator appears to have fixed, per
manner set, which manner goes in which protocol slot:

    manner sets observed in >=2 training sessions: mean 1.12 distinct orderings out of 6,
    modal ordering covers 0.957 of their sessions.

Leave-one-USER-out, prior only, no physical evidence, triple-exact:

    independent slot prior (champion's)                 199/265 = 0.751
    set-conditioned ordering, on covered triples        140/154 = 0.909  (vs 0.747)

The gain concentrates where a set is seen once elsewhere (n=115: 0.748 -> 0.948).
This is the same class of train-derived structure as the slot prior itself, just conditioned
on the set rather than marginalised over it.
"""
import os, sys, itertools, collections
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from core import load_all, gt_letters, opts


def training_orders(tr, hold_users=()):
    """frozenset(manners) -> Counter over ordered tuples, from training users only."""
    em = tr[(tr.category == 'emotion') & (~tr.user.isin(list(hold_users)))].copy()
    em['blk'] = list(zip(em.user, em.aa, em.bb))
    em['lab'] = [str(r[gt_letters(r)[0]]).strip() for _, r in em.iterrows()]
    jnt = collections.defaultdict(collections.Counter)
    for b, g in em.groupby('blk'):
        g = g.sort_values('cc')
        if len(g) != 3:
            continue
        v = tuple(g.lab)
        jnt[frozenset(v)][v] += 1
    return jnt


def canonical(jnt, I):
    """Modal ordering for this manner set, plus its support and concentration."""
    S = frozenset(I)
    c = jnt.get(S)
    if not c:
        return None, 0, 0.0
    order, top = c.most_common(1)[0]
    return order, sum(c.values()), top / sum(c.values())


def assign(order, k):
    """Candidate (clip -> manner) assignments consistent with the canonical slot order.

    k == 3: the canonical order itself.
    k == 2: the two clips occupy two of the three slots in order, so only the 3
            order-preserving sub-assignments are admissible, not all 6.
    """
    if k == 3:
        return [list(order)]
    return [[order[i], order[j]] for i, j in itertools.combinations(range(3), 2)]
