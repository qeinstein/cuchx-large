"""Vectorised joint block decoder.

Per block we precompute, once:
  B  : (n!, n*n) boolean, B[p, a*n+b] = 'a precedes b under permutation p'
  Lq : (n!,) int, index into PERMS of the 4-option permutation induced on question q
Any weighted combination of pairwise evidence sources then costs one matvec.
"""
import itertools
import numpy as np

PERMS = [''.join(p) for p in itertools.permutations('ABCD')]
PIDX = {p: i for i, p in enumerate(PERMS)}


class Block:
    def __init__(self, U, qopts):
        self.U = list(U)
        n = len(U)
        self.n = n
        self.perms = np.array(list(itertools.permutations(range(n))), np.int16)
        P = self.perms
        rank = np.empty_like(P)
        np.put_along_axis(rank, P.astype(np.intp), np.broadcast_to(np.arange(n, dtype=np.int16), P.shape), axis=1)
        self.rank = rank                                   # rank[p, u] = position of u
        B = np.zeros((len(P), n * n), bool)
        for a in range(n):
            for b in range(n):
                if a != b:
                    B[:, a * n + b] = rank[:, a] < rank[:, b]
        self.B = B
        self.L = []
        for o in qopts:
            ui = [self.U.index(x) for x in o]
            r = rank[:, ui]                                # (n!, 4)
            order = np.argsort(r, axis=1)                  # positions of A,B,C,D sorted
            codes = (order * np.array([1000, 100, 10, 1])).sum(1)
            uniq, inv = np.unique(codes, return_inverse=True)
            lut = {}
            for c in uniq:
                d = [(c // 1000) % 10, (c // 100) % 10, (c // 10) % 10, c % 10]
                lut[c] = PIDX[''.join('ABCD'[i] for i in d)]
            self.L.append(np.array([lut[c] for c in uniq])[inv])

    def score(self, pair_mats, dp_terms):
        """pair_mats: list of (weight, dict[(actA,actB)] -> logodds).
        dp_terms:   list of (weight, 24-vector) aligned with qopts."""
        tot = np.zeros(len(self.perms))
        n = self.n
        for w, M in pair_mats:
            if not w:
                continue
            F = np.zeros(n * n)
            for a in range(n):
                for b in range(n):
                    if a != b:
                        F[a * n + b] = M.get((self.U[a], self.U[b]), 0.0)
            tot += w * (self.B @ F)
        for (w, v), L in zip(dp_terms, self.L):
            if w:
                tot += w * v[L]
        return tot

    def best(self, pair_mats, dp_terms):
        t = self.score(pair_mats, dp_terms)
        k = int(np.argmax(t))
        pos = self.rank[k]
        labs = []
        for L in self.L:
            labs.append(PERMS[int(L[k])])
        return labs, float(t[k]), t
