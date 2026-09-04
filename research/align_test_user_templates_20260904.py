"""Gap-tolerant alignment of ordered test HAU blocks to training-user templates."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import evaluate_user_template_pairs_20260904 as E


def lcs_alignment(a, b):
    """Return exact-signature LCS length and one index alignment."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j] and dp[i][j] == 1 + dp[i + 1][j + 1]:
            pairs.append((i, j))
            i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return dp[0][0], pairs


def main():
    users = sorted(E.USER_BLOCKS, key=lambda u: int(u[4:]))
    segments = [(0, 14), (14, 28), (28, 44), (44, 55)]
    rows = []
    for lo, hi in segments:
        test = E.TEST_SIG[lo:hi]
        print("\nsegment", lo, hi, "n", len(test))
        scored = []
        for u in users:
            donor = [E.visible_signature(b) for b in E.USER_BLOCKS[u]]
            lcs, pairs = lcs_alignment(test, donor)
            scored.append((lcs, lcs / max(1, min(len(test), len(donor))), u, pairs))
            rows.append(dict(segment_lo=lo, segment_hi=hi, donor=u,
                             test_n=len(test), donor_n=len(donor), lcs=lcs,
                             coverage=lcs / max(1, min(len(test), len(donor))),
                             alignment=";".join(f"{lo+i}:{j}" for i, j in pairs)))
        for lcs, cov, u, pairs in sorted(scored, reverse=True)[:8]:
            print(u, "donor_n", len(E.USER_BLOCKS[u]), "lcs", lcs,
                  "coverage", f"{cov:.3f}", "pairs", pairs)
    pd.DataFrame(rows).to_csv(os.path.join(E.ROOT, "research", "test_user_template_lcs.csv"), index=False)


if __name__ == "__main__":
    main()
