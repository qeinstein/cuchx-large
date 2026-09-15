"""Row-for-row (+ byte/SHA) verification of reproduced CSVs vs archived artifacts.

Usage:
  python3 repro/verify.py --variant 334 --got <csv>
  python3 repro/verify.py --variant flipall --got <csv>
Exit 0 + 'IDENTICAL' iff every row matches the archived submission.
"""
import argparse
import hashlib
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['334', 'flipall'])
    ap.add_argument('--got', required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, 'repro', 'variant_%s.json' % a.variant)))
    ref = os.path.join(ROOT, cfg['archived_artifact'])
    r = pd.read_csv(ref).set_index('qa_id').prediction.astype(str)
    g = pd.read_csv(a.got).set_index('qa_id').prediction.astype(str)
    bad = [q for q in r.index if q not in g.index or g.loc[q] != r.loc[q]]
    extra = [q for q in g.index if q not in r.index]
    print('ref rows=%d got rows=%d mismatched=%d extra=%d'
          % (len(r), len(g), len(bad), len(extra)))
    if bad[:10]:
        print('first mismatches:', [(q, g.get(q), r.loc[q]) for q in bad[:10]])
    print('ref  sha256 %s (%s)' % (sha(ref), cfg['archived_artifact']))
    print('got  sha256 %s (%s)' % (sha(a.got), a.got))
    print('pinned sha256 %s' % cfg['expected_sha256'])
    assert sha(ref) == cfg['expected_sha256'], 'archived artifact drifted!'
    if not bad and not extra and len(r) == len(g):
        print('IDENTICAL (%s): all %d rows match' % (a.variant, len(r)))
        if sha(ref) == sha(a.got):
            print('BYTE-IDENTICAL')
    else:
        raise SystemExit('MISMATCH on variant %s' % a.variant)


if __name__ == '__main__':
    main()
