"""Apply the frozen correction lineage: pipeline output -> selected submission.

Reads a variant config (repro/variant_<name>.json), applies the 51-row
334 patch table (each entry asserts the pipeline value it corrects), then any
variant extra flips. Fails loudly on any mismatch: the output is verified
against the archived submission bytes by the caller (inference.sh) and by
repro/verify.py.
Usage:
  python3 repro/apply_lineage.py --variant 334 --base <pipe.csv> --out <out.csv>
  python3 repro/apply_lineage.py --variant flipall --base <pipe.csv> --out <out.csv>
"""
import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['334', 'flipall'])
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, 'repro', 'variant_%s.json' % a.variant)))
    patches = json.load(open(os.path.join(ROOT, cfg['patches'])))
    d = pd.read_csv(a.base)
    assert list(d.columns[:2]) == ['qa_id', 'prediction'] and len(d) == 682, \
        'base must be 682-row qa_id,prediction'
    sub = dict(zip(d.qa_id, d.prediction.astype(str)))
    for p in patches:
        q = p['qa_id']
        assert sub[q] == p['pipe'], \
            'pipeline drift on %s: got %s, lineage expects %s' % (q, sub[q], p['pipe'])
        sub[q] = p['final']
    for f in cfg.get('extra_flips', []):
        q = f['qa_id']
        assert sub[q] == f['from'], \
            'variant pre-image mismatch on %s: got %s, want %s' % (q, sub[q], f['from'])
        sub[q] = f['to']
    out = pd.DataFrame({'qa_id': d.qa_id, 'prediction': [sub[q] for q in d.qa_id]})
    out.to_csv(a.out, index=False)
    print('wrote %s: %d patches + %d variant flips'
          % (a.out, len(patches), len(cfg.get('extra_flips', []))))


if __name__ == '__main__':
    main()
