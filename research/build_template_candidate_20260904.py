import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
base = pd.read_csv(os.path.join(ROOT, 'submissions/submission_093859_SUBMITTED.csv'))
audit = pd.read_csv(os.path.join(ROOT, 'research', 'test_template_order_audit.csv'))

# Strict promotion gate from the held-out audit: an exact action-pool source, complete
# six-pair coverage after transitive closure, and a conforming 2/3-clip block.  The four
# selected rows are all in ordinary blocks; orphan/split blocks and nearest-pool matches are
# intentionally excluded.
selected = audit[(audit.nsrc == 1) & (audit.closure_coverage == 6) & audit.closure_flip]
override = dict(zip(selected.qa_id, selected.closure))
base['prediction'] = [override.get(q, p) for q, p in zip(base.qa_id, base.prediction)]
out = os.path.join(ROOT, 'research', 'submission_template_order_candidate.csv')
base.to_csv(out, index=False)
print('overrides', len(override), override)
print('written', out)
