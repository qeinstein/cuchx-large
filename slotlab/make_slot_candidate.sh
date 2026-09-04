#!/bin/bash
# Real-test application of the slot-set mechanism, as a PAIRED pair of runs.
#
# Attribution has to be clean: the 0.93859 champion already contains mechanism T (block
# repair), so a candidate is only interpretable against a baseline that also has T on and
# differs from it in exactly one flag.  That is how T itself was extracted
# (champ/diff_T_repair_cf.csv), and the resulting diff is what build_correction_layer.py
# layers onto the champion.
#
# Produces:
#   submission_slotbase.csv / champ/diff_slotbase.csv   REPAIR+CONFORM, W_SLOT=0
#   submission_slotcand.csv / champ/diff_slotcand.csv   the same, plus the clock slot prior
#   slotlab/diff_slot.csv                               slotcand vs slotbase, the attributable delta
set -e
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
COMMON="CHAMP_REPAIR=1 CHAMP_CONFORM_FIRST=1"

echo "=== paired baseline (T on, no slot prior) ==="
env $COMMON CHAMP_W_SLOT=0 \
    venv/bin/python champ/make_candidate.py submission_slotbase \
    > slotlab/log_test_slotbase.txt 2>&1
tail -4 slotlab/log_test_slotbase.txt

echo "=== candidate (T on, clock slot prior, adjacency-only) ==="
env $COMMON CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=clock CHAMP_SLOT_ADJ_ONLY=1 \
    venv/bin/python champ/make_candidate.py submission_slotcand \
    > slotlab/log_test_slotcand.txt 2>&1
tail -4 slotlab/log_test_slotcand.txt

echo "=== attributable delta ==="
venv/bin/python - <<'PY'
import pandas as pd
te = pd.read_csv('test_qa.csv')
b = pd.read_csv('submission_slotbase.csv').rename(columns={'prediction': 'prediction_b'})
c = pd.read_csv('submission_slotcand.csv').rename(columns={'prediction': 'prediction_slot'})
ch = pd.read_csv('submission_093859_SUBMITTED.csv').rename(columns={'prediction': 'prediction_champ'})
m = te[['qa_id', 'source', 'category']].merge(b, on='qa_id').merge(c, on='qa_id').merge(ch, on='qa_id')
d = m[m.prediction_b != m.prediction_slot].copy()
print('rows changed by the slot prior alone: %d' % len(d))
print(d.category.value_counts().to_string())
print()
print('of those, rows that also differ from the 0.93859 champion (genuine overrides): %d'
      % int((d.prediction_slot != d.prediction_champ).sum()))
print()
print(d[['qa_id', 'category', 'prediction_champ', 'prediction_b', 'prediction_slot']].to_string(index=False))
d.to_csv('slotlab/diff_slot.csv', index=False)
print('\nwrote slotlab/diff_slot.csv')
PY
