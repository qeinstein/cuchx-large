#!/bin/bash
# Sequential slot-mechanism arms.  Each pairstress run peaks around 4-5 GB (the lazily filled
# dense-logit presence-stat cache grows per fold), so three concurrent runs exhaust swap on a
# 17 GB machine and thrash to a standstill -- run them one at a time.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

run () {
  name=$1; shift
  echo "=== $name : $* ===" >&2
  env "$@" venv/bin/python champ/eval_pairstress.py "$name" 0.38 "$POLICY" \
      > "slotlab/log_$name.txt" 2>&1
  echo "  $name exit=$? $(grep -h 'TOTAL' "slotlab/log_$name.txt" | tail -1)" >&2
}

# --- matched-policy arms: the real test withholds only end trials (pi = 1.000)
POLICY=ends
run ends_base      CHAMP_W_SLOT=0
run ends_oracle    CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=clock CHAMP_SLOT_ORACLE=1
run ends_adjonly   CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=clock CHAMP_SLOT_ADJ_ONLY=1
run ends_clock     CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=clock
run ends_gbm       CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=gbm

# --- robustness: the same mechanisms under the uniform policy they are NOT tuned for
POLICY=uniform
run unif_oracle    CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=clock CHAMP_SLOT_ORACLE=1
run unif_clock     CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=clock
run unif_adjonly   CHAMP_W_SLOT=1 CHAMP_SLOT_MODEL=clock CHAMP_SLOT_ADJ_ONLY=1

echo "=== all arms done ===" >&2
