#!/bin/bash
# Per-block-size pool-inclusion bias sweep.  Sequential: three concurrent pairstress runs
# exhaust swap on this machine.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
for b in 0.5 1.0 2.0; do
  tag="pb2_${b/./p}"
  echo "=== $tag : bias on 2-clip blocks = $b ===" >&2
  CHAMP_POOL_BIAS="{\"2\": $b}" venv/bin/python champ/eval_pairstress.py "$tag" 0.38 ends \
      > "slotlab/log_$tag.txt" 2>&1
  echo "  $tag exit=$? $(grep -h TOTAL slotlab/log_$tag.txt | tail -1)" >&2
done
echo "=== pool bias sweep done ===" >&2
