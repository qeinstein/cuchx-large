#!/usr/bin/env bash
# Frozen solution inference entry point (Team Fluxx, CUHK-X Large Model Track).
# Usage: bash inference.sh <data_dir> <output_csv> [variant]
#   <data_dir>   official label-free Testing data (published layout); test_qa.csv
#                is located inside it (up to 4 levels deep).
#   <output_csv> destination for the submission-format CSV (qa_id,prediction).
#   [variant]    334 (default, keeper) or flipall (private lottery ticket).
# Must be run from the package root. No network access. CPU-only.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: bash inference.sh <data_dir> <output_csv> [variant]" >&2
  exit 2
fi
DATA_DIR="$1"; OUT_CSV="$2"; VARIANT="${3:-334}"
if [ "$VARIANT" != "334" ] && [ "$VARIANT" != "flipall" ]; then
  echo "variant must be 334 or flipall (got: $VARIANT)" >&2
  exit 2
fi
if [ ! -d "$DATA_DIR" ]; then echo "data_dir not found: $DATA_DIR" >&2; exit 2; fi
if [ ! -f "repro/variant_${VARIANT}.json" ]; then
  echo "must run from the package root (repro/variant_${VARIANT}.json missing)" >&2
  exit 2
fi

if [ -f "$DATA_DIR/test_qa.csv" ]; then
  TEST_QA="$DATA_DIR/test_qa.csv"
else
  TEST_QA="$(find "$DATA_DIR" -maxdepth 4 -name "test_qa.csv" 2>/dev/null | head -1)"
fi
if [ -z "$TEST_QA" ]; then
  echo "WARN: test_qa.csv not found under $DATA_DIR; using committed frozen copy" >&2
  TEST_QA="test_qa.csv"
else
  echo "test_qa: $TEST_QA"
fi
mkdir -p "$(dirname "$OUT_CSV")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CHAMP_TEST_QA="$TEST_QA" CHAMP_OUT="$WORK/pipe.csv"

echo "[1/2] pipeline solve..."
python3 champ/make_submission.py 2>&1 | tail -3
echo "[2/2] frozen lineage ($VARIANT)..."
python3 repro/apply_lineage.py --variant "$VARIANT" --base "$WORK/pipe.csv" --out "$OUT_CSV"
python3 - "$OUT_CSV" "$TEST_QA" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); t = pd.read_csv(sys.argv[2])
assert list(d.columns[:2]) == ['qa_id', 'prediction'], 'bad columns'
assert len(d) == len(t) == 682, 'row count %d/%d' % (len(d), len(t))
assert set(d.qa_id) == set(t.qa_id), 'qa_id set mismatch vs input test_qa'
assert d.prediction.map(lambda s: isinstance(s, str) and len(s) > 0 and all(c in 'ABCD' for c in s)).all()
print('validated: 682 rows, qa_ids match input, letters clean')
EOF
echo "done: $OUT_CSV"
sha256sum "$OUT_CSV"
