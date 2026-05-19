#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MODULE_PATH="${MODULE_PATH:-$ROOT_DIR/build/embedding_search_benchmark.so}"
DATASET="${DATASET:-$ROOT_DIR/python/out/1_2M_random_out_mixedbread.parquet}"
QUERIES="${QUERIES:-$ROOT_DIR/python/query_embeddings/combined.jsonl}"
DIM="${DIM:-1024}"
K="${K:-100}"
REPEATS="${REPEATS:-1}"
QUERY_LIMIT="${QUERY_LIMIT:-0}"
SIZES="${SIZES:-60000 300000 1200000}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$MODULE_PATH" ]] || { echo "Missing benchmark module: $MODULE_PATH" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 1; }
[[ -f "$QUERIES" ]] || { echo "Missing queries: $QUERIES" >&2; exit 1; }

mkdir -p results
export PYTHONPATH="$ROOT_DIR/build:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(cat results/build_info/pyarrow_dir.txt 2>/dev/null || true):${LD_LIBRARY_PATH:-}"

INSPECT_JSON="$(mktemp)"
"$PYTHON_BIN" scripts/inspect_parquet_dataset.py --json --warn-filename --json-output "$INSPECT_JSON" "$DATASET"
ACTUAL_ROWS="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_rows"])' "$INSPECT_JSON")"
ACTUAL_DIM="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["embedding_dim_inferred"])' "$INSPECT_JSON")"
rm -f "$INSPECT_JSON"
echo "Dataset path: $DATASET"
echo "Dataset actual rows: $ACTUAL_ROWS"
echo "Dataset embedding dim: $ACTUAL_DIM"
[[ "$ACTUAL_DIM" == "$DIM" ]] || { echo "Embedding dim mismatch: expected $DIM, found $ACTUAL_DIM" >&2; exit 1; }
(( ACTUAL_ROWS >= K )) || { echo "Invalid run: N=$ACTUAL_ROWS < k=$K" >&2; exit 1; }

if (( ACTUAL_ROWS < 60000 )); then
  echo "SMOKE_TEST_ONLY: max available N=$ACTUAL_ROWS is below 60000; canonical Priority 2 outputs will not be overwritten" >&2
  OUT_DIR="results/smoke_tests"
  RAW="$OUT_DIR/priority2_scaling_raw_smoke.csv"
  SUMMARY="$OUT_DIR/priority2_scaling_summary_smoke.csv"
  REPORT="$OUT_DIR/priority2_scaling_verification_smoke.md"
  SIZES="${SMOKE_SIZES:-100 500 $ACTUAL_ROWS}"
else
  OUT_DIR="results"
  RAW="results/priority2_scaling_raw.csv"
  SUMMARY="results/priority2_scaling_summary.csv"
  REPORT="results/priority2_scaling_verification.md"
fi
mkdir -p "$OUT_DIR"
rm -f "$RAW" "$SUMMARY" "$REPORT"

first=1
for n in $SIZES; do
  if (( n < K )); then
    echo "Skipping N=$n because N < k=$K" >&2
    continue
  fi
  if (( n > ACTUAL_ROWS )); then
    echo "Skipping N=$n because dataset only has $ACTUAL_ROWS rows" >&2
    continue
  fi
  append=()
  if [[ "$first" -eq 0 ]]; then
    append=(--append-csv)
  fi
  "$PYTHON_BIN" python/benchmark/benchmark_v2.py \
    -f "$DATASET" -m query -q "$QUERIES" -k "$K" --embedding-dim "$DIM" \
    --revision-csv --methods float32_avx2,binary,two_step_RF10,two_step_mf_RF10 \
    --max-vectors "$n" --query-limit "$QUERY_LIMIT" --repeats "$REPEATS" \
    --csv-output "$RAW" "${append[@]}"
  first=0
done

"$PYTHON_BIN" scripts/summarize_priority2.py "$RAW" "$SUMMARY" "$REPORT"
