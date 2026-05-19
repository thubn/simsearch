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

RAW="results/priority2_scaling_raw.csv"
SUMMARY="results/priority2_scaling_summary.csv"
REPORT="results/priority2_scaling_verification.md"
rm -f "$RAW" "$SUMMARY" "$REPORT"

first=1
for n in $SIZES; do
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
