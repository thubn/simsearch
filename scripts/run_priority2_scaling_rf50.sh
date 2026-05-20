#!/usr/bin/env bash
# Run the missing Priority 2 cells for RF=50 (two_step_RF50, two_step_mf_RF50)
# at N=300000 and N=1200000 only.  Writes raw rows to a temp file, then merges
# them into results/priority2_scaling_raw.csv (preserving existing rows) and
# regenerates the summary + verification report over the combined data.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MODULE_PATH="${MODULE_PATH:-$ROOT_DIR/build/embedding_search_benchmark.so}"
DATASET="${DATASET:-$ROOT_DIR/python/out/wiki_mxbai_1024_N1200000_seed42.parquet}"
QUERIES="${QUERIES:-$ROOT_DIR/python/query_embeddings/combined.jsonl}"
DIM="${DIM:-1024}"
K="${K:-100}"
REPEATS="${REPEATS:-10}"
QUERY_LIMIT="${QUERY_LIMIT:-100}"
SIZES="${SIZES:-300000 1200000}"
METHODS="${METHODS:-two_step_RF50,two_step_mf_RF50}"

CANONICAL_RAW="results/priority2_scaling_raw.csv"
CANONICAL_SUMMARY="results/priority2_scaling_summary.csv"
CANONICAL_REPORT="results/priority2_scaling_verification.md"
NEW_RAW="results/priority2_scaling_raw_rf50_new.csv"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$MODULE_PATH" ]] || { echo "Missing benchmark module: $MODULE_PATH" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 1; }
[[ -f "$QUERIES" ]] || { echo "Missing queries: $QUERIES" >&2; exit 1; }
[[ -f "$CANONICAL_RAW" ]] || { echo "Missing canonical raw CSV: $CANONICAL_RAW" >&2; exit 1; }

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

rm -f "$NEW_RAW"

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
    --revision-csv --methods "$METHODS" \
    --max-vectors "$n" --query-limit "$QUERY_LIMIT" --repeats "$REPEATS" \
    --csv-output "$NEW_RAW" "${append[@]}"
  first=0
done

# Sanity check: same header as the canonical raw CSV
HEADER_CANON="$(head -n 1 "$CANONICAL_RAW")"
HEADER_NEW="$(head -n 1 "$NEW_RAW")"
[[ "$HEADER_CANON" == "$HEADER_NEW" ]] || {
  echo "Schema mismatch between $CANONICAL_RAW and $NEW_RAW" >&2
  echo "Canonical: $HEADER_CANON" >&2
  echo "New:       $HEADER_NEW" >&2
  exit 1
}

# Merge: keep canonical rows verbatim, then append the new data rows (skip header)
cp "$CANONICAL_RAW" "$CANONICAL_RAW.merge_in_progress"
tail -n +2 "$NEW_RAW" >> "$CANONICAL_RAW.merge_in_progress"
mv "$CANONICAL_RAW.merge_in_progress" "$CANONICAL_RAW"

# Regenerate summary + verification report over the combined raw
"$PYTHON_BIN" scripts/summarize_priority2.py "$CANONICAL_RAW" "$CANONICAL_SUMMARY" "$CANONICAL_REPORT"

echo "Done. New rows merged into $CANONICAL_RAW; summary at $CANONICAL_SUMMARY."
