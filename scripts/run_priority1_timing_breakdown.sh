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
SANITY_N="${SANITY_N:-1000}"
SANITY_QUERIES="${SANITY_QUERIES:-3}"
SANITY_REPEATS="${SANITY_REPEATS:-2}"
FULL_N="${FULL_N:-0}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$MODULE_PATH" ]] || { echo "Missing benchmark module: $MODULE_PATH" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 1; }
[[ -f "$QUERIES" ]] || { echo "Missing queries: $QUERIES" >&2; exit 1; }

mkdir -p results
export PYTHONPATH="$ROOT_DIR/build:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(cat results/build_info/pyarrow_dir.txt 2>/dev/null || true):${LD_LIBRARY_PATH:-}"

SANITY_RAW="results/priority1_sanity_raw.csv"
RAW="results/priority1_timing_breakdown.csv"
SUMMARY="results/priority1_timing_summary.csv"
rm -f "$SANITY_RAW" "$RAW" "$SUMMARY"

"$PYTHON_BIN" python/benchmark/benchmark_v2.py \
  -f "$DATASET" -m query -q "$QUERIES" -k "$K" --embedding-dim "$DIM" \
  --revision-csv --methods float32_avx2,binary,two_step_RF10,two_step_mf_RF10 \
  --max-vectors "$SANITY_N" --query-limit "$SANITY_QUERIES" \
  --repeats "$SANITY_REPEATS" --csv-output "$SANITY_RAW"

"$PYTHON_BIN" - <<'PY'
import csv, math
rows = list(csv.DictReader(open("results/priority1_sanity_raw.csv", newline="")))
problems = []
for r in rows:
    total = float(r["T_total_ms"])
    parts = sum(float(r[k]) for k in ["T_query_sketch_ms","T_binary_scan_ms","T_candidate_selection_ms","T_rescore_ms","T_final_topk_ms"])
    if any(float(r[k]) < 0 for k in ["T_query_sketch_ms","T_binary_scan_ms","T_candidate_selection_ms","T_rescore_ms","T_final_topk_ms","T_total_ms"]):
        problems.append(f"negative timing in {r}")
    if r["method"].startswith("two_step") and abs(total - parts) > max(0.1, total * 0.25):
        problems.append(f"component sum differs from total for {r['method']} q={r['query_id']} repeat={r['repeat_id']}: total={total} parts={parts}")
with open("results/priority1_sanity_check.txt", "w") as f:
    f.write("rows=%d\n" % len(rows))
    f.write("status=%s\n" % ("PASS" if not problems else "WARN"))
    for p in problems:
        f.write(p + "\n")
PY

"$PYTHON_BIN" python/benchmark/benchmark_v2.py \
  -f "$DATASET" -m query -q "$QUERIES" -k "$K" --embedding-dim "$DIM" \
  --revision-csv --methods float32_avx2,binary,two_step_RF10,two_step_RF50,two_step_mf_RF10,two_step_mf_RF50 \
  --max-vectors "$FULL_N" --repeats "$REPEATS" --csv-output "$RAW"

"$PYTHON_BIN" scripts/summarize_priority1.py "$RAW" "$SUMMARY"
