# Revision Code Audit

Date: 2026-05-19

Baseline for audit: `0bbeb0653a569e5394ac94a502549b079f7a055c`.

## A1. Diff Summary

Commands requested:

```bash
git diff --stat 0bbeb0653a569e5394ac94a502549b079f7a055c..HEAD
git diff --name-only 0bbeb0653a569e5394ac94a502549b079f7a055c..HEAD
```

The committed revision work changed build wiring, timing data structures, Python revision CSV mode, C++ timing wrappers, scripts, docs, and generated small-N smoke-test results. This follow-up additionally adds safe dataset generation, parquet inspection, row-count gating, and output-equivalence checks.

Changed source/header files from the baseline:

| file | why it changed | algorithm semantics changed? | timing / CLI / safety only? | risk |
|---|---|---|---|---|
| `CMakeLists.txt` | Add pyarrow/Python extension build support needed by benchmark bindings. | No ranking semantics intended. | Build/extension wiring. | Medium |
| `include/timing_breakdown.h` | Add `TimingBreakdown` struct for per-query component timing. | No. | Timing only. | Low |
| `include/common_structs.h` | Thread `max_vectors` through searcher initialization. | No ranking change for full default load; revision mode can limit N intentionally. | Dataset-size control. | Medium |
| `include/embedding_io.h` | Add/load max-vector parameter. | No ranking change unless caller supplies max-vector limit. | Dataset-size control. | Medium |
| `src/embedding_io.cpp` | Stop reading after requested max vectors. | No default change when max is 0. | Dataset-size control. | Medium |
| `include/embedding_search_float.h` | Expose timing overload support. | No. | Timing only. | Low |
| `include/embedding_search_mapped_float.h` | Expose timing overload for candidate rescoring. | No. | Timing only. | Low |
| `src/embedding_search_mapped_float.cpp` | Add timing around mapped-float rescoring/final top-k. | No intended ranking change. | Timing only. | Low |
| `include/optimized_embedding_search_avx2.h` | Expose timing overload for candidate rescoring. | No. | Timing only. | Low |
| `src/optimized_embedding_search_avx2.cpp` | Add timing around AVX2 rescoring/final top-k. | No intended ranking change. | Timing only. | Low |
| `include/optimized_embedding_search_binary_avx2.h` | Expose timing overload for binary scan/candidate selection. | No. | Timing only. | Low |
| `src/optimized_embedding_search_binary_avx2.cpp` | Add timing around binary scan and candidate selection; uses `min(k, N)` for result slicing. | Ranking semantics for valid `k <= N` are unchanged. `k > N` is made safer. | Timing/safety. | Low |
| `src/embedding_search_benchmark_bindings.cpp` | Add timed binary/two-step binding methods and `max_vectors` load parameter. | Existing method names still call the same implementation paths. Timed methods are additive. | Timing/CLI support. | Medium |
| `python/benchmark/benchmark_v2.py` | Add revision CSV mode, method selection, max-vector/query-limit/repeats flags, timing columns, and N/k/RF warnings. | Existing non-revision method names and JSON output path remain. A safety failure is now raised when `N < k`. | CLI/safety/timing. | Medium |
| `python/create_embeddings.py` | Ensure incremental ParquetWriter is closed. | No embedding semantics change. | Safety/correctness. | Low |
| `python/start_create_embeddings_mixedbread.py` | Replace misleading `1_2M` + `random_rows=1000` path with safe N1000 wrapper. | Dataset generation target/name changed intentionally. | Dataset safety. | Low |
| `python/start_create_embeddings_mpnet.py` | Replace misleading `1_2M` + `random_rows=1000` path with safe N1000 wrapper. | Dataset generation target/name changed intentionally. | Dataset safety. | Low |
| `python/generate_document_embeddings.py` | New incremental generation CLI with verified metadata sidecar. | New script only. | Dataset safety. | Low |
| `scripts/inspect_parquet_dataset.py` | New parquet metadata inspection utility. | No. | Dataset safety. | Low |
| `scripts/generate_mxbai_dataset.sh` | New safe wrapper; default `TARGET_ROWS=1000`. | No benchmark ranking change. | Dataset generation safety. | Low |
| `scripts/generate_mpnet_dataset.sh` | New safe wrapper; default `TARGET_ROWS=1000`. | No benchmark ranking change. | Dataset generation safety. | Low |
| `scripts/run_priority1_timing_breakdown.sh` | Add parquet validation and smoke-test output routing for N < 60000. | No ranking change. | Safety/output hygiene. | Low |
| `scripts/run_priority2_scaling.sh` | Add parquet validation and smoke-test output routing for N < 60000. | No ranking change. | Safety/output hygiene. | Low |
| `scripts/summarize_priority1.py` | Summarize unaccounted timing. | No. | Reporting only. | Low |
| `scripts/summarize_priority2.py` | Summarize unaccounted timing. | No. | Reporting only. | Low |
| `scripts/check_output_equivalence.py` | New equivalence check comparing timed and old-style outputs. | No. | Verification only. | Low |

No source/header change appears unrelated to timing instrumentation, CLI control, dataset-size control, or safety checking. No source file was reverted.

## A2. Default-Behavior Preservation

- Existing Python method names are unchanged: `float`, `avx2`, `binary`, `int8`, `float16`, `mf`, `pca*`, `twostep_rf*`, and `ts_mf_rf*`.
- Existing non-revision JSON output behavior in `benchmark_v2.py` is still the default unless `--revision-csv` is supplied.
- Existing default `k=25` is unchanged.
- Existing query loading reads JSONL query files as before.
- Existing result ranking paths are unchanged for the old methods. Timed methods are additive and call the same underlying optimized binary, AVX2, and mapped-float searchers.
- The earlier behavior that silently dropped rescoring factors when `k*RF > N` has been removed. The code now warns instead. This avoids silently changing requested methods.
- New safety behavior: `N < k` raises a clear error rather than allowing a crash or undefined result.

## A3. Output-Equivalence Test

Command run:

```bash
export PYTHONPATH="$PWD/build:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(cat results/build_info/pyarrow_dir.txt 2>/dev/null || true):${LD_LIBRARY_PATH:-}"
.venv/bin/python scripts/check_output_equivalence.py \
  --dataset python/out/1_2M_random_out_mixedbread.parquet \
  --queries python/query_embeddings/combined.jsonl \
  --embedding-dim 1024 \
  --max-vectors 1000 \
  --query-limit 5 \
  --k 100 \
  --rf 10 \
  --csv-output results/output_equivalence_check.csv \
  --md-output results/output_equivalence_check.md
```

Result: **PASS**.

Compared methods:

- `float32_avx2`
- `binary`
- `two_step_RF10`
- `two_step_mf_RF10`

The check compares old-style search methods against timed methods in the same build. All compared methods produced the same top-k ID list or set for 5 queries at N=1000, k=100, RF=10.

Artifacts:

- `results/output_equivalence_check.csv`
- `results/output_equivalence_check.md`

## Current Limitations

The equivalence check uses the current local 1,000-row dataset. It validates instrumentation semantics, not manuscript-scale performance.

The required dataset-generation smoke test did not complete because the active `.venv` lacks `datasets`, `sentence-transformers`, and `torch`. The wrapper failed before dataset loading or embedding generation.
