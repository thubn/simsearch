# Implementation Safety Checklist

Date: 2026-05-19

## C1. Python Method to C++ Implementation Mapping

Inspected:

- `src/embedding_search_benchmark_bindings.cpp`
- `python/benchmark/benchmark_v2.py`

Mappings:

| Python method | C++ binding call | C++ implementation object | Notes |
|---|---|---|---|
| `avx2` | `EmbeddingSearch.search_avx2` | `OptimizedEmbeddingSearchAVX2` via `searchers->oavx2` | Used as `float32_avx2` in revision CSV mode. |
| `binary` | `EmbeddingSearch.search_binary` | `OptimizedEmbeddingSearchBinaryAVX2` via `searchers->obinary_avx2` | Existing `binary` already uses optimized binary AVX2. No separate `binary_opt_avx2` is needed. |
| `binary` timed | `EmbeddingSearch.search_binary_timed` | `OptimizedEmbeddingSearchBinaryAVX2::similarity_search_with_timing` | Same optimized binary path with timing fields. |
| `twostep_rf10` / `twostep_rf50` | `EmbeddingSearch.search_twostep` | candidate generation: `OptimizedEmbeddingSearchBinaryAVX2`; rescore: `OptimizedEmbeddingSearchAVX2` | Existing two-step path. |
| `two_step_RF10` / `two_step_RF50` revision CSV | `EmbeddingSearch.search_twostep_timed` | same binary candidate generation + AVX2 rescore | Timed additive path. |
| `ts_mf_rf10` / `ts_mf_rf50` | `EmbeddingSearch.search_twostep_mf` | candidate generation: `OptimizedEmbeddingSearchBinaryAVX2`; rescore: `EmbeddingSearchMappedFloat` | Existing mapped-float two-step path. |
| `two_step_mf_RF10` / `two_step_mf_RF50` revision CSV | `EmbeddingSearch.search_twostep_mf_timed` | same binary candidate generation + mapped-float rescore | Timed additive path. |
| `mf` | `EmbeddingSearch.search_mf` | `EmbeddingSearchMappedFloat` via `searchers->mappedFloat` | Full mapped-float scan. |

Conclusion: the binary-only revision method matches the optimized binary path used in two-step candidate generation. Existing `binary` semantics were not silently changed.

## C2. Timing Scope

The per-query timing fields are collected inside search calls after the dataset and searchers are already initialized.

Reported query-time timing excludes:

- parquet loading
- query JSONL loading
- searcher construction
- binary sketch construction for the whole dataset
- mapped-float partition construction
- result file writing
- Python process startup

Reported query-time timing includes:

- `T_query_sketch_ms`: query float-to-binary sketch conversion for binary/two-step methods.
- `T_binary_scan_ms`: optimized binary AVX2 scan.
- `T_candidate_selection_ms`: binary candidate top-k / top-k*RF selection.
- `T_rescore_ms`: AVX2 or mapped-float rescoring of candidate survivors.
- `T_final_topk_ms`: final top-k selection after rescoring.
- `T_total_ms`: timed method end-to-end query call wall time.

Unavoidable wrapper/conversion overhead is not forced into a component bucket. It is reported as unaccounted time.

## C3. Total-vs-Component Timing Accounting

Future revision CSV output now includes:

- `T_component_sum_ms`
- `T_unaccounted_ms`

Definitions:

```text
T_component_sum_ms =
  T_query_sketch_ms
  + T_binary_scan_ms
  + T_candidate_selection_ms
  + T_rescore_ms
  + T_final_topk_ms

T_unaccounted_ms = T_total_ms - T_component_sum_ms
```

The code does not force equality. Summarizers now include `geomean_unaccounted_ms` using the absolute unaccounted value.

Note: the old N=1000 smoke CSVs were generated before these two columns were added and have been moved to `results/smoke_tests/`.

## C4. Mapped-Float Partition Validation

Current implementation behavior:

- `EmbeddingSearchMappedFloat` constructs partitions from the active loaded embeddings in memory.
- It does not load an external partition file for the Python benchmark path.
- Therefore there is no current risk of accidentally using `mapped_float_partitions_768.txt` for a 1024-d mxbai run in the active benchmark code.

Validation rule for future changes:

- If an external partition file is introduced, the runner must record the partition file path, expected dimensionality, and active embedding dimensionality, then fail if they disagree.

Current mapped-float dimensionality is coupled to the active loaded dataset and embedding dimension through `searchers->mappedFloat.setEmbeddings(base.getEmbeddings(), 10.0)`.

## Dataset and Parameter Safety

Implemented safeguards:

- `scripts/inspect_parquet_dataset.py` reports actual parquet row count and inferred embedding dimension.
- Priority scripts inspect the dataset before running.
- If a filename contains `1_2M` but the parquet has fewer than 1,200,000 rows, a warning is printed.
- `N < k` fails clearly before search.
- `N <= k*RF` prints a warning that RF comparison is saturated and not meaningful.
- Priority 1 writes smoke outputs under `results/smoke_tests/` when `N < 60000`.
- Priority 2 writes smoke outputs under `results/smoke_tests/` when max available N is below 60000 and does not overwrite canonical manuscript-intended outputs.

## Rerun Policy

- `N < 5000`: smoke-test instrumentation only.
- `5000 <= N < 60000`: small-scale validation only; RF10/RF50 is technically meaningful if `N > k*RF`.
- `N >= 60000`: manuscript-minimum Priority 1 can run.
- `N = 60000, 300000, and one larger size`: manuscript-minimum Priority 2 can run.
- `N = 60000, 300000, 1200000`: preferred Priority 2.
