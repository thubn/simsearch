# Revision Experiment Results Summary

## 1. Git commit used

- Initial commit: `0bbeb0653a569e5394ac94a502549b079f7a055c`
- Branch: `tecs-revision-minimal-experiments`
- Final experiment commit: see repository `HEAD` after the final commit.

## 2. Machine information summary

- Machine: `meow1`
- CPU: AMD Ryzen 9 9950X 16-Core Processor
- Full machine info files:
  - `results/build_info/lscpu.txt`
  - `results/build_info/memory.txt`
  - `results/build_info/uname.txt`

## 3. Compiler and build flags

- Compiler: GCC 15.2.0
- CMake: 3.31.6
- Release flags from `CMakeLists.txt`:
  `-O3 -march=native -fopenmp -mavx2 -DNO_THREADS`
- Build logs:
  - `results/build_info/cmake_configure_before_changes.log`
  - `results/build_info/cmake_configure_with_pyarrow.log`
  - `results/build_info/cmake_configure_after_changes.log`
  - `results/build_info/cmake_build_after_changes.log`

## 4. Dataset path and query path

After the first implementation pass, local parquet files were added under
`python/out/`. The rerun used:

- Dataset path: `python/out/1_2M_random_out_mixedbread.parquet`
- Query path: `python/query_embeddings/combined.jsonl`
- Dataset rows: 1,000
- Dimension: 1024

The file name still says `1_2M`, but pyarrow metadata reports 1,000 rows. This
is the largest local mxbai document embedding parquet currently available.

## 5. Dataset sizes used

- Priority 1 run: `N=1000`
- Priority 1 sanity check: `N=1000`, 3 queries, 2 repeats
- Priority 2 run: `N=100`, `N=500`, `N=1000`

The requested `60000`, `300000`, and `1200000` sizes could not be run because
the largest local mxbai parquet has only 1,000 rows.

## 6. Methods run

Priority 1:

- `float32_avx2`
- `binary`
- `two_step_RF10`
- `two_step_RF50`
- `two_step_mf_RF10`
- `two_step_mf_RF50`

Priority 2:

- `float32_avx2`
- `binary`
- `two_step_RF10`
- `two_step_mf_RF10`

## 7. Exact commands run

Build:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyarrow numpy pandas matplotlib
PYARROW_DIR="$(python - <<'PY'
import os, pyarrow
print(os.path.dirname(pyarrow.__file__))
PY
)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPYARROW_DIR="$PYARROW_DIR" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j"$(nproc)"
```

Priority 1:

```bash
. .venv/bin/activate
DATASET="$PWD/python/out/1_2M_random_out_mixedbread.parquet" \
REPEATS=1 SANITY_N=1000 SANITY_QUERIES=3 SANITY_REPEATS=2 FULL_N=0 \
scripts/run_priority1_timing_breakdown.sh
```

Priority 2:

```bash
. .venv/bin/activate
DATASET="$PWD/python/out/1_2M_random_out_mixedbread.parquet" \
SIZES="100 500 1000" QUERY_LIMIT=100 REPEATS=1 \
scripts/run_priority2_scaling.sh
```

Plots:

```bash
. .venv/bin/activate
scripts/plot_revision_results.py
```

## 8. Priority 1 summary table

See `results/priority1_timing_summary.csv`.

Key `N=1000` summary:

- `float32_avx2`: geomean total about `0.0419 ms`
- `binary`: geomean total about `0.0131 ms`
- `two_step_RF10`: geomean total about `0.0702 ms`
- `two_step_mf_RF10`: geomean total about `0.309 ms`
- `two_step_RF50`: geomean total about `0.0641 ms`
- `two_step_mf_RF50`: geomean total about `0.302 ms`

Because `N=1000`, both RF10 and RF50 saturate at all available candidates for
`k=100`, so RF50 cannot show the intended larger-survivor behavior on this
dataset.

## 9. Priority 2 summary table

See `results/priority2_scaling_summary.csv` and
`results/priority2_scaling_verification.md`.

The local sizes are useful for script/instrumentation verification, but they
are still too small to support manuscript-scale conclusions.

## 10. Sanity-check results

`results/priority1_sanity_check.txt` reports:

```text
rows=24
status=PASS
```

No negative timing fields were observed in the sanity CSV. Timing component sums
were within the sanity threshold for two-step methods.

## 11. Whether timing components sum to total latency

For the sanity run, the component sums were close enough to total latency for the
new instrumentation. The `T_total_ms` value includes query normalization and
wrapper overhead, so exact equality is not expected.

## 12. Whether Step 1 or Step 2 dominates

On the local `N=1000` data:

- `two_step_RF10` was dominated by rescoring plus candidate selection rather
  than binary scan alone.
- This is expected with only 1,000 vectors because the two-step candidate set
  saturates to the full dataset for `k=100, RF=10`.
- No manuscript-scale Step 1/Step 2 conclusion should be drawn until the larger
  dataset is available.

## 13. Whether RF50 has higher/equal rescoring cost than RF10

On the local `N=1000` data, RF10 and RF50 both select all available candidates
for `k=100`, so survivor counts and rescore costs are saturated. This check
requires a larger dataset, ideally at least `k * 50 = 5000` vectors and
preferably the full paper dataset.

## 14. Whether RF50 has higher/equal accuracy than RF10

On the local data, RF10 and RF50 saturate to the same candidate set, so this
comparison is not meaningful.

## 15. Whether scaling with N is approximately linear

The local `N=100`, `500`, `1000` run is too small for a strong scaling
conclusion. `float32_avx2` increased with N in the summary, but the binary and
two-step numbers include fixed overheads and small absolute runtimes.

## 16. Unexpected results

- The existing full AVX2 search path assumes `k <= N`; an earlier fallback
  scaling run with `N=60` and `k=100` segfaulted. The successful reruns used all
  sizes `>= k`.
- The mapped-float initialization prints many partition lines to stdout; this is
  existing behavior and was not refactored.
- With only 1,000 vectors, RF10/RF50 candidate counts saturate, so the RF
  comparison is not representative.

## 17. Known limitations

- The local mxbai parquet has 1,000 rows, not 1.2M.
- Priority 1 and Priority 2 were executed on the largest local mxbai parquet.
- These results are useful for code and instrumentation verification, not for
  TECS manuscript-scale claims.
- Full experiments should be rerun after providing a larger document embedding
  parquet, ideally the 1.2M Wikipedia/mxbai file.

## 18. Files changed

- `CMakeLists.txt`
- `include/timing_breakdown.h`
- `include/common_structs.h`
- `include/embedding_io.h`
- `include/embedding_search_float.h`
- `include/embedding_search_mapped_float.h`
- `include/optimized_embedding_search_avx2.h`
- `include/optimized_embedding_search_binary_avx2.h`
- `src/embedding_io.cpp`
- `src/embedding_search_benchmark_bindings.cpp`
- `src/embedding_search_mapped_float.cpp`
- `src/optimized_embedding_search_avx2.cpp`
- `src/optimized_embedding_search_binary_avx2.cpp`
- `python/benchmark/benchmark_v2.py`
- `scripts/collect_machine_info.sh`
- `scripts/run_priority1_timing_breakdown.sh`
- `scripts/run_priority2_scaling.sh`
- `scripts/summarize_priority1.py`
- `scripts/summarize_priority2.py`
- `scripts/plot_revision_results.py`
- `docs/codex_repo_understanding.md`
- `docs/dataset_discovery_and_download_plan.md`
- `docs/build_diagnosis.md`

## 19. Files generated

- `initial_commit.txt`
- `initial_git_status.txt`
- `results/build_info/*`
- `results/priority1_sanity_check.txt`
- `results/priority1_sanity_raw.csv`
- `results/priority1_timing_breakdown.csv`
- `results/priority1_timing_summary.csv`
- `results/priority2_scaling_raw.csv`
- `results/priority2_scaling_summary.csv`
- `results/priority2_scaling_verification.md`
- `figures/revision_priority1_timing_breakdown.pdf`
- `figures/revision_priority2_scaling.pdf`

The generated/local parquet files under `python/out/` are ignored by git.
