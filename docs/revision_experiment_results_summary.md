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

The requested full Wikipedia/mxbai document embedding parquet was not present in
the checkout:

- Missing preferred path: `python/out/1_2M_random_out_mixedbread.parquet`
- Query path used: `python/query_embeddings/combined.jsonl`

To run sanity checks and verify the new instrumentation, a small local fallback
parquet was generated from the existing 1024-dimensional query embeddings:

- Fallback dataset: `python/out/local_query_embeddings_1024.parquet`
- Fallback dataset size: 313 vectors
- Dimension: 1024

This fallback is not a substitute for the paper-scale Wikipedia document
embedding dataset.

## 5. Dataset sizes used

- Priority 1 fallback run: `N=313`
- Priority 1 sanity check: `N=100`, 3 queries, 2 repeats
- Priority 2 fallback run: `N=100`, `N=200`, `N=313`

The requested `60000`, `300000`, and `1200000` sizes could not be run because
the full embedding parquet was unavailable.

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

Fallback parquet generation:

```bash
. .venv/bin/activate
mkdir -p python/out
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
rows=[]
with open('python/query_embeddings/combined.jsonl') as f:
    for i,line in enumerate(f):
        j=json.loads(line)
        row={'formatted_text': j.get('formatted_query') or j.get('query') or f'query {i}'}
        for d,v in enumerate(j['embedding']):
            row[f'embedding_{d}']=float(v)
        rows.append(row)
out=Path('python/out/local_query_embeddings_1024.parquet')
pd.DataFrame(rows).to_parquet(out, index=False)
print(out, len(rows), len(rows[0])-1)
PY
```

Priority 1 fallback:

```bash
. .venv/bin/activate
DATASET="$PWD/python/out/local_query_embeddings_1024.parquet" \
REPEATS=1 SANITY_N=100 SANITY_QUERIES=3 SANITY_REPEATS=2 FULL_N=0 \
scripts/run_priority1_timing_breakdown.sh
```

Priority 2 fallback:

```bash
. .venv/bin/activate
DATASET="$PWD/python/out/local_query_embeddings_1024.parquet" \
SIZES="100 200 313" QUERY_LIMIT=25 REPEATS=1 \
scripts/run_priority2_scaling.sh
```

Plots:

```bash
. .venv/bin/activate
scripts/plot_revision_results.py
```

## 8. Priority 1 summary table

See `results/priority1_timing_summary.csv`.

Key fallback summary:

- `float32_avx2`: geomean total about `0.0116 ms`
- `binary`: geomean total about `0.00593 ms`
- `two_step_RF10`: geomean total about `0.0214 ms`
- `two_step_mf_RF10`: geomean total about `0.0978 ms`

Because `N=313`, RF10 and RF50 both saturate at all available candidates, so
RF50 cannot show the intended larger-survivor behavior on this fallback data.

## 9. Priority 2 summary table

See `results/priority2_scaling_summary.csv` and
`results/priority2_scaling_verification.md`.

Fallback sizes were too small to support manuscript-scale conclusions.

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

On the fallback `N=313` data:

- `two_step_RF10` was dominated by rescoring rather than binary scan.
- This is an artifact of the tiny dataset and candidate saturation.
- No manuscript-scale conclusion should be drawn from this fallback run.

## 13. Whether RF50 has higher/equal rescoring cost than RF10

On the fallback `N=313` data, RF10 and RF50 both select all available candidates
for `k=100`, so survivor counts and rescore costs are saturated. This check
requires a larger dataset, ideally at least `k * 50 = 5000` vectors and
preferably the full paper dataset.

## 14. Whether RF50 has higher/equal accuracy than RF10

On the fallback data, RF10 and RF50 saturate to the same candidate set, so this
comparison is not meaningful.

## 15. Whether scaling with N is approximately linear

The fallback `N=100`, `200`, `313` run is too small for a strong scaling
conclusion. `float32_avx2` increased with N in the fallback summary, but the
binary and two-step numbers include fixed overheads and tiny absolute runtimes.

## 16. Unexpected results

- The existing full AVX2 search path assumes `k <= N`; an attempted fallback
  scaling run with `N=60` and `k=100` segfaulted. The successful fallback run
  used all sizes `>= k`.
- The mapped-float initialization prints many partition lines to stdout; this is
  existing behavior and was not refactored.
- On the tiny fallback dataset, two-step accuracy decreases as `N` grows because
  `k=100` and candidate saturation interact with using query embeddings as the
  document set.

## 17. Known limitations

- Full Wikipedia/mxbai document embeddings were unavailable locally.
- Priority 1 and Priority 2 were executed only on a generated 313-vector
  fallback parquet.
- The fallback parquet is useful for code and instrumentation verification, not
  for TECS manuscript claims.
- Full experiments should be rerun after providing
  `python/out/1_2M_random_out_mixedbread.parquet` or another real document
  embedding parquet.

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

The generated fallback parquet under `python/out/` is ignored by git.
