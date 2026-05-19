# Repository Understanding for TECS Revision Experiments

Initial inspection was performed on branch `reproduce-v1` at commit
`0bbeb0653a569e5394ac94a502549b079f7a055c`. The implementation branch is
`tecs-revision-minimal-experiments`.

## 1. Build system

- The project uses CMake and builds both a C++ executable and Python extension
  bindings.
- Main CMake target:
  - executable: `simsearch`
  - Python module: `embedding_search_benchmark.so`
- `CMakeLists.txt` fetches `nlohmann/json`, Eigen 3.4, and pybind11 2.11.1.
- CMake requires `PYARROW_DIR` and links Arrow and Parquet libraries from that
  directory.
- CMake requires OpenMP and Python3 interpreter/development headers.
- `CMAKE_CXX_STANDARD` is set to C++23.
- Existing root `README.md` build command:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

- Because this checkout's `CMakeLists.txt` requires `PYARROW_DIR`, a complete
  build command usually needs:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYARROW_DIR=/path/to/pyarrow
cmake --build build -j"$(nproc)"
```

- Existing CMake presets:
  - `CMakePresets.json` has one configure preset named
    `Configure preset using toolchain file`. It uses Ninja, writes under
    `out/build/${presetName}`, and sets `CMAKE_BUILD_TYPE=Debug`.
  - There is no preset named `default` in the repository `CMakePresets.json`.
  - `CMakeUserPresets.json` includes Conan-generated preset files under
    `build/Release/generators/` and `build/Debug/generators/`, but those files
    were not present in this checkout during inspection.
- Compiler/SIMD flags in `CMakeLists.txt`:
  - global: `-Wno-ignored-attributes`
  - Debug: `-O0 -g -march=native -DNO_THREADS`
  - Release: `-O3 -march=native -fopenmp -mavx2 -DNO_THREADS`
  - AVX2 is explicitly enabled in Release via `-mavx2`.
  - FMA is used in source via intrinsics; `-march=native` should enable FMA on
    the target Ryzen if supported.
- Release mode is not the only preset mode; the repository preset uses Debug,
  while README commands explicitly use Release.

## 2. Main executable and benchmark path

- The CMake executable is `simsearch`.
- `src/main.cpp` contains a C++ benchmark/query driver. It accepts:
  - `--file` / `-f`
  - `--query-file` / `-q`
  - `--topk` / `-k`
  - `--runs` / `-r`
  - `--rescoring-factor` / `-s`
- `src/main.cpp` initializes searchers and can run either random-embedding
  benchmark mode or query-file mode. It exports JSON via
  `include/stats_exporter.h`.
- `python/benchmark/benchmark_v2.py` is the primary existing benchmark driver
  used by the revision scripts. It imports `EmbeddingSearch` from
  `embedding_search_benchmark.so`, loads a parquet embedding file, runs methods,
  computes quality metrics against float reference results, and writes JSON
  result files.
- Existing benchmark result export:
  - Python benchmark writes timestamped JSON files plus timestamped summary JSON.
  - C++ executable writes `query_search_results_<timestamp>.json` or
    `random_emb_search_results_<timestamp>.json`.
- `include/stats_exporter.h` is used by `src/main.cpp`, not by
  `python/benchmark/benchmark_v2.py`.

## 3. Search methods

Current method map:

| Logical method | Existing names / entry points | Source files |
| --- | --- | --- |
| scalar float32 | `EmbeddingSearchFloat::similarity_search` | `include/embedding_search_float.h`, `src/embedding_search_float.cpp` |
| AVX2 float32 | `EmbeddingSearchAVX2`; optimized driver uses `OptimizedEmbeddingSearchAVX2` | `include/embedding_search_avx2.h`, `src/embedding_search_avx2.cpp`, `include/optimized_embedding_search_avx2.h`, `src/optimized_embedding_search_avx2.cpp` |
| binary sketch construction | `EmbeddingUtils::convertSingleFloatToBinaryAVX2`; `OptimizedEmbeddingSearchBinaryAVX2::convert_float_to_binary_avx2` | `include/embedding_utils.h`, `src/embedding_utils.cpp`, `src/optimized_embedding_search_binary_avx2.cpp` |
| binary search | `EmbeddingSearchBinary::similarity_search` | `include/embedding_search_binary.h`, `src/embedding_search_binary.cpp` |
| AVX2 binary search | `EmbeddingSearchBinaryAVX2::similarity_search` | `include/embedding_search_binary_avx2.h`, `src/embedding_search_binary_avx2.cpp` |
| optimized AVX2 binary search | `OptimizedEmbeddingSearchBinaryAVX2::similarity_search` | `include/optimized_embedding_search_binary_avx2.h`, `src/optimized_embedding_search_binary_avx2.cpp` |
| two-step search | Python binding `PyEmbeddingSearch::search_twostep`: optimized binary candidates then optimized AVX2 float rescoring | `src/embedding_search_benchmark_bindings.cpp`, `src/optimized_embedding_search_binary_avx2.cpp`, `src/optimized_embedding_search_avx2.cpp` |
| two-step mapped-float search | Python binding `PyEmbeddingSearch::search_twostep_mf`: optimized binary candidates then mapped-float rescoring | `src/embedding_search_benchmark_bindings.cpp`, `src/embedding_search_mapped_float.cpp` |
| mapped-float search | `EmbeddingSearchMappedFloat::similarity_search` | `include/embedding_search_mapped_float.h`, `src/embedding_search_mapped_float.cpp` |
| uint8 / int8 AVX2 search | `EmbeddingSearchUint8AVX2`, `OptimizedEmbeddingSearchUint8AVX2` | `include/embedding_search_uint8_avx2.h`, `src/embedding_search_uint8_avx2.cpp`, `include/optimized_embedding_search_uint8_avx2.h`, `src/optimized_embedding_search_uint8_avx2.cpp` |
| top-k selection | `std::partial_sort` in every search implementation inspected | search implementation `.cpp` files |
| NDCG / Jaccard | C++ `EmbeddingUtils::calculateNDCG`, `calculateJaccardIndex`; Python `calculate_ndcg` plus set Jaccard | `include/embedding_utils.h`, `python/benchmark/benchmark_v2.py` |

Existing Python benchmark method names:

- `float`
- `avx2`
- `binary`
- `int8`
- `float16`
- `mf`
- `pca2`, `pca4`, `pca8`, `pca16`, `pca32`
- `twostep_rf<RF>`
- `ts_mf_rf<RF>`

For the revision experiments, use this mapping:

- `float32_avx2` -> existing `avx2`
- `binary` -> existing `binary`
- `two_step_RF10` -> existing `twostep_rf10`
- `two_step_RF50` -> existing `twostep_rf50`
- `two_step_mf_RF10` -> existing `ts_mf_rf10`
- `two_step_mf_RF50` -> existing `ts_mf_rf50`

## 4. Dataset assumptions

- Expected primary embedding file paths from scripts:
  - mxbai / mixedbread 1024-d:
    `python/out/1_2M_random_out_mixedbread.parquet`
  - older mxbai script name:
    `python/out/1_2M_random_out.parquet`
  - mpnet 768-d:
    `python/out/1_2M_random_out_mpnet.parquet`
- Expected embedding format:
  - Parquet file with `formatted_text` string column.
  - Embedding columns named `embedding_0`, `embedding_1`, ..., up to
    `embedding_<d-1>`.
  - `EmbeddingIO::load_parquet` reads every row and the configured number of
    embedding columns.
- Expected query file paths:
  - mxbai / mixedbread 1024-d:
    `python/query_embeddings/combined.jsonl`
  - mpnet 768-d:
    `python/query_embeddings/combined_mpnet.jsonl`
- Query format:
  - JSON Lines with keys `query`, `formatted_query`, and `embedding`.
  - `combined.jsonl` has 313 valid 1024-dimensional queries.
  - `combined_mpnet.jsonl` has 313 valid 768-dimensional queries.
- mxbai / 1024-d selection:
  - Existing benchmark script uses `--embedding-dim 1024`,
    `../out/1_2M_random_out_mixedbread.parquet`, and
    `../query_embeddings/combined.jsonl`.
- mpnet / 768-d selection:
  - Existing benchmark script uses `--embedding-dim 768`,
    `../out/1_2M_random_out_mpnet.parquet`, and
    `../query_embeddings/combined_mpnet.jsonl`.
- `k` selection:
  - `config.json` and `python/benchmark/config.json` set `algorithm.k=100`.
  - Python benchmark CLI default is `k=25`, but existing run scripts pass
    `-k 100`, `-k 25`, or `-k 10`.
- Query repetition count:
  - Python query-file mode currently runs each query once and does not use
    `--runs`.
  - Random mode uses `--runs`, default `100`; existing run scripts use `250`.
  - Revision scripts need an explicit repeat loop for query-file experiments.
- Repository contains small and full query embedding sets, but no document
  embedding parquet files were found locally under the checkout.

## 5. Dataset download / regeneration findings

See `docs/dataset_discovery_and_download_plan.md` for details. In short:

- The paper dataset is Wikimedia Wikipedia from Hugging Face:
  `wikimedia/wikipedia`, config `20231101.en`, split `train`.
- Preferred embedding model is `mixedbread-ai/mxbai-embed-large-v1`.
- Expected output for the primary experiment is
  `python/out/1_2M_random_out_mixedbread.parquet`.
- Existing helper scripts currently set `random_rows=1000` despite the 1.2M file
  name; this must be adjusted for full regeneration.
- Large download/regeneration should not happen until the path, desired size,
  and runtime budget are approved.

## 6. Minimal implementation plan

Files to modify:

- `include/common_structs.h` or a narrow new header for a `TimingBreakdown`
  struct.
- `include/optimized_embedding_search_binary_avx2.h`
- `src/optimized_embedding_search_binary_avx2.cpp`
- `include/optimized_embedding_search_avx2.h`
- `src/optimized_embedding_search_avx2.cpp`
- `include/embedding_search_mapped_float.h`
- `src/embedding_search_mapped_float.cpp`
- `src/embedding_search_benchmark_bindings.cpp`
- `python/benchmark/benchmark_v2.py`

Files to add:

- `scripts/collect_machine_info.sh`
- `scripts/run_priority1_timing_breakdown.sh`
- `scripts/run_priority2_scaling.sh`
- `scripts/summarize_priority1.py`
- `scripts/summarize_priority2.py`
- `scripts/plot_revision_results.py`
- generated result/report files under `results/`
- final summary: `docs/revision_experiment_results_summary.md`

Functions/classes to instrument:

- `PyEmbeddingSearch::search_binary`
- `PyEmbeddingSearch::search_twostep`
- `PyEmbeddingSearch::search_twostep_mf`
- helper paths on `OptimizedEmbeddingSearchBinaryAVX2`,
  `OptimizedEmbeddingSearchAVX2`, and `EmbeddingSearchMappedFloat` to separate
  scan/rescore work from `partial_sort` top-k selection.

CLI/config options to add:

- Prefer extending `python/benchmark/benchmark_v2.py` with revision-only options:
  `--methods`, `--max-vectors`, `--query-limit`, `--repeats`,
  `--timing-breakdown`, `--csv-output`, and metadata fields.
- Keep existing benchmark behavior as the default when those options are not
  supplied.

Priority 1 run:

- Dataset: mxbai/mixedbread 1024-d parquet if available.
- Queries: `python/query_embeddings/combined.jsonl`.
- Methods: `float32_avx2`, `binary`, `two_step_RF10`, `two_step_RF50`,
  `two_step_mf_RF10`, `two_step_mf_RF50`.
- Output: `results/priority1_timing_breakdown.csv`,
  `results/priority1_timing_summary.csv`,
  `results/priority1_sanity_check.txt`.

Priority 2 run:

- Sizes: `60000`, `300000`, `1200000`, or the largest feasible/local fallback.
- Methods: `float32_avx2`, `binary`, `two_step_RF10`, `two_step_mf_RF10`.
- Output: `results/priority2_scaling_raw.csv`,
  `results/priority2_scaling_summary.csv`,
  `results/priority2_scaling_verification.md`.

Validation:

- Build original code before source changes.
- Sanity-check timing components against total latency.
- Verify no negative timing fields.
- Verify RF50 has at least as many survivors as RF10 and generally at least as
  much rescore time.
- Verify `float32_avx2` quality against scalar float reference remains exact or
  near exact under existing metric logic.
- Report unexpected data directly rather than forcing linearity or accuracy
  conclusions.
