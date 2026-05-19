# Dataset Forensics Report

Date: 2026-05-19

## 1. Executive Conclusion

Conclusion: **C. multiple datasets / mixed evidence depending on artifact.**

The current local parquet files in `python/out/` with `1_2M` in their names are **not** 1.2M-vector datasets. PyArrow metadata shows:

- `python/out/1_2M_random_out_mixedbread.parquet`: 1,000 rows, 1,024 embedding dimensions.
- `python/out/1_2M_random_out_mpnet.parquet`: 1,000 rows, 768 embedding dimensions.

The current embedding-generation start scripts also hardcode `random_rows=1000` while writing to `1_2M...parquet` output paths. Those filenames are therefore misleading for the current reproducible generation path.

However, the old benchmark JSON files under `python/jupyter/results/` all record `metadata.num_vectors = 1200000`. In `python/benchmark/benchmark_v2.py`, that metadata is populated from `self.searcher.get_dimensions()` after loading embeddings, not from the filename. The mapped-float partition text files also sum exactly to `1,200,000 * 1024` and `1,200,000 * 768` scalar values. This is strong evidence that the old plotted result artifacts were produced from a 1.2M-vector loaded dataset, unless those JSON/TXT artifacts were edited or generated from a different untracked file.

Strictly, the repository does **not** currently contain parquet metadata proving that a 1.2M parquet file exists. The missing artifact is the actual parquet file used to produce the old JSONs, or a log/result file that records both the dataset path and parquet row count.

## 2. Evidence Table

| Evidence item | File/path | Observed fact | Implication | Confidence |
|---|---|---|---|---|
| Local mxbai parquet metadata | `python/out/1_2M_random_out_mixedbread.parquet` | PyArrow reports 1,000 rows, 1,025 columns, 1,024 embedding columns. | Current local mxbai file is 1,000 vectors despite `1_2M` filename. | High |
| Local mpnet parquet metadata | `python/out/1_2M_random_out_mpnet.parquet` | PyArrow reports 1,000 rows, 769 columns, 768 embedding columns. | Current local mpnet file is 1,000 vectors despite `1_2M` filename. | High |
| Local query parquet metadata | `python/out/local_query_embeddings_1024.parquet` | PyArrow reports 313 rows, 1,024 embedding dimensions. | This is a query/fallback file, not a full document embedding file. | High |
| mxbai start script | `python/start_create_embeddings_mixedbread.py` | `output_path="out/1_2M_random_out_mixedbread.parquet"` and `random_rows=1000`. | Script can create a misleading `1_2M` filename with only 1,000 rows. | High |
| mpnet start script | `python/start_create_embeddings_mpnet.py` | `output_path="out/1_2M_random_out_mpnet.parquet"` and `random_rows=1000`. | Script can create a misleading `1_2M` filename with only 1,000 rows. | High |
| Benchmark result metadata | `python/jupyter/results/*.json` | 19 JSON files all contain `metadata.num_vectors = 1200000`; no dataset path is stored. | Old result files record 1.2M vectors, but cannot be tied to a retained parquet file path. | Medium-high |
| Metadata source in benchmark code | `python/benchmark/benchmark_v2.py` | `self.num_vectors, self.vector_dim = self.searcher.get_dimensions()`, and `save_results` writes `metadata.num_vectors`. | Old JSON `num_vectors` appears to be runtime-loaded vector count, not a filename-derived label. | High for code behavior |
| Mapped-float partitions | `python/jupyter/results/mapped_float_partitions.txt` | 256 partition sizes sum to 1,228,800,000 = 1,200,000 * 1,024. | Supports a 1.2M-vector, 1,024-dim artifact for old results. | Medium |
| Mapped-float partitions 768 | `python/jupyter/results/mapped_float_partitions_768.txt` | 256 partition sizes sum to 921,600,000 = 1,200,000 * 768. | Supports a 1.2M-vector, 768-dim artifact for old results. | Medium |
| Benchmark scripts | `run_benchmarks*.sh`, `python/benchmark/run_benchmarks*.sh` | Scripts pass `1_2M...parquet` paths. | Path names alone are weak evidence and must not be treated as row-count proof. | Low |
| Notebook plot sources | `python/jupyter/plots_v2.ipynb`, `plots_v3.ipynb`, `plots_v4.ipynb` | Notebooks load result JSONs and reference `metadata.num_vectors`; labels include `1200000` and `60K`. | Plots appear based on old JSON metadata, not parquet inspection. | Medium |
| Git history | repository history | Current start scripts have `random_rows=1000` in every commit where those start scripts appear. Generic `create_embeddings.py` has long supported optional `random_rows`. | The misleading start-script configuration was not introduced only by the current working tree. | Medium-high |

## 3. Local Parquet Inventory Summary

The full inventory is written to `results/dataset_file_inventory.csv`.

| path | file size bytes | rows | columns | inferred embedding dim | has formatted text |
|---|---:|---:|---:|---:|---|
| `python/out/1_2M_random_out_mixedbread.parquet` | 7,844,010 | 1,000 | 1,025 | 1,024 | true |
| `python/out/1_2M_random_out_mpnet.parquet` | 6,350,078 | 1,000 | 769 | 768 | true |
| `python/out/local_query_embeddings_1024.parquet` | 2,803,099 | 313 | 1,025 | 1,024 | true |

No local parquet file inspected under `python/out/` has 1.2M rows.

## 4. Embedding-Generation Script Summary

| script_path | model_name | dataset_name | dataset_config | dataset_split | output_path | random_rows | random_seed | chunk_size | streaming |
|---|---|---|---|---|---|---:|---:|---:|---|
| `python/start_create_embeddings_mixedbread.py` | `mixedbread-ai/mxbai-embed-large-v1` | `wikimedia/wikipedia` | `20231101.en` | `train` | `out/1_2M_random_out_mixedbread.parquet` | 1000 | 42 | 1000 | true |
| `python/start_create_embeddings_mpnet.py` | `sentence-transformers/all-mpnet-base-v2` | `wikimedia/wikipedia` | `20231101.en` | `train` | `out/1_2M_random_out_mpnet.parquet` | 1000 | 42 | 1000 | true |

Both start scripts are explicitly misleading: the output paths contain `1_2M`, but the configured row count is 1,000.

`python/create_embeddings.py` supports optional `random_rows`. In streaming mode, the code comment says it takes the first N rows when `random_rows` is set. This means a start script using `random_rows=1000` is expected to produce only 1,000 embedded documents.

## 5. Benchmark Script Summary

| script_path | dataset_path | query_path | embedding_dim | k | runs/repeats | methods | output_path |
|---|---|---|---:|---|---|---|---|
| `python/benchmark/run_benchmarks.sh` | `../out/1_2M_random_out.parquet`, `../out/1_2M_random_out_mpnet.parquet` | `../query_embeddings/combined.jsonl`, `../query_embeddings/combined_mpnet.jsonl` | 1024, 768 | 10, 25, 100 | `-r 250` for random/random-vec; query count from query file | default `benchmark_v2.py` methods including float, avx2, binary, int8, float16, mapped-float, PCA, two-step variants | default `benchmark_results_<timestamp>.json` unless overridden |
| `python/benchmark/run_benchmarks_v2.sh` | `../out/1_2M_random_out_mixedbread.parquet`, `../out/1_2M_random_out_mpnet.parquet` | `../query_embeddings/combined.jsonl`, `../query_embeddings/combined_mpnet.jsonl` | 1024, 768 | 10, 25, 100 | script-level benchmark runs; query/random modes | default `benchmark_v2.py` methods including two-step RF variants | default `benchmark_results_<timestamp>.json` unless overridden |
| `run_benchmarks.sh` | `python/out/1_2M_random_out_mpnet.parquet` | `python/query_embeddings/combined.jsonl` for query mode | inferred by executable/file | default or `-k 1000` for random runs | executable defaults | C++ executable methods | stdout / executable behavior |
| `run_benchmarks_mxbai.sh` | `python/out/1_2M_random_out_mixedbread.parquet` | `python/query_embeddings/combined.jsonl` for query mode | inferred by executable/file | default or `-k 1000` for random runs | executable defaults | C++ executable methods | stdout / executable behavior |

These scripts identify intended paths, but they do not themselves prove row count. The only currently available parquet metadata for the same path names shows 1,000 rows.

## 6. Old Result JSON Summary

The full inventory is written to `results/old_result_file_inventory.csv`.

Summary:

- JSON files inspected: 19.
- Files with dataset path: 0.
- Files with `metadata.num_vectors`: 19.
- `num_vectors` value in every inspected JSON: 1,200,000.
- Embedding dimensions present: 1,024 and 768.
- k values present: 10, 25, 100.
- Query-mode runs generally contain 313 runs.
- Random and random-vector result files generally contain 250 runs.
- Detected methods include `float`, `avx2`, `binary`, `int8`, `float16`, `mf`, `pca2`, `pca4`, `pca8`, `pca16`, `pca32`, `twostep_rf2`, `twostep_rf5`, `twostep_rf10`, `twostep_rf25`, `twostep_rf50`, `ts_mf_rf2`, `ts_mf_rf5`, `ts_mf_rf10`, `ts_mf_rf25`, and `ts_mf_rf50`.

Important interpretation: the old JSONs are stronger than filename labels because `benchmark_v2.py` writes `metadata.num_vectors` from `searcher.get_dimensions()` after loading embeddings. They are still not a substitute for retained parquet metadata because they do not include the dataset path.

## 7. Notebook / Plot-Source Summary

| notebook_path | input_result_json_files | hardcoded labels related to 1.2M / 60K / 1024 / 768 | whether plotted data contains actual N |
|---|---|---|---|
| `python/jupyter/plots_v2.ipynb` | Loads JSONs from `python/jupyter/results/`; explicit references include `benchmark_dim1024_k100_q.json`, `benchmark_dim1024_k100_re.json`, `benchmark_dim1024_k25_q.json`, `benchmark_dim1024_k10_q.json`, `benchmark_dim768_k100_q.json`, `benchmark_dim768_k100_re.json`, `benchmark_results_1733419058.json`. | Contains `1200000`, `60K`, `1024`, and `768` in code/output text. | References `results[result_key]['metadata']['num_vectors']`; plotted data can use JSON metadata N, but notebooks do not inspect parquet metadata. |
| `python/jupyter/plots_v3.ipynb` | Same primary result JSON references as `plots_v2.ipynb`. | Contains `1200000`, `60K`, `1024`, and `768` in code/output text. | References `metadata.num_vectors`; no parquet row-count validation observed. |
| `python/jupyter/plots_v4.ipynb` | References include `benchmark_dim1024_k100_q.json`, `benchmark_dim1024_k100_re`, `benchmark_dim1024_k10_q`, `benchmark_dim1024_k25_q`, `benchmark_dim768_k100_q`, `benchmark_dim768_k100_re`. | Contains `1200000`, `60K`, `1024`, and `768` in code/output text. | References `metadata.num_vectors`; no parquet row-count validation observed. |

Static notebook inspection is consistent with figures being generated from old JSON files, not from direct parquet metadata.

## 8. Git-History Findings

The repository is not shallow (`git rev-parse --is-shallow-repository` returned `false`).

Relevant history exists for:

- `python/create_embeddings.py`
- `python/start_create_embeddings_mixedbread.py`
- `python/start_create_embeddings_mpnet.py`

`git log --all -- python/start_create_embeddings_mixedbread.py python/start_create_embeddings_mpnet.py python/create_embeddings.py python/out` showed commits including:

- `7489c8f new plotting`
- `bee17cf try to run on my intel machine`
- `a5ec1be add python`

`git grep -n "random_rows" $(git rev-list --all)` shows that `python/start_create_embeddings_mixedbread.py` and `python/start_create_embeddings_mpnet.py` contain `random_rows=1000` in each visible commit where those start scripts exist. Older history for the generic `python/create_embeddings.py` only shows optional `random_rows` support, not a hardcoded 1.2M generation command.

`git grep -n "1_2M" $(git rev-list --all)` shows many historical script references to `1_2M...parquet` paths. Those are filename/path references and are weak evidence for row count unless paired with parquet metadata or runtime dimensions.

No tracked `python/out` parquet file was found in git history. The actual large dataset artifact appears untracked or absent.

## 9. Risk Assessment for the Manuscript

Safe claims:

- It is safe to say the current local `python/out/1_2M...parquet` files are 1,000-row files.
- It is safe to say the current start scripts regenerate only 1,000 rows while using misleading `1_2M` filenames.
- It is safe to say old result JSONs record `num_vectors=1200000` and that the benchmark code normally obtains this value from loaded searcher dimensions.

Unsafe claims:

- It is unsafe to claim the current local parquet files contain 1.2M vectors.
- It is unsafe to claim the original paper definitely used only 1,000 vectors; the old JSON and partition artifacts contradict that.
- It is unsafe to claim the original paper definitely used the current `python/out/1_2M...parquet` files as they exist now.
- It is unsafe to infer row count from filenames alone.

Experiments that must be rerun:

- Any revision experiment that depends on dataset-size claims should be rerun on a parquet file whose row count is verified immediately before the run.
- Scaling claims should be rerun with recorded parquet metadata and saved dataset path.
- If the manuscript needs to defend exact 1.2M-vector results, rerun or recover the actual 1.2M parquet and save metadata alongside results.

## 10. Recommended Next Action

For a minor revision focused on timing breakdowns, **N=60K** can be enough to demonstrate the Step-1 / Step-2 timing structure if the manuscript states the exact N and does not present it as a 1.2M replication.

For scaling claims, use at least **60K and 300K**, and preferably **1.2M** if memory/runtime allows. A three-point scaling plot with verified `N=60K`, `N=300K`, and `N=1.2M` is much stronger than using a 1,000-row file with a misleading name.

The minimum defensible next step is:

1. Verify parquet metadata immediately before each experiment.
2. Save dataset path, row count, dimension, file size, and git commit into each raw result file.
3. Use the largest verified local dataset available, but label it by actual row count, not filename.

Additional artifact needed to resolve the original-paper question completely:

- The original parquet file used for the old JSON runs, or
- A contemporaneous log that records both dataset path and loaded row count, or
- A reproducible script/config that generated the old result JSONs and records the exact input file metadata.
