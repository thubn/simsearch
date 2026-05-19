# Dataset Discovery and Download/Generation Plan

Date: 2026-05-19

## Current Local Parquet Files

Inspected with `scripts/inspect_parquet_dataset.py` / PyArrow metadata.

| path | rows | embedding dim | file size | status |
|---|---:|---:|---:|---|
| `python/out/1_2M_random_out_mixedbread.parquet` | 1,000 | 1,024 | 7,844,010 bytes | Misleading filename; not 1.2M. |
| `python/out/1_2M_random_out_mpnet.parquet` | 1,000 | 768 | 6,350,078 bytes | Misleading filename; not 1.2M. |
| `python/out/local_query_embeddings_1024.parquet` | 313 | 1,024 | 2,803,099 bytes | Query/fallback embedding file, not full document dataset. |

No local parquet currently verifies as 60K, 300K, or 1.2M document vectors.

## Current Query Files

Relevant query files:

| query file | count |
|---|---:|
| `python/query_embeddings/combined.jsonl` | 313 |
| `python/query_embeddings/combined_mpnet.jsonl` | 313 |
| `python/query_embeddings/spec_wiki_queries_emb.jsonl` | 50 |
| `python/query_embeddings/wiki_queries_emb.jsonl` | 101 |
| `python/query_embeddings/specific_queries_emb.jsonl` | 51 |
| `python/query_embeddings/stupid_queries_emb.jsonl` | 111 |

For the revision experiments, use `python/query_embeddings/combined.jsonl` for mxbai 1024-d unless runtime forces `query_limit=100`.

## Cause of the 1,000-row Misleading Files

Previous start scripts used:

```python
output_path="out/1_2M_random_out_mixedbread.parquet"
random_rows=1000
```

and:

```python
output_path="out/1_2M_random_out_mpnet.parquet"
random_rows=1000
```

This is the exact script/config variable that caused the mismatch. The old scripts have been replaced with safe N1000 wrappers that write filenames containing `N1000`, not `1_2M`.

## New Generation Path

New CLI:

```bash
python python/generate_document_embeddings.py \
  --model-name mixedbread-ai/mxbai-embed-large-v1 \
  --dataset-name wikimedia/wikipedia \
  --dataset-config 20231101.en \
  --dataset-split train \
  --target-rows 60000 \
  --embedding-dim 1024 \
  --batch-size 32 \
  --chunk-size 1000 \
  --selection-mode streaming_prefix \
  --random-seed 42 \
  --output-path python/out/wiki_mxbai_1024_N60000_seed42.parquet \
  --metadata-path python/out/wiki_mxbai_1024_N60000_seed42.metadata.json
```

Wrapper:

```bash
TARGET_ROWS=60000 scripts/generate_mxbai_dataset.sh
```

The generator uses HuggingFace streaming mode and writes incrementally with `pyarrow.parquet.ParquetWriter`. It verifies the output after writing and creates a metadata sidecar.

## Recommended Dataset Sizes

- Minimal meaningful RF test: `N >= 10000`, because `k=100` and `RF=50` requests 5,000 survivors.
- Manuscript minimum: `N >= 60000`.
- Preferred scaling set: `N = 60000, 300000, 1200000`.

Decision table:

| available N | allowed interpretation |
|---:|---|
| `< 5000` | Smoke-test instrumentation only. Do not compare RF10 vs RF50. |
| `5000 <= N < 60000` | Small-scale validation only. |
| `>= 60000` | Manuscript-minimum Priority 1. |
| `60000, 300000, and one larger N` | Manuscript-minimum Priority 2. |
| `60000, 300000, 1200000` | Preferred Priority 2. |

## Estimated Storage

Float32 embedding payload only, excluding text/parquet overhead:

| N | 1024-d float32 bytes | approximate |
|---:|---:|---:|
| 60,000 | 245,760,000 | 234 MiB |
| 300,000 | 1,228,800,000 | 1.14 GiB |
| 1,200,000 | 4,915,200,000 | 4.58 GiB |

Parquet file size may differ due to compression and stored `formatted_text`.

## Estimated Runtime

Runtime depends on GPU/CPU availability and model cache state. The script processes in chunks and does not retain all embeddings in memory. It must still encode every document:

- 60K: likely minutes on a capable GPU; longer on CPU.
- 300K: likely tens of minutes to hours depending on hardware.
- 1.2M: likely hours and requires enough disk space for several GiB plus intermediate memory per chunk.

The current `.venv` is missing `datasets`, `sentence-transformers`, and `torch`, so generation cannot run until dependencies are installed.

## Commands for Real Generation

Do not run these until approved.

```bash
TARGET_ROWS=60000 scripts/generate_mxbai_dataset.sh
TARGET_ROWS=300000 scripts/generate_mxbai_dataset.sh
TARGET_ROWS=1200000 scripts/generate_mxbai_dataset.sh
```

Mpnet equivalents:

```bash
TARGET_ROWS=60000 scripts/generate_mpnet_dataset.sh
TARGET_ROWS=300000 scripts/generate_mpnet_dataset.sh
TARGET_ROWS=1200000 scripts/generate_mpnet_dataset.sh
```

## Recommended Approval Command

After dependencies are installed and disk space is confirmed, the next practical command is:

```bash
TARGET_ROWS=60000 scripts/generate_mxbai_dataset.sh
```

Then inspect:

```bash
scripts/inspect_parquet_dataset.py python/out/wiki_mxbai_1024_N60000_seed42.parquet
```

Only after that should Priority 1 be run on the verified N=60000 parquet.
