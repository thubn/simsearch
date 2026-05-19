# Dataset Generation Update

Date: 2026-05-19

## 1. Files Modified / Added

Modified:

- `python/create_embeddings.py`
- `python/start_create_embeddings_mixedbread.py`
- `python/start_create_embeddings_mpnet.py`

Added:

- `python/generate_document_embeddings.py`
- `scripts/inspect_parquet_dataset.py`
- `scripts/generate_mxbai_dataset.sh`
- `scripts/generate_mpnet_dataset.sh`

## 2. New CLI Usage

Mxbai 1024-d example:

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

Mpnet wrapper:

```bash
TARGET_ROWS=60000 scripts/generate_mpnet_dataset.sh
```

Default wrapper behavior is safe:

```bash
TARGET_ROWS=1000
```

## 3. Incremental vs In-Memory

The new generation path is incremental:

- It streams the HuggingFace dataset.
- It batches rows into chunks.
- It encodes each chunk.
- It writes each chunk with `pyarrow.parquet.ParquetWriter`.
- It does not keep all embeddings in memory.

`python/create_embeddings.py` was also fixed to close its `ParquetWriter` in a `finally` block.

## 4. Smoke-Test Command Run

Command attempted:

```bash
TARGET_ROWS=1000 scripts/generate_mxbai_dataset.sh
```

Log:

```text
results/smoke_tests/generate_mxbai_N1000_smoke.log
```

Result: **blocked before generation**.

The active `.venv` is missing required generation dependencies:

- `datasets`
- `sentence-transformers`
- `torch`

The wrapper failed before downloading dataset rows or encoding embeddings. No large generation was run.

## 5. Smoke-Test Output Parquet Path

Intended path:

```text
python/out/wiki_mxbai_1024_N1000_seed42.parquet
```

This file was not produced because dependencies are missing.

## 6. Verified Row Count and Embedding Dimension

No newly generated smoke parquet could be verified.

Existing local file inspection still verifies the problem:

```bash
.venv/bin/python scripts/inspect_parquet_dataset.py --json --warn-filename \
  python/out/1_2M_random_out_mixedbread.parquet
```

Observed:

- rows: 1,000
- embedding dimension: 1,024
- warning: filename contains `1_2M` but parquet has only 1,000 rows

## 7. Estimated Command for N=60000

```bash
TARGET_ROWS=60000 scripts/generate_mxbai_dataset.sh
```

Expected output:

```text
python/out/wiki_mxbai_1024_N60000_seed42.parquet
python/out/wiki_mxbai_1024_N60000_seed42.metadata.json
```

## 8. Estimated Command for N=300000

```bash
TARGET_ROWS=300000 scripts/generate_mxbai_dataset.sh
```

Expected output:

```text
python/out/wiki_mxbai_1024_N300000_seed42.parquet
python/out/wiki_mxbai_1024_N300000_seed42.metadata.json
```

## 9. Estimated Command for N=1200000

```bash
TARGET_ROWS=1200000 scripts/generate_mxbai_dataset.sh
```

Expected output:

```text
python/out/wiki_mxbai_1024_N1200000_seed42.parquet
python/out/wiki_mxbai_1024_N1200000_seed42.metadata.json
```

## 10. Risks and Runtime / Storage

Risks:

- Generation cannot run until `python/requirements.txt` dependencies are installed.
- CPU-only embedding generation may be slow.
- 1.2M x 1024 float32 embeddings require about 4.58 GiB for raw embedding values before text/parquet overhead.
- The script verifies exact row count and fails if the generated parquet does not match `TARGET_ROWS`.

Install dependencies before retrying generation:

```bash
.venv/bin/python -m pip install -r python/requirements.txt
```

Then rerun the 1,000-row smoke test before any large generation:

```bash
TARGET_ROWS=1000 scripts/generate_mxbai_dataset.sh
```
