# Dataset Discovery and Download Plan

## Local discovery

No document embedding parquet, safetensors, or ndjson files were found under the
repository checkout during inspection. The repository does contain query
embeddings under `python/query_embeddings/`.

Useful local query files:

| File | Valid queries | Dimension |
| --- | ---: | ---: |
| `python/query_embeddings/combined.jsonl` | 313 | 1024 |
| `python/query_embeddings/combined_mpnet.jsonl` | 313 | 768 |
| `python/query_embeddings/3_queries_emb.jsonl` | 3 | 1024 |
| `python/query_embeddings/3_queries_emb_mpnet.jsonl` | 3 | 768 |
| `python/query_embeddings/spec_wiki_queries_emb.jsonl` | 50 | 1024 |
| `python/query_embeddings/wiki_queries_emb.jsonl` | 101 | 1024 |

`python/query_embeddings/query_example.jsonl` is illustrative and is not valid
JSONL because it contains ellipses in the embedding arrays.

## Dataset used by the artifact

- Dataset name: Wikimedia Wikipedia article dataset.
- Hugging Face dataset id: `wikimedia/wikipedia`.
- Dataset config: `20231101.en`.
- Split: `train`.
- Text fields used by the embedding generator: `title` and `text`.
- Formatting template in `python/create_embeddings.py`:
  `title: <title> text: <text>`, truncated to the model-specific max length.
- Primary embedding model for the TECS revision:
  `mixedbread-ai/mxbai-embed-large-v1`.
- Expected dimension: 1024.
- Expected number of vectors for the paper-scale run: 1,200,000.
- Expected local path: `python/out/1_2M_random_out_mixedbread.parquet`.
- Older script/path variant: `python/out/1_2M_random_out.parquet`.
- Comparison model:
  `sentence-transformers/all-mpnet-base-v2`, dimension 768, expected path
  `python/out/1_2M_random_out_mpnet.parquet`.

## Expected embedding format

The C++ loader expects a parquet file with:

- `formatted_text` string column.
- One float column per dimension named `embedding_0`, `embedding_1`, ...,
  `embedding_1023` for mxbai, or `embedding_767` for mpnet.

`EmbeddingIO::load_parquet` loads all rows in the file and reads the embedding
dimension passed by the caller.

## Regeneration scripts

The repository includes:

- `python/create_embeddings.py`
- `python/start_create_embeddings_mixedbread.py`
- `python/start_create_embeddings_mpnet.py`

The mixedbread helper currently contains:

```python
generator = ParquetEmbeddingGenerator(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    batch_size=32
)

generator.process_parquet_file(
    file_path="wikimedia/wikipedia",
    chunk_size=1000,
    dataset_config="20231101.en",
    dataset_split="train",
    output_path="out/1_2M_random_out_mixedbread.parquet",
    random_rows=1000,
    random_seed=42,
    streaming=True
)
```

Important caveat: despite the `1_2M` output filename, the checked-in script uses
`random_rows=1000`. For a 1.2M embedding file this must be changed or replaced
with a command/script that sets `random_rows=1200000` or an equivalent max-row
policy.

## Commands to obtain or regenerate data

Install Python dependencies, preferably in a virtual environment:

```bash
python3 -m pip install -r python/requirements.txt
```

Generate a small smoke-test mxbai parquet:

```bash
cd python
python3 start_create_embeddings_mixedbread.py
```

Generate a full 1.2M mxbai parquet after editing
`python/start_create_embeddings_mixedbread.py` so `random_rows=1200000`:

```bash
cd python
python3 start_create_embeddings_mixedbread.py
```

Alternative: use a temporary local copy of the helper script and set:

- `output_path="out/1_2M_random_out_mixedbread.parquet"`
- `random_rows=1200000`
- `random_seed=42`
- `streaming=True`

The script writes parquet incrementally using pyarrow.

## Expected size and runtime

Rough size estimate for 1.2M 1024-dimensional float embeddings:

- Raw vector payload: `1,200,000 * 1024 * 4` bytes = about 4.9 GB.
- Parquet plus text/metadata overhead and compression can vary substantially.
- Practical disk budget should allow at least 6 to 10 GB for the mxbai parquet.

Runtime depends on GPU/CPU, model cache state, network, and Hugging Face
streaming throughput. Full regeneration of 1.2M embeddings is expected to be a
long-running job and should not be started without approval.

## Fallback plan

1. Prefer an existing local `python/out/1_2M_random_out_mixedbread.parquet` if
   present after user supplies or mounts data.
2. If full 1.2M is unavailable, use the largest local mxbai parquet file found
   and document exact `N`.
3. If no document parquet is available, run only build/script sanity where
   possible and document the dataset blocker.
4. For quick functionality checks, generate or use a small parquet with
   `random_rows=1000` and run:
   - `N=1000`
   - `queries=3`
   - `repeats=2`

## Approval requirement

Do not download or regenerate the full Wikipedia embedding dataset until the
desired output path, target `N`, disk budget, and expected runtime are approved.
