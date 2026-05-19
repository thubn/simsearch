# Output Equivalence Check

- Dataset: `python/out/1_2M_random_out_mixedbread.parquet`
- Queries: `python/query_embeddings/combined.jsonl`
- N loaded: 1000
- k: 100
- query_limit: 5
- Status: PASS

Timed and old-style output lists are compared inside the same build.

All compared methods produced the same top-k ID list or set.
