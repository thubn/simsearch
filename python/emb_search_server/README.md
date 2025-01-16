# Embedding Search WebUI
You need the `embedding_search_benchmark.so` library in this folder or path to run this. You also need to adjust the embedding path here:
```
...
searcher.load(
    "../out/1_2M_random_out.parquet", # adjust path
    embedding_dim=1024,
...
```