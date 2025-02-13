# Git repo for Bachelors Thesis: Vector Similarity Search Optimization by Thorben Simon
This repository contains all the files used in this thesis.
This repo is availible on:
- <https://github.com/thubn/simsearch>
- <https://git.thubn.de/thubn/simsearch>

## Project Index
- [Main Project](./README.md)
- [python files](./python/README.md) used to create the embeddings, embedding search webserver, benchmarks.
  - [Benchmark](./python/benchmark/README.md) contains the python script for the automated benchmarks.
  - [Embedding Search Server](./python/emb_search_server/README.md) Try out the simsearch program with your own queries!
  - [Jupyter](./python/jupyter/README.md) contains the Jupyter Notebook that generates the plots used in the thesis.
- [memory benchmark](./memory_benchmark/README.md) used to test the effect of strided memory access vs. full sequential access.

The embeddings used are stored in this repo: <https://github.com/thubn/simsearch_embeddings> Alternatively you can create then yourself: [python files](./python/README.md)

## Gettings started with the c++ program "simsearch"
### Install dependencies
#### Fedora
```
dnf install eigen3-devel json-devel python3-devel pybind11-devel pybind11-json-devel libarrow-devel parquet-libs-devel
```

### Compiling

```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
OR (for debugging)
```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

This will `simsearch` executable and the `embedding_search_benchmark.so` library that can be used with the python programs in the [python](./python/) folder.