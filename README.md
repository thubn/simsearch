# simsearch

SIMD-optimized two-step vector similarity search for resource-constrained systems.
Companion code for the TECS paper *"Efficient Two-Step Vector Search with Quantization for Embedded Systems"*.

The core idea: a fast binary pre-filter (XOR + popcount over 1-bit sketches) narrows the candidate set, then a precise float rescore picks the final top-k. This achieves >90× speedup over brute-force float32 scan at ~98% NDCG@100.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Linux | x86-64 with AVX2, or ARM with NEON |
| CMake | ≥ 3.15 |
| C++ compiler | GCC or Clang with C++23 support |
| OpenMP | any recent version |
| Python | ≥ 3.10 |

---

## 1. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

---

## 2. Build the C++ benchmark library

The build requires knowing where PyArrow's shared libraries live (used for Parquet I/O). Export `PYARROW_DIR` from your virtual environment, then configure and build:

```bash
source .venv/bin/activate  # if not already active

PYARROW_DIR=$(python3 -c "import pyarrow, os; print(os.path.dirname(pyarrow.__file__))")

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYARROW_DIR="$PYARROW_DIR"
cmake --build . -j$(nproc)
cd ..
```

This produces two artifacts in `build/`:
- `simsearch` — standalone CLI tool
- `embedding_search_benchmark.so` — Python-callable benchmark module


---

## 3. Generate the datasets

The paper uses two corpora, both embedded with [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (1024-dim).

### Wikipedia (primary dataset — Tables 1–3, all main figures)

```bash
TARGET_ROWS=1200000 bash scripts/generate_mxbai_dataset.sh
```

Output: `python/out/wiki_mxbai_1024_N1200000_seed42.parquet` (~10 GB download, ≈1.2 M vectors).

### arXiv abstracts (cross-corpus generalization experiment)

Used to show that the two-step method generalises beyond Wikipedia. Dataset: [`gfissore/arxiv-abstracts-2021`](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) on HuggingFace.

```bash
source .venv/bin/activate
python3 python/generate_document_embeddings.py \
  --model-name mixedbread-ai/mxbai-embed-large-v1 \
  --dataset-name gfissore/arxiv-abstracts-2021 \
  --dataset-config default \
  --dataset-split train \
  --target-rows 1200000 \
  --embedding-dim 1024 \
  --text-column abstract \
  --title-column title \
  --output-path python/out/arxiv_mxbai_1024_N1200000_seed42.parquet \
  --metadata-path python/out/arxiv_mxbai_1024_N1200000_seed42.metadata.json
```

> To use the mpnet (768-dim) variant instead, run `scripts/generate_mpnet_dataset.sh` with the same `TARGET_ROWS` variable.

---

## 4. Reproduce Table 1 (per-component timing breakdown)

Table 1 reports per-component query latency and speedup for six methods on the Wikipedia dataset at N=60 000.

**Prerequisites:** complete steps 1–3, then set up the Python path so the benchmark module is importable:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/build:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(python3 -c 'import pyarrow, os; print(os.path.dirname(pyarrow.__file__))'):${LD_LIBRARY_PATH:-}"
```

**Run the benchmark:**

```bash
mkdir -p results

python3 python/benchmark/benchmark.py \
  -f python/out/wiki_mxbai_1024_N1200000_seed42.parquet \
  -m query \
  -q python/query_embeddings/combined.jsonl \
  -k 100 --embedding-dim 1024 \
  --component-csv \
  --methods float32_avx2,binary,two_step_RF10,two_step_RF50,two_step_mf_RF10,two_step_mf_RF50 \
  --max-vectors 60000 \
  --repeats 10 \
  --csv-output results/priority1_timing_breakdown.csv
```

Key flags:
- `--max-vectors 60000` — limits the index to the first 60 k vectors, matching the N used in Table 1
- `--repeats 10` — number of timing repeats per query (paper uses 10)
- `--component-csv` — enables per-component timing output (Step 1 / Step 2 breakdown)

Output: `results/priority1_timing_breakdown.csv` with columns `method`, `T_binary_scan_ms`, `T_rescore_ms`, `T_total_ms`, `ndcg`, etc.

### FAISS baseline (Table 1, last row)

```bash
pip install faiss-cpu
python3 python/benchmark/faiss_indexflat_benchmark.py
```

This runs FAISS `IndexFlatIP` (exact brute-force) on the same workload and writes results to `python/out/`.
