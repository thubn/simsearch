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

## 1. Clone and set up Python environment

```bash
git clone https://github.com/thubn/simsearch.git
cd simsearch

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

> **Tip:** For a debug build, replace `Release` with `Debug`. The debug build disables `-march=native` SIMD flags and enables AddressSanitizer-friendly settings.

---

## 3. Generate the dataset

The experiments use Wikipedia embeddings encoded with [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (1024-dim).

### Quick smoke-test dataset (≈1 000 vectors, downloads fast)

```bash
TARGET_ROWS=1000 bash scripts/generate_mxbai_dataset.sh
```

### Full dataset used in the paper (≈1.2 M vectors, ~10 GB download)

```bash
TARGET_ROWS=1200000 bash scripts/generate_mxbai_dataset.sh
```

The output parquet file is written to `python/out/` and the path is printed at the end (`OUTPUT_PATH=...`). The experiment scripts below default to looking for this file.

> To use the mpnet (768-dim) variant instead, run `scripts/generate_mpnet_dataset.sh` with the same `TARGET_ROWS` variable.

---

## 4. Reproduce paper experiments

Both scripts auto-detect the dataset size. If `N < 60 000` (e.g. smoke-test), outputs go to `results/smoke_tests/` and a warning is printed; the canonical `results/` files are not overwritten.

Before running, store the PyArrow directory so the scripts can set `LD_LIBRARY_PATH`:

```bash
source .venv/bin/activate
mkdir -p results/build_info
python3 -c "import pyarrow, os; print(os.path.dirname(pyarrow.__file__))" \
  > results/build_info/pyarrow_dir.txt
```

### Priority 1 — per-component timing breakdown (Table 1 in the paper)

```bash
bash scripts/run_priority1_timing_breakdown.sh
```

Outputs:
- `results/priority1_timing_breakdown.csv` — raw per-query timings
- `results/priority1_timing_summary.csv` — aggregated means

Override defaults via environment variables:
```bash
DATASET=/path/to/your.parquet \
REPEATS=10 \
bash scripts/run_priority1_timing_breakdown.sh
```

### Priority 2 — latency scaling with dataset size (Figure in §7)

```bash
bash scripts/run_priority2_scaling.sh
```

Outputs:
- `results/priority2_scaling_raw.csv`
- `results/priority2_scaling_summary.csv`

By default this sweeps `N ∈ {60 000, 300 000, 1 200 000}`. Override with:
```bash
SIZES="60000 300000" bash scripts/run_priority2_scaling.sh
```

---

## 5. Generate figures

```bash
source .venv/bin/activate
pip install matplotlib seaborn  # if not already installed

python3 scripts/plot_revision_priority1_priority2.py
```

Figures are written to `figures/`:
- `revision_priority1_ablation_n60k.pdf/.png`
- `revision_priority2_scaling.pdf/.png`

---

## Optional: Interactive demo

A small Flask/FastAPI web UI lets you run live queries against a loaded embedding index:

```bash
source .venv/bin/activate
cp build/embedding_search_benchmark.so python/emb_search_server/

# Edit python/emb_search_server/main.py to point to your parquet file, then:
python3 python/emb_search_server/main.py
```

Open `http://localhost:8000` in your browser.

---

## Optional: Memory bandwidth benchmark

A standalone C micro-benchmark measuring the effect of strided vs. sequential memory access (used in §7.3):

```bash
cd memory_benchmark
gcc -O3 memory_benchmark.c -o memory_benchmark
./memory_benchmark
```

Edit `#define ARRAY_SIZE` in `memory_benchmark.c` to change the allocation size (default: 12 GB).

---

## Repository layout

```
simsearch/
├── src/                        C++ source files
├── include/                    C++ headers
├── python/
│   ├── generate_document_embeddings.py   dataset generation
│   ├── benchmark/              Python benchmark driver
│   ├── emb_search_server/      interactive web UI
│   └── query_embeddings/       pre-encoded query vectors
├── scripts/
│   ├── generate_mxbai_dataset.sh
│   ├── generate_mpnet_dataset.sh
│   ├── run_priority1_timing_breakdown.sh
│   ├── run_priority2_scaling.sh
│   ├── plot_revision_priority1_priority2.py
│   └── ...
├── figures/                    output figures (generated)
├── memory_benchmark/           standalone C bandwidth tool
└── CMakeLists.txt
```
