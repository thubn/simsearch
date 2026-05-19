# Resume TODO: Dataset Generation and Revision Runs

Created: 2026-05-19
Updated: 2026-05-19 (all steps completed)

## Status

All datasets generated on rangel (RTX 3090) and copied locally; Priority 1 and Priority 2 timing runs completed on this host (meow1). Canonical outputs in `results/`:

- `results/priority1_sanity_raw.csv`, `priority1_sanity_check.txt` (status=PASS, 24 rows)
- `results/priority1_timing_breakdown.csv`, `priority1_timing_summary.csv` (60k, k=100, 313 queries, 10 repeats, 6 methods)
- `results/priority2_scaling_raw.csv`, `priority2_scaling_summary.csv`, `priority2_scaling_verification.md` (60k/300k/1.2M, k=100, 100 queries, 10 repeats, 4 methods)

Datasets (all float32, normalized, formatted_text column):

- `python/out/wiki_mxbai_1024_N60000_seed42.parquet` (397 MB)
- `python/out/wiki_mxbai_1024_N300000_seed42.parquet` (1.97 GB)
- `python/out/wiki_mxbai_1024_N1200000_seed42.parquet` (7.66 GB)

### Bug fixed during this work

`python/generate_document_embeddings.py` originally wrote float16 embeddings because sentence-transformers 5.x with mxbai-embed-large-v1 returns float16 from `encode()`. The Rust SIMD benchmark reads embedding bytes as float32, so float16 input crashed Priority 1 sanity (`Value falls between partitions`, partition values ~3e38). Fix: explicit `np.asarray(embeddings, dtype=np.float32)` cast before writing. `scripts/inspect_parquet_dataset.py` now reports `embedding_dtype` and accepts `--expect-dtype float` to catch this regression earlier.

## Current State (original)

Working directory:

```bash
/home/yunchih/code/simsearch
```

Branch:

```bash
tecs-revision-minimal-experiments
```

User approved running:

1. Python generation dependency installation.
2. N=1000 mxbai smoke generation.
3. N=60000 mxbai generation.
4. N=300000 mxbai generation.
5. N=1200000 mxbai generation.
6. Priority 1 timing run on verified N=60000.

Do not run Priority 2 until the required datasets exist and the current Priority 1 run is complete.

## Completed

Dependency install completed:

```bash
.venv/bin/python -m pip install -r python/requirements.txt
```

Installed/import-checked:

```text
torch 2.12.0+cu130
datasets 4.8.5
sentence-transformers 5.5.0
```

CUDA check:

```text
torch.cuda.is_available() == False
```

Generation will run on CPU unless the environment changes.

Disk check:

```text
/home has about 1.5T free at the time this file was written.
```

## In Progress / Next Step

Run the N=1000 smoke generation:

```bash
TARGET_ROWS=1000 scripts/generate_mxbai_dataset.sh 2>&1 | tee results/smoke_tests/generate_mxbai_N1000_smoke.log
```

Expected outputs:

```text
python/out/wiki_mxbai_1024_N1000_seed42.parquet
python/out/wiki_mxbai_1024_N1000_seed42.metadata.json
```

Verify:

```bash
.venv/bin/python scripts/inspect_parquet_dataset.py \
  --expect-rows 1000 \
  --expect-dim 1024 \
  --require-formatted-text \
  python/out/wiki_mxbai_1024_N1000_seed42.parquet
```

## Large Dataset Generation Commands

Run these only after the N=1000 smoke generation succeeds.

### N=60000

```bash
TARGET_ROWS=60000 scripts/generate_mxbai_dataset.sh 2>&1 | tee results/smoke_tests/generate_mxbai_N60000.log
```

Expected outputs:

```text
python/out/wiki_mxbai_1024_N60000_seed42.parquet
python/out/wiki_mxbai_1024_N60000_seed42.metadata.json
```

Verify:

```bash
.venv/bin/python scripts/inspect_parquet_dataset.py \
  --expect-rows 60000 \
  --expect-dim 1024 \
  --require-formatted-text \
  python/out/wiki_mxbai_1024_N60000_seed42.parquet
```

### N=300000

```bash
TARGET_ROWS=300000 scripts/generate_mxbai_dataset.sh 2>&1 | tee results/smoke_tests/generate_mxbai_N300000.log
```

Expected outputs:

```text
python/out/wiki_mxbai_1024_N300000_seed42.parquet
python/out/wiki_mxbai_1024_N300000_seed42.metadata.json
```

Verify:

```bash
.venv/bin/python scripts/inspect_parquet_dataset.py \
  --expect-rows 300000 \
  --expect-dim 1024 \
  --require-formatted-text \
  python/out/wiki_mxbai_1024_N300000_seed42.parquet
```

### N=1200000

```bash
TARGET_ROWS=1200000 scripts/generate_mxbai_dataset.sh 2>&1 | tee results/smoke_tests/generate_mxbai_N1200000.log
```

Expected outputs:

```text
python/out/wiki_mxbai_1024_N1200000_seed42.parquet
python/out/wiki_mxbai_1024_N1200000_seed42.metadata.json
```

Verify:

```bash
.venv/bin/python scripts/inspect_parquet_dataset.py \
  --expect-rows 1200000 \
  --expect-dim 1024 \
  --require-formatted-text \
  python/out/wiki_mxbai_1024_N1200000_seed42.parquet
```

## Priority 1 Manuscript-Minimum Run

Run after N=60000 verifies:

```bash
DATASET="$PWD/python/out/wiki_mxbai_1024_N60000_seed42.parquet" \
QUERIES="$PWD/python/query_embeddings/combined.jsonl" \
DIM=1024 K=100 REPEATS=10 FULL_N=60000 \
scripts/run_priority1_timing_breakdown.sh 2>&1 | tee results/priority1_N60000_run.log
```

Expected canonical outputs if N>=60000:

```text
results/priority1_sanity_raw.csv
results/priority1_sanity_check.txt
results/priority1_timing_breakdown.csv
results/priority1_timing_summary.csv
```

## Priority 2 Later

Do not run Priority 2 until at least these verified datasets exist:

```text
python/out/wiki_mxbai_1024_N60000_seed42.parquet
python/out/wiki_mxbai_1024_N300000_seed42.parquet
python/out/wiki_mxbai_1024_N1200000_seed42.parquet
```

Current `scripts/run_priority2_scaling.sh` operates against a single parquet plus max-vector prefixes. If the N=1200000 file exists, use it for all sizes:

```bash
DATASET="$PWD/python/out/wiki_mxbai_1024_N1200000_seed42.parquet" \
QUERIES="$PWD/python/query_embeddings/combined.jsonl" \
DIM=1024 K=100 REPEATS=10 QUERY_LIMIT=100 \
SIZES="60000 300000 1200000" \
scripts/run_priority2_scaling.sh 2>&1 | tee results/priority2_N60000_300000_1200000_run.log
```

Expected canonical outputs:

```text
results/priority2_scaling_raw.csv
results/priority2_scaling_summary.csv
results/priority2_scaling_verification.md
```

## Existing Important Files

Safe generation and inspection:

```text
python/generate_document_embeddings.py
scripts/generate_mxbai_dataset.sh
scripts/generate_mpnet_dataset.sh
scripts/inspect_parquet_dataset.py
```

Audit/report files already created:

```text
docs/revision_code_audit.md
docs/implementation_safety_checklist.md
docs/dataset_discovery_and_download_plan.md
docs/revision_experiment_results_summary.md
docs/dataset_generation_update.md
docs/dataset_forensics_report.md
results/output_equivalence_check.md
results/output_equivalence_check.csv
```

Known bad/misleading local files:

```text
python/out/1_2M_random_out_mixedbread.parquet  # actual rows: 1000
python/out/1_2M_random_out_mpnet.parquet       # actual rows: 1000
```

Do not use those for manuscript results.

## If Resuming After Interruption

1. Check whether a generation command is still running:

```bash
ps -ef | rg 'generate_document_embeddings|generate_mxbai_dataset|sentence_transformers|python'
```

2. Inspect generated parquet files:

```bash
for f in python/out/wiki_mxbai_1024_N*_seed42.parquet; do
  .venv/bin/python scripts/inspect_parquet_dataset.py "$f"
done
```

3. Continue from the first missing or failed expected output.

4. Do not overwrite manuscript result CSVs with N<60000 data.
