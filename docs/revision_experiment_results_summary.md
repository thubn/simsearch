# Revision Experiment Results Summary

Date: 2026-05-19

## Status

Current result status: **Instrumentation smoke test on N=1000**.

These numbers must not be used in the manuscript. They only verify that the code builds, scripts run, timing fields are populated, and basic sanity checks pass.

The local files named `1_2M_random_out_*.parquet` are not 1.2M-vector datasets. PyArrow metadata shows they contain 1,000 rows. The previous Priority 1 and Priority 2 outputs were therefore moved out of manuscript-intended filenames.

## Moved Smoke-Test Artifacts

Moved to:

- `results/smoke_tests/priority1_timing_breakdown_smoke.csv`
- `results/smoke_tests/priority1_timing_summary_smoke.csv`
- `results/smoke_tests/priority1_sanity_raw_smoke.csv`
- `results/smoke_tests/priority1_sanity_check_smoke.txt`
- `results/smoke_tests/priority2_scaling_raw_smoke.csv`
- `results/smoke_tests/priority2_scaling_summary_smoke.csv`
- `results/smoke_tests/priority2_scaling_verification_smoke.md`
- `figures/smoke_tests/revision_priority1_timing_breakdown_smoke.pdf`
- `figures/smoke_tests/revision_priority2_scaling_smoke.pdf`

Canonical manuscript-result paths should remain unused until a verified larger parquet exists.

## Dataset

Current local mxbai document parquet:

```text
python/out/1_2M_random_out_mixedbread.parquet
```

Actual metadata:

- rows: 1,000
- embedding dimension: 1,024
- formatted text: present

This dataset is sufficient for instrumentation smoke testing only.

## Output Equivalence

Output-equivalence check:

- `results/output_equivalence_check.md`
- `results/output_equivalence_check.csv`

Status: **PASS**.

Compared old-style and timed paths for:

- `float32_avx2`
- `binary`
- `two_step_RF10`
- `two_step_mf_RF10`

Settings:

- N = 1,000
- query_limit = 5
- k = 100
- RF = 10

All compared methods produced the same top-k ID list or set.

## Generation Smoke Test

Command attempted:

```bash
TARGET_ROWS=1000 scripts/generate_mxbai_dataset.sh
```

Status: blocked before generation because the active `.venv` lacks:

- `datasets`
- `sentence-transformers`
- `torch`

No 60K, 300K, or 1.2M generation was run.

## Safety Changes

Implemented:

- Parquet row-count/dimension inspector.
- Safe incremental dataset generator with metadata sidecar.
- Dataset-generation wrappers with safe `TARGET_ROWS=1000` default.
- Refusal to write a `1_2M` filename unless `target_rows == 1200000`.
- Priority scripts inspect row count before running.
- Priority scripts warn when filename suggests 1.2M but metadata does not.
- `N < k` fails clearly.
- `N <= k*RF` warns that RF comparison is saturated.
- Priority scripts route small-N outputs to `results/smoke_tests/`.
- Future revision CSVs include `T_component_sum_ms` and `T_unaccounted_ms`.

## Dataset Sufficiency

Current dataset size is not sufficient for final experiments.

- N=1000: smoke only.
- N>=10000: minimal meaningful RF50 validation.
- N>=60000: manuscript-minimum Priority 1.
- N=60000, 300000, and one larger size: manuscript-minimum Priority 2.
- N=60000, 300000, 1200000: preferred Priority 2.

## Next Commands

Install generation dependencies:

```bash
.venv/bin/python -m pip install -r python/requirements.txt
```

Retry the smoke generation:

```bash
TARGET_ROWS=1000 scripts/generate_mxbai_dataset.sh
```

Generate manuscript-minimum mxbai N=60000 after smoke passes:

```bash
TARGET_ROWS=60000 scripts/generate_mxbai_dataset.sh
```

Then run Priority 1 on a verified N>=60000 parquet:

```bash
DATASET="$PWD/python/out/wiki_mxbai_1024_N60000_seed42.parquet" \
QUERIES="$PWD/python/query_embeddings/combined.jsonl" \
DIM=1024 K=100 REPEATS=10 FULL_N=60000 \
scripts/run_priority1_timing_breakdown.sh
```

Do not run Priority 2 as manuscript-scale until at least N=60000, N=300000, and one larger verified dataset are available.
