#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

TARGET_ROWS="${TARGET_ROWS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
SEED="${SEED:-42}"
OUT_DIR="$ROOT_DIR/python/out"
OUTPUT="$OUT_DIR/wiki_mxbai_1024_N${TARGET_ROWS}_seed${SEED}.parquet"
METADATA="$OUT_DIR/wiki_mxbai_1024_N${TARGET_ROWS}_seed${SEED}.metadata.json"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" python/generate_document_embeddings.py \
  --model-name mixedbread-ai/mxbai-embed-large-v1 \
  --dataset-name wikimedia/wikipedia \
  --dataset-config 20231101.en \
  --dataset-split train \
  --target-rows "$TARGET_ROWS" \
  --embedding-dim 1024 \
  --batch-size "$BATCH_SIZE" \
  --chunk-size "$CHUNK_SIZE" \
  --selection-mode streaming_prefix \
  --random-seed "$SEED" \
  --output-path "$OUTPUT" \
  --metadata-path "$METADATA"

"$PYTHON_BIN" scripts/inspect_parquet_dataset.py \
  --expect-rows "$TARGET_ROWS" \
  --expect-dim 1024 \
  --require-formatted-text \
  "$OUTPUT"

echo "OUTPUT_PATH=$OUTPUT"
echo "METADATA_PATH=$METADATA"
