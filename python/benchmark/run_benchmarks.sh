#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/build:${PYTHONPATH:-}"

MX_EMBED="../out/1_2M_random_out_mixedbread.parquet"
MX_QUERIES="../query_embeddings/combined.jsonl"
MPNET_EMBED="../out/1_2M_random_out_mpnet.parquet"
MPNET_QUERIES="../query_embeddings/combined_mpnet.jsonl"

cd "$SCRIPT_DIR"

# Only run what plots_v2.ipynb consumes:
# - 1024-d mixedbread: k=100 query & random_embeddings, plus k=10/25 query
# - 768-d mpnet: k=100 query & random_embeddings

# mxbai / mixedbread (dim 1024)
python benchmark.py -f "$MX_EMBED" -m query -q "$MX_QUERIES" -k 100 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
python benchmark.py -f "$MX_EMBED" -m random -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
# python benchmark.py -f "$MX_EMBED" -m random-vec -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024

# mpnet (dim 768)
python benchmark.py -f "$MPNET_EMBED" -m query -q "$MPNET_QUERIES" -k 100 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
python benchmark.py -f "$MPNET_EMBED" -m random -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
# python benchmark.py -f "$MPNET_EMBED" -m random-vec -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768

# mxbai / mixedbread (dim 1024) for smaller k
python benchmark.py -f "$MX_EMBED" -m query -q "$MX_QUERIES" -k 10 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
python benchmark.py -f "$MX_EMBED" -m query -q "$MX_QUERIES" -k 25 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
# python benchmark.py -f "$MX_EMBED" -m random -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
# python benchmark.py -f "$MX_EMBED" -m random -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
# python benchmark.py -f "$MX_EMBED" -m random-vec -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
# python benchmark.py -f "$MX_EMBED" -m random-vec -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024

# mpnet (dim 768) for smaller k (not used in plots, so left commented)
# python benchmark.py -f "$MPNET_EMBED" -m query -q "$MPNET_QUERIES" -k 10 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
# python benchmark.py -f "$MPNET_EMBED" -m query -q "$MPNET_QUERIES" -k 25 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
# python benchmark.py -f "$MPNET_EMBED" -m random -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
# python benchmark.py -f "$MPNET_EMBED" -m random -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
# python benchmark.py -f "$MPNET_EMBED" -m random-vec -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
# python benchmark.py -f "$MPNET_EMBED" -m random-vec -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768

#systemctl poweroff
