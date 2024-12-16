#!/bin/bash

python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m query -q ../query_embeddings/combined.jsonl -k 100 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m random -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m random-vec -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024

python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m query -q ../query_embeddings/combined_mpnet.jsonl -k 100 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m random -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m random-vec -k 100 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768


python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m query -q ../query_embeddings/combined.jsonl -k 10 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m query -q ../query_embeddings/combined.jsonl -k 25 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024

python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m random -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m random -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024

python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m random-vec -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024
python benchmark_v2.py -f ../out/1_2M_random_out.parquet -m random-vec -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 1024

python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m query -q ../query_embeddings/combined_mpnet.jsonl -k 10 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m query -q ../query_embeddings/combined_mpnet.jsonl -k 25 --rescoring-factor 2,5,10,25,50 --embedding-dim 768

python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m random -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m random -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768

python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m random-vec -k 10 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768
python benchmark_v2.py -f ../out/1_2M_random_out_mpnet.parquet -m random-vec -k 25 -r 250 --rescoring-factor 2,5,10,25,50 --embedding-dim 768

#systemctl poweroff
