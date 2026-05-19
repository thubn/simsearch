#!/bin/bash

# run benchark with queries from embeddings themselves
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -k 1000 -s 10
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -k 1000 -s 25

./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -q python/query_embeddings/combined.jsonl -s 2
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -q python/query_embeddings/combined.jsonl -s 5
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -q python/query_embeddings/combined.jsonl -s 10
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -q python/query_embeddings/combined.jsonl -s 25
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -q python/query_embeddings/combined.jsonl -s 50
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -q python/query_embeddings/combined.jsonl -s 100
./build/simsearch -f python/out/1_2M_random_out_mixedbread.parquet -q python/query_embeddings/combined.jsonl -s 1000