#!/bin/bash

# run benchark with queries from embeddings themselves
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -k 1000 -s 10
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -k 1000 -s 25

./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -q ../../create_embeddings/query_embeddings/combined.json -s 2
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -q ../../create_embeddings/query_embeddings/combined.json -s 5
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -q ../../create_embeddings/query_embeddings/combined.json -s 10
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -q ../../create_embeddings/query_embeddings/combined.json -s 25
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -q ../../create_embeddings/query_embeddings/combined.json -s 50
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -q ../../create_embeddings/query_embeddings/combined.json -s 100
./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -q ../../create_embeddings/query_embeddings/combined.json -s 1000