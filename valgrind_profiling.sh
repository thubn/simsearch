#!/bin/bash
cd build
valgrind --tool=callgrind --instr-atstart=no --collectat-start=no --cache-sim=yes --dump-instr=yes --collect-jumps=yes ./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -r 1