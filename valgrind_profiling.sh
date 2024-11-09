#!/bin/bash
cd build
valgrind --tool=callgrind --instr-atstart=no --collectat-start=no --cache-sim=yes --dump-instr=yes --collect-jumps=yes ./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -r 1
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -r 1
valgrind --tool=massif --heap=yes --stacks=yes --detailed-freq=10 --max-snapshots=100 --time-unit=ms --massif-out-file=massif.out.%p ./simsearch -f ../../create_embeddings/out/1_2M_random_out.parquet -r 1