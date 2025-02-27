python benchmark_v2.py -f ../../embeddings/60k_random_out.parquet -m query -q ../../embeddings/query_embeddings/combined.jsonl -k 25 --embedding-dim 1024 --rescoring-factor=10,25 --measure-power --power-duration 30
sleep 10
python benchmark_v2.py -f ../../embeddings/60k_random_out.parquet -m random -r 1000 -k 25 --embedding-dim 1024 --rescoring-factor=10,25 --measure-power --power-duration 30
