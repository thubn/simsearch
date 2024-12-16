# Initialize the embedding generator
from create_embeddings import ParquetEmbeddingGenerator


generator = ParquetEmbeddingGenerator(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    #model_name="sentence-transformers/all-mpnet-base-v2",
    batch_size=32
)

df_with_embeddings = generator.process_parquet_file(
    file_path="wikimedia/wikipedia",
    chunk_size=1000,
    dataset_config="20231101.en",
    dataset_split="train",
    output_path="out/1_2M_random_out_mpnet.parquet",
    random_rows=1_200_000,
    random_seed=42
)