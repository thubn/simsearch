from generate_document_embeddings import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--model-name",
                "mixedbread-ai/mxbai-embed-large-v1",
                "--dataset-name",
                "wikimedia/wikipedia",
                "--dataset-config",
                "20231101.en",
                "--dataset-split",
                "train",
                "--target-rows",
                "1000",
                "--embedding-dim",
                "1024",
                "--batch-size",
                "32",
                "--chunk-size",
                "1000",
                "--selection-mode",
                "streaming_prefix",
                "--random-seed",
                "42",
                "--output-path",
                "out/wiki_mxbai_1024_N1000_seed42.parquet",
                "--metadata-path",
                "out/wiki_mxbai_1024_N1000_seed42.metadata.json",
            ]
        )
    )
