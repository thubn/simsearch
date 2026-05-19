#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _embedding_columns(schema) -> list[str]:
    cols = [name for name in schema.names if name.startswith("embedding_")]
    return sorted(cols, key=lambda name: int(name.rsplit("_", 1)[1]))


def inspect_output(path: Path) -> dict:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    embedding_cols = _embedding_columns(schema)
    return {
        "parquet_num_rows_verified": parquet_file.metadata.num_rows,
        "parquet_embedding_dim_verified": len(embedding_cols),
        "has_formatted_text": "formatted_text" in schema.names,
        "file_size_bytes": path.stat().st_size,
    }


def iter_hf_rows(args):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing generation dependency 'datasets'. Install dependencies with: "
            ".venv/bin/python -m pip install -r python/requirements.txt"
        ) from exc

    if args.selection_mode == "streaming_prefix":
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            streaming=True,
        )
    elif args.selection_mode == "streaming_shuffle":
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            streaming=True,
        ).shuffle(seed=args.random_seed, buffer_size=args.shuffle_buffer_size)
    else:
        raise ValueError(f"Unsupported selection mode: {args.selection_mode}")

    count = 0
    buffer = []
    for row in dataset:
        title = row.get(args.title_column)
        text = row.get(args.text_column)
        if title is None or text is None:
            continue
        buffer.append((str(title), str(text)))
        count += 1
        if len(buffer) >= args.chunk_size:
            yield buffer
            buffer = []
        if count >= args.target_rows:
            break
    if buffer:
        yield buffer


def write_embeddings(args) -> int:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from create_embeddings import ParquetEmbeddingGenerator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing generation dependency. Install dependencies with: "
            ".venv/bin/python -m pip install -r python/requirements.txt"
        ) from exc

    if "1_2M" in os.path.basename(args.output_path) and args.target_rows != 1_200_000:
        raise ValueError("Refusing to write a 1_2M filename unless --target-rows is 1200000")

    output_path = Path(args.output_path)
    metadata_path = Path(args.metadata_path or f"{args.output_path}.metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    generator = ParquetEmbeddingGenerator(
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )

    writer = None
    actual_rows = 0
    try:
        for chunk in iter_hf_rows(args):
            formatted_texts = [generator.format_text(title, text) for title, text in chunk]
            embeddings = generator.generate_embeddings(formatted_texts)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.shape[1] != args.embedding_dim:
                raise ValueError(
                    f"model produced dimension {embeddings.shape[1]}, expected {args.embedding_dim}"
                )

            result_data = {"formatted_text": formatted_texts}
            for i in range(args.embedding_dim):
                result_data[f"embedding_{i}"] = embeddings[:, i]

            table = pa.Table.from_pandas(pd.DataFrame(result_data), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            actual_rows += len(formatted_texts)
            print(f"processed_rows={actual_rows}", flush=True)

            del formatted_texts, embeddings, result_data, table
    finally:
        if writer is not None:
            writer.close()

    if actual_rows != args.target_rows:
        raise RuntimeError(f"generated {actual_rows} rows, expected {args.target_rows}")

    verified = inspect_output(output_path)
    if verified["parquet_num_rows_verified"] != args.target_rows:
        raise RuntimeError(
            f"verified {verified['parquet_num_rows_verified']} rows, expected {args.target_rows}"
        )
    if verified["parquet_embedding_dim_verified"] != args.embedding_dim:
        raise RuntimeError(
            f"verified dim {verified['parquet_embedding_dim_verified']}, expected {args.embedding_dim}"
        )
    if not verified["has_formatted_text"]:
        raise RuntimeError("formatted_text column missing from generated parquet")

    metadata = {
        "output_path": str(output_path),
        "actual_rows": actual_rows,
        "target_rows": args.target_rows,
        "embedding_dim": args.embedding_dim,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "selection_mode": args.selection_mode,
        "random_seed": args.random_seed,
        "shuffle_buffer_size": args.shuffle_buffer_size if args.selection_mode == "streaming_shuffle" else None,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script": "python/generate_document_embeddings.py",
        **verified,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"output_path={output_path}")
    print(f"metadata_path={metadata_path}")
    print(f"verified_rows={verified['parquet_num_rows_verified']}")
    print(f"verified_embedding_dim={verified['parquet_embedding_dim_verified']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate document embedding parquet files incrementally")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", default="20231101.en")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--target-rows", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--selection-mode", choices=["streaming_prefix", "streaming_shuffle"], default="streaming_prefix")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metadata-path")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return write_embeddings(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
