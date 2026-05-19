#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq


def inspect_parquet(path: str) -> dict:
    p = Path(path)
    info = {
        "path": str(p),
        "exists": p.exists(),
        "file_size_bytes": p.stat().st_size if p.exists() else 0,
        "num_rows": 0,
        "num_columns": 0,
        "embedding_dim_inferred": 0,
        "has_formatted_text": False,
        "first_embedding_column": "",
        "last_embedding_column": "",
        "embedding_dtype": "",
        "estimated_float32_embedding_bytes": 0,
    }
    if not p.exists():
        return info

    parquet_file = pq.ParquetFile(p)
    schema = parquet_file.schema_arrow
    names = schema.names
    embedding_cols = sorted(
        [name for name in names if name.startswith("embedding_")],
        key=lambda name: int(name.rsplit("_", 1)[1]) if name.rsplit("_", 1)[1].isdigit() else -1,
    )
    embedding_dtype = str(schema.field(embedding_cols[0]).type) if embedding_cols else ""

    info.update(
        {
            "num_rows": parquet_file.metadata.num_rows,
            "num_columns": len(names),
            "embedding_dim_inferred": len(embedding_cols),
            "has_formatted_text": "formatted_text" in names,
            "first_embedding_column": embedding_cols[0] if embedding_cols else "",
            "last_embedding_column": embedding_cols[-1] if embedding_cols else "",
            "embedding_dtype": embedding_dtype,
            "estimated_float32_embedding_bytes": parquet_file.metadata.num_rows
            * len(embedding_cols)
            * 4,
        }
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect parquet embedding dataset metadata")
    parser.add_argument("path", help="Parquet file path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    parser.add_argument("--json-output", help="Optional path to write JSON metadata")
    parser.add_argument("--expect-rows", type=int, default=None, help="Fail unless row count matches")
    parser.add_argument("--expect-dim", type=int, default=None, help="Fail unless inferred embedding dim matches")
    parser.add_argument("--require-formatted-text", action="store_true", help="Fail unless formatted_text exists")
    parser.add_argument("--expect-dtype", default=None, help="Fail unless embedding dtype matches (e.g. 'float')")
    parser.add_argument("--warn-filename", action="store_true", help="Warn if filename suggests a larger dataset than metadata")
    args = parser.parse_args()

    info = inspect_parquet(args.path)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(info, f, indent=2)

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key in [
            "path",
            "exists",
            "file_size_bytes",
            "num_rows",
            "num_columns",
            "embedding_dim_inferred",
            "has_formatted_text",
            "first_embedding_column",
            "last_embedding_column",
            "embedding_dtype",
            "estimated_float32_embedding_bytes",
        ]:
            print(f"{key}: {info[key]}")

    errors = []
    if not info["exists"]:
        errors.append(f"missing parquet file: {args.path}")
    if args.expect_rows is not None and info["num_rows"] != args.expect_rows:
        errors.append(f"expected {args.expect_rows} rows, found {info['num_rows']}")
    if args.expect_dim is not None and info["embedding_dim_inferred"] != args.expect_dim:
        errors.append(
            f"expected embedding dim {args.expect_dim}, found {info['embedding_dim_inferred']}"
        )
    if args.require_formatted_text and not info["has_formatted_text"]:
        errors.append("formatted_text column is missing")
    if args.expect_dtype is not None and info["embedding_dtype"] != args.expect_dtype:
        errors.append(
            f"expected embedding dtype {args.expect_dtype}, found {info['embedding_dtype']}"
        )

    if args.warn_filename and "1_2M" in os.path.basename(args.path) and info["num_rows"] < 1_200_000:
        print(
            f"WARNING: filename contains 1_2M but parquet has only {info['num_rows']} rows",
            file=sys.stderr,
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
