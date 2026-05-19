#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def ids(results):
    return [int(row[1]) for row in results]


def score_deltas(left, right):
    n = min(len(left), len(right))
    if n == 0:
        return 0.0
    return max(abs(float(left[i][0]) - float(right[i][0])) for i in range(n))


def load_queries(path, limit):
    queries = []
    with open(path) as f:
        for line in f:
            queries.append(json.loads(line))
            if limit and len(queries) >= limit:
                break
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare old-style and timed search outputs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--max-vectors", type=int, default=1000)
    parser.add_argument("--query-limit", type=int, default=5)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--rf", type=int, default=10)
    parser.add_argument("--csv-output", default="results/output_equivalence_check.csv")
    parser.add_argument("--md-output", default="results/output_equivalence_check.md")
    args = parser.parse_args()

    sys.path.insert(0, str(Path("build").resolve()))
    from embedding_search_benchmark import EmbeddingSearch
    from python.benchmark.benchmark_v2 import calculate_ndcg

    searcher = EmbeddingSearch()
    searcher.load(
        filename=args.dataset,
        embedding_dim=args.embedding_dim,
        init_pca=False,
        init_avx2=True,
        init_binary=True,
        init_int8=False,
        init_float16=False,
        init_mf=True,
        max_vectors=args.max_vectors,
    )
    n, d = searcher.get_dimensions()
    if n < args.k:
        raise SystemExit(f"N < k is invalid: N={n}, k={args.k}")

    methods = [
        ("float32_avx2", lambda q: searcher.search_avx2(q, args.k), lambda q: searcher.search_avx2(q, args.k)),
        ("binary", lambda q: searcher.search_binary(q, args.k), lambda q: searcher.search_binary_timed(q, args.k)[:2]),
        ("two_step_RF10", lambda q: searcher.search_twostep(q, args.k, args.rf), lambda q: searcher.search_twostep_timed(q, args.k, args.rf)[:2]),
        ("two_step_mf_RF10", lambda q: searcher.search_twostep_mf(q, args.k, args.rf), lambda q: searcher.search_twostep_mf_timed(q, args.k, args.rf)[:2]),
    ]

    rows = []
    passed = True
    for qid, query in enumerate(load_queries(args.queries, args.query_limit)):
        vector = np.array(query["embedding"], dtype=np.float32)
        float_results, _ = searcher.search_avx2(vector, args.k)
        for method, old_func, timed_func in methods:
            old_results, _ = old_func(vector)
            timed_results, _ = timed_func(vector)
            old_ids = ids(old_results)
            timed_ids = ids(timed_results)
            exact_list = old_ids == timed_ids
            same_set = set(old_ids) == set(timed_ids)
            max_score_delta = score_deltas(old_results, timed_results)
            ndcg_old = calculate_ndcg(float_results, old_results)
            ndcg_timed = calculate_ndcg(float_results, timed_results)
            status = "PASS" if exact_list or same_set else "FAIL"
            passed = passed and status == "PASS"
            rows.append(
                {
                    "method": method,
                    "query_id": qid,
                    "N": n,
                    "d": d,
                    "k": args.k,
                    "RF": args.rf if "RF" in method else 0,
                    "exact_same_topk_id_list": exact_list,
                    "same_topk_id_set": same_set,
                    "max_score_delta": max_score_delta,
                    "ndcg_old": ndcg_old,
                    "ndcg_timed": ndcg_timed,
                    "status": status,
                }
            )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(args.md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    failures = [r for r in rows if r["status"] != "PASS"]
    with open(md_path, "w") as f:
        f.write("# Output Equivalence Check\n\n")
        f.write(f"- Dataset: `{args.dataset}`\n")
        f.write(f"- Queries: `{args.queries}`\n")
        f.write(f"- N loaded: {n}\n")
        f.write(f"- k: {args.k}\n")
        f.write(f"- query_limit: {args.query_limit}\n")
        f.write(f"- Status: {'PASS' if passed else 'FAIL'}\n\n")
        f.write("Timed and old-style output lists are compared inside the same build.\n\n")
        if failures:
            f.write("## Failures\n\n")
            for row in failures:
                f.write(f"- {row['method']} query {row['query_id']}\n")
        else:
            f.write("All compared methods produced the same top-k ID list or set.\n")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
