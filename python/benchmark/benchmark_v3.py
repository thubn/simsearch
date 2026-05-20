"""benchmark_v3.py — like benchmark_v2.py but every per-query method call is
repeated --inner-repeats times (default 10), each timing kept individually.

The on-disk JSON shape is a strict superset of v2's:

  runs[i].searches[m].metrics.time_us         # per-repeat MEAN (back-compat)
  runs[i].searches[m].metrics.time_us_repeats # NEW: list[float], length = inner_repeats

  summary.<method>.time_us.{mean,std,min,max,median}
                            ^ now over ALL (query * inner_repeats) measurements
  summary.<method>.time_us.geomean    # NEW
  summary.<method>.inner_repeats      # NEW (echoes args.inner_repeats)

NDCG / jaccard / overlap depend only on the search result set, not on timing,
so they are computed once per (query, method) — not once per repeat.

Only --revision-csv mode is left untouched; this v3 only changes the
legacy-style query-mode JSON output that the plots_v4.ipynb consumes.
"""

import argparse
import csv
import numpy as np
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from math import log2, exp
from embedding_search_benchmark import EmbeddingSearch

def calculate_ndcg(ground_truth: List[Tuple[float, int, str]],
                  prediction: List[Tuple[float, int, str]]) -> float:
    if not ground_truth or not prediction:
        return 0.0
    k = min(len(ground_truth), len(prediction))
    truth_positions = {idx: pos for pos, (_, idx, _) in enumerate(ground_truth[:k])}
    dcg = 0.0
    for i in range(k):
        pred_idx = prediction[i][1]
        if pred_idx in truth_positions:
            position_diff = abs(float(truth_positions[pred_idx] - i))
            relevance = exp(-position_diff / k)
            dcg += relevance / log2(i + 2)
    idcg = sum(1.0 / log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def _geomean(values: List[float]) -> float:
    if not values:
        return 0.0
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return float(np.exp(np.log(clipped).mean()))


class VectorSearchBenchmarkV3:
    def __init__(
        self,
        embedding_file: str,
        k: int = 25,
        runs: int = 100,
        rescoring_factors: List[int] = None,
        embedding_dim: int = 1024,
        max_vectors: int = 0,
        inner_repeats: int = 10,
        init_pca: bool = True,
        init_int8: bool = True,
        init_float16: bool = True,
        init_mf: bool = True,
    ):
        self.searcher = EmbeddingSearch()
        self.searcher.load(
            filename=embedding_file,
            embedding_dim=embedding_dim,
            init_pca=init_pca,
            init_avx2=True,
            init_binary=True,
            init_int8=init_int8,
            init_float16=init_float16,
            init_mf=init_mf,
            max_vectors=max_vectors,
        )
        self.k = k
        self.runs = runs
        self.inner_repeats = max(1, int(inner_repeats))
        self.num_vectors, self.vector_dim = self.searcher.get_dimensions()

        self.rescoring_factors = rescoring_factors or []
        if self.num_vectors < self.k:
            raise ValueError(
                f"N < k is invalid for benchmark search: N={self.num_vectors}, k={self.k}"
            )
        if self.rescoring_factors:
            saturated = [rf for rf in self.rescoring_factors if self.k * rf >= self.num_vectors]
            if saturated:
                print(
                    "WARNING: candidate set saturates because N <= k*RF for "
                    f"RF={saturated}. RF comparison is not meaningful."
                )

        print(f"Loaded {self.num_vectors} vectors of dimension {self.vector_dim}")
        print(f"Inner repeats per (query, method): {self.inner_repeats}")
        if rescoring_factors:
            print(f"Will run two-step search with rescoring factors: {rescoring_factors}")

    def benchmark_random_embeddings(self) -> Dict[str, List[Dict[str, Any]]]:
        results = []
        random_indices = np.random.randint(0, self.num_vectors, size=self.runs)
        for run in range(self.runs):
            query = self.searcher.get_float_embedding(random_indices[run])
            run_results = self._run_all_searches(query)
            results.append({
                "run": run,
                "query_index": int(random_indices[run]),
                "searches": run_results,
            })
            if (run + 1) % 10 == 0:
                print(f"Completed {run + 1}/{self.runs} runs")
        return {"mode": "random_embeddings", "results": results}

    def benchmark_query_file(self, query_file: str) -> Dict[str, List[Dict[str, Any]]]:
        results = []
        queries = self._load_queries(query_file)
        for i, query_data in enumerate(queries):
            query_vector = np.array(query_data["embedding"], dtype=np.float32)
            run_results = self._run_all_searches(query_vector)
            results.append({
                "run": i,
                "query_text": query_data["query"],
                "formatted_query": query_data["formatted_query"],
                "searches": run_results,
            })
            if (i + 1) % 10 == 0:
                print(f"Completed query {i + 1}/{len(queries)}")
        return {"mode": "query_file", "results": results}

    def benchmark_random_vectors(self) -> Dict[str, List[Dict[str, Any]]]:
        results = []
        for run in range(self.runs):
            query = np.random.randn(self.vector_dim).astype(np.float32)
            query = query / np.linalg.norm(query)
            run_results = self._run_all_searches(query)
            results.append({"run": run, "searches": run_results})
            if (run + 1) % 10 == 0:
                print(f"Completed {run + 1}/{self.runs} runs")
        return {"mode": "random_vectors", "results": results}

    def _run_all_searches(self, query: np.ndarray) -> List[Dict[str, Any]]:
        search_methods = [
            ("float", lambda q, k: self.searcher.search_float(q, k)),
            ("avx2", lambda q, k: self.searcher.search_avx2(q, k)),
            ("binary", lambda q, k: self.searcher.search_binary(q, k)),
            ("int8", lambda q, k: self.searcher.search_int8(q, k)),
            ("float16", lambda q, k: self.searcher.search_float16(q, k)),
            ("mf", lambda q, k: self.searcher.search_mf(q, k)),
            ("pca2", lambda q, k: self.searcher.search_pca2(q, k)),
            ("pca4", lambda q, k: self.searcher.search_pca4(q, k)),
            ("pca8", lambda q, k: self.searcher.search_pca8(q, k)),
            ("pca16", lambda q, k: self.searcher.search_pca16(q, k)),
            ("pca32", lambda q, k: self.searcher.search_pca32(q, k)),
        ]
        for factor in self.rescoring_factors:
            search_methods.append((
                f"twostep_rf{factor}",
                lambda q, k, rf=factor: self.searcher.search_twostep(q, k, rf),
            ))
        for factor in self.rescoring_factors:
            search_methods.append((
                f"ts_mf_rf{factor}",
                lambda q, k, rf=factor: self.searcher.search_twostep_mf(q, k, rf),
            ))

        first_float_results = None
        search_results = []
        for method_name, search_func in search_methods:
            try:
                per_repeat_times: List[float] = []
                last_results = None
                for _ in range(self.inner_repeats):
                    results, search_time = search_func(query, self.k)
                    per_repeat_times.append(float(search_time))
                    last_results = results

                if method_name == "float":
                    first_float_results = last_results
                    float_indices = set(idx for _, idx, _ in last_results)

                # NDCG / overlap depend only on result sets, not on timing,
                # so compute once per (query, method) — not per repeat.
                metrics: Dict[str, Any] = {
                    "time_us": float(np.mean(per_repeat_times)),
                    "time_us_repeats": per_repeat_times,
                    "results": [
                        (score, int(idx), text[:100])
                        for score, idx, text in last_results[:5]
                    ],
                }

                if method_name != "float":
                    result_indices = set(idx for _, idx, _ in last_results)
                    metrics.update({
                        "overlap_with_float": len(float_indices & result_indices),
                        "jaccard_index": (
                            len(float_indices & result_indices)
                            / len(float_indices | result_indices)
                        ),
                        "ndcg": calculate_ndcg(first_float_results, last_results),
                        "ndcg_10": calculate_ndcg(first_float_results[:10], last_results[:10]),
                    })

                search_results.append({"method": method_name, "metrics": metrics})

            except Exception as e:
                print(
                    f"Error in {method_name} search: {str(e)}\nline: "
                    f"{e.__traceback__.tb_lineno}"
                )
                continue

        return search_results

    def _load_queries(self, query_file: str) -> List[Dict[str, Any]]:
        queries = []
        with open(query_file, "r") as f:
            for line in f:
                queries.append(json.loads(line))
        return queries

    def save_results(self, results: Dict[str, Any], output_file: str):
        timestamp = int(time.time())
        output_path = Path(output_file)
        final_output = output_path.with_stem(f"{output_path.stem}_{timestamp}")

        method_stats: Dict[str, Dict[str, List[float]]] = {}
        for run in results["results"]:
            for search in run["searches"]:
                method_name = search["method"]
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        "times_us": [],
                        "jaccard_indices": [],
                        "ndcg_scores": [],
                        "overlap_counts": [],
                    }
                metrics = search["metrics"]
                method_stats[method_name]["times_us"].extend(metrics["time_us_repeats"])
                if method_name != "float":
                    method_stats[method_name]["jaccard_indices"].append(metrics["jaccard_index"])
                    method_stats[method_name]["ndcg_scores"].append(metrics["ndcg"])
                    method_stats[method_name]["overlap_counts"].append(metrics["overlap_with_float"])

        summary_stats: Dict[str, Dict[str, Any]] = {}
        for method, stats in method_stats.items():
            times = stats["times_us"]
            summary_stats[method] = {
                "time_us": {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                    "median": float(np.median(times)),
                    "geomean": _geomean(times),
                },
                "inner_repeats": self.inner_repeats,
                "num_measurements": len(times),
            }
            if method != "float":
                summary_stats[method].update({
                    "jaccard_index": {
                        "mean": float(np.mean(stats["jaccard_indices"])),
                        "std": float(np.std(stats["jaccard_indices"])),
                    },
                    "ndcg": {
                        "mean": float(np.mean(stats["ndcg_scores"])),
                        "std": float(np.std(stats["ndcg_scores"])),
                    },
                    "overlap": {
                        "mean": float(np.mean(stats["overlap_counts"])),
                        "std": float(np.std(stats["overlap_counts"])),
                    },
                })

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(x) for x in obj]
            return obj

        analysis_ready = {
            "metadata": {
                "num_vectors": self.num_vectors,
                "vector_dim": self.vector_dim,
                "k": self.k,
                "runs": self.runs,
                "inner_repeats": self.inner_repeats,
                "timestamp": timestamp,
                "timestamp_human": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(timestamp)
                ),
                "mode": results["mode"],
                "rescoring_factors": self.rescoring_factors,
                "schema_version": "v3",
            },
            "summary": summary_stats,
            "method_stats": method_stats,
            "runs": convert_numpy(results["results"]),
        }

        with open(final_output, "w") as f:
            json.dump(analysis_ready, f, indent=2)
        print(f"Results saved to {final_output}")

        summary_output = final_output.with_stem(f"{final_output.stem}_summary")
        with open(summary_output, "w") as f:
            json.dump(
                {"metadata": analysis_ready["metadata"], "summary": analysis_ready["summary"]},
                f,
                indent=2,
            )
        print(f"Summary saved to {summary_output}")


def main():
    parser = argparse.ArgumentParser(description="Vector Similarity Search Benchmark v3")
    parser.add_argument("--embedding-file", "-f", required=True)
    parser.add_argument("--mode", "-m", choices=["random", "query", "random-vec"], required=True)
    parser.add_argument("--query-file", "-q")
    parser.add_argument("--k", "-k", type=int, default=25)
    parser.add_argument("--runs", "-r", type=int, default=100)
    parser.add_argument("--output", "-o", default="benchmark_results.json")
    parser.add_argument("--rescoring-factor", type=str)
    parser.add_argument("--embedding-dim", "-d", type=int, default=1024)
    parser.add_argument("--max-vectors", type=int, default=0)
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=10,
        help="Per (query, method) call this many times and record each timing",
    )

    args = parser.parse_args()
    print(args.embedding_dim)

    if args.mode == "query" and not args.query_file:
        parser.error("Query file is required for query mode")

    rescoring_factors = None
    if args.rescoring_factor:
        try:
            rescoring_factors = [int(x) for x in args.rescoring_factor.split(",")]
        except ValueError:
            parser.error("Rescoring factors must be comma-separated integers")

    if args.mode == "random":
        bench = VectorSearchBenchmarkV3(
            args.embedding_file, args.k, args.runs, rescoring_factors,
            embedding_dim=args.embedding_dim, max_vectors=args.max_vectors,
            inner_repeats=args.inner_repeats,
        )
        results = bench.benchmark_random_embeddings()
    elif args.mode == "query":
        bench = VectorSearchBenchmarkV3(
            args.embedding_file, args.k, args.runs, rescoring_factors,
            embedding_dim=args.embedding_dim, max_vectors=args.max_vectors,
            inner_repeats=args.inner_repeats,
        )
        results = bench.benchmark_query_file(args.query_file)
    else:
        bench = VectorSearchBenchmarkV3(
            args.embedding_file, args.k, args.runs, rescoring_factors,
            embedding_dim=args.embedding_dim, max_vectors=args.max_vectors,
            inner_repeats=args.inner_repeats,
        )
        results = bench.benchmark_random_vectors()

    bench.save_results(results, args.output)


if __name__ == "__main__":
    main()
