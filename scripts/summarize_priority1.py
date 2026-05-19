#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from collections import defaultdict

FIELDS = [
    "machine", "cpu_model", "dataset", "N", "d", "k", "RF", "method",
    "num_queries", "num_repeats", "geomean_total_ms", "mean_total_ms",
    "median_total_ms", "p95_total_ms", "geomean_query_sketch_ms",
    "geomean_binary_scan_ms", "geomean_candidate_selection_ms",
    "geomean_rescore_ms", "geomean_final_topk_ms", "mean_ndcg100",
    "geomean_unaccounted_ms", "std_ndcg100", "mean_jaccard",
    "mean_num_survivors",
]

def f(row, key):
    return float(row[key])

def geomean(vals):
    vals = [max(v, 1e-12) for v in vals]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0

def p95(vals):
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)
    return vals[idx]

def main(raw_path, out_path):
    groups = defaultdict(list)
    with open(raw_path, newline="") as src:
        for row in csv.DictReader(src):
            key = (row["machine"], row["cpu_model"], row["dataset"], row["N"],
                   row["d"], row["k"], row["RF"], row["method"])
            groups[key].append(row)

    with open(out_path, "w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDS)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            totals = [f(r, "T_total_ms") for r in rows]
            ndcg = [f(r, "ndcg100") for r in rows]
            jaccard = [f(r, "jaccard") for r in rows]
            writer.writerow({
                "machine": key[0],
                "cpu_model": key[1],
                "dataset": key[2],
                "N": key[3],
                "d": key[4],
                "k": key[5],
                "RF": key[6],
                "method": key[7],
                "num_queries": len({r["query_id"] for r in rows}),
                "num_repeats": len({r["repeat_id"] for r in rows}),
                "geomean_total_ms": geomean(totals),
                "mean_total_ms": statistics.fmean(totals),
                "median_total_ms": statistics.median(totals),
                "p95_total_ms": p95(totals),
                "geomean_query_sketch_ms": geomean([f(r, "T_query_sketch_ms") for r in rows]),
                "geomean_binary_scan_ms": geomean([f(r, "T_binary_scan_ms") for r in rows]),
                "geomean_candidate_selection_ms": geomean([f(r, "T_candidate_selection_ms") for r in rows]),
                "geomean_rescore_ms": geomean([f(r, "T_rescore_ms") for r in rows]),
                "geomean_final_topk_ms": geomean([f(r, "T_final_topk_ms") for r in rows]),
                "geomean_unaccounted_ms": geomean([abs(f(r, "T_unaccounted_ms")) for r in rows]),
                "mean_ndcg100": statistics.fmean(ndcg),
                "std_ndcg100": statistics.pstdev(ndcg) if len(ndcg) > 1 else 0.0,
                "mean_jaccard": statistics.fmean(jaccard),
                "mean_num_survivors": statistics.fmean([f(r, "num_survivors") for r in rows]),
            })

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_priority1.py RAW_CSV SUMMARY_CSV")
    main(sys.argv[1], sys.argv[2])
