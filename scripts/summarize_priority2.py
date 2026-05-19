#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from collections import defaultdict

FIELDS = [
    "machine", "cpu_model", "dataset", "N", "d", "k", "RF", "method",
    "num_queries", "num_repeats", "geomean_total_ms", "mean_total_ms",
    "median_total_ms", "p95_total_ms", "mean_ndcg100", "std_ndcg100",
    "mean_jaccard", "geomean_binary_scan_ms", "geomean_rescore_ms",
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

def write_verification(summary_path, report_path):
    rows = list(csv.DictReader(open(summary_path, newline="")))
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    lines = ["# Priority 2 Scaling Verification", ""]
    for method in sorted(by_method):
        vals = sorted(by_method[method], key=lambda r: int(r["N"]))
        totals = ", ".join(f"N={r['N']}: {float(r['geomean_total_ms']):.6g} ms" for r in vals)
        rescore = ", ".join(f"N={r['N']}: {float(r.get('geomean_rescore_ms', 0.0)):.6g} ms" for r in vals)
        acc = ", ".join(f"N={r['N']}: ndcg={float(r['mean_ndcg100']):.4f}, jaccard={float(r['mean_jaccard']):.4f}" for r in vals)
        lines += [
            f"## {method}",
            f"- Total latency by N: {totals}",
            f"- Rescore latency by N: {rescore}",
            f"- Accuracy by N: {acc}",
            "",
        ]
    lines += [
        "## Checklist",
        "- Float32 and binary linearity should be assessed from the table above; no conclusion is forced by this script.",
        "- Two-step Step 1 dominance should be assessed by comparing binary scan plus candidate selection against rescore plus final top-k.",
        "- Unexpected trends should be reported in `docs/revision_experiment_results_summary.md` after the full run.",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def main(raw_path, out_path, report_path):
    groups = defaultdict(list)
    with open(raw_path, newline="") as src:
        for row in csv.DictReader(src):
            key = (row["machine"], row["cpu_model"], row["dataset"], row["N"],
                   row["d"], row["k"], row["RF"], row["method"])
            groups[key].append(row)
    with open(out_path, "w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDS)
        writer.writeheader()
        for key, rows in sorted(groups.items(), key=lambda kv: (int(kv[0][3]), kv[0][7])):
            totals = [f(r, "T_total_ms") for r in rows]
            ndcg = [f(r, "ndcg100") for r in rows]
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
                "mean_ndcg100": statistics.fmean(ndcg),
                "std_ndcg100": statistics.pstdev(ndcg) if len(ndcg) > 1 else 0.0,
                "mean_jaccard": statistics.fmean([f(r, "jaccard") for r in rows]),
                "geomean_binary_scan_ms": geomean([f(r, "T_binary_scan_ms") for r in rows]),
                "geomean_rescore_ms": geomean([f(r, "T_rescore_ms") for r in rows]),
                "mean_num_survivors": statistics.fmean([f(r, "num_survivors") for r in rows]),
            })
    write_verification(out_path, report_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: summarize_priority2.py RAW_CSV SUMMARY_CSV REPORT_MD")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
