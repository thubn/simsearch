#!/usr/bin/env python3
from pathlib import Path
import csv

def try_plot_priority1():
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open("results/priority1_timing_summary.csv", newline="")))
    methods = [r["method"] for r in rows]
    scan = [float(r["geomean_binary_scan_ms"]) for r in rows]
    rescore = [float(r["geomean_rescore_ms"]) for r in rows]
    other = [float(r["geomean_query_sketch_ms"]) + float(r["geomean_candidate_selection_ms"]) + float(r["geomean_final_topk_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, scan, label="binary scan")
    ax.bar(methods, rescore, bottom=scan, label="rescore")
    ax.bar(methods, other, bottom=[a+b for a,b in zip(scan,rescore)], label="other")
    ax.set_ylabel("geomean ms")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/revision_priority1_timing_breakdown.pdf")

def try_plot_priority2():
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open("results/priority2_scaling_summary.csv", newline="")))
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        vals = sorted([r for r in rows if r["method"] == m], key=lambda r: int(r["N"]))
        ax.plot([int(r["N"]) for r in vals], [float(r["geomean_total_ms"]) for r in vals], marker="o", label=m)
    ax.set_xlabel("N")
    ax.set_ylabel("geomean ms")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/revision_priority2_scaling.pdf")

def main():
    Path("figures").mkdir(exist_ok=True)
    try:
        try_plot_priority1()
    except Exception as e:
        Path("figures/revision_priority1_timing_breakdown.txt").write_text(f"plot unavailable: {e}\n")
    try:
        try_plot_priority2()
    except Exception as e:
        Path("figures/revision_priority2_scaling.txt").write_text(f"plot unavailable: {e}\n")

if __name__ == "__main__":
    main()
