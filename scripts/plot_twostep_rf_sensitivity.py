#!/usr/bin/env python3
"""Regenerate the RF sensitivity plots for two-step search from the v3
benchmark JSON. One panel per family (Two-Step / Two-Step MF), each showing
mean search time (left y) and accuracy metrics (right y) vs rescoring factor.

Inputs:
    simsearch/python/jupyter/results/benchmark_dim1024_k100_q.json (v3)

Outputs:
    bilder/plots/twostep_comparison_combined_benchmark_dim1024_k100_q.png    (Two-Step)
    bilder/plots/twostep_comparison_combined_benchmark_dim1024_k100_q_mf.png (Two-Step MF)

Matches the legacy layout (dual y-axis, RF on the x-axis) so the existing
manuscript figure reference does not need to change.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

JSON_PATH = Path("simsearch/python/jupyter/results/benchmark_dim1024_k100_q.json")
OUT_DIR = Path("bilder/plots")

RF_VALUES = [2, 5, 10, 25, 50]

FAMILIES = {
    "twostep_rf": {
        "title": "Two-Step (RF sensitivity)",
        "out": "twostep_comparison_combined_benchmark_dim1024_k100_q.png",
    },
    "ts_mf_rf": {
        "title": "Two-Step MF (RF sensitivity)",
        "out": "twostep_comparison_combined_benchmark_dim1024_k100_q_mf.png",
    },
}


def geomean(values):
    log_vals = [math.log(v) for v in values if v > 0]
    return math.exp(sum(log_vals) / len(log_vals))


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    for prefix, meta in FAMILIES.items():
        time_ms, ndcg, jaccard = [], [], []
        for rf in RF_VALUES:
            raw = f"{prefix}{rf}"
            st = data["method_stats"][raw]
            times = [t / 1000.0 for t in st["times_us"] if t > 0]
            time_ms.append(geomean(times))
            ndcg.append(data["summary"][raw]["ndcg"]["mean"])
            jaccard.append(data["summary"][raw]["jaccard_index"]["mean"])

        fig, ax1 = plt.subplots(figsize=(7.5, 3.4))
        ax2 = ax1.twinx()

        l1, = ax1.plot(RF_VALUES, time_ms, marker="o", color="#1b7837",
                       linewidth=2.0, markersize=7, label="Time")
        ax1.set_xlabel("Rescoring factor (RF)", fontsize=11)
        ax1.set_ylabel("Geo. mean search time (ms)", fontsize=11, color="#1b7837")
        ax1.tick_params(axis="y", colors="#1b7837")
        ax1.set_xticks(RF_VALUES)
        # Headroom on both axes so the value annotations at RF=50 don't bump
        # into the top of the panel where the two curves visually converge.
        time_top = max(time_ms) * 1.18
        time_bot = min(time_ms) * 0.85
        ax1.set_ylim(time_bot, time_top)
        ax1.grid(True, linestyle=":", alpha=0.4)
        ax1.set_axisbelow(True)

        l2, = ax2.plot(RF_VALUES, ndcg, marker="s", linestyle="--",
                       color="#2c7fb8", linewidth=1.8, markersize=6,
                       label="NDCG@100")
        l3, = ax2.plot(RF_VALUES, jaccard, marker="^", linestyle=":",
                       color="#d95f02", linewidth=1.8, markersize=6,
                       label="Jaccard")
        ax2.set_ylabel("Accuracy score", fontsize=11)
        ax2.set_ylim(0.55, 1.04)

        # Combined legend in a place that won't overlap the curves
        ax1.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
                   loc="center right", framealpha=0.95, fontsize=9)

        # Annotate each point with its value, offset to avoid clutter
        for x, y in zip(RF_VALUES, time_ms):
            ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, -12), ha="center", fontsize=8,
                         color="#1b7837")
        for x, y in zip(RF_VALUES, ndcg):
            ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8,
                         color="#2c7fb8")

        fig.tight_layout()
        out_path = OUT_DIR / meta["out"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Wrote {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
