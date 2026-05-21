#!/usr/bin/env python3
"""Generate manuscript figures for TECS revision Priority 1 (ablation) and
Priority 2 (scaling).

Inputs:
    simsearch/results/priority1_timing_breakdown.csv
    simsearch/results/priority2_scaling_raw.csv

Outputs (written to bilder/plots/):
    revision_priority1_ablation_n60k.pdf  + .png
    revision_priority2_scaling.pdf        + .png

Design choices documented inline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Consistent component colors across both figures. Using a colorblind-safe
# palette. Order is the natural temporal flow of a query.
COMPONENTS = [
    ("T_query_sketch_ms",       "Query sketch"),
    ("T_binary_scan_ms",        "Binary scan"),
    ("T_candidate_selection_ms","Candidate selection"),
    ("T_rescore_ms",            "Rescore"),
    ("T_final_topk_ms",         "Final top-k"),
]
COMPONENT_COLORS = dict(zip(
    [c[1] for c in COMPONENTS],
    sns.color_palette("colorblind", n_colors=len(COMPONENTS)),
))
# A distinct gray for "fused scan" (float32_avx2: no decomposable components,
# all time lands in T_unaccounted_ms by instrumentation design).
FUSED_COLOR = "#888888"

METHOD_DISPLAY = {
    "binary":            "Binary only",
    "two_step_RF10":     "Two-step (RF=10)",
    "two_step_mf_RF10":  "Two-step MF (RF=10)",
    "two_step_RF50":     "Two-step (RF=50)",
    "two_step_mf_RF50":  "Two-step MF (RF=50)",
    "float32_avx2":      "Float32 AVX2 (baseline)",
}

# Method order chosen so that the eye scans from cheapest to most expensive,
# with the baseline placed at the bottom as the visual reference.
METHOD_ORDER_P1 = [
    "binary",
    "two_step_RF10",
    "two_step_mf_RF10",
    "two_step_RF50",
    "two_step_mf_RF50",
    "float32_avx2",
]

METHOD_ORDER_P2 = [
    "binary",
    "two_step_RF10",
    "two_step_mf_RF10",
    "float32_avx2",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def per_method_means(df: pd.DataFrame, group_cols=("method",)) -> pd.DataFrame:
    """Mean over (query_id, repeat_id) for every group."""
    agg_cols = [c for c, _ in COMPONENTS] + [
        "T_total_ms", "T_component_sum_ms", "T_unaccounted_ms",
        "ndcg100", "jaccard", "num_survivors",
    ]
    return (
        df.groupby(list(group_cols), as_index=False)[agg_cols].mean()
    )


# ---------------------------------------------------------------------------
# Figure 1 — Priority 1 ablation at N = 60,000
# ---------------------------------------------------------------------------
def plot_priority1(p1_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(p1_csv)
    means = per_method_means(df).set_index("method")

    # Leave room at the top for an external legend rather than overlaying bars.
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    y_positions = np.arange(len(METHOD_ORDER_P1))[::-1]  # top bar = first method

    for yi, method in zip(y_positions, METHOD_ORDER_P1):
        row = means.loc[method]
        if method == "float32_avx2":
            # Fused single-pass scan; the instrumentation puts the whole cost
            # in T_unaccounted_ms intentionally. Plot it as one solid bar.
            total = row["T_total_ms"]
            ax.barh(yi, total, color=FUSED_COLOR, edgecolor="white", linewidth=0.5)
        else:
            left = 0.0
            for col, label in COMPONENTS:
                v = row[col]
                # Skip the synthetic 1e-12 placeholders.
                if v < 1e-6:
                    continue
                ax.barh(yi, v, left=left, color=COMPONENT_COLORS[label],
                        edgecolor="white", linewidth=0.5)
                left += v

        # Right-side annotation: total ms, NDCG, and speedup vs baseline.
        baseline_total = means.loc["float32_avx2", "T_total_ms"]
        speedup = baseline_total / row["T_total_ms"]
        ndcg = row["ndcg100"]
        annot = f"  {row['T_total_ms']:5.2f} ms   NDCG={ndcg:.3f}   {speedup:5.1f}×"
        ax.text(row["T_total_ms"], yi, annot, va="center", ha="left",
                fontsize=9, family="monospace")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER_P1])
    ax.set_xlabel("Mean query latency (ms)")
    # Title sits above the legend.
    ax.set_title("Priority 1: Per-component query latency at N=60,000 (k=100)",
                 pad=42)

    # Make room for the right-side annotations.
    xmax = means["T_total_ms"].max()
    ax.set_xlim(0, xmax * 1.55)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Custom legend placed above the plot so it never overlaps the bars.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COMPONENT_COLORS[label])
        for _, label in COMPONENTS
    ]
    legend_labels = [label for _, label in COMPONENTS]
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=FUSED_COLOR))
    legend_labels.append("Fused scan (no decomposition)")
    ax.legend(legend_handles, legend_labels,
              loc="lower center", bbox_to_anchor=(0.5, 1.04),
              ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"revision_priority1_ablation_n60k.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Priority 2 scaling
# ---------------------------------------------------------------------------
def plot_priority2(p2_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(p2_csv)
    means = per_method_means(df, group_cols=("method", "N"))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})
    ax_scaling, ax_compose = axes

    # ---------------- Panel A: log-log scaling ----------------
    method_palette = {
        "binary":           "#1f77b4",
        "two_step_RF10":    "#2ca02c",
        "two_step_mf_RF10": "#9467bd",
        "float32_avx2":     "#d62728",
    }
    method_markers = {
        "binary":           "o",
        "two_step_RF10":    "s",
        "two_step_mf_RF10": "^",
        "float32_avx2":     "D",
    }

    Ns = sorted(means["N"].unique())
    for method in METHOD_ORDER_P2:
        sub = means[means["method"] == method].sort_values("N")
        ax_scaling.plot(sub["N"], sub["T_total_ms"],
                        color=method_palette[method],
                        marker=method_markers[method], markersize=7,
                        linewidth=1.8, label=METHOD_DISPLAY[method])

    # Linear-scaling reference: anchor at smallest N for float32_avx2.
    anchor_N = Ns[0]
    anchor_t = means[(means["method"] == "float32_avx2") & (means["N"] == anchor_N)]["T_total_ms"].iloc[0]
    xs = np.array([Ns[0], Ns[-1]], dtype=float)
    ys = anchor_t * xs / anchor_N
    ax_scaling.plot(xs, ys, linestyle=":", color="gray", linewidth=1.2,
                    label="Ideal linear (slope 1)")

    # Speedup callout: bracket the gap between baseline and two_step_RF10
    # at the largest N rather than firing an arrow across the panel.
    big_N = Ns[-1]
    t_base = means[(means["method"] == "float32_avx2") & (means["N"] == big_N)]["T_total_ms"].iloc[0]
    t_two  = means[(means["method"] == "two_step_RF10") & (means["N"] == big_N)]["T_total_ms"].iloc[0]
    speedup = t_base / t_two
    # Vertical double-headed arrow between the two points at big_N.
    ax_scaling.annotate(
        "", xy=(big_N, t_base), xytext=(big_N, t_two),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.1),
    )
    # Geometric midpoint on the log-scaled y axis.
    mid_y = np.sqrt(t_base * t_two)
    ax_scaling.annotate(
        f"{speedup:.1f}× speedup\nat N={big_N/1_000_000:.1f}M",
        xy=(big_N, mid_y),
        xytext=(big_N * 0.92, mid_y),
        textcoords="data",
        ha="right", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7d6", ec="gray", lw=0.6),
    )

    ax_scaling.set_xscale("log")
    ax_scaling.set_yscale("log")
    ax_scaling.set_xticks(Ns)
    ax_scaling.get_xaxis().set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v/1000)}K" if v < 1_000_000 else f"{v/1_000_000:.1f}M"))
    ax_scaling.set_xlabel("Dataset size N (vectors)")
    ax_scaling.set_ylabel("Mean query latency (ms, log scale)")
    ax_scaling.set_title("(a) Latency scaling with N (k=100)")
    ax_scaling.grid(which="both", linestyle="--", alpha=0.45)
    # Legend in lower right keeps it clear of both the high-cost float32 curve
    # and the speedup callout near the top.
    ax_scaling.legend(loc="lower right", frameon=True, framealpha=0.95,
                      fontsize=8.5)

    # ---------------- Panel B: composition of two_step_RF10 by N ----------------
    method_focus = "two_step_RF10"
    sub = means[means["method"] == method_focus].sort_values("N").set_index("N")

    width = 0.55
    x = np.arange(len(Ns))

    for ni, N in enumerate(Ns):
        row = sub.loc[N]
        bottom = 0.0
        for col, label in COMPONENTS:
            v = row[col]
            if v < 1e-6:
                continue
            ax_compose.bar(ni, v, width, bottom=bottom,
                           color=COMPONENT_COLORS[label],
                           edgecolor="white", linewidth=0.5,
                           label=label if ni == 0 else None)
            bottom += v
        rescore_frac = row["T_rescore_ms"] / row["T_total_ms"]
        ax_compose.text(ni, bottom + sub["T_total_ms"].max() * 0.02,
                        f"{row['T_total_ms']:.2f} ms\n"
                        f"NDCG={row['ndcg100']:.3f}\n"
                        f"rescore={rescore_frac*100:.0f}%",
                        ha="center", va="bottom", fontsize=8.5)

    ax_compose.set_xticks(x)
    ax_compose.set_xticklabels([f"{int(N/1000)}K" if N < 1_000_000 else f"{N/1_000_000:.1f}M"
                                for N in Ns])
    ax_compose.set_xlabel("Dataset size N")
    ax_compose.set_ylabel("Latency (ms)")
    ax_compose.set_title("(b) Two-step (RF=10) component composition as N grows")
    ax_compose.set_ylim(0, sub["T_total_ms"].max() * 1.30)
    ax_compose.grid(axis="y", linestyle="--", alpha=0.45)
    ax_compose.set_axisbelow(True)

    # Shared legend at the bottom (uses panel B handles).
    handles, labels = ax_compose.get_legend_handles_labels()
    ax_compose.legend(handles, labels, loc="upper left", fontsize=8.5,
                      frameon=True, framealpha=0.95)

    fig.suptitle("Priority 2: Scaling and where time is spent in two-step search",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"revision_priority2_scaling.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_priority1(args.results_dir / "priority1_timing_breakdown.csv", args.out_dir)
    plot_priority2(args.results_dir / "priority2_scaling_raw.csv", args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
