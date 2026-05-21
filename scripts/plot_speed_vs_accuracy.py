#!/usr/bin/env python3
"""Regenerate the speed-vs-accuracy scatter plot from the v3 9950X benchmark
JSON.

The plot is rendered to sit side-by-side with
plots/memory_vs_accuracy_manual_benchmark_dim1024_k100_q.png, so it inherits
that figure's aspect ratio, font sizes, color palette, label-next-to-dot
style, and the absence of a legend.

Inputs:
    simsearch/python/jupyter/results/benchmark_dim1024_k100_q.json (v3)

Output:
    plots/speed_vs_accuracy_manual_benchmark_dim1024_k100_q.png  (+ .pdf)

Label placement: each dot's label is placed by trying candidate offset
directions (E, W, N, S, NE, NW, SE, SW, ...) in order; the first direction
whose bounding box stays inside the axes and doesn't overlap any other
already-placed label or any data dot is chosen. A final overlap check
either confirms `OK: no overlapping labels` or prints which pairs collide.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

JSON_PATH = Path("simsearch/python/jupyter/results/benchmark_dim1024_k100_q.json")
OUT_PATH = Path("plots/speed_vs_accuracy_manual_benchmark_dim1024_k100_q.png")

# Match the style of memory_vs_accuracy_manual_benchmark_dim1024_k100_q.png:
# larger label fonts, no top/right spines, dashed grid, no legend, label
# next to each dot.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})


METHODS = [
    # raw_key, display name, category
    ("avx2",         "Float32 (AVX2)",      "baseline"),
    ("binary",       "Binary",              "binary"),
    ("int8",         "Int8",                "quant_dark"),
    ("float16",      "Float16",             "quant_dark"),
    ("pca2",         "PCA (/2)",            "pca_light"),
    ("pca4",         "PCA (/4)",            "pca_light"),
    ("pca8",         "PCA (/8)",            "pca_light"),
    ("pca16",        "PCA (/16)",           "pca_light"),
    ("pca32",        "PCA (/32)",           "pca_light"),
    # Only endpoints; intermediate RFs live in the RF-sensitivity figure.
    ("twostep_rf2",  "Two-Step (RF=2)",     "twostep"),
    ("twostep_rf50", "Two-Step (RF=50)",    "twostep"),
    ("ts_mf_rf2",    "Two-Step MF (RF=2)",  "twostep_mf"),
    ("ts_mf_rf50",   "Two-Step MF (RF=50)", "twostep_mf"),
]

# Palette chosen to match the companion memory_vs_accuracy plot:
# dark navy for the "standalone quantization" family (binary, int8, float16,
# mapped float -- the last is excluded from this view), light blue for PCA,
# orange for the AVX2 baseline, two greens for the two two-step families.
CATEGORY_COLOR = {
    "baseline":    "#e67e22",
    "binary":      "#1f4e6e",
    "quant_dark":  "#1f4e6e",
    "pca_light":   "#3498db",
    "twostep":     "#27ae60",
    "twostep_mf":  "#16723d",
}

CANDIDATE_OFFSETS = [
    (  8,   0, "left",   "center"),    # E
    ( -8,   0, "right",  "center"),    # W
    (  0,  10, "center", "bottom"),    # N
    (  0, -10, "center", "top"),       # S
    (  7,   7, "left",   "bottom"),    # NE
    ( -7,   7, "right",  "bottom"),    # NW
    (  7,  -7, "left",   "top"),       # SE
    ( -7,  -7, "right",  "top"),       # SW
    ( 16,   0, "left",   "center"),    # far E
    (-16,   0, "right",  "center"),    # far W
    (  0,  18, "center", "bottom"),    # far N
    (  0, -18, "center", "top"),       # far S
    ( 14,  10, "left",   "bottom"),    # far NE
    (-14,  10, "right",  "bottom"),    # far NW
    ( 14, -10, "left",   "top"),       # far SE
    (-14, -10, "right",  "top"),       # far SW
    ( 22, -16, "left",   "top"),       # far far SE
    ( 22,  16, "left",   "bottom"),    # far far NE
]


def geomean(values):
    log_vals = [math.log(v) for v in values if v > 0]
    return math.exp(sum(log_vals) / len(log_vals))


def load_data():
    with open(JSON_PATH) as f:
        d = json.load(f)
    rows = []
    for raw, display, cat in METHODS:
        st = d["method_stats"][raw]
        times_ms = [t / 1000.0 for t in st["times_us"] if t > 0]
        latency = geomean(times_ms)
        ndcg = d["summary"][raw].get("ndcg", {}).get("mean", 1.0)
        rows.append({
            "raw": raw, "label": display, "category": cat,
            "latency_ms": latency, "ndcg": ndcg,
        })
    return rows


def bbox_overlap(a, b, pad=2.0):
    return not (
        a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or
        a.y1 + pad < b.y0 or b.y1 + pad < a.y0
    )


def dot_bbox(ax, fig, x, y, radius_pts=5.0):
    px, py = ax.transData.transform((x, y))
    return mtrans.Bbox.from_extents(
        px - radius_pts, py - radius_pts, px + radius_pts, py + radius_pts,
    )


def place_label(ax, fig, x, y, text, occupied, dot_bboxes,
                fontsize=12, color="#1f2933"):
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer=renderer)
    for dx, dy, ha, va in CANDIDATE_OFFSETS:
        offset = mtrans.offset_copy(
            ax.transData, fig=fig, x=dx, y=dy, units="points",
        )
        t = ax.text(x, y, text, transform=offset, ha=ha, va=va,
                    fontsize=fontsize, color=color, zorder=5)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        inside = (bb.x0 >= ax_bb.x0 - 1 and bb.x1 <= ax_bb.x1 + 1
                  and bb.y0 >= ax_bb.y0 - 1 and bb.y1 <= ax_bb.y1 + 1)
        clash = any(bbox_overlap(bb, ob) for ob in occupied) \
                or any(bbox_overlap(bb, db, pad=1.0) for db in dot_bboxes)
        if inside and not clash:
            occupied.append(bb)
            return t
        t.remove()
    # Fallback: pick the first candidate that's at least inside the axes.
    for dx, dy, ha, va in CANDIDATE_OFFSETS:
        offset = mtrans.offset_copy(
            ax.transData, fig=fig, x=dx, y=dy, units="points",
        )
        t = ax.text(x, y, text, transform=offset, ha=ha, va=va,
                    fontsize=fontsize, color=color, zorder=5)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        if (bb.x0 >= ax_bb.x0 and bb.x1 <= ax_bb.x1
                and bb.y0 >= ax_bb.y0 and bb.y1 <= ax_bb.y1):
            occupied.append(bb)
            return t
        t.remove()
    dx, dy, ha, va = CANDIDATE_OFFSETS[0]
    offset = mtrans.offset_copy(
        ax.transData, fig=fig, x=dx, y=dy, units="points",
    )
    t = ax.text(x, y, text, transform=offset, ha=ha, va=va,
                fontsize=fontsize, color=color, zorder=5)
    fig.canvas.draw()
    occupied.append(t.get_window_extent(renderer=renderer))
    return t


def main():
    rows = load_data()

    # Aspect ratio matched to memory_vs_accuracy_manual...png; slightly
    # taller to give the dense Two-Step cluster more vertical room for
    # labels.
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    for r in rows:
        ax.scatter(
            r["latency_ms"], r["ndcg"],
            s=110, color=CATEGORY_COLOR[r["category"]],
            edgecolor="white", linewidth=1.0, zorder=3,
        )

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("NDCG Score")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)

    fig.canvas.draw()
    dot_bboxes = [
        dot_bbox(ax, fig, r["latency_ms"], r["ndcg"]) for r in rows
    ]

    # Place labels: most-isolated NDCG values first so they grab open space.
    occupied = []
    label_rows = sorted(rows, key=lambda r: abs(r["ndcg"] - 0.6), reverse=True)
    for r in label_rows:
        place_label(ax, fig, r["latency_ms"], r["ndcg"], r["label"],
                    occupied, dot_bboxes, fontsize=11)

    # Final overlap audit
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_artists = [a for a in ax.get_children()
                     if isinstance(a, plt.Text) and a.get_text()
                     and a is not ax.title
                     and a is not ax.xaxis.label
                     and a is not ax.yaxis.label]
    bboxes = [a.get_window_extent(renderer=renderer) for a in label_artists]
    overlaps = []
    for i in range(len(label_artists)):
        for j in range(i + 1, len(label_artists)):
            if bbox_overlap(bboxes[i], bboxes[j], pad=0.5):
                overlaps.append((label_artists[i].get_text(),
                                 label_artists[j].get_text()))
    if overlaps:
        print("WARNING: residual label overlaps detected:")
        for a, b in overlaps:
            print(f"  {a!r}  <->  {b!r}")
    else:
        print("OK: no overlapping labels.")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
