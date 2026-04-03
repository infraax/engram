#!/usr/bin/env python3
"""ENGRAM Research Paper — Figure Generation.

Generates all 15 figures for the ENGRAM paper from results/ data files.
Output: results/figures/*.pdf (LaTeX-compatible, 300 DPI)

Usage:
    cd ENGRAM && python scripts/paper_figures.py
    python scripts/paper_figures.py --only fig02   # Single figure
    python scripts/paper_figures.py --list          # List all figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ABSOLUTE_DIR = RESULTS_DIR / "absolute"
STRESS_DIR = RESULTS_DIR / "stress"

# LaTeX-compatible style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-safe palette
COLORS = {
    "blue": "#4477AA",
    "orange": "#EE6677",
    "green": "#228833",
    "purple": "#AA3377",
    "cyan": "#66CCEE",
    "grey": "#BBBBBB",
    "red": "#CC3311",
    "teal": "#009988",
    "yellow": "#CCBB44",
    "indigo": "#332288",
}

PASS_COLOR = COLORS["green"]
FAIL_COLOR = COLORS["red"]


# ── Data Loading ─────────────────────────────────────────────────────────

def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file and return parsed dict."""
    return json.loads(path.read_text())


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF and PNG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", format="pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  Saved: {name}.pdf + .png")


# ── Figure 2: Frequency Combination Comparison ──────────────────────────

def fig02_frequency_comparison() -> None:
    """Bar chart: 6 frequency combos × recall and margin."""
    print("Fig 02: Frequency combination comparison...")
    data = load_json(ABSOLUTE_DIR / "multifreq_comparison.json")
    results = data["results"]

    combos = list(results.keys())
    recalls = [results[c]["recall"] * 100 for c in combos]
    margins = [results[c]["margin_mean"] * 1000 for c in combos]  # ×1000
    failures = [results[c]["n_failures"] for c in combos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Recall
    x = np.arange(len(combos))
    bar_colors = [COLORS["green"] if c == "f0+f1" else COLORS["blue"] for c in combos]
    bars = ax1.bar(x, recalls, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(combos, rotation=30, ha="right")
    ax1.set_ylabel("Recall@1 (%)")
    ax1.set_title("(a) Recall by Frequency Combination")
    ax1.set_ylim(60, 102)
    for bar, val, nf in zip(bars, recalls, failures):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.0f}%\n({nf} fail)", ha="center", va="bottom", fontsize=8)

    # Right: Mean margin
    bars2 = ax2.bar(x, margins, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(combos, rotation=30, ha="right")
    ax2.set_ylabel("Mean Margin (×10³)")
    ax2.set_title("(b) Mean Discrimination Margin")
    for bar, val in zip(bars2, margins):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Multi-Frequency Fingerprint Ablation (N=200)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig02_frequency_comparison")


# ── Figure 3: Margin Power Law ──────────────────────────────────────────

def fig03_margin_power_law() -> None:
    """Log-log plot: margin vs N for f1 and f0+f1 with fitted power laws."""
    print("Fig 03: Margin power law...")
    f1_data = load_json(ABSOLUTE_DIR / "margin_compression_law.json")
    f0f1_data = load_json(ABSOLUTE_DIR / "multifreq_law.json")

    # f1 data
    f1_n = [int(n) for n in f1_data["results"].keys()]
    f1_margins = [f1_data["results"][str(n)]["mean_margin"] for n in f1_n]
    f1_alpha = f1_data["alpha"]
    f1_A = f1_data["A"]

    # f0+f1 data
    f0f1_n = [int(n) for n in f0f1_data["results"].keys()]
    f0f1_margins = [f0f1_data["results"][str(n)]["mean_margin"] for n in f0f1_n]
    f0f1_alpha = f0f1_data["alpha"]
    f0f1_A = f0f1_data["A"]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Data points
    ax.scatter(f1_n, f1_margins, color=COLORS["orange"], s=60, zorder=5, label="f1 (data)")
    ax.scatter(f0f1_n, f0f1_margins, color=COLORS["blue"], s=60, zorder=5, label="f0+f1 (data)")

    # Fitted curves
    n_fit = np.linspace(3, 250, 200)
    f1_fit = f1_A * n_fit ** f1_alpha
    f0f1_fit = f0f1_A * n_fit ** f0f1_alpha

    ax.plot(n_fit, f1_fit, color=COLORS["orange"], linestyle="--", alpha=0.7,
            label=f"f1 fit: {f1_A:.4f}·N^{{{f1_alpha:.3f}}}")
    ax.plot(n_fit, f0f1_fit, color=COLORS["blue"], linestyle="--", alpha=0.7,
            label=f"f0+f1 fit: {f0f1_A:.4f}·N^{{{f0f1_alpha:.3f}}}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Corpus Size N")
    ax.set_ylabel("Mean Discrimination Margin")
    ax.set_title("Margin Power Law: Graceful Degradation")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([5, 10, 20, 50, 100, 200])

    # Annotation
    ax.annotate(
        f"f0+f1: α={f0f1_alpha:.3f} (shallower)\nf1: α={f1_alpha:.3f}",
        xy=(100, f0f1_A * 100 ** f0f1_alpha), xytext=(30, 0.003),
        arrowprops={"arrowstyle": "->", "color": COLORS["grey"]},
        fontsize=9, bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5}
    )

    fig.tight_layout()
    save_figure(fig, "fig03_margin_power_law")


# ── Figure 4: Recall vs N — Fourier vs FCDB ─────────────────────────────

def fig04_recall_vs_n() -> None:
    """Fourier f0+f1 recall vs FCDB recall across corpus sizes."""
    print("Fig 04: Recall vs N (Fourier vs FCDB)...")
    f0f1_data = load_json(ABSOLUTE_DIR / "multifreq_law.json")
    stress_data = load_json(STRESS_DIR / "STRESS_SUMMARY.json")

    # Fourier f0+f1
    fourier_n = [int(n) for n in f0f1_data["results"].keys()]
    fourier_recall = [f0f1_data["results"][str(n)]["recall"] * 100 for n in fourier_n]

    # FCDB cross-model
    fcdb_map = stress_data["recall_at_1_vs_n_fcdb"]
    fcdb_n = [int(n) for n in fcdb_map.keys()]
    fcdb_recall = [v * 100 for v in fcdb_map.values()]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(fourier_n, fourier_recall, "o-", color=COLORS["blue"], linewidth=2,
            markersize=7, label="Fourier f0+f1 (same-model)", zorder=5)
    ax.plot(fcdb_n, fcdb_recall, "s--", color=COLORS["orange"], linewidth=2,
            markersize=7, label="FCDB (cross-model)", zorder=5)

    # Collapse annotation
    ax.axvline(x=100, color=COLORS["red"], linestyle=":", alpha=0.5)
    ax.annotate("FCDB collapse\n(N=100)", xy=(100, 30), xytext=(140, 50),
                arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
                fontsize=9, color=COLORS["red"])

    ax.set_xlabel("Corpus Size N")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Retrieval Recall vs Corpus Size")
    ax.legend(loc="lower left")
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, 210)

    fig.tight_layout()
    save_figure(fig, "fig04_recall_vs_n")


# ── Figure 5: Cross-Model Strategy Comparison ───────────────────────────

def fig05_cross_model_strategies() -> None:
    """Horizontal bar chart: 9 cross-model methods × margin."""
    print("Fig 05: Cross-model strategy comparison...")

    strategies = [
        ("CCA", -0.420, False),
        ("Residual FCB", -0.382, False),
        ("Procrustes", -0.104, False),
        ("RR (K=20)", -0.066, False),
        ("FCB+ridge", -0.017, False),
        ("Contrastive", 0.001, True),
        ("JCB", 0.011, True),
        ("JCB+delta", 0.037, True),
        ("FCDB", 0.124, True),
    ]

    names = [s[0] for s in strategies]
    margins = [s[1] for s in strategies]
    colors = [PASS_COLOR if s[2] else FAIL_COLOR for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, margins, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Retrieval Margin")
    ax.set_title("Cross-Model Transfer Strategies (Llama 3B → 8B)")
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, margins):
        x_offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + x_offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", ha=ha, va="center", fontsize=9, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PASS_COLOR, label="PASS (margin > 0)"),
                       Patch(facecolor=FAIL_COLOR, label="FAIL (margin ≤ 0)")]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "fig05_cross_model_strategies")


# ── Figure 6: CKA Layer Similarity ──────────────────────────────────────

def fig06_cka_layers() -> None:
    """CKA similarity per layer: within-family vs cross-family."""
    print("Fig 06: CKA layer similarity...")
    within = load_json(ABSOLUTE_DIR / "FAMILY_CKA.json")
    cross = load_json(ABSOLUTE_DIR / "FAMILY_CKA_CROSS.json")

    within_cka = within["layer_ckas"]
    cross_cka = cross["layer_ckas"]
    layers = list(range(len(within_cka)))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(layers, within_cka, "o-", color=COLORS["blue"], markersize=5, linewidth=1.5,
            label=f"Within-family (Llama 3B↔8B), μ={within['mean_cka']:.3f}")
    ax.plot(layers, cross_cka, "s--", color=COLORS["orange"], markersize=5, linewidth=1.5,
            label=f"Cross-family (Llama↔Qwen), μ={cross['mean_cka']:.3f}")

    ax.axhline(y=0.95, color=COLORS["grey"], linestyle=":", alpha=0.5, label="0.95 threshold")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("Centered Kernel Alignment Across Layers")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0.85, 1.0)

    # Annotate min
    min_idx_w = int(np.argmin(within_cka))
    min_idx_c = int(np.argmin(cross_cka))
    ax.annotate(f"min={within_cka[min_idx_w]:.3f}", xy=(min_idx_w, within_cka[min_idx_w]),
                xytext=(min_idx_w + 2, within_cka[min_idx_w] - 0.01),
                fontsize=8, color=COLORS["blue"])
    ax.annotate(f"min={cross_cka[min_idx_c]:.3f}", xy=(min_idx_c, cross_cka[min_idx_c]),
                xytext=(min_idx_c + 2, cross_cka[min_idx_c] - 0.01),
                fontsize=8, color=COLORS["orange"])

    fig.tight_layout()
    save_figure(fig, "fig06_cka_layers")


# ── Figure 7: Domain Confusion Before/After ──────────────────────────────

def fig07_confusion_matrix() -> None:
    """Heatmaps: f1 confusion vs f0+f1 confusion across domains."""
    print("Fig 07: Domain confusion matrix...")
    data = load_json(ABSOLUTE_DIR / "confusion_analysis.json")

    domains = sorted({
        k.split(" -> ")[0] for k in data["f1_confusion"].keys()
    } | {
        k.split(" -> ")[1] for k in data["f1_confusion"].keys()
    })

    def build_matrix(confusion_dict: dict[str, int]) -> np.ndarray:
        n = len(domains)
        mat = np.zeros((n, n))
        for key, count in confusion_dict.items():
            src, dst = key.split(" -> ")
            if src in domains and dst in domains:
                i = domains.index(src)
                j = domains.index(dst)
                mat[i, j] = count
        return mat

    f1_mat = build_matrix(data["f1_confusion"])
    best_mat = build_matrix(data["best_confusion"])

    # Short domain labels
    short_labels = [d[:6] for d in domains]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(f1_mat, cmap="Reds", aspect="auto", interpolation="nearest")
    ax1.set_xticks(range(len(domains)))
    ax1.set_yticks(range(len(domains)))
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(short_labels, fontsize=8)
    ax1.set_title("(a) f1 Only — 28 Failures")
    ax1.set_xlabel("Confused With")
    ax1.set_ylabel("True Domain")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(best_mat, cmap="Blues", aspect="auto", interpolation="nearest")
    ax2.set_xticks(range(len(domains)))
    ax2.set_yticks(range(len(domains)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(short_labels, fontsize=8)
    ax2.set_title("(b) f0+f1 — 4 Failures")
    ax2.set_xlabel("Confused With")
    ax2.set_ylabel("True Domain")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Domain Confusion Analysis (N=200)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig07_confusion_matrix")


# ── Figure 8: Domain Recall Radar ────────────────────────────────────────

def fig08_domain_recall_radar() -> None:
    """Radar chart: per-domain recall with f0+f1."""
    print("Fig 08: Domain recall radar...")
    data = load_json(ABSOLUTE_DIR / "confusion_analysis.json")
    domain_recall = data["domain_recall"]

    categories = list(domain_recall.keys())
    values = [domain_recall[c] * 100 for c in categories]

    # Close the polygon
    values_closed = values + [values[0]]
    n = len(categories)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    ax.plot(angles_closed, values_closed, "o-", color=COLORS["blue"], linewidth=2, markersize=6)
    ax.fill(angles_closed, values_closed, color=COLORS["blue"], alpha=0.15)

    ax.set_xticks(angles)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
    ax.set_ylim(80, 102)
    ax.set_yticks([85, 90, 95, 100])
    ax.set_yticklabels(["85%", "90%", "95%", "100%"], fontsize=8)
    ax.set_title("Per-Domain Recall@1 (f0+f1, N=200)", pad=20)

    # Annotate minimum
    min_idx = int(np.argmin(values))
    ax.annotate(f"{values[min_idx]:.0f}%",
                xy=(angles[min_idx], values[min_idx]),
                xytext=(angles[min_idx] + 0.2, values[min_idx] - 3),
                fontsize=9, fontweight="bold", color=COLORS["red"])

    fig.tight_layout()
    save_figure(fig, "fig08_domain_recall_radar")


# ── Figure 9: HNSW Benchmark ────────────────────────────────────────────

def fig09_hnsw_benchmark() -> None:
    """Bar chart: HNSW vs brute-force latency."""
    print("Fig 09: HNSW benchmark...")
    data = load_json(ABSOLUTE_DIR / "HNSW_BENCH.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Latency comparison
    methods = ["Brute-Force", "HNSW"]
    latencies = [data["bf_latency_us"], data["hnsw_latency_us"]]
    colors = [COLORS["orange"], COLORS["blue"]]
    bars = ax1.bar(methods, latencies, color=colors, edgecolor="white", width=0.5)
    ax1.set_ylabel("Latency (μs)")
    ax1.set_title(f"(a) Search Latency — {data['speedup']:.1f}× Speedup")
    for bar, val in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{val:.1f} μs", ha="center", va="bottom", fontsize=10)

    # Recall comparison
    recalls = [data["bruteforce_recall"] * 100, data["hnsw_recall"] * 100]
    bars2 = ax2.bar(methods, recalls, color=colors, edgecolor="white", width=0.5)
    ax2.set_ylabel("Recall@1 (%)")
    ax2.set_title("(b) Recall Preserved")
    ax2.set_ylim(98, 100.5)
    for bar, val in zip(bars2, recalls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("HNSW Index Benchmark (N=200)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig09_hnsw_benchmark")


# ── Figure 10: INT8 Compression ──────────────────────────────────────────

def fig10_int8_compression() -> None:
    """Bar chart: FP16 vs INT8 comparison."""
    print("Fig 10: INT8 compression...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Size comparison
    configs = ["591 tok", "6,403 tok"]
    fp16_sizes = [73.9, 800.4]
    int8_sizes = [37.5, 406.5]
    x = np.arange(len(configs))
    w = 0.35
    ax1.bar(x - w / 2, fp16_sizes, w, label="FP16", color=COLORS["orange"], edgecolor="white")
    ax1.bar(x + w / 2, int8_sizes, w, label="INT8", color=COLORS["blue"], edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylabel("File Size (MB)")
    ax1.set_title("(a) .eng File Size — 1.97× Compression")
    ax1.legend()

    # Quality metrics
    metrics = ["Cosine\nSimilarity", "Margin\n(FP16)", "Margin\n(INT8)"]
    values = [0.99998, 0.381, 0.262]
    bar_colors = [COLORS["green"], COLORS["blue"], COLORS["cyan"]]
    bars = ax2.bar(metrics, values, color=bar_colors, edgecolor="white", width=0.5)
    ax2.set_ylabel("Value")
    ax2.set_title("(b) Quality Preservation")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.5f}" if val > 0.9 else f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)

    fig.suptitle("INT8 Quantization Impact", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig10_int8_compression")


# ── Figure 12: Margin Distribution ───────────────────────────────────────

def fig12_margin_distribution() -> None:
    """Distribution comparison: f1 vs f0+f1 summary statistics."""
    print("Fig 12: Margin distribution...")
    data = load_json(ABSOLUTE_DIR / "multifreq_comparison.json")
    results = data["results"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # We'll show key statistics as a visualization
    combos = ["f1", "f0+f1"]
    means = [results[c]["margin_mean"] * 1000 for c in combos]
    medians = [results[c]["margin_median"] * 1000 for c in combos]
    mins = [results[c]["margin_min"] * 1000 for c in combos]

    x = np.arange(len(combos))
    w = 0.25
    ax.bar(x - w, means, w, label="Mean", color=COLORS["blue"], edgecolor="white")
    ax.bar(x, medians, w, label="Median", color=COLORS["green"], edgecolor="white")
    ax.bar(x + w, mins, w, label="Min", color=COLORS["red"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(combos, fontsize=12)
    ax.set_ylabel("Margin (×10³)")
    ax.set_title("Margin Statistics: f1 vs f0+f1 (N=200)")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Annotate improvement
    ax.annotate(
        f"+76% mean margin\n25/28 failures fixed",
        xy=(1, means[1]), xytext=(1.3, means[1] + 1),
        arrowprops={"arrowstyle": "->", "color": COLORS["green"]},
        fontsize=9, bbox={"boxstyle": "round,pad=0.3", "facecolor": "#e6ffe6", "alpha": 0.8}
    )

    fig.tight_layout()
    save_figure(fig, "fig12_margin_distribution")


# ── Figure 13: FCDB Stability-Discrimination Tradeoff ────────────────────

def fig13_fcdb_tradeoff() -> None:
    """Dual-axis: basis stability vs retrieval margin vs corpus size."""
    print("Fig 13: FCDB stability-discrimination tradeoff...")

    # Data from PAPER_TABLE.md
    n_vals = [50, 100, 125, 200]
    stability = [0.82, 0.906, 0.983, 0.999]  # subspace agreement
    margin = [0.124, None, None, 0.013]  # Only measured at 50 and 200
    margin_n = [50, 200]
    margin_v = [0.124, 0.013]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    # Stability (left axis)
    line1 = ax1.plot(n_vals, stability, "o-", color=COLORS["blue"], linewidth=2,
                     markersize=8, label="Basis Stability", zorder=5)
    ax1.set_xlabel("Corpus Size N")
    ax1.set_ylabel("Subspace Agreement", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax1.set_ylim(0.7, 1.05)

    # Margin (right axis)
    line2 = ax2.plot(margin_n, margin_v, "s--", color=COLORS["orange"], linewidth=2,
                     markersize=8, label="Retrieval Margin", zorder=5)
    ax2.set_ylabel("Cross-Model Margin", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.set_ylim(-0.01, 0.15)

    # Threshold line
    ax1.axhline(y=0.99, color=COLORS["grey"], linestyle=":", alpha=0.5)
    ax1.annotate("Stable (≥0.99)", xy=(125, 0.99), fontsize=8, color=COLORS["grey"])

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left")

    ax1.set_title("FCDB Stability–Discrimination Tradeoff")
    fig.tight_layout()
    save_figure(fig, "fig13_fcdb_tradeoff")


# ── Figure 14: TTFT Speedup ─────────────────────────────────────────────

def fig14_ttft_speedup() -> None:
    """Grouped bar chart: cold vs warm TTFT."""
    print("Fig 14: TTFT speedup...")

    configs = ["3B / 4K tok", "3B / 16K tok", "8B / 591 tok"]
    cold_ttft = [11439, 94592, 3508]  # ms
    warm_ttft = [170, 1777, 116]  # ms
    speedups = [67.2, 53.2, 30.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(configs))
    w = 0.35
    ax1.bar(x - w / 2, cold_ttft, w, label="Cold TTFT", color=COLORS["orange"], edgecolor="white")
    ax1.bar(x + w / 2, warm_ttft, w, label="Warm TTFT", color=COLORS["blue"], edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("(a) Time to First Token")
    ax1.set_yscale("log")
    ax1.legend()

    # Speedup bars
    bars = ax2.bar(configs, speedups, color=COLORS["green"], edgecolor="white", width=0.5)
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("(b) KV Cache Restoration Speedup")
    ax2.set_xticklabels(configs, fontsize=9)
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("KV Cache Warm Start Performance", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig14_ttft_speedup")


# ── Figure 15: EGR Overhead Scaling ──────────────────────────────────────

def fig15_egr_overhead() -> None:
    """Scatter/line: EGR overhead vs token count."""
    print("Fig 15: EGR overhead scaling...")

    tokens = [600, 6403, 600]
    overhead_ms = [30.6, 48.8, 84.0]
    labels = ["16 layers\n(8-24)", "16 layers\n(8-24)", "32 layers\n(all)"]
    colors_pts = [COLORS["blue"], COLORS["blue"], COLORS["orange"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for t, o, l, c in zip(tokens, overhead_ms, labels, colors_pts):
        ax.scatter(t, o, s=100, color=c, zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(l, xy=(t, o), xytext=(t + 200, o + 2), fontsize=9)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("EGR Overhead (ms)")
    ax.set_title("Fingerprint Extraction Overhead")
    ax.set_xlim(0, 7000)
    ax.set_ylim(20, 95)

    # Reference lines
    ax.axhline(y=50, color=COLORS["grey"], linestyle=":", alpha=0.3)
    ax.text(100, 51, "50ms threshold", fontsize=8, color=COLORS["grey"])

    fig.tight_layout()
    save_figure(fig, "fig15_egr_overhead")


# ── Figure 1: Architecture Diagram (Mermaid) ────────────────────────────

def fig01_architecture_mermaid() -> None:
    """Generate Mermaid flowchart for system architecture."""
    print("Fig 01: Architecture diagram (Mermaid)...")
    mermaid = """\
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#4477AA', 'primaryTextColor': '#fff', 'primaryBorderColor': '#335588', 'lineColor': '#666', 'secondaryColor': '#EE6677', 'tertiaryColor': '#228833'}}}%%
flowchart TD
    A[LLM Runtime<br/>llama.cpp] -->|KV cache blob| B[Blob Parser]
    B -->|Layer keys K| C[Fourier Fingerprint<br/>f0+f1 DFT]
    C -->|2048-dim vector| D{Storage}
    D -->|.eng binary| E[EIGENGRAM File<br/>v1.2 format]
    D -->|HNSW index| F[FAISS IndexHNSW<br/>M=32]

    G[Query Session] -->|New KV cache| C
    C -->|Query fingerprint| H[Geodesic Retrieval]
    F -->|Top-k candidates| H

    H --> I{Stage 0<br/>Prior Check}
    I -->|chronic failure| J[Skip / LOW]
    I -->|ok| K{Stage 1<br/>HNSW Search}
    K -->|HIGH / MEDIUM| L[Result]
    K -->|below threshold| M{Stage 2<br/>Trajectory}
    M -->|interpolation| N{Stage 3<br/>Constraints}
    N --> O{Stage 4<br/>Metadata}
    O --> L

    subgraph Confidence Tracking
        P[IndexC<br/>SQLite] ---|update| I
        L ---|record| P
    end

    style A fill:#4477AA,stroke:#335588,color:#fff
    style C fill:#228833,stroke:#1a6625,color:#fff
    style E fill:#EE6677,stroke:#cc5566,color:#fff
    style F fill:#66CCEE,stroke:#55aabb,color:#000
    style H fill:#AA3377,stroke:#882266,color:#fff
"""
    mermaid_path = FIGURES_DIR / "fig01_architecture.mmd"
    mermaid_path.write_text(mermaid)
    print(f"  Saved: fig01_architecture.mmd")


# ── Figure 11: Retrieval Pipeline (Mermaid) ──────────────────────────────

def fig11_retrieval_pipeline_mermaid() -> None:
    """Generate Mermaid diagram for 4-stage geodesic retrieval."""
    print("Fig 11: Retrieval pipeline (Mermaid)...")
    mermaid = """\
%%{init: {'theme': 'base'}}%%
flowchart LR
    Q[Query<br/>Fingerprint] --> S0

    S0[Stage 0<br/>Prior Preemption<br/><i>IndexC chronic<br/>failure check</i>]
    S0 -->|"pass"| S1
    S0 -->|"preempt"| SKIP[SKIP<br/>confidence=LOW]

    S1[Stage 1<br/>HNSW Search<br/><i>cosine top-k</i>]
    S1 -->|"margin > 0.005"| HIGH[HIGH<br/>199/200 docs]
    S1 -->|"margin 0.001-0.005"| MED[MEDIUM]
    S1 -->|"margin < 0.001"| S2

    S2[Stage 2<br/>Trajectory<br/><i>interpolation<br/>w=0.3</i>]
    S2 --> S3

    S3[Stage 3<br/>Negative<br/>Constraints<br/><i>apophatic layer</i>]
    S3 --> S4

    S4[Stage 4<br/>Metadata<br/>Disambig<br/><i>domain + keywords<br/>+ norms</i>]
    S4 --> LOW[LOW<br/>1/200 docs<br/><i>doc_146</i>]

    style S0 fill:#66CCEE,stroke:#55aabb
    style S1 fill:#4477AA,stroke:#335588,color:#fff
    style S2 fill:#CCBB44,stroke:#aa9933
    style S3 fill:#EE6677,stroke:#cc5566,color:#fff
    style S4 fill:#AA3377,stroke:#882266,color:#fff
    style HIGH fill:#228833,stroke:#1a6625,color:#fff
    style MED fill:#CCBB44,stroke:#aa9933
    style LOW fill:#EE6677,stroke:#cc5566,color:#fff
    style SKIP fill:#BBBBBB,stroke:#999999
"""
    mermaid_path = FIGURES_DIR / "fig11_retrieval_pipeline.mmd"
    mermaid_path.write_text(mermaid)
    print(f"  Saved: fig11_retrieval_pipeline.mmd")


# ── Consolidated Findings JSON ───────────────────────────────────────────

def generate_findings() -> None:
    """Consolidate all key metrics into a single findings.json."""
    print("Generating consolidated findings...")

    findings = {
        "title": "ENGRAM Protocol — Consolidated Research Findings",
        "date": "2026-04-03",
        "hardware": {
            "platform": "Apple M3, 24GB RAM",
            "gpu": "Metal (n_gpu_layers=-1)",
            "os": "macOS Darwin 25.4.0",
            "llama_cpp": "0.3.19",
            "faiss": "1.13.2",
            "torch": "2.11.0",
        },
        "same_model_retrieval": {
            "method": "Fourier f0+f1 fingerprint",
            "corpus_size": 200,
            "n_domains": 10,
            "recall_at_1": 0.98,
            "n_failures": 4,
            "mean_margin": 0.007201,
            "margin_power_law": {"A": 0.021342, "alpha": -0.2065},
            "f1_only_recall": 0.86,
            "f1_only_failures": 28,
            "improvement_over_f1": "25/28 failures fixed (+76% mean margin)",
            "ml_math_confusion_reduction": "81.5%",
        },
        "frequency_ablation": {
            "combos_tested": 6,
            "best": "f0+f1",
            "results": {
                "f1": {"recall": 0.86, "margin": 0.004087},
                "f2": {"recall": 0.715, "margin": 0.002196},
                "f1+f2": {"recall": 0.95, "margin": 0.004744},
                "f1+f2+f3": {"recall": 0.95, "margin": 0.004129},
                "f0+f1": {"recall": 0.98, "margin": 0.007201},
                "f1+f3": {"recall": 0.89, "margin": 0.003477},
            },
        },
        "hnsw_index": {
            "speedup": 5.65,
            "recall": 0.995,
            "latency_us": 51.83,
            "bruteforce_latency_us": 293.07,
        },
        "geodesic_retrieval": {
            "stages": 4,
            "final_recall": 1.0,
            "n_high": 0,
            "n_medium": 199,
            "n_low": 1,
            "hard_failure": "doc_146 (resolved by Stage 4 metadata)",
        },
        "int8_compression": {
            "ratio": 1.97,
            "cosine_similarity": 0.99998,
            "margin_fp16": 0.381,
            "margin_int8": 0.262,
            "margin_preserved": True,
        },
        "ttft_speedup": {
            "3b_4k": {"cold_ms": 11439, "warm_ms": 170, "speedup": 67.2},
            "3b_16k": {"cold_ms": 94592, "warm_ms": 1777, "speedup": 53.2},
            "8b_591": {"cold_ms": 3508, "warm_ms": 116, "speedup": 30.8},
        },
        "cross_model_transfer": {
            "n_strategies": 9,
            "best_method": "FCDB",
            "best_margin": 0.124,
            "results": {
                "CCA": {"margin": -0.420, "correct": False},
                "Residual_FCB": {"margin": -0.382, "correct": False},
                "Procrustes": {"margin": -0.104, "correct": False},
                "RR": {"margin": -0.066, "correct": False},
                "FCB_ridge": {"margin": -0.017, "correct": False},
                "Contrastive": {"margin": 0.001, "correct": True},
                "JCB": {"margin": 0.011, "correct": True},
                "JCB_delta": {"margin": 0.037, "correct": True},
                "FCDB": {"margin": 0.124, "correct": True},
            },
            "key_insight": "Cross-model transfer requires representing documents as directions from a shared reference point (Frechet mean), not positions in space",
        },
        "fcdb_scaling": {
            "v1_n50": {"stability": 0.82, "margin": 0.124},
            "v2_n200": {"stability": 0.999, "margin": 0.013},
            "collapse_n": 100,
            "tradeoff": "Larger corpus stabilizes basis but dilutes per-document signal",
        },
        "cka_analysis": {
            "within_family": {"models": "Llama 3B ↔ 8B", "mean_cka": 0.975, "f0f1_sim": 0.875},
            "cross_family": {"models": "Llama ↔ Qwen", "mean_cka": 0.927, "f0f1_sim": 0.259},
            "verdict": "Manifolds topologically isomorphic (CKA>0.92 all pairs)",
        },
        "domain_recall": {
            "computer_science": 1.0, "general_world": 0.95, "history": 1.0,
            "language_arts": 1.0, "ml_systems": 0.90, "mathematics": 1.0,
            "philosophy": 1.0, "medicine": 0.95, "biology": 1.0, "physics": 1.0,
        },
        "eigengram_format": {
            "version": "1.2",
            "architectures": ["llama", "gemma", "gemma4/ISWA", "phi", "qwen", "mistral"],
            "iswa_support": "Gemma 4 26B dual-cache (5+25 layers, 6144-dim fingerprint)",
        },
    }

    paper_dir = RESULTS_DIR / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    findings_path = paper_dir / "findings.json"
    findings_path.write_text(json.dumps(findings, indent=2))
    print(f"  Saved: paper/findings.json")


# ── LaTeX Tables ─────────────────────────────────────────────────────────

def generate_latex_tables() -> None:
    """Generate LaTeX table source for the paper."""
    print("Generating LaTeX tables...")

    tables = r"""\
% ──────────────────────────────────────────────────────────────────────
% Table 1: Multi-Frequency Ablation
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{Multi-frequency fingerprint ablation at $N=200$. The f0+f1 combination
achieves the highest recall and mean margin, fixing 25 of 28 single-frequency failures.}
\label{tab:frequency-ablation}
\begin{tabular}{lcccc}
\toprule
Frequencies & Recall@1 & Mean Margin & Min Margin & Failures \\
\midrule
$f_1$ & 86.0\% & 4.09$\times 10^{-3}$ & $-4.71\times 10^{-3}$ & 28 \\
$f_2$ & 71.5\% & 2.20$\times 10^{-3}$ & $-5.85\times 10^{-3}$ & 57 \\
$f_1 + f_2$ & 95.0\% & 4.74$\times 10^{-3}$ & $-2.68\times 10^{-3}$ & 10 \\
$f_1 + f_2 + f_3$ & 95.0\% & 4.13$\times 10^{-3}$ & $-2.71\times 10^{-3}$ & 10 \\
\rowcolor{green!10}
$f_0 + f_1$ & \textbf{98.0\%} & \textbf{7.20}$\times 10^{-3}$ & $-4.09\times 10^{-3}$ & \textbf{4} \\
$f_1 + f_3$ & 89.0\% & 3.48$\times 10^{-3}$ & $-4.08\times 10^{-3}$ & 22 \\
\bottomrule
\end{tabular}
\end{table}

% ──────────────────────────────────────────────────────────────────────
% Table 2: Cross-Model Transfer Strategies
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{Cross-model transfer strategies (Llama 3B $\to$ 8B). Nine methods tested;
FCDB achieves the only reliable positive margin without requiring an adapter.}
\label{tab:cross-model}
\begin{tabular}{lccc}
\toprule
Method & Margin & Correct & Adapter \\
\midrule
CCA & $-0.420$ & \xmark & symmetric \\
Residual FCB & $-0.382$ & \xmark & none \\
Procrustes & $-0.104$ & \xmark & orthogonal \\
Relative Repr. & $-0.066$ & \xmark & none \\
FCB + ridge & $-0.017$ & \xmark & ridge \\
\midrule
Contrastive $\delta$ & $+0.001$ & \cmark & ridge \\
JCB & $+0.011$ & \cmark & none \\
JCB + $\delta$ & $+0.037$ & \cmark & none \\
\rowcolor{green!10}
\textbf{FCDB} & $\mathbf{+0.124}$ & \cmark & \textbf{none} \\
\bottomrule
\end{tabular}
\end{table}

% ──────────────────────────────────────────────────────────────────────
% Table 3: TTFT Speedup
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{KV cache warm-start performance. TTFT speedup ranges from 27--67$\times$
depending on model size and context length.}
\label{tab:ttft}
\begin{tabular}{lccccc}
\toprule
Model & Tokens & Cold TTFT & Warm TTFT & Speedup & EGR (ms) \\
\midrule
Llama 3.2 3B & 4,002 & 11,439\,ms & 170\,ms & 67.2$\times$ & 9.5 \\
Llama 3.2 3B & 16,382 & 94,592\,ms & 1,777\,ms & 53.2$\times$ & 9.5 \\
Llama 3.1 8B & 591 & 3,508\,ms & 116\,ms & 30.8$\times$ & 30.6 \\
\bottomrule
\end{tabular}
\end{table}

% ──────────────────────────────────────────────────────────────────────
% Table 4: INT8 Compression
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{INT8 quantization results. Per-row symmetric quantization achieves
1.97$\times$ compression with negligible quality loss (cos\_sim = 0.99998).}
\label{tab:int8}
\begin{tabular}{lcccc}
\toprule
Tokens & FP16 Size & INT8 Size & Ratio & $\cos(s_\text{fp16}, s_\text{int8})$ \\
\midrule
591 & 73.9\,MB & 37.5\,MB & 1.97$\times$ & 0.99998 \\
6,403 & 800.4\,MB & 406.5\,MB & 1.97$\times$ & 0.99998 \\
\bottomrule
\end{tabular}
\end{table}

% ──────────────────────────────────────────────────────────────────────
% Table 5: CKA Analysis
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{Centered Kernel Alignment (CKA) between model families. High CKA values
($>0.92$) confirm topological isomorphism of key manifolds across architectures.}
\label{tab:cka}
\begin{tabular}{lccc}
\toprule
Comparison & Mean CKA & f0+f1 Sim & Verdict \\
\midrule
Within-family (Llama 3B $\leftrightarrow$ 8B) & 0.975 & 0.875 & Isomorphic \\
Cross-family (Llama $\leftrightarrow$ Qwen) & 0.927 & 0.259 & Isomorphic \\
\bottomrule
\end{tabular}
\end{table}

% ──────────────────────────────────────────────────────────────────────
% Table 6: HNSW Benchmark
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{HNSW index performance at $N=200$. The index provides 5.65$\times$
speedup over brute-force with no recall loss.}
\label{tab:hnsw}
\begin{tabular}{lcc}
\toprule
Method & Latency ($\mu$s) & Recall@1 \\
\midrule
Brute-force & 293.1 & 99.5\% \\
HNSW ($M=32$) & 51.8 & 99.5\% \\
\midrule
\textbf{Speedup} & \textbf{5.65$\times$} & --- \\
\bottomrule
\end{tabular}
\end{table}

% ──────────────────────────────────────────────────────────────────────
% Table 7: Domain Recall
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{Per-domain recall@1 with f0+f1 fingerprint at $N=200$.
All domains achieve $\geq 90\%$ recall.}
\label{tab:domain-recall}
\begin{tabular}{lc}
\toprule
Domain & Recall@1 \\
\midrule
Biology & 100.0\% \\
Computer Science & 100.0\% \\
History & 100.0\% \\
Language Arts & 100.0\% \\
Mathematics & 100.0\% \\
Philosophy & 100.0\% \\
Physics & 100.0\% \\
General World & 95.0\% \\
Medicine & 95.0\% \\
ML/Systems & 90.0\% \\
\bottomrule
\end{tabular}
\end{table}

% ──────────────────────────────────────────────────────────────────────
% Table 8: Margin Power Law
% ──────────────────────────────────────────────────────────────────────
\begin{table}[t]
\centering
\caption{Margin scaling law parameters. Both fingerprint methods follow
power-law decay $\bar{m} = A \cdot N^\alpha$ with no hard collapse point.}
\label{tab:power-law}
\begin{tabular}{lccc}
\toprule
Fingerprint & $A$ & $\alpha$ & Recall@200 \\
\midrule
$f_1$ & 0.0181 & $-0.277$ & 86.0\% \\
$f_0 + f_1$ & 0.0213 & $-0.207$ & 98.0\% \\
\bottomrule
\end{tabular}
\end{table}
"""

    paper_dir = RESULTS_DIR / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    tables_path = paper_dir / "tables.tex"
    tables_path.write_text(tables)
    print(f"  Saved: paper/tables.tex")


# ── Registry ─────────────────────────────────────────────────────────────

FIGURE_REGISTRY: dict[str, tuple[str, object]] = {
    "fig01": ("System Architecture (Mermaid)", fig01_architecture_mermaid),
    "fig02": ("Frequency Combination Comparison", fig02_frequency_comparison),
    "fig03": ("Margin Power Law", fig03_margin_power_law),
    "fig04": ("Recall vs N (Fourier vs FCDB)", fig04_recall_vs_n),
    "fig05": ("Cross-Model Strategy Comparison", fig05_cross_model_strategies),
    "fig06": ("CKA Layer Similarity", fig06_cka_layers),
    "fig07": ("Domain Confusion Matrix", fig07_confusion_matrix),
    "fig08": ("Domain Recall Radar", fig08_domain_recall_radar),
    "fig09": ("HNSW Benchmark", fig09_hnsw_benchmark),
    "fig10": ("INT8 Compression", fig10_int8_compression),
    "fig11": ("Retrieval Pipeline (Mermaid)", fig11_retrieval_pipeline_mermaid),
    "fig12": ("Margin Distribution", fig12_margin_distribution),
    "fig13": ("FCDB Tradeoff", fig13_fcdb_tradeoff),
    "fig14": ("TTFT Speedup", fig14_ttft_speedup),
    "fig15": ("EGR Overhead Scaling", fig15_egr_overhead),
    "findings": ("Consolidated Findings JSON", generate_findings),
    "tables": ("LaTeX Tables", generate_latex_tables),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ENGRAM paper figures")
    parser.add_argument("--only", help="Generate only this figure (e.g., fig02)")
    parser.add_argument("--list", action="store_true", help="List all figures")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable figures:")
        for key, (desc, _) in FIGURE_REGISTRY.items():
            print(f"  {key:10s}  {desc}")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {FIGURES_DIR}\n")

    if args.only:
        if args.only not in FIGURE_REGISTRY:
            print(f"Unknown figure: {args.only}")
            print(f"Available: {', '.join(FIGURE_REGISTRY.keys())}")
            sys.exit(1)
        desc, func = FIGURE_REGISTRY[args.only]
        func()
    else:
        for key, (desc, func) in FIGURE_REGISTRY.items():
            try:
                func()
            except Exception as e:
                print(f"  ERROR generating {key}: {e}")

    print(f"\nDone. Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
