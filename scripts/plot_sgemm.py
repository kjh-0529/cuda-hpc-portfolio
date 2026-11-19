"""Plot SGEMM results: % of cuBLAS per optimization step, as a horizontal bar chart.

Usage:
    python scripts/plot_sgemm.py [csv] [--size 2048]

Reads the benchmark CSV (default: 01-sgemm/results/kaggle-t4.csv) and writes
01-sgemm/results/sgemm_steps.png and sgemm_steps_dark.png (for GitHub dark mode).
Requires matplotlib.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984",
                  grid="#e6e5e0", bar="#2a78d6"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8f8e88",
                 grid="#33332f", bar="#3987e5"),
}

STEP_NAMES = {
    "01_naive": "01  Naive",
    "02_coalesced": "02  Coalesced access",
    "03_smem_tiling": "03  Shared-memory tiling",
    "04_register_blocking": "04  Register blocking",
    "05_vectorized": "05  float4 loads",
    "06_double_buffering": "06  Double buffering",
}


def load_rows(path, size):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["size"]) == size and r["kernel"] != "cublas"]
    if not rows:
        raise SystemExit(f"no kernel rows for size {size} in {path}")
    return rows


def rounded_bar(ax, y, width, height, color, radius_px, fig):
    """Bar with a rounded data end and a square baseline end."""
    # Convert the radius from pixels to data units on each axis.
    x0, x1 = ax.get_xlim()
    bbox = ax.get_window_extent()
    rx = radius_px * (x1 - x0) / bbox.width
    y0, y1 = ax.get_ylim()
    ry = radius_px * abs(y1 - y0) / bbox.height
    r = min(rx, width / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=ry / rx if rx else 1,
                                linewidth=0, facecolor=color))
    # Square off the baseline end.
    ax.add_patch(Rectangle((0, y - height / 2), min(width, 2 * r), height,
                           linewidth=0, facecolor=color))


def plot(rows, size, gpu, theme_name, out):
    t = THEMES[theme_name]
    labels = [STEP_NAMES.get(r["kernel"], r["kernel"]) for r in rows]
    pct = [float(r["pct_of_cublas"]) for r in rows]
    gflops = [float(r["gflops"]) for r in rows]
    speedup = gflops[-1] / gflops[0]
    best = max(range(len(pct)), key=lambda i: pct[i])

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    plt.rcParams["font.family"] = "DejaVu Sans"

    n = len(rows)
    ys = list(range(n))[::-1]  # first step at the top
    ax.set_xlim(0, 106)
    ax.set_ylim(-0.7, n - 0.3)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.12)

    for x in (25, 50, 75):
        ax.axvline(x, color=t["grid"], linewidth=0.8, zorder=0)
    ax.axvline(100, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(100, n - 0.38, "cuBLAS = 100%", color=t["text2"], fontsize=8.5,
            ha="center", va="bottom")

    bar_h = 0.36
    for y, p, g in zip(ys, pct, gflops):
        rounded_bar(ax, y, p, bar_h, t["bar"], 4, fig)
        label = f"{p:.1f}%  ·  {g:,.0f} GFLOPS"
        if p > 50:
            # Long bars: the label fits inside, clear of the 100% reference line.
            ax.text(p - 1.5, y, label, color="#ffffff", fontsize=9, va="center",
                    ha="right", fontweight="bold")
        else:
            ax.text(p + 1.2, y, label, color=t["text"], fontsize=9, va="center", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10, color=t["text"])
    for i, lbl in enumerate(ax.get_yticklabels()):
        if i == best:
            lbl.set_fontweight("bold")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5, color=t["text2"])
    ax.tick_params(axis="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.02, 0.94, f"Hand-written SGEMM reaches {pct[best]:.0f}% of cuBLAS in 6 steps",
             fontsize=13.5, fontweight="bold", color=t["text"], ha="left")
    fig.text(0.02, 0.875,
             f"FP32, {size}×{size}×{size}, {gpu}, sustained clocks · "
             f"{speedup:.0f}x faster than the naive kernel",
             fontsize=9.5, color=t["text2"], ha="left")
    fig.text(0.02, 0.03, "Each kernel compared with cuBLAS timed right before it (median of 5 batches).",
             fontsize=8, color=t["muted"], ha="left")

    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "01-sgemm/results/kaggle-t4.csv"))
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--gpu", default="Tesla T4")
    args = ap.parse_args()
    rows = load_rows(args.csv, args.size)
    outdir = ROOT / "01-sgemm/results"
    plot(rows, args.size, args.gpu, "light", outdir / "sgemm_steps.png")
    plot(rows, args.size, args.gpu, "dark", outdir / "sgemm_steps_dark.png")


if __name__ == "__main__":
    main()
