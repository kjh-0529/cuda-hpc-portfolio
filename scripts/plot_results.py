"""Plot headline charts for projects 2-4 from their committed CSVs, in the style of plot_sgemm.py.

Usage:
    python scripts/plot_results.py

Writes 02-llm-kernels/results/softmax_steps.png, 03-ddp-training/results/ddp_throughput.png
and 04-memory-bandwidth/results/gpu_bandwidth.png. Requires matplotlib.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_sgemm import ROOT, THEMES, rounded_bar


def read(path):
    with open(ROOT / path, newline="") as f:
        return list(csv.DictReader(f))


def hbar(labels, values, texts, xmax, ref, ref_label, title, subtitle, footnote, out):
    """Horizontal bars, first row on top, with a dashed reference line at `ref`."""
    t = THEMES["light"]
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    n = len(labels)
    ys = list(range(n))[::-1]
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.12)

    ax.axvline(ref, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(ref, n - 0.38, ref_label, color=t["text2"], fontsize=8.5, ha="center", va="bottom")
    for y, v, txt in zip(ys, values, texts):
        rounded_bar(ax, y, v, 0.36, t["bar"], 4, fig)
        if v > xmax * 0.5:
            ax.text(v - xmax * 0.015, y, txt, color="#ffffff", fontsize=9, va="center",
                    ha="right", fontweight="bold")
        else:
            ax.text(v + xmax * 0.012, y, txt, color=t["text"], fontsize=9, va="center", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10, color=t["text"])
    ax.set_xticks([])
    ax.tick_params(axis="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.02, 0.94, title, fontsize=13.5, fontweight="bold", color=t["text"], ha="left")
    fig.text(0.02, 0.875, subtitle, fontsize=9.5, color=t["text2"], ha="left")
    fig.text(0.02, 0.03, footnote, fontsize=8, color=t["muted"], ha="left")
    fig.savefig(ROOT / out, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def softmax():
    names = {"01_naive": "01  Naive", "02_block_smem": "02  Block + shared mem",
             "03_warp_shuffle": "03  Warp shuffle", "04_online": "04  Online softmax",
             "05_auto": "05  Auto-select"}
    rows = [r for r in read("02-llm-kernels/softmax/results/kaggle-t4.csv")
            if r["rows"] == "4096" and r["cols"] == "4096" and r["kernel"] != "copy"]
    pct = [float(r["pct_of_copy"]) for r in rows]
    gbps = [float(r["eff_gbps"]) for r in rows]
    hbar([names[r["kernel"]] for r in rows], pct,
         [f"{p:.0f}%  ·  {g:.0f} GB/s" for p, g in zip(pct, gbps)], 110, 100,
         "memory copy = 100%",
         f"Softmax kernel matches memory-copy bandwidth ({pct[-1]:.1f}%)",
         f"FP32, 4096 rows × 4096 cols, Tesla T4 · {gbps[-1] / gbps[0]:.1f}x faster than naive",
         "Copy = a plain read+write of the same bytes, the ceiling for a memory-bound op.",
         "02-llm-kernels/results/softmax_steps.png")


def ddp():
    runs = {"1gpu_fp32": "1 GPU, FP32", "1gpu_fp16": "1 GPU, FP16",
            "2gpu_fp16": "2 GPUs, FP16", "2gpu_accum4_nosync": "2 GPUs, FP16, no_sync()"}
    rows = {r["run"]: r for r in read("03-ddp-training/results/kaggle-t4x2.csv")}
    tps = [float(rows[k]["tokens_per_s"]) / 1000 for k in runs]
    base, one = tps[0], tps[1]
    eff = tps[2] / (2 * one) * 100
    hbar(list(runs.values()), tps, [f"{v:.0f}k tok/s  ·  {v / base:.1f}x" for v in tps],
         max(tps) * 1.08, 2 * one, "2x one FP16 GPU",
         f"FP16 is {one / base:.1f}x faster; 2 GPUs scale at {eff:.0f}%",
         "Small GPT trained with PyTorch DDP + NCCL, 2x Tesla T4 · same convergence as 1 GPU",
         "Training throughput in tokens per second; multipliers are relative to 1 GPU FP32.",
         "03-ddp-training/results/ddp_throughput.png")


def bandwidth():
    peak = 320
    rows = read("04-memory-bandwidth/results/kaggle-t4.csv")
    pick = [r for r in rows if r["mode"] == "device" and r["tbsize"] in ("0", "1024")]
    names = {"cudaMemcpy": "cudaMemcpy", "copy": "copy", "mul": "scale",
             "add": "add", "triad": "triad", "dot": "dot (read-only)"}
    gbps = [float(r["gbps"]) for r in pick]
    first = [float(r["gbps"]) for r in rows if r["kernel"] == "first_touch_copy" and r["tbsize"] == "1024"][0]
    hbar([names[r["kernel"]] for r in pick], gbps,
         [f"{g:.0f} GB/s  ·  {g / peak * 100:.0f}%" for g in gbps], peak * 1.06, peak,
         f"peak {peak} GB/s",
         f"Hand-written kernels reach {min(gbps[1:]) / peak * 100:.0f}–{max(gbps) / peak * 100:.0f}% of peak, above cudaMemcpy",
         f"STREAM-style kernels, device memory, 1024 threads/block, Tesla T4 · managed-memory first touch: {first:.1f} GB/s",
         "Bytes moved per second; percentages are of the T4's theoretical 320 GB/s.",
         "04-memory-bandwidth/results/gpu_bandwidth.png")


if __name__ == "__main__":
    (ROOT / "02-llm-kernels/results").mkdir(exist_ok=True)
    softmax()
    ddp()
    bandwidth()
