"""Print a Markdown table from the JSON results written by train.py.

Usage: python summarize.py results/*.json
Scaling efficiency compares each multi-GPU run with the single-GPU run that has
the same precision and per-GPU micro-batch: tokens/s(N GPUs) / (N x tokens/s(1 GPU)).
"""
import json
import sys


def main(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        r["file"] = p.split("/")[-1]
        runs.append(r)
    runs.sort(key=lambda r: (r["amp"], r["gpus"], r["grad_accum"], r["file"]))
    single = {(r["amp"], r["micro_batch"]): r for r in runs if r["gpus"] == 1 and r["grad_accum"] == 1}
    print("| Run | GPUs | Precision | Global batch (tokens) | tokens/s | per GPU | MFU | Peak mem (GB) | Scaling eff. | Final val loss |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        base = single.get((r["amp"], r["micro_batch"]))
        eff = "—"
        if r["gpus"] > 1 and base:
            eff = f"{100 * r['tokens_per_s'] / (r['gpus'] * base['tokens_per_s']):.0f}%"
        print(f"| {r['file']} | {r['gpus']} | {r['amp']} | {r['global_batch_tokens']:,} | "
              f"{r['tokens_per_s']:,} | {r['tokens_per_s_per_gpu']:,} | {r['mfu_pct']}% | "
              f"{r['peak_mem_gb']} | {eff} | {r['final_val_loss']:.4f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
