"""Compare the extension with PyTorch's own softmax and LayerNorm.

Usage (from 02-llm-kernels/torch_ext):
    python bench_torch.py [--out results/kaggle-t4.csv]

Timing matches the C++ benchmarks: GPU warm-up, then each time is the median of
5 batches of ~50 ms measured with CUDA events. Bandwidth counts x read once and
y written once.
"""
import argparse
import csv
import statistics
import sys

import torch
import torch.nn.functional as F

import llm_kernels

SHAPES = [(8192, 256), (4096, 1024), (4096, 4096), (2048, 8192), (1024, 32768)]


def time_ms(fn, target_ms=50.0, reps=5):
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); fn(); stop.record(); stop.synchronize()
    est = max(start.elapsed_time(stop), 1e-3)
    iters = max(1, min(10000, int(target_ms / est + 0.999)))
    per_call = []
    for _ in range(reps):
        start.record()
        for _ in range(iters):
            fn()
        stop.record(); stop.synchronize()
        per_call.append(start.elapsed_time(stop) / iters)
    return statistics.median(per_call)


def warm_up(ms=2000.0):
    x = torch.randn(4096, 4096, device="cuda")
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    elapsed = 0.0
    while elapsed < ms:
        start.record()
        for _ in range(10):
            torch.softmax(x, -1)
        stop.record(); stop.synchronize()
        elapsed += start.elapsed_time(stop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="also write the CSV to this file")
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}  PyTorch {torch.__version__}  CUDA {torch.version.cuda}",
          file=sys.stderr)
    warm_up()
    rows_out = []
    header = ["op", "rows", "cols", "impl", "ms", "gbps", "speedup_vs_torch"]
    print(",".join(header))
    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda") * 5
        w = 1 + 0.5 * torch.rand(cols, device="cuda")
        b = 0.5 * torch.rand(cols, device="cuda") - 0.25
        nbytes = 2 * x.numel() * 4
        cases = {
            "softmax": {
                "torch": lambda: torch.softmax(x, -1),
                "ours": lambda: llm_kernels.softmax(x),
            },
            "layer_norm": {
                "torch": lambda: F.layer_norm(x, (cols,), w, b, 1e-5),
                "ours": lambda: llm_kernels.layer_norm(x, w, b, 1e-5),
            },
        }
        for op, impls in cases.items():
            # Re-time torch right before ours, as the C++ benchmarks do with their baselines.
            t_torch = time_ms(impls["torch"])
            t_ours = time_ms(impls["ours"])
            for impl, t in (("torch", t_torch), ("ours", t_ours)):
                row = [op, rows, cols, impl, f"{t:.4f}", f"{nbytes / (t * 1e-3) / 1e9:.1f}",
                       f"{t_torch / t:.2f}"]
                rows_out.append(row)
                print(",".join(map(str, row)))
    if args.out:
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerows([header] + rows_out)


if __name__ == "__main__":
    main()
