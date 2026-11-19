# LLM Kernels: Softmax & LayerNorm

Memory-bandwidth-bound kernels used in every transformer layer, optimized step by step and later exposed to PyTorch as a C++/CUDA extension.

Status: softmax steps 01–03 implemented and measured on a Tesla T4.

## Softmax

Row-wise softmax over a `[rows x cols]` FP32 matrix:

```text
y[r, c] = exp(x[r, c] - max_r) / sum_c exp(x[r, c] - max_r)
```

Subtracting the row maximum keeps `exp()` from overflowing. Shapes cover short rows (256), typical hidden sizes (1024, 4096) and vocabulary-sized rows (32768).

### How it is measured

Softmax does very little math per byte, so speed is limited by memory bandwidth. The metric is **effective bandwidth**: bytes of `x` read once plus bytes of `y` written once, divided by time. The baseline is a device-to-device copy of the same size, the fastest possible way to read `x` and write `y` once, so `pct_of_copy` shows how close a kernel is to the memory limit.

Every kernel is checked against a double-precision CPU reference. Timing uses the same method as the SGEMM benchmark: GPU warm-up, median of 5 batches of about 50 ms, and the copy baseline re-timed right before every kernel.

### Results

GPU: Tesla T4 (sm_75) on Kaggle, CUDA 13.0. Raw data: [`softmax/results/kaggle-t4.csv`](softmax/results/kaggle-t4.csv). Cells show effective bandwidth in GB/s and, in parentheses, % of the copy baseline.

| Kernel | 8192×256 | 4096×1024 | 4096×4096 | 1024×32768 | Main change |
|---|---|---|---|---|---|
| copy (baseline) | 228 | 232 | 232 | 234 | reads x and writes y once |
| 01 naive | 40 (17%) | 38 (17%) | 38 (16%) | 20 (9%) | one thread per row |
| 02 block + shared memory | 92 (40%) | 223 (96%) | 167 (72%) | 117 (50%) | one block per row, coalesced, tree reduction |
| 03 warp shuffle | 123 (54%) | 243 (105%) | 170 (73%) | 117 (50%) | reductions in registers with `__shfl_xor_sync` |
| 04 online softmax | — | — | — | — | max and sum in one pass |
| 05 vectorized, row in registers | — | — | — | — | `float4`, read x once |

The copy reaches about 230 GB/s, roughly 72% of the T4's 320 GB/s peak, which is typical for a real kernel.

What the numbers show:

- **Coalescing is worth 3–6x** (step 01 → 02), as in SGEMM.
- **Row length decides everything after that.** Step 03 reads x three times, but at 4096×1024 it matches the copy (105%): a 4 KB row stays in cache between the three passes, so DRAM sees x only once. At 1024×32768 a row is 128 KB, every pass goes back to DRAM, and the kernel moves 3 reads + 1 write instead of 1 + 1. It reaches exactly the predicted 2/4 = **50%** of the copy. 4096×4096 (16 KB rows) sits in between at 73%.
- **Warp shuffles only matter for short rows.** With 256 columns each thread handles one element, so the reductions and barriers dominate, and shuffles lift 40% to 54%. For long rows the kernel is waiting on memory and step 03 equals step 02.

### Steps

**01 naive.** One thread computes a whole row in three passes (max, sum of exponentials, output). Neighbouring threads work on different rows, so a warp reads 32 addresses `cols` floats apart: nothing is coalesced. Only `rows` threads run, too few to fill the GPU for long rows.

**02 one block per row, shared-memory reduction.** 256 threads stride over the row together, so each warp reads 32 consecutive floats. The row max and the row sum are combined with a shared-memory tree: 8 rounds, each followed by `__syncthreads()`. Global traffic: x is read three times and y written once.

**03 warp-shuffle reduction.** Each reduction first runs inside a warp with `__shfl_xor_sync` (registers only, no barrier). Only the 8 per-warp results go through shared memory. Per reduction this replaces 8 shared-memory rounds and 8 barriers with 5 shuffles and 2 barriers. Measured: 40% → 54% of the copy for 256-column rows, no change for long rows, because long rows are limited by memory, not by the reduction.

**Next.** Step 04 (online softmax) computes the max and the sum in a single pass, cutting reads of x from three to two; for long rows that predicts about 2/3 = 67% of the copy. Step 05 keeps the row in registers and uses `float4`, so x is read only once; that works for rows up to a few thousand columns, while vocabulary-sized rows rely on step 04. Short rows (256) would benefit from one warp per row instead of one block.

### Run

```bash
cmake -S .. -B ../build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ../build -j
cd softmax && ../../build/02-llm-kernels/softmax/softmax_bench | tee results/<gpu-name>.csv
```

Custom shapes: `softmax_bench 4096x1024 1024x32768`.

## Planned

- LayerNorm with Welford's algorithm (mean and variance in one pass)
- FP16 / BF16 inputs
- PyTorch C++/CUDA extension (`torch.utils.cpp_extension`), compared with `torch.softmax` and `torch.nn.LayerNorm`
- Optional: the same softmax in Triton
