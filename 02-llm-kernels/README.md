# LLM Kernels: Softmax & LayerNorm

Memory-bandwidth-bound kernels used in every transformer layer, optimized step by step and later exposed to PyTorch as a C++/CUDA extension.

Status: softmax steps 01–04 measured on a Tesla T4; step 05 (register kernel + dispatcher) implemented, results pending.

## Softmax

Row-wise softmax over a `[rows x cols]` FP32 matrix:

```text
y[r, c] = exp(x[r, c] - max_r) / sum_c exp(x[r, c] - max_r)
```

Subtracting the row maximum keeps `exp()` from overflowing. Shapes cover short rows (256), typical hidden sizes (1024, 4096, 8192) and vocabulary-sized rows (32768).

### How it is measured

Softmax does very little math per byte, so speed is limited by memory bandwidth. The metric is **effective bandwidth**: bytes of `x` read once plus bytes of `y` written once, divided by time. The baseline is a device-to-device copy of the same size, the fastest possible way to read `x` and write `y` once, so `pct_of_copy` shows how close a kernel is to the memory limit.

Every kernel is checked against a double-precision CPU reference. Timing uses the same method as the SGEMM benchmark: GPU warm-up, median of 5 batches of about 50 ms, and the copy baseline re-timed right before every kernel.

### Results

GPU: Tesla T4 (sm_75) on Kaggle, CUDA 13.0. Raw data: [`softmax/results/kaggle-t4.csv`](softmax/results/kaggle-t4.csv). Cells show effective bandwidth in GB/s and, in parentheses, % of the copy baseline. The 2048×8192 column was added with step 05 and will be filled by the next run.

| Kernel | 8192×256 | 4096×1024 | 4096×4096 | 2048×8192 | 1024×32768 | Main change |
|---|---|---|---|---|---|---|
| copy (baseline) | 228 | 232 | 232 | TBD | 234 | reads x and writes y once |
| 01 naive | 40 (17%) | 38 (16%) | 38 (16%) | TBD | 20 (9%) | one thread per row |
| 02 block + shared memory | 92 (40%) | 219 (94%) | 167 (72%) | TBD | 117 (50%) | one block per row, coalesced, tree reduction |
| 03 warp shuffle | 118 (52%) | 243 (105%) | 169 (73%) | TBD | 117 (50%) | reductions in registers with `__shfl_xor_sync` |
| 04 online softmax | 105 (46%) | 218 (94%) | 181 (78%) | TBD | **150 (64%)** | max and sum in one pass |
| 05 auto: row in registers + dispatcher | TBD | TBD | TBD | TBD | TBD | `float4`, read x once; picks the kernel by row length |

The copy reaches about 230 GB/s, roughly 72% of the T4's 320 GB/s peak, which is typical for a real kernel.

What the numbers show:

- **Coalescing is worth 3–6x** (step 01 → 02), as in SGEMM.
- **Row length decides everything after that.** Step 03 reads x three times, but at 4096×1024 it matches the copy (105%): a 4 KB row stays in cache between the three passes, so DRAM sees x only once. At 1024×32768 a row is 128 KB, every pass goes back to DRAM, and the kernel moves 3 reads + 1 write instead of 1 + 1. It reaches exactly the predicted 2/4 = **50%** of the copy. 4096×4096 (16 KB rows) sits in between at 73%.
- **Online softmax wins only where DRAM is the bottleneck.** For 128 KB rows it goes from 50% to 64% of the copy (predicted 67%) and for 16 KB rows from 73% to 78%. Where rows already stay in cache (4 KB) or are short (256), DRAM traffic was not the limit, and the extra work of online softmax (a branch per element, two `exp` calls per pair merge) makes it slower than step 03. Production libraries pick a kernel by row length for exactly this reason.
- **Warp shuffles only matter for short rows.** With 256 columns each thread handles one element, so the reductions and barriers dominate, and shuffles lift 40% to 52–54%. For long rows the kernel is waiting on memory and step 03 equals step 02.

### Steps

**01 naive.** One thread computes a whole row in three passes (max, sum of exponentials, output). Neighbouring threads work on different rows, so a warp reads 32 addresses `cols` floats apart: nothing is coalesced. Only `rows` threads run, too few to fill the GPU for long rows.

**02 one block per row, shared-memory reduction.** 256 threads stride over the row together, so each warp reads 32 consecutive floats. The row max and the row sum are combined with a shared-memory tree: 8 rounds, each followed by `__syncthreads()`. Global traffic: x is read three times and y written once.

**03 warp-shuffle reduction.** Each reduction first runs inside a warp with `__shfl_xor_sync` (registers only, no barrier). Only the 8 per-warp results go through shared memory. Per reduction this replaces 8 shared-memory rounds and 8 barriers with 5 shuffles and 2 barriers. Measured: 40% → 52–54% of the copy for 256-column rows, no change for long rows, because long rows are limited by memory, not by the reduction.

**04 online softmax.** Each thread keeps a running pair (m, d): the max so far and the sum of exp(x − m) so far. When a larger value arrives, the old sum is rescaled instead of recomputed:

```text
x > m:  d = d * exp(m - x) + 1,  m = x
else:   d = d + exp(x - m)
```

Two partial pairs merge the same way, `m = max(m1, m2)`, `d = d1·exp(m1 − m) + d2·exp(m2 − m)`, so the block reduces pairs with warp shuffles as in step 03. The max and the sum now come from one pass over x, and a second pass writes y: 2 reads + 1 write instead of 3 + 1. The empty value is −FLT_MAX rather than −∞, so `exp(m_i − m)` never becomes NaN. A float32 simulation of the exact reduction order, including rows with large spikes that force many rescalings, matched a double-precision reference within 2e-6 relative error.

- Prediction (written before measuring): long rows 50% → 67% of the copy; no change where rows fit in cache.
- Measured on T4: 1024×32768 50% → **64%**, 1.29x faster than step 03, close to the prediction. 4096×4096 73% → 78%.
- Cache-resident and short rows got slower (4096×1024: 105% → 94%; 8192×256: 52% → 46%). There, reading x again was nearly free, so saving a pass gains nothing while the per-element branch and the costlier pair merge add work.

**05 row in registers + dispatcher.** A group of threads loads its whole row into registers with `float4` loads, then computes the max, the sum and the outputs from registers: x is read once, the same traffic as a copy. Two group sizes:

| Row length | Configuration | Floats per thread |
|---|---|---|
| ≤ 1024 | one **warp** per row, 4 rows per 128-thread block; reductions are pure shuffles, no shared memory or barriers | 4–32 |
| 1025–8192 | one **block** of 256 threads per row | 8–32 |
| > 8192 | too long for registers: falls back to step 04 (online softmax) | — |

`launch_softmax_auto()` picks the configuration from the row length; rows whose length is not a multiple of 4 cannot use `float4` and fall back to step 03 (≤ 2048 columns) or step 04. The benchmark logs which path each shape uses. Slots past the end of a row hold −FLT_MAX, which adds exp(−huge) = 0 to the sum and is never written. All seven configurations compile with 18–67 registers and no spills; a CPU simulation checked results, `float4` bounds, that every output is written exactly once, and that idle warps in the last block write nothing.

- Prediction: close to the copy (≥ 90%) for rows up to 8192 columns, since x is read once; short rows should improve most, because one warp per row stops leaving most of a 256-thread block idle. 1024×32768 stays at step 04's 64%.
- Measured on T4: TBD

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
