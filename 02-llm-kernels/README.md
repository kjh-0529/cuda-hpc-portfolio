# LLM Kernels: Softmax & LayerNorm

Memory-bandwidth-bound kernels used in every transformer layer, optimized step by step and later exposed to PyTorch as a C++/CUDA extension.

Status: softmax done (copy speed for rows up to 8192 columns on a Tesla T4). LayerNorm steps 01–03 implemented, results pending.

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

GPU: Tesla T4 (sm_75) on Kaggle, CUDA 13.0. Raw data: [`softmax/results/kaggle-t4.csv`](softmax/results/kaggle-t4.csv). Cells show effective bandwidth in GB/s and, in parentheses, % of the copy baseline.

| Kernel | 8192×256 | 4096×1024 | 4096×4096 | 2048×8192 | 1024×32768 | Main change |
|---|---|---|---|---|---|---|
| copy (baseline) | 228 | 232 | 232 | 233 | 234 | reads x and writes y once |
| 01 naive | 40 (17%) | 38 (16%) | 38 (16%) | 31 (13%) | 20 (9%) | one thread per row |
| 02 block + shared memory | 91 (40%) | 218 (94%) | 167 (72%) | 128 (55%) | 117 (50%) | one block per row, coalesced, tree reduction |
| 03 warp shuffle | 123 (54%) | 243 (105%) | 169 (73%) | 128 (55%) | 117 (50%) | reductions in registers with `__shfl_xor_sync` |
| 04 online softmax | 108 (47%) | 217 (93%) | 181 (78%) | 161 (69%) | 150 (64%) | max and sum in one pass |
| **05 auto: row in registers + dispatcher** | **232 (102%)** | **228 (98%)** | **232 (100%)** | **231 (100%)** | **150 (64%)** | `float4`, read x once; picks the kernel by row length |

The copy reaches about 230 GB/s, roughly 72% of the T4's 320 GB/s peak, which is typical for a real kernel.

What the numbers show:

- **Step 05 reaches copy speed (98–102%) for every row length up to 8192**, 6–8x faster than naive and 1.3–1.9x faster than the best earlier step. Vocabulary-sized rows stay at 64% with online softmax.
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
- Measured on T4: 98–102% of the copy at 256, 1024, 4096 and 8192 columns. The biggest gains are where predicted: short rows 54% → 102% (1.9x over step 03, from one warp per row), and 8192-column rows 69% → 100% (1.4x over step 04, from reading x once).
- The dispatcher logged the intended path for every shape, and 1024×32768 matched step 04 exactly (64%).
- One honest exception: at 4096×1024, step 03 is 7% faster (105% vs 98%). Those 4 KB rows already stay in cache, so reading x three times costs nothing, and the plain copy is not the true hardware limit. A dispatcher tuned only for the T4 could send 513–1024-column rows to step 03; I kept the register path because it does not depend on cache behaviour that changes between GPUs.
- Remaining gap for vocabulary-sized rows: a row of 32768 floats does not fit in registers (128 per thread) or a block's shared memory on the T4 (48 KB by default). Splitting such a row across several blocks, with a second small kernel to combine their (max, sum) pairs, would be the next step.

### Run

```bash
cmake -S .. -B ../build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ../build -j
cd softmax && ../../build/02-llm-kernels/softmax/softmax_bench | tee results/<gpu-name>.csv
```

Custom shapes: `softmax_bench 4096x1024 1024x32768`.

## LayerNorm

Row-wise LayerNorm over a `[rows x cols]` FP32 matrix, with `gamma` and `beta` of length `cols`:

```text
mean = sum(x) / cols
var  = sum((x - mean)^2) / cols        (biased, as in PyTorch)
y    = (x - mean) / sqrt(var + eps) * gamma + beta
```

Inputs are `100 + uniform(-5, 5)`: a large mean relative to the spread. That is where the tempting one-pass formula `var = E[x²] − E[x]²` goes wrong in FP32, because it subtracts two large, nearly equal numbers. A float32 simulation of the reduction on these inputs gives a relative variance error of about 1e-4 for that formula versus about 1e-7 for the methods below, roughly 1000x worse, and the gap grows with mean / spread. Measurement, baseline and checks are the same as for softmax (CPU double reference, copy baseline).

### Results

GPU: TBD. Raw data: `layernorm/results/`.

| Kernel | 8192×256 | 4096×1024 | 4096×4096 | 2048×8192 | 1024×32768 | Main change |
|---|---|---|---|---|---|---|
| copy (baseline) | TBD | TBD | TBD | TBD | TBD | reads x and writes y once |
| 01 two-pass | TBD | TBD | TBD | TBD | TBD | mean, then variance, then output: x read 3 times |
| 02 Welford | TBD | TBD | TBD | TBD | TBD | mean and variance in one pass: x read 2 times |
| 03 auto: row in registers + dispatcher | TBD | TBD | TBD | TBD | TBD | x read once; same dispatcher as softmax |

### Steps

**01 two-pass.** One block per row with warp-shuffle reductions, reusing what worked for softmax. Pass 1 computes the mean, pass 2 the sum of squared deviations, pass 3 the output. Numerically stable, but x is read three times.

**02 Welford.** Each thread keeps (mean, M2, count) and updates it per element; partial results from different threads merge exactly with Chan's formula, so the block reduces them with warp shuffles. Mean and variance come from one pass, and the output from a second: 2 reads of x, still numerically stable. This is LayerNorm's counterpart of online softmax.

**03 row in registers + dispatcher.** The softmax step 05 design: one warp per row up to 1024 columns, one 256-thread block per row up to 8192, `float4` loads, x read once. With the row in registers, the exact two-pass variance costs no extra memory traffic, so Welford is not needed there. Longer rows, and rows whose length is not a multiple of 4, use step 02. `gamma` and `beta` are also read with `float4`; they are shared by all rows and stay in cache.

Checks before running on a GPU: all nine kernel configurations compile with no register spills, and a float32 simulation of the Welford merge tree and the register kernel's statistics matched a double-precision reference within 2.3e-7 relative variance error.

- Prediction: the same pattern as softmax. Step 02 helps long rows (about 50% → 67% of the copy at 32768 columns); step 03 reaches about 100% of the copy up to 8192 columns.

### Run

```bash
cd layernorm && ../../build/02-llm-kernels/layernorm/layernorm_bench 2>&1 | tee results/<gpu-name>.csv
```

## Planned

- FP16 / BF16 inputs
- PyTorch C++/CUDA extension (`torch.utils.cpp_extension`), compared with `torch.softmax` and `torch.nn.LayerNorm`
- Optional: the same softmax in Triton
