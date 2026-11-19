# LLM Kernels: Softmax & LayerNorm

Memory-bandwidth-bound kernels used in every transformer layer, optimized step by step and later exposed to PyTorch as a C++/CUDA extension.

Status: softmax steps 01–03 implemented; results pending.

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

GPU: TBD. Raw data: `softmax/results/`.

| Kernel | 8192×256 | 4096×1024 | 4096×4096 | 1024×32768 | Main change |
|---|---|---|---|---|---|
| copy (baseline) | TBD | TBD | TBD | TBD | reads x and writes y once |
| 01 naive | TBD | TBD | TBD | TBD | one thread per row |
| 02 block + shared memory | TBD | TBD | TBD | TBD | one block per row, coalesced, tree reduction |
| 03 warp shuffle | TBD | TBD | TBD | TBD | reductions in registers with `__shfl_xor_sync` |
| 04 online softmax | — | — | — | — | max and sum in one pass |
| 05 vectorized, row in registers | — | — | — | — | `float4`, read x once |

### Steps

**01 naive.** One thread computes a whole row in three passes (max, sum of exponentials, output). Neighbouring threads work on different rows, so a warp reads 32 addresses `cols` floats apart: nothing is coalesced. Only `rows` threads run, too few to fill the GPU for long rows.

**02 one block per row, shared-memory reduction.** 256 threads stride over the row together, so each warp reads 32 consecutive floats. The row max and the row sum are combined with a shared-memory tree: 8 rounds, each followed by `__syncthreads()`. Global traffic: x is read three times and y written once.

**03 warp-shuffle reduction.** Each reduction first runs inside a warp with `__shfl_xor_sync` (registers only, no barrier). Only the 8 per-warp results go through shared memory. Per reduction this replaces 8 shared-memory rounds and 8 barriers with 5 shuffles and 2 barriers. Since the kernel is memory-bound, the gain is expected to be small; step 04 attacks the real cost, which is reading x three times.

**Next.** Step 04 (online softmax) computes the max and the sum in a single pass, cutting reads of x from three to two. Step 05 keeps the row in registers and uses `float4`, so x is read only once.

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
