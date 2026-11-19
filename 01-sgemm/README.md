# SGEMM Optimization: from naive to cuBLAS-level

FP32 matrix multiplication `C = alpha * A * B + beta * C` (row-major), optimized step by step and compared against cuBLAS.

## Results

Square matrices, FP32. GPU: Tesla T4 (sm_75) on Kaggle, CUDA 13.0, driver 580. Raw data: [`results/kaggle-t4.csv`](results/kaggle-t4.csv).

| Kernel | 1024³ GFLOPS | 2048³ GFLOPS | 4096³ GFLOPS | % of cuBLAS (1024³ / 2048³ / 4096³) | Main change |
|---|---|---|---|---|---|
| cuBLAS | 3825 | 4180 | 4245 | 100 | reference |
| 01 naive | 60 | 61 | 62 | 1.6 / 1.5 / 1.5 | one thread per output element |
| 02 coalesced | 609 | 581 | 571 | 16 / 14 / 14 | coalesced global memory access |
| 03 shared-memory tiling | 900 | 881 | 865 | 25 / 21 / 21 | reuse tiles of A and B from shared memory |
| 04 register blocking | 2864 | 3310 | 3146 | 82 / 81 / 76 | each thread computes an 8×8 block of C from registers |
| 05 vectorized loads | 3329 | 3603 | 3472 | 92 / 87 / 84 | `float4` loads, A tile transposed in shared memory |
| 06 double buffering | TBD | TBD | TBD | TBD | load the next tile while computing the current one |

Step 02 is about 10x faster than step 01, from changing only how threads map to the matrix. Step 03 adds 1.5x. Step 04 is the big jump: 3.2–3.8x over step 03. Step 05 adds 1.09–1.16x, reaching 84–92% of cuBLAS and about 59x naive at 2048³.

Measurement notes:

- Each percentage compares a kernel with cuBLAS timed right before it, after a warm-up, so both run under the same GPU clock (method in [How to run](#how-to-run)). The percentages are now consistent across sizes, which they were not with the first version of the benchmark.
- These are **sustained** numbers. The T4 is a 70 W card: in short bursts on a cool GPU, cuBLAS reached about 7200 GFLOPS at 2048³ in earlier runs, but under continuous load it settles near 4100 GFLOPS as the card lowers its clock to stay within power and temperature limits. All kernels here are measured in that sustained state.

(Add a GFLOPS-vs-size plot here: `results/gflops.png`.)

## Steps

### 01 naive

One thread computes one element of C. `threadIdx.x` is mapped to the row, so threads in a warp access A and C with a large stride and global memory loads are not coalesced.

- Nsight Compute: pending. Kaggle blocks access to GPU performance counters (`ERR_NVGPUCTRPERM`); all steps will be profiled together on a GPU VM with admin rights. Expected: many sectors per request (uncoalesced loads).
- Next step: map `threadIdx.x` to the column so a warp reads consecutive addresses.

### 02 coalesced

Same algorithm, but `threadIdx.x` is mapped to the column of C. The 32 threads of a warp now share one row and cover 32 consecutive columns:

| Access | Step 01 (per warp) | Step 02 (per warp) |
|---|---|---|
| `B[k*N + col]` | same address for all threads | 32 consecutive floats, one 128-byte transaction |
| `A[row*K + k]` | 32 addresses, `K` floats apart | same address, broadcast |
| `C[row*N + col]` | 32 addresses, `N` floats apart | 32 consecutive floats |

- Measured on T4: 609 GFLOPS at 1024³ (10.1x step 01), 571 GFLOPS at 4096³ (9.3x).
- GFLOPS falls slightly as the matrix grows (609 → 581 → 571): each thread still streams a full row of A and column of B from global memory, and at larger sizes less of that data stays in the 4 MB L2 cache between reuses.
- Nsight Compute: pending (see step 01). Compare `Memory Throughput` and `L1/TEX sectors per request` with step 01.
- Remaining bottleneck: every element of A and B is still loaded from global memory `N` or `M` times. Step 03 loads tiles into shared memory and reuses them.

### 03 shared-memory tiling

Each 32×32 thread block computes a 32×32 tile of C and walks along K in steps of 32:

1. Every thread loads one element of A and one of B into shared memory (coalesced).
2. `__syncthreads()`, so the whole tile is loaded.
3. Each thread does 32 multiply-adds reading only shared memory.
4. `__syncthreads()` again before the next tile overwrites shared memory.

Every value loaded from global memory is reused 32 times inside the block, so global memory traffic drops by about 32x versus step 02. In the inner loop, `As[ty][k]` is a broadcast and `Bs[k][tx]` reads 32 consecutive floats, so there are no shared-memory bank conflicts. Out-of-range elements are loaded as zero, so sizes that are not multiples of 32 stay correct.

- Measured on T4: 865 GFLOPS at 4096³, 1.52x step 02 and 14.0x step 01; 1.48–1.52x step 02 at every size.
- GFLOPS stays nearly flat with matrix size (900 → 881 → 865), since each value loaded from global memory is reused 32 times regardless of how much fits in L2.
- Why only 1.5x and not 32x: the bottleneck moved from global memory to the SM itself.
    - Each multiply-add still needs two shared-memory loads, so shared-memory bandwidth and load instructions now limit speed.
    - A 32×32 block has 1024 threads, which is the per-SM maximum on Turing (sm_75). Only one block fits per SM, so during each `__syncthreads()` there is no other block to keep the SM busy.
- Nsight Compute: pending (see step 01).
- Next: step 04 makes each thread compute several elements of C from values held in registers, cutting shared-memory loads per multiply-add and using smaller blocks.

### 04 register blocking

Each block computes a 128×128 tile of C with 256 threads, so each thread computes an 8×8 sub-block (64 results instead of 1). For each step of 8 along K:

1. The block loads a 128×8 tile of A and an 8×128 tile of B into shared memory (4 + 4 elements per thread).
2. For each k, a thread copies 8 values of A and 8 of B from shared memory into registers, then does 8×8 = 64 multiply-adds from registers.

| | Step 03 | Step 04 |
|---|---|---|
| Results per thread | 1 | 64 |
| Shared-memory loads per multiply-add | 2 | 0.25 |
| Threads per block | 1024 | 256 |
| Blocks per T4 SM | 1 | 2 (128 registers/thread, 8 KB shared memory/block) |

`ptxas` reports 128 registers per thread and no register spills on sm_75. Index math was checked with a thread-by-thread CPU simulation, including sizes that are not multiples of the tile.

- Measured on T4: 3310 GFLOPS at 2048³, 3.8x step 03 and 54x step 01, 81% of cuBLAS.
- Nsight Compute: pending (see step 01).
- Expected remaining bottleneck: A is read from shared memory one float at a time with a stride of 8, and global loads of A are only 32 bytes wide. Step 05 stores A transposed and uses `float4` loads.

### 05 vectorized loads

Same tiling as step 04 (128×128 block tile, 8×8 results per thread, 256 threads), with three changes:

1. **`float4` global loads:** each thread loads one `float4` of A and one of B per tile, 4 floats per instruction instead of 1.
2. **Transposed A tile:** A is stored in shared memory as `As[k][m]` instead of `As[m][k]`. A thread needs 8 consecutive rows of A for one k; after the transpose they are contiguous, so it reads them with two `float4` loads instead of eight strided scalar loads.
3. **`float4` for B and C:** the 8 values of B per k and the 8 results per row of C also move as two `float4` each.

| Per thread, per tile | Step 04 | Step 05 |
|---|---|---|
| Global load instructions (A + B) | 4 + 4 | 1 + 1 |
| Shared-memory load instructions in the inner loop | 128 | 32 |

`float4` needs 16-byte alignment, so the kernel requires `K % 4 == 0` and `N % 4 == 0`; other sizes fall back to the step 04 kernel. `ptxas` reports 113 registers per thread and no spills on sm_75. A thread-by-thread CPU simulation checked results, `float4` alignment and bounds, including sizes that are not multiples of the tile.

- Measured on T4: 3603 GFLOPS at 2048³, 1.09x step 04 and 59x step 01, 87% of cuBLAS. The gain is largest at 1024³ (1.16x, 92% of cuBLAS).
- Nsight Compute: pending (see step 01).
- Known cost: the transposed stores into `As` have a 2-way shared-memory bank conflict (two threads with the same row write different k into the same bank). It happens once per tile, against 32 conflict-free `float4` reads.

### 06 double buffering

Same tiling and `float4` accesses as step 05; what changes is when the next tile is loaded.

| | Step 05 | Step 06 |
|---|---|---|
| Per tile | load → store to shared → sync → compute → sync | issue loads for tile t+1 → compute tile t → store t+1 into the other buffer → sync |
| Shared-memory buffers | 1 (8 KB) | 2 (16 KB) |
| `__syncthreads()` per tile | 2 | 1 |
| Registers per thread | 113 | 127 (8 extra floats hold the prefetched tile) |

The global loads for the next tile are in flight while each thread does the 512 multiply-adds of the current tile, so global-memory latency is hidden behind computation instead of stalling the block. Turing (sm_75) has no `cp.async` (added in sm_80), so the prefetch goes through registers.

One barrier per tile is enough: tile t+1 is written into the buffer that was last read for tile t-1, and every thread passed the barrier at the end of iteration t-1 only after finishing that computation. A CPU simulation checked results, `float4` alignment and bounds, and found no shared-memory race between barriers (its race checker was confirmed to flag a deliberately racy access).

- Measured on T4: TBD
- Nsight Compute: pending (see step 01).
- Expected gain: modest. Two blocks per SM already hide part of the load latency in step 05, so double buffering mainly removes the remaining stalls at each barrier.

## How to run

```bash
cmake -S .. -B ../build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ../build -j
../build/01-sgemm/sgemm_bench 512 1024 2048 4096 | tee results/<gpu-name>.csv
```

The benchmark checks every kernel against cuBLAS (`correct` column) and exits with a non-zero code if any kernel is wrong.

How times are measured:

- **Warm-up:** before each size the GPU runs cuBLAS for 2 s (first size) or 0.5 s, so clocks are at full speed when timing starts.
- **Median of batches:** each time is the median of 5 batches. A batch repeats the kernel enough times to take about 50 ms, so fast kernels are not dominated by launch overhead and slow kernels finish quickly.
- **Adjacent cuBLAS:** cuBLAS is re-timed right before every kernel. `pct_of_cublas` compares the kernel with that run, and `cublas_gflops` shows its value. This keeps the comparison fair when the GPU lowers its clock under sustained load.

Profile one kernel with Nsight Compute:

```bash
ncu --set full -k regex:sgemm -o results/naive ../build/01-sgemm/sgemm_bench 4096
```

## Adding a step

1. Add `kernels/0N_<name>.cuh` with a `launch_sgemm_<name>(M, N, K, alpha, A, B, beta, C)` function.
2. Include it in `bench.cu` and add it to `kKernels`.
3. Run the benchmark, profile it, and fill in the tables above.
