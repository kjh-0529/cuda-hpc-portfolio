# SGEMM Optimization: from naive to cuBLAS-level

FP32 matrix multiplication `C = alpha * A * B + beta * C` (row-major), optimized step by step and compared against cuBLAS.

## Results

Square matrices, FP32. GPU: Tesla T4 (sm_75) on Kaggle, CUDA 13.0, driver 580. Raw data: [`results/kaggle-t4.csv`](results/kaggle-t4.csv).

| Kernel | 1024³ GFLOPS | 2048³ GFLOPS | 4096³ GFLOPS | % of cuBLAS (4096³) | Main change |
|---|---|---|---|---|---|
| cuBLAS | 2596 | 6571 | 4164 | 100 | reference |
| 01 naive | 61 | 61 | 62 | 1.5 | one thread per output element |
| 02 coalesced | 799 | 593 | 564 | 13.5 | coalesced global memory access |
| 03 shared-memory tiling | 827 | 894 | 855 | 20.5 | reuse tiles of A and B from shared memory |
| 04 register blocking | TBD | TBD | TBD | TBD | each thread computes an 8×8 block of C from registers |
| 05 vectorized loads | — | — | — | — | `float4` loads, no shared-memory bank conflicts |
| 06 double buffering | — | — | — | — | overlap tile loads with computation |

Step 02 is 9–13x faster than step 01 at every size, from changing only how threads map to the matrix. Step 03 adds another 1.5x at 2048³ and 4096³ and, unlike step 02, holds its speed as the matrix grows.

Measurement notes:

- The T4 is a 70 W card and lowers its clock under sustained load. cuBLAS reaches 6500–7200 GFLOPS at 2048³ but only about 4100 GFLOPS at 4096³, right after several seconds of slow naive runs. The 4096³ percentages are therefore approximate; compare kernels within the same run.
- At 1024³ a cuBLAS call takes under 1 ms, so launch overhead lowers its GFLOPS.

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

- Measured on T4: 799 GFLOPS at 1024³ (13.2x step 01), 564 GFLOPS at 4096³ (9.1x).
- GFLOPS falls as the matrix grows: each thread still streams a full row of A and column of B from global memory, and at larger sizes less of that data stays in the 4 MB L2 cache between reuses.
- Nsight Compute: pending (see step 01). Compare `Memory Throughput` and `L1/TEX sectors per request` with step 01.
- Remaining bottleneck: every element of A and B is still loaded from global memory `N` or `M` times. Step 03 loads tiles into shared memory and reuses them.

### 03 shared-memory tiling

Each 32×32 thread block computes a 32×32 tile of C and walks along K in steps of 32:

1. Every thread loads one element of A and one of B into shared memory (coalesced).
2. `__syncthreads()`, so the whole tile is loaded.
3. Each thread does 32 multiply-adds reading only shared memory.
4. `__syncthreads()` again before the next tile overwrites shared memory.

Every value loaded from global memory is reused 32 times inside the block, so global memory traffic drops by about 32x versus step 02. In the inner loop, `As[ty][k]` is a broadcast and `Bs[k][tx]` reads 32 consecutive floats, so there are no shared-memory bank conflicts. Out-of-range elements are loaded as zero, so sizes that are not multiples of 32 stay correct.

- Measured on T4: 855 GFLOPS at 4096³, 1.52x step 02 and 13.9x step 01. At 1024³ the gain is only 1.04x: step 02 was already at its best size there (799 GFLOPS), with many of its repeated loads served from the L2 cache.
- GFLOPS no longer drops with matrix size (827 → 894 → 855), since global memory traffic no longer grows with the cache misses that hurt step 02.
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

- Measured on T4: TBD
- Nsight Compute: pending (see step 01).
- Expected remaining bottleneck: A is read from shared memory one float at a time with a stride of 8, and global loads of A are only 32 bytes wide. Step 05 stores A transposed and uses `float4` loads.

## How to run

```bash
cmake -S .. -B ../build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ../build -j
../build/01-sgemm/sgemm_bench 512 1024 2048 4096 | tee results/<gpu-name>.csv
```

The benchmark checks every kernel against cuBLAS (`correct` column) and exits with a non-zero code if any kernel is wrong.

Profile one kernel with Nsight Compute:

```bash
ncu --set full -k regex:sgemm -o results/naive ../build/01-sgemm/sgemm_bench 4096
```

## Adding a step

1. Add `kernels/0N_<name>.cuh` with a `launch_sgemm_<name>(M, N, K, alpha, A, B, beta, C)` function.
2. Include it in `bench.cu` and add it to `kKernels`.
3. Run the benchmark, profile it, and fill in the tables above.
