# SGEMM Optimization: from naive to cuBLAS-level

FP32 matrix multiplication `C = alpha * A * B + beta * C` (row-major), optimized step by step and compared against cuBLAS.

## Results

Square matrices, FP32. GPU: Tesla T4 (sm_75) on Kaggle, CUDA 13.0, driver 580. Raw data: [`results/kaggle-t4.csv`](results/kaggle-t4.csv).

| Kernel | 1024³ GFLOPS | 2048³ GFLOPS | 4096³ GFLOPS | % of cuBLAS (4096³) | Main change |
|---|---|---|---|---|---|
| cuBLAS | 2593 | 7176 | 4058 | 100 | reference |
| 01 naive | 58 | 62 | 62 | 1.5 | one thread per output element |
| 02 coalesced | 744 | 579 | 549 | 13.5 | coalesced global memory access |
| 03 shared-memory tiling | TBD | TBD | TBD | TBD | reuse tiles of A and B from shared memory |
| 04 register blocking | — | — | — | — | each thread computes a small block of C |
| 05 vectorized loads | — | — | — | — | `float4` loads, no shared-memory bank conflicts |
| 06 double buffering | — | — | — | — | overlap tile loads with computation |

Step 02 is 9–13x faster than step 01 at every size, from changing only how threads map to the matrix.

Measurement notes:

- The T4 is a 70 W card and lowers its clock under sustained load. cuBLAS reached 7176 GFLOPS at 2048³ but only 4058 GFLOPS at 4096³, right after several seconds of slow naive runs. The 4096³ percentages are therefore approximate; compare kernels within the same run.
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

- Measured on T4: 744 GFLOPS at 1024³ (12.7x step 01), 549 GFLOPS at 4096³ (8.9x).
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

- Measured on T4: TBD
- Nsight Compute: pending (see step 01).
- Expected remaining bottleneck: each multiply-add still needs two shared-memory loads, so shared-memory bandwidth and instruction overhead limit performance. Step 04 makes each thread compute several elements of C and keeps values in registers.

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
