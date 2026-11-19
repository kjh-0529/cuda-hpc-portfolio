# SGEMM Optimization: from naive to cuBLAS-level

FP32 matrix multiplication `C = alpha * A * B + beta * C` (row-major), optimized step by step and compared against cuBLAS.

## Results

Square matrices, FP32. GPU: TBD, CUDA: TBD.

| Kernel | GFLOPS (4096³) | % of cuBLAS | Main change |
|---|---|---|---|
| cuBLAS | TBD | 100 | reference |
| 01 naive | TBD | TBD | one thread per output element |
| 02 coalesced | — | — | coalesced global memory access |
| 03 shared-memory tiling | — | — | reuse tiles of A and B from shared memory |
| 04 register blocking | — | — | each thread computes a small block of C |
| 05 vectorized loads | — | — | `float4` loads, no shared-memory bank conflicts |
| 06 double buffering | — | — | overlap tile loads with computation |

(Add a GFLOPS-vs-size plot here: `results/gflops.png`.)

## Steps

### 01 naive

One thread computes one element of C. `threadIdx.x` is mapped to the row, so threads in a warp access A and C with a large stride and global memory loads are not coalesced.

- Nsight Compute: TBD (expected: low memory throughput efficiency, many sectors per request)
- Next step: map `threadIdx.x` to the column so a warp reads consecutive addresses.

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
