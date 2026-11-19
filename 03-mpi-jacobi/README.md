# MPI + CUDA 2D Jacobi Solver (planned)

2D heat equation solved with Jacobi iteration, scaled from CPU to multiple GPUs.

Planned steps:

1. CPU baseline (C + OpenMP)
2. Single-GPU CUDA, then shared-memory tiling
3. MPI domain decomposition with halo exchange between GPUs (CUDA-aware MPI or host staging)
4. Overlap interior computation with halo exchange using CUDA streams
5. Nsight Systems timeline to confirm the overlap

Metrics: grid updates per second, CPU vs 1 GPU vs 2 GPUs, strong-scaling efficiency.
