// Step 3: shared-memory tiling.
//
// Each 32x32 thread block computes a 32x32 tile of C. It walks along K in steps
// of 32: the block loads a 32x32 tile of A and a 32x32 tile of B into shared
// memory (one element per thread, coalesced), synchronizes, and every thread then
// reads 32 values of A and 32 of B from fast shared memory instead of global memory.
//
// Each element loaded from global memory is now reused 32 times inside the block,
// so global memory traffic drops by about 32x compared with step 2.
//
// Shared-memory bank conflicts: in the inner loop, As[ty][k] is the same address
// for the whole warp (broadcast), and Bs[k][tx] reads 32 consecutive floats, which
// fall in 32 different banks. Neither access conflicts.
#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

template <int TILE>
__global__ void sgemm_smem_tiling_kernel(int M, int N, int K, float alpha,
                                         const float* __restrict__ A,
                                         const float* __restrict__ B, float beta,
                                         float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        // Load one tile of A and one tile of B. Out-of-range elements are zero,
        // so sizes that are not multiples of TILE still give correct results.
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();  // tiles fully loaded before anyone reads them

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // everyone done reading before the next tile overwrites
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void launch_sgemm_smem_tiling(int M, int N, int K, float alpha, const float* A,
                                     const float* B, float beta, float* C) {
    constexpr int kTile = 32;
    dim3 block(kTile, kTile);
    dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    sgemm_smem_tiling_kernel<kTile><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
}
