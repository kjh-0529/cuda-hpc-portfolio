// Step 2: coalesced global memory access.
//
// Same work per thread as step 1 (one thread per element of C), but threadIdx.x
// is now mapped to the COLUMN of C. Within a warp, the 32 threads share the same
// row and have consecutive columns, so:
//   - B[k * N + col]   : 32 consecutive floats -> one coalesced 128-byte load
//   - A[row * K + k]   : the same address for all 32 threads -> one broadcast load
//   - C[row * N + col] : 32 consecutive floats -> one coalesced store
// In step 1 each of these touched up to 32 different memory segments per warp.
#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

__global__ void sgemm_coalesced_kernel(int M, int N, int K, float alpha,
                                       const float* __restrict__ A,
                                       const float* __restrict__ B, float beta,
                                       float* __restrict__ C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void launch_sgemm_coalesced(int M, int N, int K, float alpha, const float* A,
                                   const float* B, float beta, float* C) {
    constexpr int kBlock = 32;
    dim3 block(kBlock, kBlock);
    // grid.x now covers columns (N) and grid.y covers rows (M).
    dim3 grid((N + kBlock - 1) / kBlock, (M + kBlock - 1) / kBlock);
    sgemm_coalesced_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
}
