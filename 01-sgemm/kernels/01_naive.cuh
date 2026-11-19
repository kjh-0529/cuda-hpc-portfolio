// Step 1: naive SGEMM. One thread computes one element of C.
//
// C = alpha * A * B + beta * C, all row-major.
// A is M x K, B is K x N, C is M x N.
//
// threadIdx.x is mapped to the ROW of C on purpose. Threads in a warp then read
// A and write C with a stride of K or N floats, so global memory accesses are not
// coalesced. Step 2 fixes this by mapping threadIdx.x to the column instead.
#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void launch_sgemm_naive(int M, int N, int K, float alpha, const float* A,
                               const float* B, float beta, float* C) {
    constexpr int kBlock = 32;
    dim3 block(kBlock, kBlock);
    dim3 grid((M + kBlock - 1) / kBlock, (N + kBlock - 1) / kBlock);
    sgemm_naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
}
