// Step 2: one thread block per row, reductions in shared memory.
//
// The block's 256 threads stride over the row together, so a warp reads 32
// consecutive floats (coalesced). The row max and the row sum are each combined
// with a shared-memory tree reduction: log2(256) = 8 rounds, each followed by
// __syncthreads().
//
// Global traffic per element: read x three times (max, sum, output), write y once.
#pragma once

#include "cuda_check.cuh"
#include "kernels/common.cuh"

template <int THREADS>
__global__ void softmax_block_smem_kernel(const float* __restrict__ x,
                                          float* __restrict__ y, int rows, int cols) {
    __shared__ float red[THREADS];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    // Row max.
    float m = -FLT_MAX;
    for (int c = tid; c < cols; c += THREADS) m = fmaxf(m, xr[c]);
    red[tid] = m;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();  // everyone has read red[0] before it is reused

    // Row sum of exp(x - max).
    float s = 0.0f;
    for (int c = tid; c < cols; c += THREADS) s += expf(xr[c] - m);
    red[tid] = s;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];

    for (int c = tid; c < cols; c += THREADS) yr[c] = expf(xr[c] - m) * inv;
}

inline void launch_softmax_block_smem(const float* x, float* y, int rows, int cols) {
    constexpr int kThreads = 256;
    softmax_block_smem_kernel<kThreads><<<rows, kThreads>>>(x, y, rows, cols);
    CUDA_CHECK_LAST();
}
