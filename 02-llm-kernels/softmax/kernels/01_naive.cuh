// Step 1: naive softmax. One thread handles one whole row in three passes:
// find the max, sum the exponentials, write the outputs.
//
// Neighbouring threads work on different rows, so at any moment a warp reads 32
// addresses that are `cols` floats apart: global memory accesses are not
// coalesced. It also uses only `rows` threads, often too few to fill the GPU.
#pragma once

#include "cuda_check.cuh"
#include "kernels/common.cuh"

__global__ void softmax_naive_kernel(const float* __restrict__ x, float* __restrict__ y,
                                     int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    float m = -FLT_MAX;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
    float s = 0.0f;
    for (int c = 0; c < cols; ++c) s += expf(xr[c] - m);
    const float inv = 1.0f / s;
    for (int c = 0; c < cols; ++c) yr[c] = expf(xr[c] - m) * inv;
}

inline void launch_softmax_naive(const float* x, float* y, int rows, int cols) {
    constexpr int kThreads = 128;
    softmax_naive_kernel<<<(rows + kThreads - 1) / kThreads, kThreads>>>(x, y, rows, cols);
    CUDA_CHECK_LAST();
}
