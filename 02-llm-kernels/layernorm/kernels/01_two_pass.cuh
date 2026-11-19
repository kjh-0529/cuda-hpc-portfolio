// Step 1: two-pass LayerNorm, one block per row.
//
// Pass 1 sums x for the mean; pass 2 sums (x - mean)^2 for the variance; pass 3
// writes y. Reductions use warp shuffles (the best reduction from the softmax
// work). x is read three times.
//
// Why not one pass with sum(x) and sum(x^2), var = E[x^2] - E[x]^2? In FP32 that
// formula subtracts two large, nearly equal numbers. For values around 100 +- 5
// (the benchmark inputs) a float32 simulation gives a relative variance error of
// about 1e-4, versus about 1e-7 for two-pass or Welford: roughly 1000x worse. The
// error grows with mean / spread and can make the variance negative.
#pragma once

#include "cuda_check.cuh"
#include "kernels/common.cuh"

template <int THREADS>
__global__ void layernorm_two_pass_kernel(const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          float* __restrict__ y, int rows, int cols,
                                          float eps) {
    __shared__ float scratch[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    float s = 0.0f;
    for (int c = tid; c < cols; c += THREADS) s += xr[c];
    const float mean = block_reduce<THREADS, false>(s, scratch) / cols;

    float q = 0.0f;
    for (int c = tid; c < cols; c += THREADS) {
        const float d = xr[c] - mean;
        q += d * d;
    }
    const float var = block_reduce<THREADS, false>(q, scratch) / cols;
    const float rstd = rsqrtf(var + eps);

    for (int c = tid; c < cols; c += THREADS)
        yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
}

inline void launch_layernorm_two_pass(const float* x, const float* gamma, const float* beta,
                                      float* y, int rows, int cols, float eps) {
    constexpr int kThreads = 256;
    layernorm_two_pass_kernel<kThreads><<<rows, kThreads>>>(x, gamma, beta, y, rows, cols, eps);
    CUDA_CHECK_LAST();
}
