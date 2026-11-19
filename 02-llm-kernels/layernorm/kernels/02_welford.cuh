// Step 2: Welford's algorithm, one block per row. Mean and variance in ONE pass.
//
// Each thread keeps (mean, M2, n), where M2 = sum of squared deviations from the
// running mean, and updates it per element:
//     n += 1;  delta = x - mean;  mean += delta / n;  M2 += delta * (x - mean)
// Partial results from different threads merge exactly (Chan et al.):
//     n = na + nb;  delta = mean_b - mean_a
//     mean = mean_a + delta * nb / n
//     M2   = M2_a + M2_b + delta^2 * na * nb / n
// Like online softmax, this removes one pass: 2 reads of x instead of 3, while
// staying numerically stable (no E[x^2] - E[x]^2 cancellation).
#pragma once

#include "cuda_check.cuh"
#include "kernels/common.cuh"

struct Welford {
    float mean;
    float m2;
    float n;
};

__device__ __forceinline__ Welford welford_merge(Welford a, Welford b) {
    const float n = a.n + b.n;
    if (n == 0.0f) return a;
    const float delta = b.mean - a.mean;
    const float wb = b.n / n;
    return {a.mean + delta * wb, a.m2 + b.m2 + delta * delta * a.n * wb, n};
}

__device__ __forceinline__ Welford warp_reduce_welford(Welford v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        Welford o;
        o.mean = __shfl_xor_sync(0xffffffffu, v.mean, offset);
        o.m2 = __shfl_xor_sync(0xffffffffu, v.m2, offset);
        o.n = __shfl_xor_sync(0xffffffffu, v.n, offset);
        v = welford_merge(v, o);
    }
    return v;
}

template <int THREADS>
__device__ __forceinline__ Welford block_reduce_welford(Welford v, Welford* scratch) {
    constexpr int kWarps = THREADS / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_reduce_welford(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        Welford w = (lane < kWarps) ? scratch[lane] : Welford{0.0f, 0.0f, 0.0f};
        w = warp_reduce_welford(w);
        if (lane == 0) scratch[0] = w;
    }
    __syncthreads();
    return scratch[0];
}

template <int THREADS>
__global__ void layernorm_welford_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y, int rows, int cols,
                                         float eps) {
    __shared__ Welford scratch[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    Welford w{0.0f, 0.0f, 0.0f};
    for (int c = tid; c < cols; c += THREADS) {
        const float v = xr[c];
        w.n += 1.0f;
        const float delta = v - w.mean;
        w.mean += delta / w.n;
        w.m2 += delta * (v - w.mean);
    }
    const Welford r = block_reduce_welford<THREADS>(w, scratch);
    const float rstd = rsqrtf(r.m2 / cols + eps);

    for (int c = tid; c < cols; c += THREADS)
        yr[c] = (xr[c] - r.mean) * rstd * gamma[c] + beta[c];
}

inline void launch_layernorm_welford(const float* x, const float* gamma, const float* beta,
                                     float* y, int rows, int cols, float eps) {
    constexpr int kThreads = 256;
    layernorm_welford_kernel<kThreads><<<rows, kThreads>>>(x, gamma, beta, y, rows, cols, eps);
    CUDA_CHECK_LAST();
}
