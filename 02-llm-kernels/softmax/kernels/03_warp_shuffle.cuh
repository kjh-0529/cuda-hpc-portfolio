// Step 3: reductions with warp shuffles.
//
// Same work split as step 2, but each reduction first happens inside a warp with
// __shfl_xor_sync (registers only, no shared memory, no barrier). Only the 8 warp
// results go through shared memory, then warp 0 combines them. That replaces
// 8 rounds of shared-memory traffic plus 8 barriers per reduction with
// 5 shuffles, one small shared-memory exchange and 2 barriers.
#pragma once

#include "cuda_check.cuh"
#include "kernels/common.cuh"

// Block-wide reduction built from warp reductions. Returns the result to every thread.
template <int THREADS, bool IS_MAX>
__device__ __forceinline__ float block_reduce(float v, float* scratch) {
    constexpr int kWarps = THREADS / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    v = IS_MAX ? warp_reduce_max(v) : warp_reduce_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = (lane < kWarps) ? scratch[lane] : (IS_MAX ? -FLT_MAX : 0.0f);
        w = IS_MAX ? warp_reduce_max(w) : warp_reduce_sum(w);
        if (lane == 0) scratch[0] = w;
    }
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();  // scratch can be reused by the next reduction
    return result;
}

template <int THREADS>
__global__ void softmax_warp_shuffle_kernel(const float* __restrict__ x,
                                            float* __restrict__ y, int rows, int cols) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "THREADS must be warps, <= 1024");
    __shared__ float scratch[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    float m = -FLT_MAX;
    for (int c = tid; c < cols; c += THREADS) m = fmaxf(m, xr[c]);
    m = block_reduce<THREADS, true>(m, scratch);

    float s = 0.0f;
    for (int c = tid; c < cols; c += THREADS) s += expf(xr[c] - m);
    s = block_reduce<THREADS, false>(s, scratch);
    const float inv = 1.0f / s;

    for (int c = tid; c < cols; c += THREADS) yr[c] = expf(xr[c] - m) * inv;
}

inline void launch_softmax_warp_shuffle(const float* x, float* y, int rows, int cols) {
    constexpr int kThreads = 256;
    softmax_warp_shuffle_kernel<kThreads><<<rows, kThreads>>>(x, y, rows, cols);
    CUDA_CHECK_LAST();
}
