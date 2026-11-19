// Warp- and block-level reductions shared by the softmax and LayerNorm kernels.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// Block-wide reduction built from warp reductions: each warp reduces with
// shuffles, the per-warp results go through shared memory, warp 0 combines them.
// Returns the result to every thread. `scratch` must hold at least 32 floats.
template <int THREADS, bool IS_MAX>
__device__ __forceinline__ float block_reduce(float v, float* scratch) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "THREADS must be warps, <= 1024");
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
