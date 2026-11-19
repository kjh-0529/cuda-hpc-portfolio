// Step 4: online softmax (Milakov & Gimelshein, 2018).
//
// Steps 1-3 read x three times: once for the max, once for the sum, once for the
// output. The sum needs the max first, because every term is exp(x - max).
//
// Online softmax keeps a running pair (m, d): the max seen so far and the sum of
// exp(x - m) so far. When a larger value arrives, the old sum is rescaled:
//     x > m:  d = d * exp(m - x) + 1,  m = x
//     else:   d = d + exp(x - m)
// Two partial pairs combine the same way:
//     m = max(m1, m2),  d = d1 * exp(m1 - m) + d2 * exp(m2 - m)
// so the max and the sum come out of ONE pass over x, and a second pass writes y.
// Global traffic drops from 3 reads + 1 write to 2 reads + 1 write.
//
// The pair is reduced across the block with warp shuffles, as in step 3.
#pragma once

#include "cuda_check.cuh"
#include "kernels/common.cuh"

struct MaxSum {
    float m;  // running max
    float d;  // running sum of exp(x - m)
};

__device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.m, b.m);
    // -FLT_MAX (not -inf) as the empty value keeps exp(m_i - m) finite: never NaN.
    return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

__device__ __forceinline__ MaxSum warp_reduce_maxsum(MaxSum v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        MaxSum o;
        o.m = __shfl_xor_sync(0xffffffffu, v.m, offset);
        o.d = __shfl_xor_sync(0xffffffffu, v.d, offset);
        v = combine(v, o);
    }
    return v;
}

template <int THREADS>
__device__ __forceinline__ MaxSum block_reduce_maxsum(MaxSum v, MaxSum* scratch) {
    constexpr int kWarps = THREADS / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    v = warp_reduce_maxsum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        MaxSum w = (lane < kWarps) ? scratch[lane] : MaxSum{-FLT_MAX, 0.0f};
        w = warp_reduce_maxsum(w);
        if (lane == 0) scratch[0] = w;
    }
    __syncthreads();
    return scratch[0];
}

template <int THREADS>
__global__ void softmax_online_kernel(const float* __restrict__ x, float* __restrict__ y,
                                      int rows, int cols) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "THREADS must be warps, <= 1024");
    __shared__ MaxSum scratch[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    // Pass 1: running max and sum in one sweep, one exp per element.
    MaxSum t{-FLT_MAX, 0.0f};
    for (int c = tid; c < cols; c += THREADS) {
        const float v = xr[c];
        if (v > t.m) {
            t.d = t.d * expf(t.m - v) + 1.0f;
            t.m = v;
        } else {
            t.d += expf(v - t.m);
        }
    }
    const MaxSum r = block_reduce_maxsum<THREADS>(t, scratch);
    const float inv = 1.0f / r.d;

    // Pass 2: outputs.
    for (int c = tid; c < cols; c += THREADS) yr[c] = expf(xr[c] - r.m) * inv;
}

inline void launch_softmax_online(const float* x, float* y, int rows, int cols) {
    constexpr int kThreads = 256;
    softmax_online_kernel<kThreads><<<rows, kThreads>>>(x, y, rows, cols);
    CUDA_CHECK_LAST();
}
