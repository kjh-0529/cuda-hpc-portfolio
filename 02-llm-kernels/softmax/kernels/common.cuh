// Shared helpers for the softmax kernels.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>

// Row-wise softmax: y[r, c] = exp(x[r, c] - max_r) / sum_c exp(x[r, c] - max_r).
// Subtracting the row maximum keeps exp() from overflowing.
using SoftmaxLaunchFn = void (*)(const float* x, float* y, int rows, int cols);

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
