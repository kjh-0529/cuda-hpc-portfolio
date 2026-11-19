// Shared helpers for the softmax kernels.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>

#include "reduce.cuh"  // warp_reduce_max/sum, block_reduce (02-llm-kernels/common)

// Row-wise softmax: y[r, c] = exp(x[r, c] - max_r) / sum_c exp(x[r, c] - max_r).
// Subtracting the row maximum keeps exp() from overflowing.
using SoftmaxLaunchFn = void (*)(const float* x, float* y, int rows, int cols);
