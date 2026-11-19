// Shared definitions for the LayerNorm kernels.
//
// LayerNorm over each row of a [rows x cols] matrix:
//   mean = sum(x) / cols
//   var  = sum((x - mean)^2) / cols            (biased, as in PyTorch)
//   y    = (x - mean) / sqrt(var + eps) * gamma + beta
// gamma and beta have `cols` elements and are shared by every row.
#pragma once

#include <cuda_runtime.h>

#include "reduce.cuh"

using LayerNormLaunchFn = void (*)(const float* x, const float* gamma, const float* beta,
                                   float* y, int rows, int cols, float eps);
