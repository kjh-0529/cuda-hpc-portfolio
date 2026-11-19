// Step 3: row in registers + row-length dispatcher (same design as softmax step 5).
//
// A group of threads loads its row into registers with float4 loads and computes
// everything from registers: x is read once, like a copy. Because the row is in
// registers, the variance can use the exact two-pass formula (mean first, then
// sum of squared deviations) at no extra memory cost; Welford is not needed here.
//
//   <= 1024 columns: one warp per row, 4 rows per 128-thread block, shuffle-only
//   <= 8192 columns: one 256-thread block per row
//   longer rows, or cols % 4 != 0: Welford kernel (step 2)
//
// gamma and beta are read with float4 too; they are shared by all rows and stay
// in cache, so they add little DRAM traffic.
#pragma once

#include "cuda_check.cuh"
#include "kernels/02_welford.cuh"
#include "kernels/common.cuh"

template <int ROW_THREADS, int VEC>
__global__ void __launch_bounds__(ROW_THREADS == 32 ? 128 : ROW_THREADS)
    layernorm_registers_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                               const float* __restrict__ beta, float* __restrict__ y,
                               int rows, int cols, float eps) {
    constexpr int kRowsPerBlock = (ROW_THREADS == 32) ? 4 : 1;
    __shared__ float scratch[32];  // block-per-row variant only

    const int t = threadIdx.x % ROW_THREADS;
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / ROW_THREADS;
    const bool active = row < rows;
    const int vec_cols = cols / 4;
    const float4* xr = reinterpret_cast<const float4*>(x + static_cast<size_t>(row) * cols);
    float4* yr = reinterpret_cast<float4*>(y + static_cast<size_t>(row) * cols);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);

    float v[VEC * 4];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        const int idx = i * ROW_THREADS + t;
        float4 q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (active && idx < vec_cols) q = xr[idx];
        v[4 * i + 0] = q.x; v[4 * i + 1] = q.y; v[4 * i + 2] = q.z; v[4 * i + 3] = q.w;
        s += (q.x + q.y) + (q.z + q.w);  // padding slots are 0
    }
    if constexpr (ROW_THREADS == 32) s = warp_reduce_sum(s);
    else s = block_reduce<ROW_THREADS, false>(s, scratch);
    const float mean = s / cols;

    float q2 = 0.0f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        if (i * ROW_THREADS + t < vec_cols) {  // skip padding, it is not part of the row
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float d = v[4 * i + k] - mean;
                q2 += d * d;
            }
        }
    }
    if constexpr (ROW_THREADS == 32) q2 = warp_reduce_sum(q2);
    else q2 = block_reduce<ROW_THREADS, false>(q2, scratch);
    const float rstd = rsqrtf(q2 / cols + eps);

    if (!active) return;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        const int idx = i * ROW_THREADS + t;
        if (idx < vec_cols) {
            const float4 g = g4[idx];
            const float4 b = b4[idx];
            yr[idx] = make_float4((v[4 * i + 0] - mean) * rstd * g.x + b.x,
                                  (v[4 * i + 1] - mean) * rstd * g.y + b.y,
                                  (v[4 * i + 2] - mean) * rstd * g.z + b.z,
                                  (v[4 * i + 3] - mean) * rstd * g.w + b.w);
        }
    }
}

template <int ROW_THREADS, int VEC>
inline void launch_ln_registers_config(const float* x, const float* gamma, const float* beta,
                                       float* y, int rows, int cols, float eps) {
    if constexpr (ROW_THREADS == 32) {
        layernorm_registers_kernel<32, VEC>
            <<<(rows + 3) / 4, 128>>>(x, gamma, beta, y, rows, cols, eps);
    } else {
        layernorm_registers_kernel<ROW_THREADS, VEC>
            <<<rows, ROW_THREADS>>>(x, gamma, beta, y, rows, cols, eps);
    }
    CUDA_CHECK_LAST();
}

inline const char* layernorm_auto_path(int cols) {
    if (cols % 4 != 0) return "02_welford (cols % 4 != 0)";
    if (cols <= 128) return "03 warp/row, 4 floats/thread";
    if (cols <= 256) return "03 warp/row, 8 floats/thread";
    if (cols <= 512) return "03 warp/row, 16 floats/thread";
    if (cols <= 1024) return "03 warp/row, 32 floats/thread";
    if (cols <= 2048) return "03 block/row, 8 floats/thread";
    if (cols <= 4096) return "03 block/row, 16 floats/thread";
    if (cols <= 8192) return "03 block/row, 32 floats/thread";
    return "02_welford (row too long for registers)";
}

inline void launch_layernorm_auto(const float* x, const float* gamma, const float* beta,
                                  float* y, int rows, int cols, float eps) {
    if (cols % 4 != 0 || cols > 8192)
        return launch_layernorm_welford(x, gamma, beta, y, rows, cols, eps);
    if (cols <= 128) return launch_ln_registers_config<32, 1>(x, gamma, beta, y, rows, cols, eps);
    if (cols <= 256) return launch_ln_registers_config<32, 2>(x, gamma, beta, y, rows, cols, eps);
    if (cols <= 512) return launch_ln_registers_config<32, 4>(x, gamma, beta, y, rows, cols, eps);
    if (cols <= 1024) return launch_ln_registers_config<32, 8>(x, gamma, beta, y, rows, cols, eps);
    if (cols <= 2048) return launch_ln_registers_config<256, 2>(x, gamma, beta, y, rows, cols, eps);
    if (cols <= 4096) return launch_ln_registers_config<256, 4>(x, gamma, beta, y, rows, cols, eps);
    launch_ln_registers_config<256, 8>(x, gamma, beta, y, rows, cols, eps);
}
