// Step 5: keep the whole row in registers, load it with float4, read x ONCE.
//
// Steps 1-4 read x two or three times. Here a group of threads loads its row
// into registers with float4 loads, computes the max and the sum from those
// registers, and writes y from them: 1 read + 1 write, the same traffic as a copy.
//
// Two group sizes, chosen at compile time:
//   - ROW_THREADS = 32: one WARP per row, for short rows (<= 1024 columns).
//     Reductions are pure warp shuffles: no shared memory, no __syncthreads().
//     Each 128-thread block handles 4 rows, so short rows no longer leave most of
//     a 256-thread block idle (the weakness of steps 2-4 at 256 columns).
//   - ROW_THREADS = 256: one BLOCK per row, for medium rows (<= 8192 columns).
// Each thread holds VEC float4s (4 * VEC floats). Thread t of the group loads
// float4 number v * ROW_THREADS + t, so consecutive threads read consecutive
// 16-byte chunks (coalesced). Slots past the end of the row hold -FLT_MAX, which
// adds exp(-huge) = 0 to the sum and is never written.
//
// Requires cols % 4 == 0 (float4 alignment) and cols <= ROW_THREADS * VEC * 4.
// launch_softmax_auto() below picks the configuration, or another kernel.
#pragma once

#include "cuda_check.cuh"
#include "03_warp_shuffle.cuh"
#include "04_online.cuh"
#include "common.cuh"

template <int ROW_THREADS, int VEC>
__global__ void __launch_bounds__(ROW_THREADS == 32 ? 128 : ROW_THREADS)
    softmax_registers_kernel(const float* __restrict__ x, float* __restrict__ y, int rows,
                             int cols) {
    constexpr int kRowsPerBlock = (ROW_THREADS == 32) ? 4 : 1;
    __shared__ float scratch[32];  // only used by the block-per-row variant

    const int t = threadIdx.x % ROW_THREADS;  // index inside the row group
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / ROW_THREADS;
    // A warp-per-row block may run past the last row; those warps only skip
    // memory accesses (they still take part in nothing block-wide).
    const bool active = row < rows;

    const int vec_cols = cols / 4;
    const float4* xr = reinterpret_cast<const float4*>(x + static_cast<size_t>(row) * cols);
    float4* yr = reinterpret_cast<float4*>(y + static_cast<size_t>(row) * cols);

    float v[VEC * 4];
    float m = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        const int idx = i * ROW_THREADS + t;
        float4 q = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
        if (active && idx < vec_cols) q = xr[idx];
        v[4 * i + 0] = q.x; v[4 * i + 1] = q.y; v[4 * i + 2] = q.z; v[4 * i + 3] = q.w;
        m = fmaxf(m, fmaxf(fmaxf(q.x, q.y), fmaxf(q.z, q.w)));
    }
    if constexpr (ROW_THREADS == 32) {
        m = warp_reduce_max(m);
    } else {
        m = block_reduce<ROW_THREADS, true>(m, scratch);
    }

    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < VEC * 4; ++k) {
        v[k] = expf(v[k] - m);  // padding slots become 0
        s += v[k];
    }
    if constexpr (ROW_THREADS == 32) {
        s = warp_reduce_sum(s);
    } else {
        s = block_reduce<ROW_THREADS, false>(s, scratch);
    }
    const float inv = 1.0f / s;

    if (!active) return;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        const int idx = i * ROW_THREADS + t;
        if (idx < vec_cols) {
            yr[idx] = make_float4(v[4 * i] * inv, v[4 * i + 1] * inv, v[4 * i + 2] * inv,
                                  v[4 * i + 3] * inv);
        }
    }
}

template <int ROW_THREADS, int VEC>
inline void launch_registers_config(const float* x, float* y, int rows, int cols, cudaStream_t stream = 0) {
    if constexpr (ROW_THREADS == 32) {
        softmax_registers_kernel<32, VEC><<<(rows + 3) / 4, 128, 0, stream>>>(x, y, rows, cols);
    } else {
        softmax_registers_kernel<ROW_THREADS, VEC><<<rows, ROW_THREADS, 0, stream>>>(x, y, rows, cols);
    }
    CUDA_CHECK_LAST();
}

// Which implementation launch_softmax_auto() uses for a shape (for logging).
inline const char* softmax_auto_path(int cols) {
    if (cols % 4 != 0) return cols <= 2048 ? "03_warp_shuffle (cols % 4 != 0)"
                                           : "04_online (cols % 4 != 0)";
    if (cols <= 128) return "05 warp/row, 4 floats/thread";
    if (cols <= 256) return "05 warp/row, 8 floats/thread";
    if (cols <= 512) return "05 warp/row, 16 floats/thread";
    if (cols <= 1024) return "05 warp/row, 32 floats/thread";
    if (cols <= 2048) return "05 block/row, 8 floats/thread";
    if (cols <= 4096) return "05 block/row, 16 floats/thread";
    if (cols <= 8192) return "05 block/row, 32 floats/thread";
    return "04_online (row too long for registers)";
}

// Dispatcher: one warp per row for short rows, one block per row with the row in
// registers for medium rows, online softmax for rows too long to hold in registers.
// Rows whose length is not a multiple of 4 cannot use float4 and fall back to the
// best scalar kernel measured for that length on the T4.
inline void launch_softmax_auto(const float* x, float* y, int rows, int cols, cudaStream_t stream = 0) {
    if (cols % 4 != 0) {
        if (cols <= 2048) launch_softmax_warp_shuffle(x, y, rows, cols, stream);
        else launch_softmax_online(x, y, rows, cols, stream);
        return;
    }
    if (cols <= 128) return launch_registers_config<32, 1>(x, y, rows, cols, stream);
    if (cols <= 256) return launch_registers_config<32, 2>(x, y, rows, cols, stream);
    if (cols <= 512) return launch_registers_config<32, 4>(x, y, rows, cols, stream);
    if (cols <= 1024) return launch_registers_config<32, 8>(x, y, rows, cols, stream);
    if (cols <= 2048) return launch_registers_config<256, 2>(x, y, rows, cols, stream);
    if (cols <= 4096) return launch_registers_config<256, 4>(x, y, rows, cols, stream);
    if (cols <= 8192) return launch_registers_config<256, 8>(x, y, rows, cols, stream);
    launch_softmax_online(x, y, rows, cols, stream);
}
