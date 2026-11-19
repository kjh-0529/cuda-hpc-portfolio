// Step 6: double buffering.
//
// Same tiling and float4 accesses as step 5. The difference is WHEN the next
// tile is loaded. In step 5 every tile goes:
//     load from global -> store to shared -> sync -> compute -> sync
// so while a block waits for global memory it computes nothing (other blocks on
// the SM can hide part of that wait, but only 2 blocks fit per T4 SM).
//
// Here shared memory holds two buffers. While the block computes tile t from one
// buffer, each thread already issues the global loads for tile t+1 into
// registers; after computing, it writes those registers into the other buffer.
// The global-memory latency is overlapped with the 512 multiply-adds per thread
// of the current tile, and only one __syncthreads() is needed per tile.
//
// Turing (sm_75) has no cp.async (that arrived with sm_80), so the prefetch goes
// through registers: 8 extra floats per thread.
//
// Why one barrier is enough: tile t is computed from buffer t % 2 and tile t+1 is
// stored into buffer (t+1) % 2. That buffer was last read while computing tile
// t-1, and every thread passed the barrier at the end of iteration t-1 only after
// finishing that computation, so nobody is still reading it.
//
// Requires K % 4 == 0 and N % 4 == 0 (float4 alignment); otherwise it falls back
// to the step 4 kernel.
#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "kernels/04_register_blocking.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_double_buffering_kernel(int M, int N, int K, float alpha,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B, float beta,
                                  float* __restrict__ C) {
    constexpr int kThreads = (BM / TM) * (BN / TN);
    static_assert(BM * BK == 4 * kThreads, "one float4 of A per thread per tile");
    static_assert(BK * BN == 4 * kThreads, "one float4 of B per thread per tile");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "TM and TN must be multiples of 4");

    __shared__ __align__(16) float As[2][BK * BM];  // transposed: As[buf][k * BM + m]
    __shared__ __align__(16) float Bs[2][BK * BN];  // row-major:  Bs[buf][k * BN + n]

    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);

    const int aRow = tid / (BK / 4);
    const int aCol = (tid % (BK / 4)) * 4;
    const int bRow = tid / (BN / 4);
    const int bCol = (tid % (BN / 4)) * 4;

    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Global -> registers for the tile starting at k0.
    auto load_tile = [&](int k0, float4& va, float4& vb) {
        const int agRow = blockRow + aRow, agCol = k0 + aCol;
        va = (agRow < M && agCol < K)
                 ? *reinterpret_cast<const float4*>(&A[agRow * K + agCol])
                 : zero4;
        const int bgRow = k0 + bRow, bgCol = blockCol + bCol;
        vb = (bgRow < K && bgCol < N)
                 ? *reinterpret_cast<const float4*>(&B[bgRow * N + bgCol])
                 : zero4;
    };
    // Registers -> shared memory buffer `buf` (A transposed).
    auto store_tile = [&](int buf, const float4& va, const float4& vb) {
        As[buf][(aCol + 0) * BM + aRow] = va.x;
        As[buf][(aCol + 1) * BM + aRow] = va.y;
        As[buf][(aCol + 2) * BM + aRow] = va.z;
        As[buf][(aCol + 3) * BM + aRow] = va.w;
        *reinterpret_cast<float4*>(&Bs[buf][bRow * BN + bCol]) = vb;
    };

    float acc[TM][TN] = {};
    float regA[TM];
    float regB[TN];
    float4 nextA, nextB;

    // Prologue: tile 0 into buffer 0.
    load_tile(0, nextA, nextB);
    store_tile(0, nextA, nextB);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool has_next = k0 + BK < K;
        // Issue the global loads for the next tile now; they complete while we compute.
        if (has_next) load_tile(k0 + BK, nextA, nextB);

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 a =
                    *reinterpret_cast<const float4*>(&As[buf][k * BM + threadRow * TM + i]);
                regA[i] = a.x; regA[i + 1] = a.y; regA[i + 2] = a.z; regA[i + 3] = a.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 b =
                    *reinterpret_cast<const float4*>(&Bs[buf][k * BN + threadCol * TN + j]);
                regB[j] = b.x; regB[j + 1] = b.y; regB[j + 2] = b.z; regB[j + 3] = b.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
        }

        if (has_next) {
            store_tile(buf ^ 1, nextA, nextB);
            __syncthreads();  // next buffer complete, and everyone is done with this one
            buf ^= 1;
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = blockRow + threadRow * TM + i;
        if (row >= M) break;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            const int col = blockCol + threadCol * TN + j;
            if (col < N) {  // N % 4 == 0, so col + 3 < N as well
                float4* dst = reinterpret_cast<float4*>(&C[row * N + col]);
                float4 c = *dst;
                c.x = alpha * acc[i][j + 0] + beta * c.x;
                c.y = alpha * acc[i][j + 1] + beta * c.y;
                c.z = alpha * acc[i][j + 2] + beta * c.z;
                c.w = alpha * acc[i][j + 3] + beta * c.w;
                *dst = c;
            }
        }
    }
}

inline void launch_sgemm_double_buffering(int M, int N, int K, float alpha,
                                          const float* A, const float* B, float beta,
                                          float* C) {
    if (K % 4 != 0 || N % 4 != 0) {
        launch_sgemm_register_blocking(M, N, K, alpha, A, B, beta, C);
        return;
    }
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));  // 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_double_buffering_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
}
