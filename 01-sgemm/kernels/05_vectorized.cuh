// Step 5: vectorized (float4) loads and a transposed A tile in shared memory.
//
// Same tiling as step 4 (128x128 block tile, BK = 8, 8x8 results per thread,
// 256 threads). Three changes:
//
//   1. Global loads use float4: each thread loads one float4 of A and one float4
//      of B per tile (4 floats per instruction instead of 1).
//   2. The A tile is stored TRANSPOSED in shared memory (As[k][m] instead of
//      As[m][k]). In the inner loop a thread needs 8 consecutive rows of A for one
//      k; after the transpose those 8 values are contiguous, so they are read with
//      two float4 loads instead of eight strided scalar loads.
//   3. The 8 values of B and the 8 results per row of C are also moved as two
//      float4 each.
//
// float4 accesses need 16-byte alignment, so this kernel requires K % 4 == 0 and
// N % 4 == 0. Other sizes fall back to the step 4 kernel. Tile edges are handled
// by zero-padding whole float4s: with K and N multiples of 4, a float4 is either
// fully inside the matrix or fully outside it.
#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "kernels/04_register_blocking.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_vectorized_kernel(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B, float beta,
                            float* __restrict__ C) {
    constexpr int kThreads = (BM / TM) * (BN / TN);
    static_assert(BM * BK == 4 * kThreads, "one float4 of A per thread per tile");
    static_assert(BK * BN == 4 * kThreads, "one float4 of B per thread per tile");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "TM and TN must be multiples of 4");

    __shared__ __align__(16) float As[BK * BM];  // transposed: As[k * BM + m]
    __shared__ __align__(16) float Bs[BK * BN];  // row-major:  Bs[k * BN + n]

    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);

    // Each thread loads one float4 of the A tile and one float4 of the B tile.
    const int aRow = tid / (BK / 4);         // 0 .. BM-1
    const int aCol = (tid % (BK / 4)) * 4;   // 0 or 4
    const int bRow = tid / (BN / 4);         // 0 .. BK-1
    const int bCol = (tid % (BN / 4)) * 4;   // 0, 4, ..., BN-4

    float acc[TM][TN] = {};
    float regA[TM];
    float regB[TN];
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        {
            const int gRow = blockRow + aRow, gCol = k0 + aCol;
            const float4 v = (gRow < M && gCol < K)
                                 ? *reinterpret_cast<const float4*>(&A[gRow * K + gCol])
                                 : zero4;
            As[(aCol + 0) * BM + aRow] = v.x;
            As[(aCol + 1) * BM + aRow] = v.y;
            As[(aCol + 2) * BM + aRow] = v.z;
            As[(aCol + 3) * BM + aRow] = v.w;
        }
        {
            const int gRow = k0 + bRow, gCol = blockCol + bCol;
            const float4 v = (gRow < K && gCol < N)
                                 ? *reinterpret_cast<const float4*>(&B[gRow * N + gCol])
                                 : zero4;
            *reinterpret_cast<float4*>(&Bs[bRow * BN + bCol]) = v;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 a =
                    *reinterpret_cast<const float4*>(&As[k * BM + threadRow * TM + i]);
                regA[i] = a.x; regA[i + 1] = a.y; regA[i + 2] = a.z; regA[i + 3] = a.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 b =
                    *reinterpret_cast<const float4*>(&Bs[k * BN + threadCol * TN + j]);
                regB[j] = b.x; regB[j + 1] = b.y; regB[j + 2] = b.z; regB[j + 3] = b.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
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

inline void launch_sgemm_vectorized(int M, int N, int K, float alpha, const float* A,
                                    const float* B, float beta, float* C) {
    if (K % 4 != 0 || N % 4 != 0) {
        // float4 rows would be misaligned: use the scalar step 4 kernel instead.
        launch_sgemm_register_blocking(M, N, K, alpha, A, B, beta, C);
        return;
    }
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));  // 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_vectorized_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
}
