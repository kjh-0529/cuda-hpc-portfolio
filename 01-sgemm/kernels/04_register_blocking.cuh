// Step 4: 2D register blocking.
//
// Each block computes a BM x BN = 128 x 128 tile of C with 256 threads, so each
// thread computes a TM x TN = 8 x 8 sub-block of C (64 results instead of 1).
//
// Per step along K (BK = 8):
//   1. The block loads a 128x8 tile of A and an 8x128 tile of B into shared memory
//      (4 + 4 elements per thread; the B loads are fully coalesced).
//   2. For each of the 8 k values, a thread copies 8 values of A and 8 values of B
//      from shared memory into registers, then does 8 x 8 = 64 multiply-adds from
//      registers.
//
// Shared-memory loads per multiply-add drop from 2 (step 3) to 16 / 64 = 0.25.
// Blocks are 256 threads, so several blocks fit on one Turing SM (1024-thread
// limit) and keep it busy while another block waits at __syncthreads().
//
// Sizes that are not multiples of the tile are handled by zero-padding loads and
// skipping out-of-range stores.
#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_register_blocking_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
    constexpr int kThreads = (BM / TM) * (BN / TN);
    static_assert(kThreads % BK == 0, "A tile load needs kThreads % BK == 0");
    static_assert(kThreads % BN == 0, "B tile load needs kThreads % BN == 0");

    __shared__ float As[BM * BK];  // row-major BM x BK
    __shared__ float Bs[BK * BN];  // row-major BK x BN

    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;  // first row of C for this block
    const int blockCol = blockIdx.x * BN;  // first column of C for this block

    // Position of this thread's TM x TN sub-block inside the block tile.
    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);

    // Mapping for loading the A tile (BM x BK) and B tile (BK x BN).
    const int aRow = tid / BK, aCol = tid % BK;
    constexpr int aStride = kThreads / BK;  // rows of A loaded per pass
    const int bRow = tid / BN, bCol = tid % BN;
    constexpr int bStride = kThreads / BN;  // rows of B loaded per pass

    float acc[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int r = aRow; r < BM; r += aStride) {
            const int gRow = blockRow + r, gCol = k0 + aCol;
            As[r * BK + aCol] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        for (int r = bRow; r < BK; r += bStride) {
            const int gRow = k0 + r, gCol = blockCol + bCol;
            Bs[r * BN + bCol] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[k * BN + threadCol * TN + j];
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
        for (int j = 0; j < TN; ++j) {
            const int col = blockCol + threadCol * TN + j;
            if (col < N) C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

inline void launch_sgemm_register_blocking(int M, int N, int K, float alpha,
                                           const float* A, const float* B, float beta,
                                           float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));  // 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_blocking_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
}
