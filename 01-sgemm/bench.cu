// SGEMM benchmark: runs every registered kernel and cuBLAS on square matrices,
// checks each kernel against cuBLAS, and prints CSV to stdout.
//
// Usage: ./sgemm_bench [size ...]        (default: 512 1024 2048 4096)
// Example: ./sgemm_bench 1024 4096 > results/rtx4090.csv

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cublas_check.cuh"
#include "cuda_check.cuh"
#include "timer.cuh"
#include "verify.h"

#include "kernels/01_naive.cuh"
#include "kernels/02_coalesced.cuh"
// Add new steps here, e.g. #include "kernels/03_smem_tiling.cuh"

using LaunchFn = void (*)(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C);

struct KernelEntry {
    const char* name;
    LaunchFn launch;
};

// Register each optimization step here, in order.
static const KernelEntry kKernels[] = {
    {"01_naive", launch_sgemm_naive},
    {"02_coalesced", launch_sgemm_coalesced},
};

// cuBLAS is column-major. For row-major C = A * B we compute C^T = B^T * A^T,
// which in column-major terms is just swapping the A and B arguments.
static void cublas_sgemm(cublasHandle_t handle, int M, int N, int K, float alpha,
                         const float* A, const float* B, float beta, float* C) {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             B, N, A, K, &beta, C, N));
}

static double gflops(int M, int N, int K, float ms) {
    return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {512, 1024, 2048, 4096};

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    bool all_ok = true;

    std::printf("kernel,size,ms,gflops,pct_of_cublas,correct\n");
    for (int n : sizes) {
        const int M = n, N = n, K = n;
        const size_t bytesA = sizeof(float) * M * K;
        const size_t bytesB = sizeof(float) * K * N;
        const size_t bytesC = sizeof(float) * M * N;

        std::vector<float> hA(M * K), hB(K * N), hRef(M * N), hOut(M * N);
        fill_random(hA, 1);
        fill_random(hB, 2);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

        // Reference result and timing from cuBLAS.
        float cublas_ms = time_average_ms(
            [&] { cublas_sgemm(handle, M, N, K, alpha, dA, dB, beta, dC); });
        CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        const double cublas_gf = gflops(M, N, K, cublas_ms);
        std::printf("cublas,%d,%.4f,%.1f,100.0,ref\n", n, cublas_ms, cublas_gf);

        for (const auto& kern : kKernels) {
            // Correctness run on a cleared output buffer.
            CUDA_CHECK(cudaMemset(dC, 0, bytesC));
            kern.launch(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytesC, cudaMemcpyDeviceToHost));
            // Tolerance grows with K because FP32 summation order differs from cuBLAS.
            const bool ok = all_close(hOut, hRef, 1e-4f * K, 1e-3f);
            all_ok = all_ok && ok;

            float ms = time_average_ms(
                [&] { kern.launch(M, N, K, alpha, dA, dB, beta, dC); });
            const double gf = gflops(M, N, K, ms);
            std::printf("%s,%d,%.4f,%.1f,%.1f,%s\n", kern.name, n, ms, gf,
                        100.0 * gf / cublas_gf, ok ? "yes" : "NO");
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
