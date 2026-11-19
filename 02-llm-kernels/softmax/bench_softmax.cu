// Softmax benchmark: runs every registered kernel on row-major [rows x cols]
// FP32 inputs, checks each against a double-precision CPU reference, and prints CSV.
//
// Softmax is memory-bound, so the metric is effective bandwidth:
//   eff_gbps = (bytes of x read once + bytes of y written once) / time
// and the baseline is a device-to-device copy of the same size (the fastest way
// to read x and write y once). pct_of_copy = eff_gbps / copy_gbps.
//
// Measurement follows 01-sgemm: GPU warm-up before each shape, median of 5
// batches of ~50 ms, and the copy baseline re-timed right before every kernel.
//
// Usage: ./softmax_bench [ROWSxCOLS ...]   (default: 8192x256 4096x1024 4096x4096 1024x32768)

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "timer.cuh"
#include "verify.h"

#include "kernels/01_naive.cuh"
#include "kernels/02_block_smem.cuh"
#include "kernels/03_warp_shuffle.cuh"
// Add new steps here.

struct KernelEntry {
    const char* name;
    SoftmaxLaunchFn launch;
};

static const KernelEntry kKernels[] = {
    {"01_naive", launch_softmax_naive},
    {"02_block_smem", launch_softmax_block_smem},
    {"03_warp_shuffle", launch_softmax_warp_shuffle},
};

struct Shape {
    int rows, cols;
};

static bool parse_shape(const char* s, Shape& out) {
    return std::sscanf(s, "%dx%d", &out.rows, &out.cols) == 2 && out.rows > 0 && out.cols > 0;
}

static void softmax_reference(const std::vector<float>& x, std::vector<float>& y, int rows,
                              int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = &x[static_cast<size_t>(r) * cols];
        float* yr = &y[static_cast<size_t>(r) * cols];
        double m = xr[0];
        for (int c = 1; c < cols; ++c) m = std::fmax(m, xr[c]);
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += std::exp(xr[c] - m);
        for (int c = 0; c < cols; ++c) yr[c] = static_cast<float>(std::exp(xr[c] - m) / s);
    }
}

int main(int argc, char** argv) {
    std::vector<Shape> shapes;
    for (int i = 1; i < argc; ++i) {
        Shape sh{};
        if (!parse_shape(argv[i], sh)) {
            std::fprintf(stderr, "bad shape '%s', expected ROWSxCOLS\n", argv[i]);
            return EXIT_FAILURE;
        }
        shapes.push_back(sh);
    }
    if (shapes.empty()) shapes = {{8192, 256}, {4096, 1024}, {4096, 4096}, {1024, 32768}};

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    bool all_ok = true;
    bool first_shape = true;
    std::printf("kernel,rows,cols,ms,eff_gbps,pct_of_copy,copy_gbps,correct\n");

    for (const Shape& sh : shapes) {
        const size_t n = static_cast<size_t>(sh.rows) * sh.cols;
        const size_t bytes = n * sizeof(float);

        std::vector<float> hx(n), hy(n), href(n);
        fill_random(hx, 7);
        for (auto& v : hx) v *= 5.0f;  // logits roughly in [-5, 5]
        softmax_reference(hx, href, sh.rows, sh.cols);

        float *dx, *dy, *dtmp;
        CUDA_CHECK(cudaMalloc(&dx, bytes));
        CUDA_CHECK(cudaMalloc(&dy, bytes));
        CUDA_CHECK(cudaMalloc(&dtmp, bytes));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

        auto run_copy = [&] { CUDA_CHECK(cudaMemcpyAsync(dtmp, dx, bytes, cudaMemcpyDeviceToDevice)); };
        auto gbps = [&](float ms) { return 2.0 * bytes / (ms * 1e-3) / 1e9; };

        warm_up_gpu(run_copy, first_shape ? 2000.0f : 500.0f);
        first_shape = false;

        const float copy_ms = time_median_ms(run_copy);
        const double copy_gbps = gbps(copy_ms);
        std::printf("copy,%d,%d,%.4f,%.1f,100.0,%.1f,ref\n", sh.rows, sh.cols, copy_ms,
                    copy_gbps, copy_gbps);

        for (const auto& kern : kKernels) {
            CUDA_CHECK(cudaMemset(dy, 0, bytes));
            kern.launch(dx, dy, sh.rows, sh.cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
            const bool ok = all_close(hy, href, 1e-6f, 1e-4f);
            all_ok = all_ok && ok;

            const double near_copy = gbps(time_median_ms(run_copy));
            const float ms = time_median_ms([&] { kern.launch(dx, dy, sh.rows, sh.cols); });
            const double eff = gbps(ms);
            std::printf("%s,%d,%d,%.4f,%.1f,%.1f,%.1f,%s\n", kern.name, sh.rows, sh.cols, ms,
                        eff, 100.0 * eff / near_copy, near_copy, ok ? "yes" : "NO");
        }

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(dtmp));
    }
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
