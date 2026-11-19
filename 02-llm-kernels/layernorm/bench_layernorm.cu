// LayerNorm benchmark: runs every registered kernel on row-major [rows x cols]
// FP32 inputs, checks each against a double-precision CPU reference, and prints CSV.
//
// Inputs are 100 + uniform(-5, 5): a large mean relative to the spread, where the
// naive E[x^2] - E[x]^2 variance formula loses about 3 digits in FP32. gamma ~ U(0.5, 1.5),
// beta ~ U(-0.5, 0.5), eps = 1e-5.
//
// Metric and timing match the softmax benchmark: effective bandwidth (x read once
// + y written once) as a percentage of a same-size device-to-device copy, with
// warm-up, median of 5 batches, and the copy re-timed right before every kernel.
//
// Usage: ./layernorm_bench [ROWSxCOLS ...]
//        (default: 8192x256 4096x1024 4096x4096 2048x8192 1024x32768)

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.cuh"
#include "timer.cuh"
#include "verify.h"

#include "kernels/01_two_pass.cuh"
#include "kernels/02_welford.cuh"
#include "kernels/03_registers.cuh"

struct KernelEntry {
    const char* name;
    LayerNormLaunchFn launch;
};

static const KernelEntry kKernels[] = {
    {"01_two_pass", launch_layernorm_two_pass},
    {"02_welford", launch_layernorm_welford},
    {"03_auto", launch_layernorm_auto},
};

struct Shape {
    int rows, cols;
};

static void layernorm_reference(const std::vector<float>& x, const std::vector<float>& g,
                                const std::vector<float>& b, std::vector<float>& y, int rows,
                                int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = &x[static_cast<size_t>(r) * cols];
        float* yr = &y[static_cast<size_t>(r) * cols];
        double mean = 0.0;
        for (int c = 0; c < cols; ++c) mean += xr[c];
        mean /= cols;
        double var = 0.0;
        for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
        var /= cols;
        const double rstd = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c)
            yr[c] = static_cast<float>((xr[c] - mean) * rstd * g[c] + b[c]);
    }
}

int main(int argc, char** argv) {
    std::vector<Shape> shapes;
    for (int i = 1; i < argc; ++i) {
        Shape sh{};
        if (std::sscanf(argv[i], "%dx%d", &sh.rows, &sh.cols) != 2 || sh.rows <= 0 ||
            sh.cols <= 0) {
            std::fprintf(stderr, "bad shape '%s', expected ROWSxCOLS\n", argv[i]);
            return EXIT_FAILURE;
        }
        shapes.push_back(sh);
    }
    if (shapes.empty()) shapes = {{8192, 256}, {4096, 1024}, {4096, 4096}, {2048, 8192}, {1024, 32768}};

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const float eps = 1e-5f;
    bool all_ok = true;
    bool first_shape = true;
    std::printf("kernel,rows,cols,ms,eff_gbps,pct_of_copy,copy_gbps,correct\n");

    for (const Shape& sh : shapes) {
        const size_t n = static_cast<size_t>(sh.rows) * sh.cols;
        const size_t bytes = n * sizeof(float);
        const size_t vbytes = sh.cols * sizeof(float);

        std::vector<float> hx(n), hy(n), href(n), hg(sh.cols), hb(sh.cols);
        fill_random(hx, 11);
        for (auto& v : hx) v = 100.0f + 5.0f * v;
        fill_random(hg, 12);
        for (auto& v : hg) v = 1.0f + 0.5f * v;
        fill_random(hb, 13);
        for (auto& v : hb) v *= 0.5f;
        layernorm_reference(hx, hg, hb, href, sh.rows, sh.cols, eps);

        float *dx, *dy, *dtmp, *dg, *db;
        CUDA_CHECK(cudaMalloc(&dx, bytes));
        CUDA_CHECK(cudaMalloc(&dy, bytes));
        CUDA_CHECK(cudaMalloc(&dtmp, bytes));
        CUDA_CHECK(cudaMalloc(&dg, vbytes));
        CUDA_CHECK(cudaMalloc(&db, vbytes));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dg, hg.data(), vbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, hb.data(), vbytes, cudaMemcpyHostToDevice));

        auto run_copy = [&] { CUDA_CHECK(cudaMemcpyAsync(dtmp, dx, bytes, cudaMemcpyDeviceToDevice)); };
        auto gbps = [&](float ms) { return 2.0 * bytes / (ms * 1e-3) / 1e9; };

        warm_up_gpu(run_copy, first_shape ? 2000.0f : 500.0f);
        first_shape = false;
        std::fprintf(stderr, "%dx%d: 03_auto uses %s\n", sh.rows, sh.cols,
                     layernorm_auto_path(sh.cols));

        const float copy_ms = time_median_ms(run_copy);
        std::printf("copy,%d,%d,%.4f,%.1f,100.0,%.1f,ref\n", sh.rows, sh.cols, copy_ms,
                    gbps(copy_ms), gbps(copy_ms));

        for (const auto& kern : kKernels) {
            CUDA_CHECK(cudaMemset(dy, 0, bytes));
            kern.launch(dx, dg, db, dy, sh.rows, sh.cols, eps, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
            const bool ok = all_close(hy, href, 5e-4f, 1e-3f);
            all_ok = all_ok && ok;

            const double near_copy = gbps(time_median_ms(run_copy));
            const float ms =
                time_median_ms([&] { kern.launch(dx, dg, db, dy, sh.rows, sh.cols, eps, 0); });
            const double eff = gbps(ms);
            std::printf("%s,%d,%d,%.4f,%.1f,%.1f,%.1f,%s\n", kern.name, sh.rows, sh.cols, ms,
                        eff, 100.0 * eff / near_copy, near_copy, ok ? "yes" : "NO");
        }

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(dtmp));
        CUDA_CHECK(cudaFree(dg));
        CUDA_CHECK(cudaFree(db));
    }
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
