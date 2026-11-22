// GPU memory bandwidth: STREAM-style kernels (Copy, Mul, Add, Triad, Dot) on doubles,
// written from scratch in the style of STREAM / BabelStream.
//
// Sweeps the thread-block size and the memory type, and compares with cudaMemcpy and
// the theoretical peak:
//   device   cudaMalloc: arrays live in GPU memory
//   managed  cudaMallocManaged, initialized by the CPU: the first kernel migrates the
//            pages to the GPU (reported separately as "first_touch"), later ones run on
//            GPU-resident pages
//   host     pinned host memory mapped into the GPU (zero-copy): every access goes over PCIe
//
// Bytes counted per element: Copy/Mul 2 arrays, Add/Triad 3, Dot 2 (8 bytes each).
// Times are medians (common/timer.cuh), after a warm-up.
//
// Usage: ./gpu_stream [--n ELEMENTS] [--modes device,managed,host] [--tbsizes 64,128,...]

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "timer.cuh"

constexpr double kScalar = 0.4;
constexpr int kDotBlocks = 1024;  // partial sums, reduced on the host

__global__ void copy_k(const double* a, double* c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i];
}
__global__ void mul_k(double* b, const double* c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) b[i] = kScalar * c[i];
}
__global__ void add_k(const double* a, const double* b, double* c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
__global__ void triad_k(double* a, const double* b, const double* c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + kScalar * c[i];
}
// Grid-stride loop over a fixed number of blocks; each block writes one partial sum.
__global__ void dot_k(const double* a, const double* b, double* partial, size_t n) {
    extern __shared__ double s[];
    double sum = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        sum += a[i] * b[i];
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}
__global__ void init_k(double* a, double* b, double* c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { a[i] = 0.1; b[i] = 0.2; c[i] = 0.0; }
}

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) v.push_back(item);
    return v;
}

struct Arrays {
    double *a, *b, *c;          // pointers the kernels use
    double *ha, *hb, *hc;       // host pointers for the "host" mode (else unused)
};

static Arrays alloc(const std::string& mode, size_t n) {
    Arrays x{};
    const size_t bytes = n * sizeof(double);
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&x.a, bytes)); CUDA_CHECK(cudaMalloc(&x.b, bytes)); CUDA_CHECK(cudaMalloc(&x.c, bytes));
    } else if (mode == "managed") {
        CUDA_CHECK(cudaMallocManaged(&x.a, bytes)); CUDA_CHECK(cudaMallocManaged(&x.b, bytes));
        CUDA_CHECK(cudaMallocManaged(&x.c, bytes));
    } else if (mode == "host") {
        CUDA_CHECK(cudaHostAlloc(&x.ha, bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&x.hb, bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&x.hc, bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&x.a, x.ha, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&x.b, x.hb, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&x.c, x.hc, 0));
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        std::exit(EXIT_FAILURE);
    }
    return x;
}

static void release(const std::string& mode, Arrays& x) {
    if (mode == "host") { cudaFreeHost(x.ha); cudaFreeHost(x.hb); cudaFreeHost(x.hc); }
    else { cudaFree(x.a); cudaFree(x.b); cudaFree(x.c); }
}

// Reads one element of a GPU-visible array back to the host.
static double at(const double* p, size_t i) {
    double v;
    CUDA_CHECK(cudaMemcpy(&v, p + i, sizeof(double), cudaMemcpyDefault));
    return v;
}

int main(int argc, char** argv) {
    size_t n = size_t(1) << 25;  // 33.5M doubles = 268 MB per array, the BabelStream default
    std::vector<std::string> modes = {"device", "managed", "host"};
    std::vector<int> tbsizes = {64, 128, 256, 512, 1024};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--n")) n = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--modes")) modes = split(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--tbsizes")) {
            tbsizes.clear();
            for (const std::string& t : split(argv[i + 1])) tbsizes.push_back(std::stoi(t));
        }
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    const double peak = 2.0 * mem_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR: 2 transfers per clock
    const double bytes = n * sizeof(double);
    std::fprintf(stderr, "GPU: %s, theoretical peak %.1f GB/s, %zu doubles (%.1f MB) per array\n",
                 prop.name, peak, n, bytes / 1e6);

    std::printf("mode,tbsize,kernel,ms,gbps,pct_of_peak\n");
    auto row = [&](const std::string& mode, int tb, const char* k, float ms, double moved) {
        const double gbps = moved / (ms * 1e-3) / 1e9;
        std::printf("%s,%d,%s,%.4f,%.1f,%.1f\n", mode.c_str(), tb, k, ms, gbps, 100.0 * gbps / peak);
    };

    // Baseline: cudaMemcpy device-to-device reads and writes `bytes` once each.
    {
        double *src, *dst;
        CUDA_CHECK(cudaMalloc(&src, bytes)); CUDA_CHECK(cudaMalloc(&dst, bytes));
        CUDA_CHECK(cudaMemset(src, 0, bytes));
        auto run = [&] { CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)); };
        warm_up_gpu(run, 2000.0f);
        row("device", 0, "cudaMemcpy", time_median_ms(run), 2 * bytes);
        cudaFree(src); cudaFree(dst);
    }

    double* partial;
    CUDA_CHECK(cudaMallocManaged(&partial, kDotBlocks * sizeof(double)));
    bool all_ok = true;

    for (const std::string& mode : modes) {
        Arrays x = alloc(mode, n);
        for (int tb : tbsizes) {
            const unsigned grid = (unsigned)((n + tb - 1) / tb);
            if (mode == "managed") {
                // CPU writes the data, so the first GPU kernel has to migrate the pages.
                for (size_t i = 0; i < n; ++i) { x.a[i] = 0.1; x.b[i] = 0.2; x.c[i] = 0.0; }
                GpuTimer t;
                t.start();
                copy_k<<<grid, tb>>>(x.a, x.c, n);
                t.stop();
                CUDA_CHECK_LAST();
                row(mode, tb, "first_touch_copy", t.elapsed_ms(), 2 * bytes);
            }
            init_k<<<grid, tb>>>(x.a, x.b, x.c, n);
            CUDA_CHECK_LAST();

            // Correctness: one pass of each kernel from known values, then spot checks.
            copy_k<<<grid, tb>>>(x.a, x.c, n);
            mul_k<<<grid, tb>>>(x.b, x.c, n);
            add_k<<<grid, tb>>>(x.a, x.b, x.c, n);
            triad_k<<<grid, tb>>>(x.a, x.b, x.c, n);
            dot_k<<<kDotBlocks, tb, tb * sizeof(double)>>>(x.a, x.b, partial, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            // Expected values after one pass, in the kernels' order. Not exact: 0.4 * 0.1 is
            // not 0.04 in binary, and the GPU may fuse multiply-adds, so allow 1e-12 relative.
            double ea = 0.1, eb = 0.2, ec = 0.0;
            ec = ea; eb = kScalar * ec; ec = ea + eb; ea = eb + kScalar * ec;
            auto close = [](double got, double want) { return std::fabs(got - want) <= 1e-12 * std::fabs(want); };
            double dot = 0.0;
            for (int i = 0; i < kDotBlocks; ++i) dot += partial[i];
            bool ok = std::fabs(dot - ea * eb * n) <= 1e-8 * ea * eb * n;
            for (size_t i : {size_t(0), n / 2, n - 1})
                ok = ok && close(at(x.a, i), ea) && close(at(x.b, i), eb) && close(at(x.c, i), ec);
            if (!ok) std::fprintf(stderr, "verification FAILED: mode %s tbsize %d\n", mode.c_str(), tb);
            all_ok = all_ok && ok;

            const float budget = mode == "host" ? 300.0f : 50.0f;  // host mode is slow: fewer calls
            row(mode, tb, "copy", time_median_ms([&] { copy_k<<<grid, tb>>>(x.a, x.c, n); }, budget), 2 * bytes);
            row(mode, tb, "mul", time_median_ms([&] { mul_k<<<grid, tb>>>(x.b, x.c, n); }, budget), 2 * bytes);
            row(mode, tb, "add", time_median_ms([&] { add_k<<<grid, tb>>>(x.a, x.b, x.c, n); }, budget), 3 * bytes);
            row(mode, tb, "triad", time_median_ms([&] { triad_k<<<grid, tb>>>(x.a, x.b, x.c, n); }, budget), 3 * bytes);
            row(mode, tb, "dot", time_median_ms([&] {
                dot_k<<<kDotBlocks, tb, tb * sizeof(double)>>>(x.a, x.b, partial, n); }, budget), 2 * bytes);
            CUDA_CHECK_LAST();
        }
        release(mode, x);
    }
    cudaFree(partial);
    std::fprintf(stderr, all_ok ? "all results verified\n" : "SOME RESULTS WRONG\n");
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
