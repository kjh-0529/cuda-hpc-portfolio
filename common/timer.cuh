// GPU timing with CUDA events.
#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    void stop(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(stop_, stream)); }

    // Milliseconds between start() and stop(). Blocks until stop() has completed.
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// Runs fn() `warmup` times untimed, then `iters` times timed.
// Returns the average milliseconds per call.
template <typename Fn>
float time_average_ms(Fn fn, int warmup = 2, int iters = 10) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) fn();
    timer.stop();
    return timer.elapsed_ms() / static_cast<float>(iters);
}
