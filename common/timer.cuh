// GPU timing with CUDA events.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

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

// Keeps the GPU busy with fn() for at least `ms` milliseconds so that clocks have
// ramped up before measurement starts.
template <typename Fn>
void warm_up_gpu(Fn fn, float ms) {
    GpuTimer timer;
    float elapsed = 0.0f;
    while (elapsed < ms) {
        timer.start();
        for (int i = 0; i < 10; ++i) fn();
        timer.stop();
        elapsed += timer.elapsed_ms();
    }
}

// Robust timing: the number of calls per batch is chosen so that each batch takes
// about `target_ms`, then `reps` batches are timed and the median per-call time
// is returned. Fast kernels get many calls per batch (less launch-overhead noise);
// slow kernels get few (shorter runs, less heating of the GPU).
template <typename Fn>
float time_median_ms(Fn fn, float target_ms = 50.0f, int reps = 5) {
    fn();  // warm-up: first launch, caches, lazy initialization
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    fn();
    timer.stop();
    const float est = std::max(timer.elapsed_ms(), 1e-3f);
    const int iters = std::clamp(static_cast<int>(std::ceil(target_ms / est)), 1, 10000);
    if (est > 500.0f) reps = std::min(reps, 3);  // very slow kernels: keep runs short

    std::vector<float> per_call(reps);
    for (int r = 0; r < reps; ++r) {
        timer.start();
        for (int i = 0; i < iters; ++i) fn();
        timer.stop();
        per_call[r] = timer.elapsed_ms() / static_cast<float>(iters);
    }
    std::nth_element(per_call.begin(), per_call.begin() + reps / 2, per_call.end());
    return per_call[reps / 2];
}
