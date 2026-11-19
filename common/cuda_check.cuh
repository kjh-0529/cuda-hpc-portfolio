// Error-checking macros for CUDA runtime calls. Code that cannot exit the process
// (for example a PyTorch extension) can define its own CUDA_CHECK first.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(err_), __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
#endif

// Checks for errors from the most recent kernel launch.
#ifndef CUDA_CHECK_LAST
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
#endif
