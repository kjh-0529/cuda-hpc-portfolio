// Error-checking macros for CUDA runtime and cuBLAS calls.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

// Checks for errors from the most recent kernel launch.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
