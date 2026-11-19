// Error-checking macro for cuBLAS calls.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error %s at %s:%d\n",                \
                         cublasGetStatusName(st_), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
