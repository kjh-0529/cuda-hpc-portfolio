cmake_minimum_required(VERSION 3.24)

# Default GPU targets: T4 (75), A100 (80), RTX 30xx (86), RTX 40xx / L4 (89).
# Override with -DCMAKE_CUDA_ARCHITECTURES=native to build only for the local GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

project(cuda_hpc_portfolio LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -lineinfo keeps source line info so Nsight Compute can map metrics to code.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(CUDAToolkit REQUIRED)

add_subdirectory(01-sgemm)
