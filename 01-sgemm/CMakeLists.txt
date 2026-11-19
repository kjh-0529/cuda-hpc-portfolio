add_executable(sgemm_bench bench.cu)
target_include_directories(sgemm_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(sgemm_bench PRIVATE CUDA::cudart CUDA::cublas)
