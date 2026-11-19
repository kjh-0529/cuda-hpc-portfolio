add_executable(softmax_bench bench_softmax.cu)
target_include_directories(softmax_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(softmax_bench PRIVATE CUDA::cudart)
