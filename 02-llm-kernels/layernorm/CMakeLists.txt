add_executable(layernorm_bench bench_layernorm.cu)
target_include_directories(layernorm_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(layernorm_bench PRIVATE CUDA::cudart)
