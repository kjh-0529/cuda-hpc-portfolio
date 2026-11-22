add_executable(gpu_stream gpu_stream.cu)
target_include_directories(gpu_stream PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(gpu_stream PRIVATE CUDA::cudart)
