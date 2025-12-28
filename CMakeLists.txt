cmake_minimum_required(VERSION 3.20)
project(CatCu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

option(ENABLE_CUBLAS "Enable cuBLAS backend" ON)
option(ENABLE_CUBLASLT "Enable cuBLASLt backend" ON)

# Recommended flags for perf testing
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Let user pass -DCMAKE_CUDA_ARCHITECTURES=80;86;89 etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench
        src/Interface.cu
        main.cpp
)

target_include_directories(gemm_bench PRIVATE include)

if(ENABLE_CUBLAS)
    target_compile_definitions(gemm_bench PRIVATE ENABLE_CUBLAS=1)
    target_link_libraries(gemm_bench PRIVATE CUDA::cublas)
endif()

if(ENABLE_CUBLASLT)
    target_compile_definitions(gemm_bench PRIVATE ENABLE_CUBLASLT=1)
    # CUDA::cublasLt is available with CUDAToolkit package on modern CMake
    target_link_libraries(gemm_bench PRIVATE CUDA::cublasLt)
endif()

target_link_libraries(gemm_bench PRIVATE CUDA::cudart)
