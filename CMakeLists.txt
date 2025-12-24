cmake_minimum_required(VERSION 3.20)

project(CatCu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD  17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
add_executable(hello_cuda test/hello_cuda.cu)
target_link_libraries(hello_cuda PRIVATE CUDA::cudart)

set_property(TARGET hello_cuda PROPERTY CUDA_ARCHITECTURES 80)


