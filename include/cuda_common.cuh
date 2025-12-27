#pragma once
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>


#define CUDA_CHECK(x) do {  \
cudaError_t e = (x); \
if ( e != cudaSuccess){ \
std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
std::exit(1); \
}\
} while(0)

#define CUBLAS_CHECK(x) do { \
cublasStatus_t s = (x); \
if (s != CUBLAS_STATUS_SUCCESS) { \
std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, (int)s); \
std::exit(1); \
} \
} while(0)
