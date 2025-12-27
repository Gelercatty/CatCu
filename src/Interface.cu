#pragma once

#include <device_launch_parameters.h>
#include <unistd.h>

#include "../include/cuda_common.cuh"
#include "../include/DataArgs.h"
#include "cublas.h"
#include "cuda_runtime.h"
class IMatmul
{
public:
    virtual ~IMatmul() = default;
    virtual const char* name() const = 0;
    virtual void init(const MatmulProblem&, cudaStream_t){}
    virtual void run(const MatmulProblem& p, const MatmulArgs& a, cudaStream_t) = 0;
    virtual void fini() {}
};

class CublaseMatMul final : public IMatmul
{
private:
    cublasHandle_t h_{nullptr};
    bool tf32_ {false};
public:
    explicit CublaseMatMul(bool tf32 = false): tf32_(tf32)
    {
        CUBLAS_CHECK(cublasCreate_v2(&h_));
    }
    ~CublaseMatMul() override
    {
        if (tf32_) cublasDestroy_v2(h_);
    }
    const char* name() const override {return tf32_? "cuBLAS SGEMM (TF32?)" : "cuBLAS SGEMM";}

    void run(const MatmulProblem& p, const MatmulArgs& a, cudaStream_t stream) override
    {
        CUBLAS_CHECK(cublasSetStream_v2(h_, stream));
#if defined(CUBLAS_TF32_TENSOR_OP_MATH)
        CUBLAS_CHECK(cublasSetMathMode(h_, tf32_ ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
#endif

        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm_v2(
            h_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.M, p.N, p.K,
            &alpha,
            a.B, p.N,
            a.A, p.K,
            &beta,
            a.C, p.N
        ));
    }
};


template<int TILE>
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;
    int tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t)
    {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();


    }

}