#pragma once

#include "../include/Interface.cuh"
#include "cublas_v2.h"
class CublasMatMul final : public IMatmul
{
private:
    cublasHandle_t h_{nullptr};
    bool tf32_ {false};
public:
    explicit CublasMatMul(bool tf32 = false): tf32_(tf32)
    {
        CUBLAS_CHECK(cublasCreate_v2(&h_));
    }
    ~CublasMatMul() override
    {
        if (h_) cublasDestroy_v2(h_);
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
#pragma  unroll
        for (int i =0; i< TILE; ++i) acc +=  As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

class Tiled16Matmul final : public IMatmul
{
public:
    const char* name() const override { return "kernel tiled16"; }
    void run(const MatmulProblem& p, const MatmulArgs& a, cudaStream_t stream) override
    {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((p.N + TILE -1 ) / TILE, (p.M + TILE - 1) / TILE);
        matmul_tiled<TILE><<<grid, block, 0, stream>>>(a.A, a.B, a.C, p.M, p.N, p.K);
    }
};


std::unique_ptr<IMatmul> make_cublas(bool tf32){return std::make_unique<CublasMatMul>(tf32);}
std::unique_ptr<IMatmul> make_tiled16() { return std::make_unique<Tiled16Matmul>(); }

