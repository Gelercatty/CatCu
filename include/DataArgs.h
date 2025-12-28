#pragma once

// C_M_N = A_M_K x B_K_N
struct MatmulProblem
{
    int M, N, K;
};

struct MatmulArgs
{
    const float *A;
    const float *B;
    float* C;
};