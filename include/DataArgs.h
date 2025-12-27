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