#pragma once
#include "cuda_common.cuh"
#include "DataArgs.h"

inline void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) x = dist(rng);
}

inline void verify_host(const std::vector<float>& ref, const std::vector<float>& out)
{
    double max_abs = 0, max_rel = 0, mse = 0;
    for (size_t i = 0; i<ref.size(); ++i)
    {
        double a = ref[i], b = out[i];
        double diff = std::abs(a-b);
        max_abs = std::max(max_abs, diff);
        double denom = std::max(1e-6, std::abs(a));
        max_rel = std::max(max_rel, diff / denom);
        mse += diff * diff;
    }
    mse /= (double)ref.size();
    std::printf("Verify: max_abs=%.3e max_rel=%.3e mse=%.3e\n", max_abs, max_rel, mse);
}

inline double gflops(const MatmulProblem& p, float ms) {
    double flops = 2.0 * (double)p.M * p.N * p.K;
    return flops / (ms / 1e3) / 1e9;
}
