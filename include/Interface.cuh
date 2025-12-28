//
// Created by root on 2025/12/28.
//

#ifndef CATCU_INTERFACE_CUH
#define CATCU_INTERFACE_CUH

#include <memory>
#include "cuda_common.cuh"
#include "DataArgs.h"

class IMatmul
{
public:
    virtual ~IMatmul() = default;
    virtual const char* name() const = 0;
    virtual void init(const MatmulProblem&, cudaStream_t){}
    virtual void run(const MatmulProblem& p, const MatmulArgs& a, cudaStream_t) = 0;
    virtual void fini() {}
};

std::unique_ptr<IMatmul> make_cublas(bool tf32 = false);
std::unique_ptr<IMatmul> make_tiled16();
#endif //CATCU_INTERFACE_CUH