#include <tbb/task.h>

#include "../include/cuda_common.cuh"


class CudaStream
{
private:
    cudaStream_t s_{nullptr};
public:
    CudaStream() {CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));}
    ~CudaStream() {if (s_) cudaStreamDestroy(s_);}
    CudaStream(const CudaStream&) = delete;

    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const {return s_;}
    void sync() const {CUDA_CHECK(cudaStreamSynchronize(s_));}
};

class CudaEvent
{
private:
    cudaEvent_t e_{nullptr};
public:
    CudaEvent(){CUDA_CHECK(cudaEventCreate(&e_));}
    ~CudaEvent(){if (e_) cudaEventDestroy(e_);}
    CudaEvent(const CudaEvent&)=delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const {return e_;}
};

class GpuTimer
{
private:
    CudaEvent start_, stop_;
public:
    void tic(cudaStream_t s){CUDA_CHECK(cudaEventRecord(start_.get(), s));}
    float toc_ms(cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(stop_.get(), s));
        CUDA_CHECK(cudaEventSynchronize(stop_.get()));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_.get(), stop_.get()));
        return ms;
    }
};

template <class T>
class DeviceBuffer
{
private:
    T* p_{nullptr};
    size_t n_{0};
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n ){ reset(n);}
    ~DeviceBuffer() {if (p_) cudaFree(p_);}

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) {o.p_ = nullptr; o.n_ = 0;}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept
    {
        if (this != &o)
        {
            if (p_) cudaFree(p_);
            p_ = o.p_; n_ = o.n_;
            o.p_ = nullptr; o.n_ = 0;
        }
        return *this;
    }

    void reset(size_t n)
    {
        if (p_) cudaFree(p_);
        n_ = n;
        CUDA_CHECK(cudaMalloc(& p_, n_* sizeof(T)));
    }

    T* data() {return p_;}
    const T* data() const {return p_;}
    size_t size() const {return n_;}
};

