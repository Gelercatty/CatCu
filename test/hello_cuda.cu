#include <atomic>
#include <cuda_runtime.h>
#include <iostream>


__global__ void set_value(int* x)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *x = 123;
    }
}

int main()
{
    int *d_x = nullptr;
    int h_x = 0;
    cudaError_t err = cudaMalloc(&d_x, sizeof(int));
    if (err!=cudaSuccess)
    {
        std::cerr<<"cudaMalloc failed: "<<cudaGetErrorString(err)<<std::endl;
        return 1;
    }
    err = cudaMemcpy(d_x, &h_x, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr<<"cudaMemcpy failed: "<<cudaGetErrorString(err)<<std::endl;
        cudaFree(d_x);
        return 1;
    }
    set_value<<<1, 1>>>(d_x);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr<<"cudaGetLastError: "<<cudaGetErrorString(err)<<std::endl;
        cudaFree(d_x);
        return 1;
    }
    // wait for GPU
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr <<"cudaDeviceSynchronize: "<<cudaGetErrorString(err)<<std::endl;
        cudaFree(d_x);
        return 1;
    }
    err = cudaMemcpy(&h_x, d_x, sizeof(int), cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess )
    {
        std::cerr<<"cudaMemcpy failed: "<<cudaGetErrorString(err)<<std::endl;
        cudaFree(d_x);
        return 1;
    }
    cudaFree(d_x);
    std::cout<<"h_x = "<<h_x<<std::endl;
    return 0;


}