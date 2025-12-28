#include "include/cuda_common.cuh"
#include "include/cuda_raii.cuh"
#include "include/DataArgs.h"
#include "include/utools.cuh"
#include "src/Interface.cu"
#include <memory>
#include <chrono>

struct BenchOut {
    float ms = 0.f;
    double gflops = 0.0;
};

static BenchOut bench_one(IMatmul& impl,
                          const MatmulProblem& p,
                          const MatmulArgs& a,
                          cudaStream_t stream,
                          int warmup, int iters) {
    impl.init(p, stream);

    for (int i = 0; i < warmup; ++i) impl.run(p, a, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GpuTimer t;
    t.tic(stream);
    for (int i = 0; i < iters; ++i) impl.run(p, a, stream);
    CUDA_CHECK(cudaGetLastError());
    float total_ms = t.toc_ms(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    impl.fini();

    BenchOut out;
    out.ms = total_ms / iters;
    out.gflops = gflops(p, out.ms);
    return out;
}


// CPU ：C(MxN) = A(MxK) * B(KxN)，row-major
static void cpu_gemm_naive(const MatmulProblem& p,
                           const float* A,
                           const float* B,
                           float* C)
{
    for (int i = 0; i < p.M; ++i) {
        const float* arow = A + (size_t)i * p.K;
        float* crow = C + (size_t)i * p.N;
        for (int j = 0; j < p.N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < p.K; ++k) {
                acc += arow[k] * B[(size_t)k * p.N + j];
            }
            crow[j] = acc;
        }
    }
}

static BenchOut bench_cpu_naive(const MatmulProblem& p,
                                const float* A,
                                const float* B,
                                float* C,
                                int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) cpu_gemm_naive(p, A, B, C);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) cpu_gemm_naive(p, A, B, C);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    BenchOut out;
    out.ms = (float)(total_ms / iters);
    out.gflops = gflops(p, out.ms);
    return out;
}


// ---------------- main ----------------
int main(int argc, char** argv) {
  MatmulProblem p{80, 80, 80};
  int warmup = 3, iters = 1;

  if (argc >= 2) p.M = std::atoi(argv[1]);
  if (argc >= 3) p.N = std::atoi(argv[2]);
  if (argc >= 4) p.K = std::atoi(argv[3]);
  if (argc >= 5) iters = std::atoi(argv[4]);

  std::printf("M=%d N=%d K=%d, iters=%d\n", p.M, p.N, p.K, iters);

  CudaStream stream;

  size_t sizeA = (size_t)p.M * p.K;
  size_t sizeB = (size_t)p.K * p.N;
  size_t sizeC = (size_t)p.M * p.N;

  // host
  std::vector<float> hA(sizeA), hB(sizeB);
  std::vector<float> hRef(sizeC), hOut(sizeC);
  fill_random(hA, 123);
  fill_random(hB, 456);
  // cpu baseline
  //
  std::vector<float> hCpu(sizeC);
    int cpu_warmup = 0;
    int cpu_iters  = 1;
    auto cpuRes = bench_cpu_naive(p, hA.data(), hB.data(), hCpu.data(), cpu_warmup, cpu_iters);
    std::printf("%-20s  %.3f ms  %.2f GFLOP/s\n", "CPU naive", cpuRes.ms, cpuRes.gflops);
  // device
  DeviceBuffer<float> dA(sizeA), dB(sizeB), dC(sizeC), dRef(sizeC);
  CUDA_CHECK(cudaMemcpyAsync(dA.data(), hA.data(), sizeA*sizeof(float), cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(dB.data(), hB.data(), sizeB*sizeof(float), cudaMemcpyHostToDevice, stream.get()));
  stream.sync();

  std::vector<std::unique_ptr<IMatmul>> impls;
  impls.emplace_back(std::make_unique<CublasMatMul>(false)); // reference + 对照
  impls.emplace_back(std::make_unique<Tiled16Matmul>());     // baseline

  MatmulArgs argsRef{dA.data(), dB.data(), dRef.data()};
  auto refRes = bench_one(*impls[0], p, argsRef, stream.get(), warmup, iters);
  std::printf("%-20s  %.3f ms  %.2f GFLOP/s\n", impls[0]->name(), refRes.ms, refRes.gflops);

  // 把 ref 拷回
  CUDA_CHECK(cudaMemcpyAsync(hRef.data(), dRef.data(), sizeC*sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
    std::puts("CPU naive vs cuBLAS verify:");
    verify_host(hRef, hCpu);


  for (size_t i = 1; i < impls.size(); ++i) {
    CUDA_CHECK(cudaMemsetAsync(dC.data(), 0, sizeC*sizeof(float), stream.get()));

    MatmulArgs a{dA.data(), dB.data(), dC.data()};
    auto r = bench_one(*impls[i], p, a, stream.get(), warmup, iters);

      double speedup_vs_cpu     = cpuRes.ms / r.ms;
      double speedup_vs_cublas  = refRes.ms / r.ms;

      std::printf("%-20s  %.3f ms  %.2f GFLOP/s  "
                  "speedup(vs cpu)=%.2fx  speedup(vs cublas)=%.2fx\n",
                  impls[i]->name(), r.ms, r.gflops, speedup_vs_cpu, speedup_vs_cublas);
      std::printf("cuBLAS speedup(vs cpu)=%.2fx\n", cpuRes.ms / refRes.ms);

    // verify
    CUDA_CHECK(cudaMemcpyAsync(hOut.data(), dC.data(), sizeC*sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    verify_host(hRef, hOut);
  }

  return 0;
}