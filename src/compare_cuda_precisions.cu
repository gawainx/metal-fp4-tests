#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace cute;

struct WorkloadShape { int m, k, n; };
struct Timing { double mean, p50, min, max; };

static void check(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(status));
    std::exit(1);
  }
}

static void check(cutlass::Status status, const char* action) {
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "%s: %s\n", action, cutlassGetStatusString(status));
    std::exit(1);
  }
}

static void check(cublasStatus_t status, const char* action) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s: cuBLAS status %d\n", action, int(status));
    std::exit(1);
  }
}

static float decode(uint8_t code) {
  constexpr float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return (code & 8) ? -values[code & 7] : values[code & 7];
}

static uint8_t fp8(uint8_t code) {
  constexpr uint8_t magnitudes[] = {0, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
  return magnitudes[code & 7] | ((code & 8) << 4);
}

template <typename Element> static std::vector<uint8_t> encode(const std::vector<uint8_t>& codes);

template <> std::vector<uint8_t> encode<cutlass::bfloat16_t>(const std::vector<uint8_t>& codes) {
  std::vector<uint8_t> out(codes.size() * 2);
  for (size_t i = 0; i < codes.size(); ++i) {
    uint16_t bits = uint16_t(std::bit_cast<uint32_t>(decode(codes[i])) >> 16);
    out[i * 2] = uint8_t(bits);
    out[i * 2 + 1] = uint8_t(bits >> 8);
  }
  return out;
}

template <> std::vector<uint8_t> encode<cutlass::float_e4m3_t>(const std::vector<uint8_t>& codes) {
  std::vector<uint8_t> out(codes.size());
  for (size_t i = 0; i < codes.size(); ++i) out[i] = fp8(codes[i]);
  return out;
}

template <> std::vector<uint8_t> encode<cutlass::float_e2m1_t>(const std::vector<uint8_t>& codes) {
  std::vector<uint8_t> out(codes.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = codes[2 * i] | (codes[2 * i + 1] << 4);
  return out;
}

static Timing summarize(std::vector<double> samples) {
  double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  return {mean, samples[samples.size() / 2], samples.front(), samples.back()};
}

template <typename Element, int Alignment> struct Kernel {
  using Tile = Shape<_128, _64, _128>;
  using Cluster = Shape<_1, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, float, cutlass::layout::ColumnMajor, 4,
      float, cutlass::layout::ColumnMajor, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      Element, cutlass::layout::RowMajor, Alignment,
      Element, cutlass::layout::ColumnMajor, Alignment,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>>;
};

template <typename Element, int Alignment>
static std::pair<Timing, double> run(WorkloadShape shape, const std::vector<uint8_t>& a_codes,
                                     const std::vector<uint8_t>& b_codes,
                                     const std::vector<float>& reference) {
  auto a = encode<Element>(a_codes);
  auto b = encode<Element>(b_codes);
  void *d_a = nullptr, *d_b = nullptr, *workspace = nullptr;
  float *d_out = nullptr;
  check(cudaMalloc(&d_a, a.size()), "allocate A");
  check(cudaMalloc(&d_b, b.size()), "allocate B");
  check(cudaMalloc(&d_out, reference.size() * sizeof(float)), "allocate output");
  check(cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice), "copy A");
  check(cudaMemcpy(d_b, b.data(), b.size(), cudaMemcpyHostToDevice), "copy B");

  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "create start event");
  check(cudaEventCreate(&stop), "create stop event");
  constexpr int warmup = 3, repetitions = 30;
  std::vector<double> samples;
  auto measure = [&](auto launch) {
    for (int i = -warmup; i < repetitions; ++i) {
      check(cudaEventRecord(start), "record start");
      launch();
      check(cudaEventRecord(stop), "record stop");
      check(cudaEventSynchronize(stop), "wait for GEMM");
      if (i >= 0) {
        float ms;
        check(cudaEventElapsedTime(&ms, start, stop), "measure GEMM");
        samples.push_back(ms);
      }
    }
  };
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    cublasHandle_t handle;
    check(cublasCreate(&handle), "create cuBLAS handle");
    const float alpha = 1, beta = 0;
    measure([&] {
      check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         shape.m, shape.n, shape.k, &alpha,
                         d_a, CUDA_R_16BF, shape.k,
                         d_b, CUDA_R_16BF, shape.k,
                         &beta, d_out, CUDA_R_32F, shape.m,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "launch BF16 GEMM");
    });
    check(cublasDestroy(handle), "destroy cuBLAS handle");
  } else {
    using Gemm = typename Kernel<Element, Alignment>::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {shape.m, shape.k, 1});
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {shape.n, shape.k, 1});
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {shape.m, shape.n, 1});
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {shape.m, shape.n, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {shape.m, shape.n, shape.k, 1},
        {reinterpret_cast<Element*>(d_a), stride_a, reinterpret_cast<Element*>(d_b), stride_b},
        {{1.0f, 0.0f}, d_out, stride_c, d_out, stride_d}};
    check(cudaMalloc(&workspace, Gemm::get_workspace_size(args)), "allocate workspace");
    Gemm gemm;
    check(gemm.initialize(args, workspace), "initialize GEMM");
    measure([&] { check(gemm(), "launch GEMM"); });
  }
  std::vector<float> output(reference.size());
  check(cudaMemcpy(output.data(), d_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy output");
  double max_error = 0;
  for (size_t n = 0; n < size_t(shape.n); ++n) {
    for (size_t m = 0; m < size_t(shape.m); ++m) {
      float actual = output[n * shape.m + m];
      if (!std::isfinite(actual)) { std::fprintf(stderr, "nonfinite result\n"); std::exit(1); }
      max_error = std::max(max_error, std::abs(double(actual) - reference[m * shape.n + n]));
    }
  }
  check(cudaEventDestroy(start), "destroy event");
  check(cudaEventDestroy(stop), "destroy event");
  check(cudaFree(workspace), "free workspace");
  check(cudaFree(d_out), "free output");
  check(cudaFree(d_b), "free B");
  check(cudaFree(d_a), "free A");
  return {summarize(samples), max_error};
}

int main() {
  cudaDeviceProp device{};
  check(cudaGetDeviceProperties(&device, 0), "inspect GPU");
  if (device.major != 12 || device.minor != 0) {
    std::fprintf(stderr, "This benchmark requires SM120; found SM%d%d\n", device.major, device.minor);
    return 1;
  }
  std::ostringstream report;
  report << std::fixed << std::setprecision(6);
  report << "{\n  \"device\": \"" << device.name << "\",\n"
         << "  \"comparison\": \"cuda_bf16_fp8_e4m3_fp4_e2m1_gemm\",\n"
         << "  \"input\": \"identical E2M1 values exactly representable in all three formats; no scale plane\",\n"
         << "  \"accumulator\": \"FP32\",\n  \"warmup_iterations\": 3,\n"
         << "  \"measured_iterations_per_kernel\": 30,\n  \"workloads\": [\n";
  const WorkloadShape shapes[] = {{128, 1024, 1024}, {32, 1024, 3072}};
  const char* names[] = {"prefill_linear", "qkv_projection"};
  for (int s = 0; s < 2; ++s) {
    auto shape = shapes[s];
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> pick(0, 15);
    std::vector<uint8_t> a(size_t(shape.m) * shape.k), b_row_major(size_t(shape.k) * shape.n);
    for (auto& code : a) code = uint8_t(pick(generator));
    for (auto& code : b_row_major) code = uint8_t(pick(generator));
    std::vector<uint8_t> b(b_row_major.size());
    for (int k = 0; k < shape.k; ++k)
      for (int n = 0; n < shape.n; ++n)
        b[size_t(n) * shape.k + k] = b_row_major[size_t(k) * shape.n + n];
    std::vector<float> reference(size_t(shape.m) * shape.n);
    for (int m = 0; m < shape.m; ++m)
      for (int n = 0; n < shape.n; ++n)
        for (int k = 0; k < shape.k; ++k)
          reference[size_t(m) * shape.n + n] += decode(a[size_t(m) * shape.k + k]) * decode(b_row_major[size_t(k) * shape.n + n]);
    auto bf16 = run<cutlass::bfloat16_t, 8>(shape, a, b, reference);
    auto fp8 = run<cutlass::float_e4m3_t, 16>(shape, a, b, reference);
    auto fp4 = run<cutlass::float_e2m1_t, 128>(shape, a, b, reference);
    auto emit = [&](const char* key, const std::pair<Timing, double>& result) {
      const auto& t = result.first;
      report << "\"" << key << "\": {\"mean_gpu_ms\": " << t.mean << ", \"p50_gpu_ms\": " << t.p50
             << ", \"min_gpu_ms\": " << t.min << ", \"max_gpu_ms\": " << t.max << "}";
    };
    if (bf16.second > .001 || fp8.second > .001 || fp4.second > .001) {
      std::fprintf(stderr, "%s reference errors: BF16 %.6f, FP8 %.6f, FP4 %.6f\n",
                   names[s], bf16.second, fp8.second, fp4.second);
      return 1;
    }
    report << "    {\"name\": \"" << names[s] << "\", \"m\": " << shape.m << ", \"k\": " << shape.k
           << ", \"n\": " << shape.n << ", ";
    emit("bf16", bf16); report << ", ";
    emit("fp8_e4m3", fp8); report << ", ";
    emit("fp4_e2m1", fp4);
    report << ", \"speedup_vs_bf16\": {\"fp8_e4m3\": " << bf16.first.mean / fp8.first.mean
           << ", \"fp4_e2m1\": " << bf16.first.mean / fp4.first.mean << "}, "
           << "\"max_cpu_reference_error\": {\"bf16\": " << bf16.second
           << ", \"fp8_e4m3\": " << fp8.second << ", \"fp4_e2m1\": " << fp4.second << "}}"
           << (s == 1 ? "\n" : ",\n");
    std::printf("%s: BF16 %.4f ms, FP8 %.4f ms (%.2fx), FP4 %.4f ms (%.2fx)\n",
                names[s], bf16.first.mean, fp8.first.mean, bf16.first.mean / fp8.first.mean,
                fp4.first.mean, bf16.first.mean / fp4.first.mean);
  }
  report << "  ]\n}\n";
  std::filesystem::create_directories("results");
  std::string component(device.name);
  for (char& character : component) if (character == ' ') character = '_';
  const std::string path = "results/precision_comparison_" + component + ".json";
  std::ofstream out(path);
  out << report.str();
  std::printf("Saved %s\n", path.c_str());
}
