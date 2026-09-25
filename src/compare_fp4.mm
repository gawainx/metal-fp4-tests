#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Shape {
  uint32_t m;
  uint32_t k;
  uint32_t n;
};

struct Timing {
  double mean_ms;
  double p50_ms;
  double min_ms;
  double max_ms;
};

struct ComparisonRow {
  const char *name;
  Shape shape;
  Timing native;
  Timing software;
};

static float decodeE2M1(uint8_t code) {
  constexpr float values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[code & 7u];
  return (code & 8u) == 0u ? magnitude : -magnitude;
}

static id<MTLComputePipelineState> loadPipeline(id<MTLDevice> device, NSString *path, NSString *name) {
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
  if (library == nil) {
    std::fprintf(stderr, "%s: %s\n", path.UTF8String, error.localizedDescription.UTF8String);
    std::exit(1);
  }
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
  if (pipeline == nil) {
    std::fprintf(stderr, "%s: %s\n", name.UTF8String, error.localizedDescription.UTF8String);
    std::exit(1);
  }
  return pipeline;
}

static double execute(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> activations, id<MTLBuffer> weights, id<MTLBuffer> output,
                      id<MTLBuffer> shape_buffer, Shape shape, bool native) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:activations offset:0 atIndex:0];
  [encoder setBuffer:weights offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBuffer:shape_buffer offset:0 atIndex:3];
  if (native) {
    [encoder dispatchThreadgroups:MTLSizeMake(shape.n / 32, shape.m / 32, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  } else {
    [encoder dispatchThreads:MTLSizeMake(shape.n, shape.m, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::fprintf(stderr, "GPU command: %s\n", command.error.localizedDescription.UTF8String);
    std::exit(1);
  }
  return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

static Timing summarize(std::vector<double> samples) {
  const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  return {mean, samples[samples.size() / 2], samples.front(), samples.back()};
}

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "Metal device unavailable\n");
      return 1;
    }
    id<MTLComputePipelineState> native_pipeline = loadPipeline(device, @"build/fp4_native.metallib", @"fp4_native_matmul");
    id<MTLComputePipelineState> software_pipeline = loadPipeline(device, @"build/fp4_software_decode.metallib", @"fp4_software_matmul");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    constexpr int warmup = 3;
    constexpr int repetitions = 30;
    const Shape shapes[] = {{128, 1024, 1024}, {32, 1024, 3072}};
    const char *names[] = {"prefill_linear", "qkv_projection"};
    std::vector<ComparisonRow> rows;
    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "{\n  \"device\": \"" << device.name.UTF8String
           << "\",\n  \"comparison\": \"native_fp4_tensorops_vs_software_fp4_decode\""
           << ",\n  \"input\": \"identical packed E2M1 weights and FP16 activations; no scale plane\""
           << ",\n  \"warmup_iterations\": " << warmup
           << ",\n  \"measured_iterations_per_kernel\": " << repetitions
           << ",\n  \"workloads\": [\n";

    for (size_t workload = 0; workload < 2; ++workload) {
      const Shape shape = shapes[workload];
      std::mt19937 generator(42);
      std::uniform_int_distribution<int> activation_pick(0, 3);
      std::uniform_int_distribution<int> weight_pick(0, 15);
      const _Float16 activation_values[] = {_Float16(-1.0f), _Float16(-0.5f), _Float16(0.5f), _Float16(1.0f)};
      std::vector<_Float16> activation_data(size_t(shape.m) * shape.k);
      for (auto &value : activation_data) value = activation_values[activation_pick(generator)];
      std::vector<uint8_t> weight_data(size_t(shape.k) * shape.n / 2);
      for (auto &packed : weight_data) {
        packed = uint8_t(weight_pick(generator) | (weight_pick(generator) << 4));
      }
      const size_t output_count = size_t(shape.m) * shape.n;
      id<MTLBuffer> a = [device newBufferWithBytes:activation_data.data() length:activation_data.size() * sizeof(_Float16) options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [device newBufferWithBytes:weight_data.data() length:weight_data.size() options:MTLResourceStorageModeShared];
      id<MTLBuffer> native_output = [device newBufferWithLength:output_count * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> software_output = [device newBufferWithLength:output_count * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> parameters = [device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared];
      std::vector<double> native_samples;
      std::vector<double> software_samples;
      for (int iteration = -warmup; iteration < repetitions; ++iteration) {
        const bool native_first = (iteration & 1) == 0;
        const auto run = [&](bool native) {
          const double time = execute(queue, native ? native_pipeline : software_pipeline, a, b,
                                      native ? native_output : software_output, parameters, shape, native);
          if (iteration >= 0) (native ? native_samples : software_samples).push_back(time);
        };
        run(native_first);
        run(!native_first);
      }
      const auto *native_values = static_cast<const float *>(native_output.contents);
      const auto *software_values = static_cast<const float *>(software_output.contents);
      double max_abs_difference = 0.0;
      double max_reference_error = 0.0;
      size_t nonzero_outputs = 0;
      for (size_t index = 0; index < output_count; ++index) {
        if (!std::isfinite(native_values[index]) || !std::isfinite(software_values[index])) {
          std::fprintf(stderr, "%s: non-finite output\n", names[workload]);
          return 1;
        }
        max_abs_difference = std::max(max_abs_difference, std::fabs(double(native_values[index]) - software_values[index]));
        const size_t row = index / shape.n;
        const size_t column = index % shape.n;
        float reference = 0.0f;
        for (size_t reduction = 0; reduction < shape.k; ++reduction) {
          const size_t weight_index = reduction * shape.n + column;
          const uint8_t packed = weight_data[weight_index / 2];
          const uint8_t code = (weight_index & 1u) == 0u ? packed & 15u : packed >> 4u;
          reference += float(activation_data[row * shape.k + reduction]) * decodeE2M1(code);
        }
        max_reference_error = std::max(max_reference_error, std::fabs(double(native_values[index]) - reference));
        if (reference != 0.0f) ++nonzero_outputs;
      }
      if (max_abs_difference > 0.001 || max_reference_error > 0.001 || nonzero_outputs == 0) {
        std::fprintf(stderr, "%s: native/software difference %.6f, CPU reference error %.6f, nonzero outputs %zu\n",
                     names[workload], max_abs_difference, max_reference_error, nonzero_outputs);
        return 1;
      }
      const Timing native = summarize(native_samples);
      const Timing software = summarize(software_samples);
      rows.push_back({names[workload], shape, native, software});
      report << "    {\"name\": \"" << names[workload] << "\", \"m\": " << shape.m
             << ", \"k\": " << shape.k << ", \"n\": " << shape.n
             << ", \"native_fp4\": {\"mean_gpu_ms\": " << native.mean_ms
             << ", \"p50_gpu_ms\": " << native.p50_ms << ", \"min_gpu_ms\": " << native.min_ms
             << ", \"max_gpu_ms\": " << native.max_ms << "}"
             << ", \"software_fp4\": {\"mean_gpu_ms\": " << software.mean_ms
             << ", \"p50_gpu_ms\": " << software.p50_ms << ", \"min_gpu_ms\": " << software.min_ms
             << ", \"max_gpu_ms\": " << software.max_ms << "}"
             << ", \"software_over_native_time_ratio\": " << software.mean_ms / native.mean_ms
             << ", \"max_output_abs_difference\": " << max_abs_difference
             << ", \"max_cpu_reference_error\": " << max_reference_error << "}"
             << (workload == 1 ? "\n" : ",\n");
    }
    report << "  ]\n}\n";
    std::filesystem::create_directories("results");
    std::string component(device.name.UTF8String);
    for (char &character : component) if (character == ' ') character = '_';
    const std::string path = "results/fp4_native_vs_software_" + component + ".json";
    std::ofstream output(path);
    output << report.str();
    if (!output) {
      std::fprintf(stderr, "cannot write %s\n", path.c_str());
      return 1;
    }
    std::printf("GPU 计算耗时（%d 次平均）| %s\n", repetitions, device.name.UTF8String);
    std::puts("+----------------+----------------+----------------+----------------+----------+");
    std::puts("| Workload       | M x K x N      | Native FP4 ms  | Decode FP4 ms  | Native x |");
    std::puts("+----------------+----------------+----------------+----------------+----------+");
    for (const ComparisonRow &row : rows) {
      char dimensions[32];
      std::snprintf(dimensions, sizeof(dimensions), "%ux%ux%u", row.shape.m, row.shape.k, row.shape.n);
      std::printf("| %-14s | %-14s | %14.6f | %14.6f | %8.2f |\n",
                  row.name, dimensions, row.native.mean_ms, row.software.mean_ms,
                  row.software.mean_ms / row.native.mean_ms);
    }
    std::puts("+----------------+----------------+----------------+----------------+----------+");
    std::printf("Native x = 软件解码耗时 / 原生 FP4 耗时\nJSON 结果：%s\n", path.c_str());
    return 0;
  }
}
