#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

constexpr size_t format_count = 6;
constexpr const char *format_names[format_count] = {"BF16 native", "BF16 software", "FP8 native", "FP8 software", "FP4 native", "FP4 software"};
constexpr const char *json_names[format_count] = {"bf16_native", "bf16_software", "fp8_native", "fp8_software", "fp4_native", "fp4_software"};
constexpr const char *precision_names[] = {"BF16", "FP8 E4M3", "FP4 E2M1"};

struct ComparisonRow {
  const char *name;
  Shape shape;
  Timing timings[format_count];
  double max_error[format_count];
};

static float decodeE2M1(uint8_t code) {
  constexpr float values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[code & 7u];
  return (code & 8u) == 0u ? magnitude : -magnitude;
}

static uint8_t encodeE4M3(uint8_t fp4_code) {
  constexpr uint8_t magnitudes[] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
  return uint8_t(magnitudes[fp4_code & 7u] | ((fp4_code & 8u) << 4u));
}

static std::vector<uint8_t> packE2M1(const std::vector<uint8_t> &codes) {
  std::vector<uint8_t> packed(codes.size() / 2);
  for (size_t index = 0; index < packed.size(); ++index) {
    packed[index] = uint8_t(codes[2 * index] | (codes[2 * index + 1] << 4u));
  }
  return packed;
}

static std::vector<uint8_t> encodeFP8(const std::vector<uint8_t> &codes) {
  std::vector<uint8_t> values(codes.size());
  for (size_t index = 0; index < codes.size(); ++index) values[index] = encodeE4M3(codes[index]);
  return values;
}

static std::vector<uint16_t> encodeBF16(const std::vector<uint8_t> &codes) {
  std::vector<uint16_t> values(codes.size());
  for (size_t index = 0; index < codes.size(); ++index) {
    values[index] = uint16_t(std::bit_cast<uint32_t>(decodeE2M1(codes[index])) >> 16u);
  }
  return values;
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
                      id<MTLBuffer> shape_buffer, Shape shape, bool software) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:activations offset:0 atIndex:0];
  [encoder setBuffer:weights offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBuffer:shape_buffer offset:0 atIndex:3];
  if (software) {
    [encoder dispatchThreads:MTLSizeMake(shape.n, shape.m, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else {
    [encoder dispatchThreadgroups:MTLSizeMake(shape.n / 32, shape.m / 32, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
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

static void printTimingJSON(std::ostringstream &report, const char *name, Timing timing) {
  report << "\"" << name << "\": {\"mean_gpu_ms\": " << timing.mean_ms
         << ", \"p50_gpu_ms\": " << timing.p50_ms
         << ", \"min_gpu_ms\": " << timing.min_ms
         << ", \"max_gpu_ms\": " << timing.max_ms << "}";
}

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "Metal device unavailable\n");
      return 1;
    }
    id<MTLComputePipelineState> pipelines[format_count] = {
        loadPipeline(device, @"build/bf16_native.metallib", @"bf16_native_matmul"),
        loadPipeline(device, @"build/bf16_software_decode.metallib", @"bf16_software_matmul"),
        loadPipeline(device, @"build/fp8_native.metallib", @"fp8_native_matmul"),
        loadPipeline(device, @"build/fp8_software_decode.metallib", @"fp8_software_matmul"),
        loadPipeline(device, @"build/fp4_native.metallib", @"fp4_native_matmul"),
        loadPipeline(device, @"build/fp4_software_decode.metallib", @"fp4_software_matmul")};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    constexpr int warmup = 3;
    constexpr int repetitions = 30;
    const Shape shapes[] = {{128, 1024, 1024}, {32, 1024, 3072}};
    const char *names[] = {"prefill_linear", "qkv_projection"};
    std::vector<ComparisonRow> rows;
    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "{\n  \"device\": \"" << device.name.UTF8String
           << "\",\n  \"comparison\": \"paired_native_vs_software_bf16_fp8_fp4\""
           << ",\n  \"input\": \"identical values exactly representable in BF16, FP8 E4M3 and FP4 E2M1; no scale plane\""
           << ",\n  \"warmup_iterations\": " << warmup
           << ",\n  \"measured_iterations_per_kernel\": " << repetitions
           << ",\n  \"workloads\": [\n";

    for (size_t workload = 0; workload < 2; ++workload) {
      const Shape shape = shapes[workload];
      std::mt19937 generator(42);
      std::uniform_int_distribution<int> code_pick(0, 15);
      std::vector<uint8_t> activation_codes(size_t(shape.m) * shape.k);
      std::vector<uint8_t> weight_codes(size_t(shape.k) * shape.n);
      for (auto &code : activation_codes) code = uint8_t(code_pick(generator));
      for (auto &code : weight_codes) code = uint8_t(code_pick(generator));
      const auto activation_fp4 = packE2M1(activation_codes);
      const auto weight_fp4 = packE2M1(weight_codes);
      const auto activation_fp8 = encodeFP8(activation_codes);
      const auto weight_fp8 = encodeFP8(weight_codes);
      const auto activation_bf16 = encodeBF16(activation_codes);
      const auto weight_bf16 = encodeBF16(weight_codes);
      id<MTLBuffer> activation_buffers[format_count] = {
          [device newBufferWithBytes:activation_bf16.data() length:activation_bf16.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:activation_bf16.data() length:activation_bf16.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:activation_fp8.data() length:activation_fp8.size() options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:activation_fp8.data() length:activation_fp8.size() options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:activation_fp4.data() length:activation_fp4.size() options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:activation_fp4.data() length:activation_fp4.size() options:MTLResourceStorageModeShared]};
      id<MTLBuffer> weight_buffers[format_count] = {
          [device newBufferWithBytes:weight_bf16.data() length:weight_bf16.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:weight_bf16.data() length:weight_bf16.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:weight_fp8.data() length:weight_fp8.size() options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:weight_fp8.data() length:weight_fp8.size() options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:weight_fp4.data() length:weight_fp4.size() options:MTLResourceStorageModeShared],
          [device newBufferWithBytes:weight_fp4.data() length:weight_fp4.size() options:MTLResourceStorageModeShared]};
      const size_t output_count = size_t(shape.m) * shape.n;
      id<MTLBuffer> outputs[format_count];
      for (size_t format = 0; format < format_count; ++format) {
        outputs[format] = [device newBufferWithLength:output_count * sizeof(float) options:MTLResourceStorageModeShared];
      }
      id<MTLBuffer> parameters = [device newBufferWithBytes:&shape length:sizeof(shape) options:MTLResourceStorageModeShared];
      std::vector<double> samples[format_count];
      for (int iteration = -warmup; iteration < repetitions; ++iteration) {
        const size_t first = size_t(iteration + warmup) % format_count;
        for (size_t offset = 0; offset < format_count; ++offset) {
          const size_t format = (first + offset) % format_count;
          const double time = execute(queue, pipelines[format], activation_buffers[format],
                                      weight_buffers[format], outputs[format], parameters, shape,
                                      format % 2 == 1);
          if (iteration >= 0) samples[format].push_back(time);
        }
      }
      ComparisonRow row{};
      row.name = names[workload];
      row.shape = shape;
      const float *gpu_values[format_count];
      for (size_t format = 0; format < format_count; ++format) {
        gpu_values[format] = static_cast<const float *>(outputs[format].contents);
        row.timings[format] = summarize(samples[format]);
      }
      size_t nonzero_outputs = 0;
      for (size_t index = 0; index < output_count; ++index) {
        const size_t output_row = index / shape.n;
        const size_t output_column = index % shape.n;
        float reference = 0.0f;
        for (size_t reduction = 0; reduction < shape.k; ++reduction) {
          reference += decodeE2M1(activation_codes[output_row * shape.k + reduction]) *
                       decodeE2M1(weight_codes[reduction * shape.n + output_column]);
        }
        if (reference != 0.0f) ++nonzero_outputs;
        for (size_t format = 0; format < format_count; ++format) {
          const float actual = gpu_values[format][index];
          if (!std::isfinite(actual)) {
            std::fprintf(stderr, "%s %s: non-finite output\n", names[workload], format_names[format]);
            return 1;
          }
          row.max_error[format] = std::max(row.max_error[format], std::fabs(double(actual) - reference));
        }
      }
      for (size_t format = 0; format < format_count; ++format) {
        if (row.max_error[format] > 0.001 || nonzero_outputs == 0) {
          std::fprintf(stderr, "%s %s: CPU reference error %.6f\n", names[workload],
                       format_names[format], row.max_error[format]);
          return 1;
        }
      }
      rows.push_back(row);
      report << "    {\"name\": \"" << row.name << "\", \"m\": " << shape.m
             << ", \"k\": " << shape.k << ", \"n\": " << shape.n << ", ";
      for (size_t format = 0; format < format_count; ++format) {
        printTimingJSON(report, json_names[format], row.timings[format]);
        report << ", ";
      }
      report << "\"max_cpu_reference_error\": {";
      for (size_t format = 0; format < format_count; ++format) {
        report << "\"" << json_names[format] << "\": " << row.max_error[format]
               << (format + 1 == format_count ? "" : ", ");
      }
      report << "}}" << (workload + 1 == 2 ? "\n" : ",\n");
    }
    report << "  ]\n}\n";
    std::filesystem::create_directories("results");
    std::string component(device.name.UTF8String);
    for (char &character : component) if (character == ' ') character = '_';
    const std::string path = "results/precision_comparison_" + component + ".json";
    std::ofstream output(path);
    output << report.str();
    if (!output) {
      std::fprintf(stderr, "cannot write %s\n", path.c_str());
      return 1;
    }
    std::printf("GPU 计算耗时（%d 次平均）| %s | 越低越快\n", repetitions, device.name.UTF8String);
    std::puts("+----------------+----------+---------------+---------------+----------------+");
    std::puts("| Workload       | Format   | Native GPU ms | Software GPU ms | Native speedup |");
    std::puts("+----------------+----------+---------------+---------------+----------------+");
    for (const ComparisonRow &row : rows) {
      for (size_t precision = 0; precision < format_count / 2; ++precision) {
        const double native_ms = row.timings[2 * precision].mean_ms;
        const double software_ms = row.timings[2 * precision + 1].mean_ms;
        std::printf("| %-14s | %-8s | %13.6f | %13.6f | %13.2fx |\n",
                    row.name, precision_names[precision], native_ms, software_ms,
                    software_ms / native_ms);
      }
      std::puts("+----------------+----------+---------------+---------------+----------------+");
    }
    std::printf("Native speedup = 软件解码耗时 / 原生 Ops 耗时\nJSON 结果：%s\n", path.c_str());
    return 0;
  }
}
