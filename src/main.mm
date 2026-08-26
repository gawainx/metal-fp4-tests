#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/utsname.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

/** Describes one matrix multiplication workload used by a Transformer inference phase. */
struct Workload {
  /** Human-readable Transformer operation name. */
  std::string name;
  /** Activation row count. */
  uint32_t m;
  /** Hidden dimension shared by activations and weights. */
  uint32_t k;
  /** Output projection width. */
  uint32_t n;
};

/** Holds GPU timing and numerical-error measurements for one workload and representation. */
struct Measurement {
  /** Representation name: fp16 or fp4_e2m1_block32. */
  std::string representation;
  /** Mean GPU execution time over measured iterations. */
  double mean_ms = 0.0;
  /** Fastest measured GPU execution time. */
  double min_ms = 0.0;
  /** Median measured GPU execution time. */
  double p50_ms = 0.0;
  /** 95th percentile measured GPU execution time. */
  double p95_ms = 0.0;
  /** Slowest measured GPU execution time. */
  double max_ms = 0.0;
  /** Mean CPU wall-clock time from command submission through completion. */
  double mean_wall_ms = 0.0;
  /** Logical dense-matmul throughput. */
  double tflops = 0.0;
  /** Largest absolute difference from the FP32 CPU reference. */
  double max_abs_error = 0.0;
  /** Root mean square difference from the FP32 CPU reference. */
  double rmse = 0.0;
  /** Cosine similarity to the FP32 CPU reference. */
  double cosine_similarity = 0.0;
};

/** Shared parameters passed to the Metal matrix multiplication kernels. */
struct MatmulParams {
  /** Number of activation rows. */
  uint32_t m;
  /** Reduction dimension. */
  uint32_t k;
  /** Number of output columns. */
  uint32_t n;
};

/** Escapes a string for insertion into a JSON string literal. */
static std::string jsonEscape(const std::string &value) {
  std::string escaped;
  for (char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

/** Returns the current machine architecture reported by uname. */
static std::string machineArchitecture() {
  struct utsname system_info {};
  uname(&system_info);
  return system_info.machine;
}

/** Converts a device name into a filename component while preserving its model words. */
static std::string filenameComponent(const std::string &value) {
  std::string component;
  for (const char character : value) {
    const bool is_ascii_letter = (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z');
    const bool is_digit = character >= '0' && character <= '9';
    component.push_back(is_ascii_letter || is_digit ? character : '_');
  }
  return component;
}

/** Parses a positive command-line integer and returns the fallback for invalid input. */
static uint32_t parsePositive(const char *value, uint32_t fallback) {
  const long parsed = std::strtol(value, nullptr, 10);
  return parsed > 0 ? static_cast<uint32_t>(parsed) : fallback;
}

/** Rounds a finite scalar to the closest finite E2M1 code after applying a block scale. */
static uint8_t encodeE2M1(float normalized) {
  static constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const bool negative = normalized < 0.0f;
  const float magnitude = std::fabs(normalized);
  uint8_t best_index = 0;
  float best_distance = std::numeric_limits<float>::infinity();
  for (uint8_t index = 0; index < 8; ++index) {
    const float distance = std::fabs(magnitude - values[index]);
    if (distance < best_distance) {
      best_distance = distance;
      best_index = index;
    }
  }
  return static_cast<uint8_t>((negative ? 0x8u : 0u) | best_index);
}

/** Converts FP32 weights into FP16 values, packed E2M1 nibbles, and one FP16 scale per 32 values. */
static void quantizeWeights(const std::vector<float> &weights, uint32_t k, std::vector<_Float16> &fp16_weights,
                            std::vector<uint8_t> &packed_fp4, std::vector<_Float16> &scales) {
  fp16_weights.resize(weights.size());
  packed_fp4.assign((weights.size() + 1) / 2, 0);
  scales.resize(weights.size() / 32);
  for (size_t index = 0; index < weights.size(); ++index) {
    fp16_weights[index] = _Float16(weights[index]);
  }
  for (size_t block = 0; block < scales.size(); ++block) {
    float maximum = 0.0f;
    for (size_t element = 0; element < 32; ++element) {
      maximum = std::max(maximum, std::fabs(weights[block * 32 + element]));
    }
    const float scale = maximum == 0.0f ? 1.0f : maximum / 6.0f;
    scales[block] = _Float16(scale);
    for (size_t element = 0; element < 32; ++element) {
      const uint8_t code = encodeE2M1(weights[block * 32 + element] / scale);
      const size_t packed_index = (block * 32 + element) / 2;
      packed_fp4[packed_index] |= static_cast<uint8_t>(code << ((element & 1u) * 4u));
    }
  }
  if (weights.size() % k != 0 || k % 32 != 0) {
    std::fprintf(stderr, "[fp4] invalid weight dimensions for block-32 quantization\n");
    std::exit(2);
  }
}

/** Calculates a FP32 reference product for numerical validation. */
static std::vector<float> cpuMatmul(const std::vector<float> &activations, const std::vector<float> &weights,
                                    const Workload &workload) {
  std::vector<float> output(static_cast<size_t>(workload.m) * workload.n, 0.0f);
  for (uint32_t row = 0; row < workload.m; ++row) {
    for (uint32_t column = 0; column < workload.n; ++column) {
      float sum = 0.0f;
      for (uint32_t reduction = 0; reduction < workload.k; ++reduction) {
        sum += activations[static_cast<size_t>(row) * workload.k + reduction] *
               weights[static_cast<size_t>(column) * workload.k + reduction];
      }
      output[static_cast<size_t>(row) * workload.n + column] = sum;
    }
  }
  return output;
}

/** Computes error statistics between GPU output and the FP32 CPU reference. */
static void calculateError(Measurement &measurement, const std::vector<float> &actual, const std::vector<float> &reference) {
  double square_error = 0.0;
  double actual_norm = 0.0;
  double reference_norm = 0.0;
  double dot_product = 0.0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const double difference = static_cast<double>(actual[index]) - reference[index];
    measurement.max_abs_error = std::max(measurement.max_abs_error, std::fabs(difference));
    square_error += difference * difference;
    actual_norm += static_cast<double>(actual[index]) * actual[index];
    reference_norm += static_cast<double>(reference[index]) * reference[index];
    dot_product += static_cast<double>(actual[index]) * reference[index];
  }
  measurement.rmse = std::sqrt(square_error / actual.size());
  measurement.cosine_similarity = dot_product / std::sqrt(actual_norm * reference_norm);
}

/** Returns the nearest-rank percentile from a nonempty, sorted duration sample. */
static double percentile(const std::vector<double> &sorted_samples, double fraction) {
  const size_t rank = static_cast<size_t>(std::ceil(fraction * sorted_samples.size()));
  return sorted_samples[std::max<size_t>(1, rank) - 1];
}

/** Executes a Metal pipeline, measures GPU time, reads results, and validates them against the reference. */
static Measurement runPipeline(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                               const std::vector<id<MTLBuffer>> &buffers, uint32_t params_index,
                               const Workload &workload, const std::string &representation,
                               const std::vector<float> &reference, uint32_t iterations) {
  Measurement measurement;
  (void)params_index;
  measurement.representation = representation;
  std::vector<double> elapsed_ms;
  std::vector<double> wall_ms;
  for (uint32_t iteration = 0; iteration < iterations + 1; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      [encoder setBuffer:buffers[index] offset:0 atIndex:index];
    }
    const MTLSize grid = MTLSizeMake(workload.n, workload.m, 1);
    const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    const double elapsed_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "[fp4] GPU command failed: %s\n", command_buffer.error.localizedDescription.UTF8String);
      std::exit(3);
    }
    if (iteration > 0) {
      elapsed_ms.push_back((command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0);
      wall_ms.push_back(elapsed_wall_ms);
    }
  }
  for (const double elapsed : elapsed_ms) {
    measurement.mean_ms += elapsed;
  }
  measurement.mean_ms /= elapsed_ms.size();
  for (const double elapsed : wall_ms) {
    measurement.mean_wall_ms += elapsed;
  }
  measurement.mean_wall_ms /= wall_ms.size();
  std::sort(elapsed_ms.begin(), elapsed_ms.end());
  measurement.min_ms = elapsed_ms.front();
  measurement.p50_ms = percentile(elapsed_ms, 0.50);
  measurement.p95_ms = percentile(elapsed_ms, 0.95);
  measurement.max_ms = elapsed_ms.back();
  measurement.tflops = (2.0 * workload.m * workload.k * workload.n) / (measurement.mean_ms * 1.0e9);
  id<MTLBuffer> output_buffer = buffers[representation == "fp16" ? 2 : 3];
  std::vector<float> output(reference.size());
  std::copy_n(static_cast<const float *>(output_buffer.contents), output.size(), output.data());
  calculateError(measurement, output, reference);
  return measurement;
}

/** Writes the complete benchmark result to standard output and a JSON file. */
static void writeReport(const std::string &path, const std::string &report) {
  std::printf("%s", report.c_str());
  std::ofstream file(path);
  file << report;
  std::printf("[fp4] JSON saved to %s\n", path.c_str());
}

/** Runs capability probing and Transformer-shaped FP4/FP16 benchmarks. */
int main(int argc, const char *argv[]) {
  @autoreleasepool {
    std::string output_path;
    bool output_path_specified = false;
    std::string native_fp4_sdk = "unavailable";
    uint32_t iterations = 5;
    uint32_t hidden_size = 1024;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--output" && index + 1 < argc) {
        output_path = argv[++index];
        output_path_specified = true;
      }
      if (argument == "--native-fp4-sdk" && index + 1 < argc) native_fp4_sdk = argv[++index];
      if (argument == "--iterations" && index + 1 < argc) iterations = parsePositive(argv[++index], iterations);
      if (argument == "--hidden-size" && index + 1 < argc) hidden_size = parsePositive(argv[++index], hidden_size);
    }
    hidden_size = std::max(32u, hidden_size - (hidden_size % 32u));
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "[fp4] Metal device unavailable\n");
      return 1;
    }
    if (!output_path_specified) {
      std::filesystem::create_directories("results");
      output_path = "results/fp4_benchmark_" + filenameComponent(device.name.UTF8String) + ".json";
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError *library_error = nil;
    NSURL *library_url = [NSURL fileURLWithPath:@"build/fp4_kernels.metallib"];
    id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&library_error];
    if (library == nil) {
      std::fprintf(stderr, "[fp4] cannot load Metal library: %s\n", library_error.localizedDescription.UTF8String);
      return 1;
    }
    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> fp16_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"matmul_fp16"] error:&pipeline_error];
    id<MTLComputePipelineState> fp4_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"matmul_fp4_e2m1"] error:&pipeline_error];
    if (fp16_pipeline == nil || fp4_pipeline == nil) {
      std::fprintf(stderr, "[fp4] cannot create pipeline: %s\n", pipeline_error.localizedDescription.UTF8String);
      return 1;
    }
    const bool apple7 = [device supportsFamily:MTLGPUFamilyApple7];
    std::printf("[fp4] device=%s architecture=%s Apple7=%s native_fp4_sdk=%s\n", device.name.UTF8String,
                machineArchitecture().c_str(), apple7 ? "yes" : "no", native_fp4_sdk.c_str());
    std::vector<Workload> workloads = {
        {"prefill_linear", 128, hidden_size, hidden_size},
        {"decode_linear", 1, hidden_size, hidden_size},
        {"qkv_projection", 32, hidden_size, hidden_size * 3},
    };
    std::mt19937 generator(20260826);
    std::normal_distribution<float> distribution(0.0f, 0.25f);
    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "{\n  \"environment\": {\n    \"device\": \"" << jsonEscape(device.name.UTF8String) << "\",\n"
           << "    \"architecture\": \"" << machineArchitecture() << "\",\n"
           << "    \"os_version\": \"" << jsonEscape(NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String) << "\",\n"
           << "    \"apple_gpu_family_7_or_later\": " << (apple7 ? "true" : "false") << ",\n"
           << "    \"native_fp4_tensor_api_in_active_sdk\": " << (native_fp4_sdk == "available" ? "true" : "false") << ",\n"
           << "    \"native_fp4_status\": \"" << (native_fp4_sdk == "available" ? "requires runtime tensor execution probe" : "active SDK lacks MTLTensorDataTypeMetalFloat4E2M1") << "\"\n  },\n"
           << "  \"implementation\": \"software_fallback_packed_e2m1_block32_with_fp16_activations\",\n  \"workloads\": [\n";
    for (size_t workload_index = 0; workload_index < workloads.size(); ++workload_index) {
      const Workload &workload = workloads[workload_index];
      std::vector<float> activations(static_cast<size_t>(workload.m) * workload.k);
      std::vector<float> weights(static_cast<size_t>(workload.n) * workload.k);
      for (float &value : activations) value = distribution(generator);
      for (float &value : weights) value = distribution(generator);
      std::vector<_Float16> fp16_activations(activations.size());
      for (size_t index = 0; index < activations.size(); ++index) fp16_activations[index] = _Float16(activations[index]);
      std::vector<_Float16> fp16_weights;
      std::vector<uint8_t> packed_fp4;
      std::vector<_Float16> scales;
      quantizeWeights(weights, workload.k, fp16_weights, packed_fp4, scales);
      const std::vector<float> reference = cpuMatmul(activations, weights, workload);
      const MatmulParams params = {workload.m, workload.k, workload.n};
      id<MTLBuffer> activation_buffer = [device newBufferWithBytes:fp16_activations.data() length:fp16_activations.size() * sizeof(_Float16) options:MTLResourceStorageModeShared];
      id<MTLBuffer> fp16_weight_buffer = [device newBufferWithBytes:fp16_weights.data() length:fp16_weights.size() * sizeof(_Float16) options:MTLResourceStorageModeShared];
      id<MTLBuffer> fp4_weight_buffer = [device newBufferWithBytes:packed_fp4.data() length:packed_fp4.size() options:MTLResourceStorageModeShared];
      id<MTLBuffer> scale_buffer = [device newBufferWithBytes:scales.data() length:scales.size() * sizeof(_Float16) options:MTLResourceStorageModeShared];
      id<MTLBuffer> fp16_output = [device newBufferWithLength:reference.size() * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> fp4_output = [device newBufferWithLength:reference.size() * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> fp16_params = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
      id<MTLBuffer> fp4_params = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
      const Measurement fp16 = runPipeline(queue, fp16_pipeline, {activation_buffer, fp16_weight_buffer, fp16_output, fp16_params}, 3, workload, "fp16", reference, iterations);
      const Measurement fp4 = runPipeline(queue, fp4_pipeline, {activation_buffer, fp4_weight_buffer, scale_buffer, fp4_output, fp4_params}, 4, workload, "fp4_e2m1_block32", reference, iterations);
      const double speedup = fp16.mean_ms / fp4.mean_ms;
      std::printf("[fp4] %-16s fp16 gpu(avg/p50/p95)=%7.3f/%7.3f/%7.3f ms wall=%7.3f ms | fp4 gpu(avg/p50/p95)=%7.3f/%7.3f/%7.3f ms wall=%7.3f ms speedup=%5.3fx\n", workload.name.c_str(), fp16.mean_ms, fp16.p50_ms, fp16.p95_ms, fp16.mean_wall_ms, fp4.mean_ms, fp4.p50_ms, fp4.p95_ms, fp4.mean_wall_ms, speedup);
      report << "    {\"name\": \"" << workload.name << "\", \"m\": " << workload.m << ", \"k\": " << workload.k << ", \"n\": " << workload.n
             << ", \"fp16\": {\"mean_gpu_ms\": " << fp16.mean_ms << ", \"min_gpu_ms\": " << fp16.min_ms << ", \"p50_gpu_ms\": " << fp16.p50_ms << ", \"p95_gpu_ms\": " << fp16.p95_ms << ", \"max_gpu_ms\": " << fp16.max_ms << ", \"mean_wall_ms\": " << fp16.mean_wall_ms << ", \"tflops\": " << fp16.tflops << ", \"max_abs_error\": " << fp16.max_abs_error << ", \"rmse\": " << fp16.rmse << ", \"cosine_similarity\": " << fp16.cosine_similarity << "}"
             << ", \"fp4_e2m1_block32\": {\"mean_gpu_ms\": " << fp4.mean_ms << ", \"min_gpu_ms\": " << fp4.min_ms << ", \"p50_gpu_ms\": " << fp4.p50_ms << ", \"p95_gpu_ms\": " << fp4.p95_ms << ", \"max_gpu_ms\": " << fp4.max_ms << ", \"mean_wall_ms\": " << fp4.mean_wall_ms << ", \"tflops\": " << fp4.tflops << ", \"max_abs_error\": " << fp4.max_abs_error << ", \"rmse\": " << fp4.rmse << ", \"cosine_similarity\": " << fp4.cosine_similarity << "}"
             << ", \"fp4_vs_fp16_speedup\": " << speedup << "}" << (workload_index + 1 == workloads.size() ? "\n" : ",\n");
    }
    report << "  ]\n}\n";
    writeReport(output_path, report.str());
    return 0;
  }
}
