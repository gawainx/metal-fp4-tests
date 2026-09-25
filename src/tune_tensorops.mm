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

struct Shape { uint32_t m, k, n; const char *name; };
struct Tile { uint32_t m, n, simdgroups, block_k; const char *suffix; };
struct Timing { double mean, p50, min, max; };
struct Sample { size_t index; float reference; };

static constexpr Tile tiles[] = {
    {32, 32, 1, 1024, "m32n32_s1"}, {32, 64, 1, 1024, "m32n64_s1"},
    {32, 128, 1, 1024, "m32n128_s1"}, {64, 64, 1, 1024, "m64n64_s1"},
    {64, 64, 4, 1024, "m64n64_s4"}, {32, 128, 4, 1024, "m32n128_s4"},
    {32, 64, 4, 1024, "m32n64_s4"}, {64, 128, 4, 1024, "m64n128_s4"},
    {128, 128, 4, 1024, "m128n128_s4"},
    {32, 128, 4, 128, "m32n128_s4_k128"}, {32, 128, 4, 256, "m32n128_s4_k256"},
    {64, 64, 4, 128, "m64n64_s4_k128"}, {64, 64, 4, 256, "m64n64_s4_k256"}};
static constexpr const char *formats[] = {"bf16", "fp8", "fp4"};

struct Candidate {
  int format;
  Tile tile;
  std::string name;
  id<MTLComputePipelineState> pipeline;
  id<MTLBuffer> output;
  std::vector<double> samples;
  Timing timing{};
  double max_error = 0;
};

static float decode(uint8_t code) {
  constexpr float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return (code & 8) ? -values[code & 7] : values[code & 7];
}

static std::vector<uint8_t> packFP4(const std::vector<uint8_t> &codes) {
  std::vector<uint8_t> packed(codes.size() / 2);
  for (size_t i = 0; i < packed.size(); ++i) packed[i] = codes[2 * i] | (codes[2 * i + 1] << 4);
  return packed;
}

static std::vector<uint8_t> packFP8(const std::vector<uint8_t> &codes) {
  constexpr uint8_t magnitudes[] = {0, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
  std::vector<uint8_t> packed(codes.size());
  for (size_t i = 0; i < packed.size(); ++i)
    packed[i] = magnitudes[codes[i] & 7] | ((codes[i] & 8) << 4);
  return packed;
}

static std::vector<uint16_t> packBF16(const std::vector<uint8_t> &codes) {
  std::vector<uint16_t> packed(codes.size());
  for (size_t i = 0; i < packed.size(); ++i)
    packed[i] = uint16_t(std::bit_cast<uint32_t>(decode(codes[i])) >> 16);
  return packed;
}

static id<MTLBuffer> buffer(id<MTLDevice> device, const void *data, size_t bytes) {
  id<MTLBuffer> result = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
  if (!result) { std::fprintf(stderr, "Metal buffer allocation failed\n"); std::exit(1); }
  return result;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                             const std::string &name) {
  NSError *error = nil;
  NSString *function_name = [NSString stringWithUTF8String:name.c_str()];
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (!function) { std::fprintf(stderr, "missing kernel %s\n", name.c_str()); std::exit(1); }
  id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
  if (!result) { std::fprintf(stderr, "%s: %s\n", name.c_str(), error.localizedDescription.UTF8String); std::exit(1); }
  return result;
}

static double execute(id<MTLCommandQueue> queue, Candidate &candidate,
                      id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> parameters, Shape shape) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  command.label = [NSString stringWithUTF8String:candidate.name.c_str()];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  encoder.label = command.label;
  [encoder setComputePipelineState:candidate.pipeline];
  [encoder setBuffer:a offset:0 atIndex:0];
  [encoder setBuffer:b offset:0 atIndex:1];
  [encoder setBuffer:candidate.output offset:0 atIndex:2];
  [encoder setBuffer:parameters offset:0 atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.n / candidate.tile.n, shape.m / candidate.tile.m, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * candidate.tile.simdgroups, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::fprintf(stderr, "%s: %s\n", candidate.name.c_str(), command.error.localizedDescription.UTF8String);
    std::exit(1);
  }
  return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

static Timing summarize(std::vector<double> samples) {
  double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  return {mean, samples[samples.size() / 2], samples.front(), samples.back()};
}

static std::vector<Sample> referenceSamples(Shape shape, const std::vector<uint8_t> &a,
                                             const std::vector<uint8_t> &b) {
  std::vector<Sample> samples;
  for (uint32_t m = 0; m < shape.m; m += 32) {
    for (uint32_t n = 0; n < shape.n; n += 32) {
      for (uint32_t offset : {0u, 31u}) {
        uint32_t row = m + offset, column = n + offset;
        float value = 0;
        for (uint32_t k = 0; k < shape.k; ++k)
          value += decode(a[size_t(row) * shape.k + k]) * decode(b[size_t(k) * shape.n + column]);
        samples.push_back({size_t(row) * shape.n + column, value});
      }
    }
  }
  return samples;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    bool selected_only = false;
    std::string output_path;
    for (int argument = 1; argument < argc; ++argument) {
      if (std::string(argv[argument]) == "--selected") selected_only = true;
      else if (std::string(argv[argument]) == "--output" && argument + 1 < argc)
        output_path = argv[++argument];
      else {
        std::fprintf(stderr, "usage: %s [--selected] [--output path]\n", argv[0]);
        return 1;
      }
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { std::fprintf(stderr, "Metal device unavailable\n"); return 1; }
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@"build/tune_tensorops.metallib"] error:&error];
    if (!library) { std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const Shape shapes[] = {{128, 1024, 1024, "prefill_linear"},
                            {32, 1024, 3072, "qkv_projection"},
                            {512, 1024, 2048, "large_prefill"},
                            {32, 1024, 8192, "wide_projection"}};
    const int warmup = selected_only ? 500 : 5;
    const int repetitions = selected_only ? 100 : 40;
    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "{\n  \"device\": \"" << device.name.UTF8String
           << "\",\n  \"input\": \"identical E2M1 values encoded as BF16, FP8 E4M3, FP4 E2M1\",\n"
           << "  \"output\": \"FP32\",\n  \"warmup_iterations\": " << warmup
           << ",\n  \"measured_iterations_per_kernel\": " << repetitions << ",\n  \"workloads\": [\n";
    for (size_t workload = 0; workload < std::size(shapes); ++workload) {
      Shape shape = shapes[workload];
      std::mt19937 generator(42);
      std::uniform_int_distribution<int> pick(0, 15);
      std::vector<uint8_t> a_codes(size_t(shape.m) * shape.k), b_codes(size_t(shape.k) * shape.n);
      for (auto &code : a_codes) code = uint8_t(pick(generator));
      for (auto &code : b_codes) code = uint8_t(pick(generator));
      auto a_bf16 = packBF16(a_codes), b_bf16 = packBF16(b_codes);
      auto a_fp8 = packFP8(a_codes), b_fp8 = packFP8(b_codes);
      auto a_fp4 = packFP4(a_codes), b_fp4 = packFP4(b_codes);
      id<MTLBuffer> a_buffers[] = {buffer(device, a_bf16.data(), a_bf16.size() * 2),
                                   buffer(device, a_fp8.data(), a_fp8.size()),
                                   buffer(device, a_fp4.data(), a_fp4.size())};
      id<MTLBuffer> b_buffers[] = {buffer(device, b_bf16.data(), b_bf16.size() * 2),
                                   buffer(device, b_fp8.data(), b_fp8.size()),
                                   buffer(device, b_fp4.data(), b_fp4.size())};
      id<MTLBuffer> parameters = buffer(device, &shape, sizeof(shape));
      auto references = referenceSamples(shape, a_codes, b_codes);
      std::vector<Candidate> candidates;
      for (int format = 0; format < 3; ++format) {
        for (const Tile &tile : tiles) {
          if (shape.m % tile.m || shape.n % tile.n) continue;
          if (selected_only) {
            const bool large_prefill = std::string(shape.name) == "large_prefill";
            const char *choice = format == 2 && large_prefill ? "m32n32_s1"
                                 : format == 0 && large_prefill ? "m32n64_s1"
                                 : "m32n128_s4";
            if (std::string(tile.suffix) != choice) continue;
          }
          Candidate candidate{};
          candidate.format = format;
          candidate.tile = tile;
          candidate.name = std::string(formats[format]) + "_" + tile.suffix;
          candidate.pipeline = pipeline(device, library, candidate.name);
          candidate.output = [device newBufferWithLength:size_t(shape.m) * shape.n * sizeof(float)
                                                options:MTLResourceStorageModeShared];
          candidates.push_back(std::move(candidate));
        }
      }
      for (int iteration = -warmup; iteration < repetitions; ++iteration) {
        size_t first = size_t(iteration + warmup) % candidates.size();
        for (size_t offset = 0; offset < candidates.size(); ++offset) {
          Candidate &candidate = candidates[(first + offset) % candidates.size()];
          double elapsed = execute(queue, candidate,
                                   a_buffers[candidate.format],
                                   b_buffers[candidate.format],
                                   parameters, shape);
          if (iteration >= 0) candidate.samples.push_back(elapsed);
        }
      }
      report << "    {\"name\": \"" << shape.name << "\", \"m\": " << shape.m
             << ", \"k\": " << shape.k << ", \"n\": " << shape.n << ", \"candidates\": [\n";
      for (size_t index = 0; index < candidates.size(); ++index) {
        Candidate &candidate = candidates[index];
        candidate.timing = summarize(candidate.samples);
        const float *output = static_cast<const float *>(candidate.output.contents);
        for (size_t i = 0; i < size_t(shape.m) * shape.n; ++i) {
          if (!std::isfinite(output[i])) { std::fprintf(stderr, "%s nonfinite\n", candidate.name.c_str()); return 1; }
        }
        for (const Sample &sample : references)
          candidate.max_error = std::max(candidate.max_error, std::abs(double(output[sample.index]) - sample.reference));
        if (candidate.max_error > .001) {
          std::fprintf(stderr, "%s %s error %.6f\n", shape.name, candidate.name.c_str(), candidate.max_error);
          return 1;
        }
        const Timing &t = candidate.timing;
        std::printf("%-16s %-18s %.6f ms (p50 %.6f) error %.6f\n", shape.name,
                    candidate.name.c_str(), t.mean, t.p50, candidate.max_error);
        report << "      {\"name\": \"" << candidate.name << "\", \"format\": \"" << formats[candidate.format]
               << "\", \"tile_m\": " << candidate.tile.m << ", \"tile_n\": " << candidate.tile.n
               << ", \"simdgroups\": " << candidate.tile.simdgroups
               << ", \"block_k\": " << candidate.tile.block_k
               << ", \"mean_gpu_ms\": " << t.mean << ", \"p50_gpu_ms\": " << t.p50
               << ", \"min_gpu_ms\": " << t.min << ", \"max_gpu_ms\": " << t.max
               << ", \"max_sampled_cpu_reference_error\": " << candidate.max_error << "}"
               << (index + 1 == candidates.size() ? "\n" : ",\n");
      }
      report << "    ]}" << (workload + 1 == std::size(shapes) ? "\n" : ",\n");
    }
    report << "  ]\n}\n";
    std::filesystem::create_directories("results");
    std::string component(device.name.UTF8String);
    for (char &character : component) if (character == ' ') character = '_';
    std::string path = output_path.empty()
                           ? std::string("results/tensorops_") +
                                 (selected_only ? "selected_validation_" : "tuning_extended_") +
                                 component + ".json"
                           : output_path;
    if (std::filesystem::exists(path)) {
      if (!output_path.empty()) {
        std::fprintf(stderr, "result already exists: %s\n", path.c_str());
        return 1;
      }
      const std::string stem = path.substr(0, path.size() - 5);
      for (unsigned run = 2;; ++run) {
        path = stem + "_run" + std::to_string(run) + ".json";
        if (!std::filesystem::exists(path)) break;
      }
    }
    std::ofstream output(path);
    output << report.str();
    if (!output) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
    std::printf("Saved %s\n", path.c_str());
  }
}
