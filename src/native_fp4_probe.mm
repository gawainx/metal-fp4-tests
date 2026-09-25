#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "Metal device unavailable\n");
      return 1;
    }
    std::printf("device=%s Apple10=%s\n", device.name.UTF8String,
                [device supportsFamily:MTLGPUFamilyApple10] ? "yes" : "no");

    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:@"build/native_fp4_probe.metallib"];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil) {
      std::fprintf(stderr, "load library: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"native_fp4_matmul"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      std::fprintf(stderr, "create pipeline: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }

    constexpr size_t side = 32;
    constexpr size_t fp4_row_stride = 256;
    std::vector<_Float16> activations(side * side, _Float16(1.0f));
    std::vector<uint8_t> weights(side * fp4_row_stride / 2, 0x22);
    std::vector<_Float16> output(side * side, _Float16(0.0f));
    id<MTLBuffer> a = [device newBufferWithBytes:activations.data() length:activations.size() * sizeof(_Float16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithBytes:weights.data() length:weights.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [device newBufferWithBytes:output.data() length:output.size() * sizeof(_Float16) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:c offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "GPU execution: %s\n", command.error.localizedDescription.UTF8String);
      return 1;
    }

    const auto *result = static_cast<const _Float16 *>(c.contents);
    float max_error = 0.0f;
    for (size_t index = 0; index < output.size(); ++index) {
      max_error = std::max(max_error, std::fabs(float(result[index]) - 32.0f));
    }
    std::printf("native TensorOps FP4 matmul: expected=32, output[0]=%.1f, max_error=%.1f\n",
                float(result[0]), max_error);
    return max_error == 0.0f ? 0 : 1;
  }
}
