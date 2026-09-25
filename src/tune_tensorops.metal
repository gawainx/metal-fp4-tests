#include <metal_stdlib>
#include <metal_tensor>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct TuneShape {
  uint m;
  uint k;
  uint n;
};

#define DEFINE_TUNING_KERNEL(NAME, ELEMENT, POINTER, TILE_M, TILE_N, SIMD_GROUPS) \
kernel void NAME(device POINTER *activations [[buffer(0)]], \
                 device POINTER *weights [[buffer(1)]], \
                 device float *output [[buffer(2)]], \
                 constant TuneShape &shape [[buffer(3)]], \
                 uint2 tile [[threadgroup_position_in_grid]]) { \
  tensor<device ELEMENT, dextents<int, 2>, tensor_inline> a( \
      activations, dextents<int, 2>(int(shape.k), int(shape.m)), \
      array<int, 2>({1, int(shape.k)})); \
  tensor<device ELEMENT, dextents<int, 2>, tensor_inline> b( \
      weights, dextents<int, 2>(int(shape.n), int(shape.k)), \
      array<int, 2>({1, int(shape.n)})); \
  tensor<device float, dextents<int, 2>, tensor_inline> c( \
      output, dextents<int, 2>(int(shape.n), int(shape.m)), \
      array<int, 2>({1, int(shape.n)})); \
  constexpr auto descriptor = matmul2d_descriptor(TILE_M, TILE_N, 1024); \
  matmul2d<descriptor, execution_simdgroups<SIMD_GROUPS>> operation; \
  auto a_tile = a.slice<1024, TILE_M>(0, int(tile.y) * TILE_M); \
  auto b_tile = b.slice<TILE_N, 1024>(int(tile.x) * TILE_N, 0); \
  auto c_tile = c.slice<TILE_N, TILE_M>(int(tile.x) * TILE_N, int(tile.y) * TILE_M); \
  operation.run(a_tile, b_tile, c_tile); \
}

#define DEFINE_FORMAT_VARIANTS(PREFIX, ELEMENT, POINTER) \
DEFINE_TUNING_KERNEL(PREFIX##_m32n32_s1, ELEMENT, POINTER, 32, 32, 1) \
DEFINE_TUNING_KERNEL(PREFIX##_m32n64_s1, ELEMENT, POINTER, 32, 64, 1) \
DEFINE_TUNING_KERNEL(PREFIX##_m32n128_s1, ELEMENT, POINTER, 32, 128, 1) \
DEFINE_TUNING_KERNEL(PREFIX##_m64n64_s1, ELEMENT, POINTER, 64, 64, 1) \
DEFINE_TUNING_KERNEL(PREFIX##_m64n64_s4, ELEMENT, POINTER, 64, 64, 4) \
DEFINE_TUNING_KERNEL(PREFIX##_m32n128_s4, ELEMENT, POINTER, 32, 128, 4) \
DEFINE_TUNING_KERNEL(PREFIX##_m32n64_s4, ELEMENT, POINTER, 32, 64, 4) \
DEFINE_TUNING_KERNEL(PREFIX##_m64n128_s4, ELEMENT, POINTER, 64, 128, 4) \
DEFINE_TUNING_KERNEL(PREFIX##_m128n128_s4, ELEMENT, POINTER, 128, 128, 4)

DEFINE_FORMAT_VARIANTS(bf16, bfloat, bfloat)
DEFINE_FORMAT_VARIANTS(fp8, metal_fp8_e4m3_format, uchar)
DEFINE_FORMAT_VARIANTS(fp4, metal_fp4_e2m1_format, uchar)

#define DEFINE_K_TUNING_KERNEL(NAME, ELEMENT, POINTER, TILE_M, TILE_N, BLOCK_K) \
kernel void NAME(device POINTER *activations [[buffer(0)]], \
                 device POINTER *weights [[buffer(1)]], \
                 device float *output [[buffer(2)]], \
                 constant TuneShape &shape [[buffer(3)]], \
                 uint2 tile [[threadgroup_position_in_grid]]) { \
  tensor<device ELEMENT, dextents<int, 2>, tensor_inline> a( \
      activations, dextents<int, 2>(int(shape.k), int(shape.m)), \
      array<int, 2>({1, int(shape.k)})); \
  tensor<device ELEMENT, dextents<int, 2>, tensor_inline> b( \
      weights, dextents<int, 2>(int(shape.n), int(shape.k)), \
      array<int, 2>({1, int(shape.n)})); \
  tensor<device float, dextents<int, 2>, tensor_inline> c( \
      output, dextents<int, 2>(int(shape.n), int(shape.m)), \
      array<int, 2>({1, int(shape.n)})); \
  constexpr auto mode = matmul2d_descriptor::mode::multiply_accumulate; \
  constexpr auto descriptor = matmul2d_descriptor(TILE_M, TILE_N, BLOCK_K, false, false, false, mode); \
  matmul2d<descriptor, execution_simdgroups<4>> operation; \
  auto first_a = a.slice<BLOCK_K, TILE_M>(0, int(tile.y) * TILE_M); \
  auto first_b = b.slice<TILE_N, BLOCK_K>(int(tile.x) * TILE_N, 0); \
  auto accumulator = operation.get_destination_cooperative_tensor<decltype(first_a), decltype(first_b), float>(); \
  for (uint k = 0; k < shape.k; k += BLOCK_K) { \
    threadgroup_barrier(mem_flags::mem_none); \
    auto a_tile = a.slice<BLOCK_K, TILE_M>(int(k), int(tile.y) * TILE_M); \
    auto b_tile = b.slice<TILE_N, BLOCK_K>(int(tile.x) * TILE_N, int(k)); \
    operation.run(a_tile, b_tile, accumulator); \
  } \
  auto c_tile = c.slice<TILE_N, TILE_M>(int(tile.x) * TILE_N, int(tile.y) * TILE_M); \
  accumulator.store(c_tile); \
}

#define DEFINE_K_VARIANTS(PREFIX, ELEMENT, POINTER) \
DEFINE_K_TUNING_KERNEL(PREFIX##_m32n128_s4_k128, ELEMENT, POINTER, 32, 128, 128) \
DEFINE_K_TUNING_KERNEL(PREFIX##_m32n128_s4_k256, ELEMENT, POINTER, 32, 128, 256) \
DEFINE_K_TUNING_KERNEL(PREFIX##_m64n64_s4_k128, ELEMENT, POINTER, 64, 64, 128) \
DEFINE_K_TUNING_KERNEL(PREFIX##_m64n64_s4_k256, ELEMENT, POINTER, 64, 64, 256)

DEFINE_K_VARIANTS(bf16, bfloat, bfloat)
DEFINE_K_VARIANTS(fp8, metal_fp8_e4m3_format, uchar)
DEFINE_K_VARIANTS(fp4, metal_fp4_e2m1_format, uchar)
