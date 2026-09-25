#include <metal_stdlib>
#include <metal_tensor>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct Shape {
  uint m;
  uint k;
  uint n;
};

// Metal 4.1 TensorOps consumes the packed FP4 operand directly.
kernel void fp4_native_matmul(device half *activations [[buffer(0)]],
                              device uchar *packed_weights [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              constant Shape &shape [[buffer(3)]],
                              uint2 tile [[threadgroup_position_in_grid]]) {
  tensor<device half, dextents<int, 2>, tensor_inline> a(
      activations, dextents<int, 2>(int(shape.k), int(shape.m)),
      array<int, 2>({1, int(shape.k)}));
  tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline> b(
      packed_weights, dextents<int, 2>(int(shape.n), int(shape.k)),
      array<int, 2>({1, int(shape.n)}));
  tensor<device float, dextents<int, 2>, tensor_inline> c(
      output, dextents<int, 2>(int(shape.n), int(shape.m)),
      array<int, 2>({1, int(shape.n)}));
  constexpr auto descriptor = matmul2d_descriptor(32, 32, 1024);
  matmul2d<descriptor, execution_simdgroup> operation;
  auto a_tile = a.slice<1024, 32>(0, int(tile.y) * 32);
  auto b_tile = b.slice<32, 1024>(int(tile.x) * 32, 0);
  auto c_tile = c.slice<32, 32>(int(tile.x) * 32, int(tile.y) * 32);
  operation.run(a_tile, b_tile, c_tile);
}
