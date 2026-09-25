#include <metal_stdlib>
#include <metal_tensor>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void native_fp4_matmul(device half *a [[buffer(0)]],
                              device uchar *b [[buffer(1)]],
                              device half *c [[buffer(2)]]) {
  tensor<device half, dextents<int, 2>, tensor_inline> ta(a, dextents<int, 2>(32, 32), array<int, 2>({1, 32}));
  tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline> tb(b, dextents<int, 2>(32, 32), array<int, 2>({1, 256}));
  tensor<device half, dextents<int, 2>, tensor_inline> tc(c, dextents<int, 2>(32, 32), array<int, 2>({1, 32}));
  constexpr auto desc = matmul2d_descriptor(32, 32, 32);
  matmul2d<desc, execution_simdgroup> op;
  op.run(ta, tb, tc);
}
