#include <metal_stdlib>
using namespace metal;

struct Shape {
  uint m;
  uint k;
  uint n;
};

inline float decode_bf16(ushort code) {
  return as_type<float>(uint(code) << 16u);
}

kernel void bf16_software_matmul(device const ushort *activations [[buffer(0)]],
                                 device const ushort *weights [[buffer(1)]],
                                 device float *output [[buffer(2)]],
                                 constant Shape &shape [[buffer(3)]],
                                 uint2 element [[thread_position_in_grid]]) {
  if (element.x >= shape.n || element.y >= shape.m) return;
  float sum = 0.0f;
  for (uint reduction = 0; reduction < shape.k; ++reduction) {
    sum += decode_bf16(activations[element.y * shape.k + reduction]) *
           decode_bf16(weights[reduction * shape.n + element.x]);
  }
  output[element.y * shape.n + element.x] = sum;
}
