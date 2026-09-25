#include <metal_stdlib>
using namespace metal;

struct Shape {
  uint m;
  uint k;
  uint n;
};

inline float decode_e2m1(uchar code) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = magnitudes[code & 7u];
  return (code & 8u) == 0u ? value : -value;
}

// Reads the exact same packed bytes as fp4_native_matmul, then decodes each nibble in the shader.
kernel void fp4_software_matmul(device const half *activations [[buffer(0)]],
                                device const uchar *packed_weights [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant Shape &shape [[buffer(3)]],
                                uint2 element [[thread_position_in_grid]]) {
  if (element.x >= shape.n || element.y >= shape.m) return;
  float sum = 0.0f;
  for (uint reduction = 0; reduction < shape.k; ++reduction) {
    const uint weight_index = reduction * shape.n + element.x;
    const uchar packed = packed_weights[weight_index / 2u];
    const uchar code = (weight_index & 1u) == 0u ? packed & 15u : packed >> 4u;
    sum += float(activations[element.y * shape.k + reduction]) * decode_e2m1(code);
  }
  output[element.y * shape.n + element.x] = sum;
}
