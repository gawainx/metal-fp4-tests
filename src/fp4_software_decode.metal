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

// Reads the exact same packed bytes as fp4_native_matmul, then decodes both operands in the shader.
kernel void fp4_software_matmul(device const uchar *activations [[buffer(0)]],
                                device const uchar *packed_weights [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant Shape &shape [[buffer(3)]],
                                uint2 element [[thread_position_in_grid]]) {
  if (element.x >= shape.n || element.y >= shape.m) return;
  float sum = 0.0f;
  for (uint reduction = 0; reduction < shape.k; ++reduction) {
    const uint activation_index = element.y * shape.k + reduction;
    const uint weight_index = reduction * shape.n + element.x;
    const uchar activation_byte = activations[activation_index / 2u];
    const uchar weight_byte = packed_weights[weight_index / 2u];
    const uchar activation_code = (activation_index & 1u) == 0u ? activation_byte & 15u : activation_byte >> 4u;
    const uchar weight_code = (weight_index & 1u) == 0u ? weight_byte & 15u : weight_byte >> 4u;
    sum += decode_e2m1(activation_code) * decode_e2m1(weight_code);
  }
  output[element.y * shape.n + element.x] = sum;
}
