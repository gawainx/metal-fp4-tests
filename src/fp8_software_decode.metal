#include <metal_stdlib>
using namespace metal;

struct Shape {
  uint m;
  uint k;
  uint n;
};

inline float decode_e4m3(uchar code) {
  const uint exponent = (code >> 3u) & 15u;
  const uint mantissa = code & 7u;
  const float magnitude = exponent == 0u
      ? float(mantissa) * 0.001953125f
      : (1.0f + float(mantissa) * 0.125f) * exp2(float(exponent) - 7.0f);
  return (code & 128u) == 0u ? magnitude : -magnitude;
}

kernel void fp8_software_matmul(device const uchar *activations [[buffer(0)]],
                                device const uchar *weights [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant Shape &shape [[buffer(3)]],
                                uint2 element [[thread_position_in_grid]]) {
  if (element.x >= shape.n || element.y >= shape.m) return;
  float sum = 0.0f;
  for (uint reduction = 0; reduction < shape.k; ++reduction) {
    sum += decode_e4m3(activations[element.y * shape.k + reduction]) *
           decode_e4m3(weights[reduction * shape.n + element.x]);
  }
  output[element.y * shape.n + element.x] = sum;
}
