#include <metal_stdlib>
using namespace metal;

/** Parameters shared by the FP16 and packed-FP4 matrix multiplication kernels. */
struct MatmulParams {
  /** Number of rows in the activation matrix. */
  uint m;
  /** Shared reduction dimension. It must be a multiple of 32. */
  uint k;
  /** Number of output columns and weight rows. */
  uint n;
};

/** Decodes an E2M1 finite-number nibble into a floating-point value. */
inline float decode_e2m1(uchar bits) {
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[bits & 0x7u];
  return (bits & 0x8u) == 0 ? magnitude : -magnitude;
}

/** Computes one FP16 activation by FP16 weight output element. */
kernel void matmul_fp16(device const half *activations [[buffer(0)]],
                         device const half *weights [[buffer(1)]],
                         device float *output [[buffer(2)]],
                         constant MatmulParams &params [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.n || gid.y >= params.m) {
    return;
  }
  float sum = 0.0f;
  const uint activation_base = gid.y * params.k;
  const uint weight_base = gid.x * params.k;
  for (uint k_index = 0; k_index < params.k; ++k_index) {
    sum += float(activations[activation_base + k_index]) * float(weights[weight_base + k_index]);
  }
  output[gid.y * params.n + gid.x] = sum;
}

/** Computes one FP16 activation by packed E2M1 FP4 weight output element with 32-value block scaling. */
kernel void matmul_fp4_e2m1(device const half *activations [[buffer(0)]],
                             device const uchar *packed_weights [[buffer(1)]],
                             device const half *block_scales [[buffer(2)]],
                             device float *output [[buffer(3)]],
                             constant MatmulParams &params [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.n || gid.y >= params.m) {
    return;
  }
  float sum = 0.0f;
  const uint activation_base = gid.y * params.k;
  const uint weight_base = gid.x * params.k;
  const uint scale_base = gid.x * (params.k / 32u);
  for (uint k_index = 0; k_index < params.k; ++k_index) {
    const uchar packed = packed_weights[(weight_base + k_index) >> 1u];
    const uchar nibble = (k_index & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u);
    const float weight = decode_e2m1(nibble) * float(block_scales[scale_base + (k_index / 32u)]);
    sum += float(activations[activation_base + k_index]) * weight;
  }
  output[gid.y * params.n + gid.x] = sum;
}
