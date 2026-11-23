// float -> int8/uint8 affine quantisation: clamp(round_half_even(v / scale) + zp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace executorch_rknn {

// Adding and subtracting 1.5 * 2^23 rounds half to even under the default
// rounding mode, for |x| < 2^22; callers clamp first. Much cheaper than
// nearbyint on uClibc.
constexpr float kRoundMagic = 12582912.0f;
constexpr float kQuantLimit = 1.0e6f;
constexpr size_t kPrefetch = 64;  // floats ahead

template <typename T>
inline T quantize(float v, float inv_scale, int32_t zp) {
  float x = std::min(std::max(v * inv_scale, -kQuantLimit), kQuantLimit);
  x = (x + kRoundMagic) - kRoundMagic;
  const int32_t q = static_cast<int32_t>(x) + zp;
  return static_cast<T>(std::min<int32_t>(std::max<int32_t>(q, std::numeric_limits<T>::min()),
                                          std::numeric_limits<T>::max()));
}

#if defined(__ARM_NEON)
// Quantises 8 floats to int8. Advanced SIMD always rounds to nearest even. No
// float clamp is needed: values too large for the magic-number rounding stay
// too large, and the saturating conversion, narrowing and add clamp them.
// With the prefetch in quantize_row_s8, a 224x224x3 input takes 1.5 ms on a
// Cortex-A7, against 2.5 ms with a float clamp and no prefetch.
inline int8x8_t quantize8_s8(const float* p, float32x4_t inv_scale, int16x8_t zp, float32x4_t magic) {
  float32x4_t a = vmulq_f32(vld1q_f32(p), inv_scale);
  float32x4_t b = vmulq_f32(vld1q_f32(p + 4), inv_scale);
  a = vsubq_f32(vaddq_f32(a, magic), magic);
  b = vsubq_f32(vaddq_f32(b, magic), magic);
  const int16x8_t q = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(a)), vqmovn_s32(vcvtq_s32_f32(b)));
  return vqmovn_s16(vqaddq_s16(q, zp));
}
#endif

// One row of NCHW float (channel c at src + c * channel_step) -> NHWC int8.
inline void quantize_row_s8(const float* src, size_t channel_step, int8_t* dst, uint32_t C, uint32_t W,
                            float inv_scale, int32_t zp) {
  uint32_t w = 0;
#if defined(__ARM_NEON)
  const float32x4_t inv = vdupq_n_f32(inv_scale), magic = vdupq_n_f32(kRoundMagic);
  const int16x8_t z = vdupq_n_s16(static_cast<int16_t>(zp));
  if (C == 1) {
    for (; w + 8 <= W; w += 8) vst1_s8(dst + w, quantize8_s8(src + w, inv, z, magic));
  } else if (C == 3) {
    const float *c0 = src, *c1 = src + channel_step, *c2 = src + 2 * channel_step;
    for (; w + 8 <= W; w += 8) {
      // The A7 does not prefetch three input streams well on its own.
      __builtin_prefetch(c0 + w + kPrefetch);
      __builtin_prefetch(c1 + w + kPrefetch);
      __builtin_prefetch(c2 + w + kPrefetch);
      int8x8x3_t q;
      q.val[0] = quantize8_s8(c0 + w, inv, z, magic);
      q.val[1] = quantize8_s8(c1 + w, inv, z, magic);
      q.val[2] = quantize8_s8(c2 + w, inv, z, magic);
      vst3_s8(dst + 3 * w, q);
    }
  } else if (C == 4) {
    const float *c0 = src, *c1 = src + channel_step, *c2 = src + 2 * channel_step, *c3 = src + 3 * channel_step;
    for (; w + 8 <= W; w += 8) {
      __builtin_prefetch(c0 + w + kPrefetch);
      __builtin_prefetch(c1 + w + kPrefetch);
      __builtin_prefetch(c2 + w + kPrefetch);
      __builtin_prefetch(c3 + w + kPrefetch);
      int8x8x4_t q;
      q.val[0] = quantize8_s8(c0 + w, inv, z, magic);
      q.val[1] = quantize8_s8(c1 + w, inv, z, magic);
      q.val[2] = quantize8_s8(c2 + w, inv, z, magic);
      q.val[3] = quantize8_s8(c3 + w, inv, z, magic);
      vst4_s8(dst + 4 * w, q);
    }
  }
#endif
  for (; w < W; ++w)
    for (uint32_t c = 0; c < C; ++c) dst[w * C + c] = quantize<int8_t>(src[c * channel_step + w], inv_scale, zp);
}

// NCHW float -> NHWC int8 with rows w_stride elements apart. Padding is left as is.
inline void quantize_nchw_to_nhwc_s8(const float* src, int8_t* dst, uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                                     size_t w_stride, float inv_scale, int32_t zp) {
  const size_t channel_step = static_cast<size_t>(H) * W;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      quantize_row_s8(src + (static_cast<size_t>(n) * C * H + h) * W, channel_step,
                      dst + (static_cast<size_t>(n) * H + h) * w_stride * C, C, W, inv_scale, zp);
}

}  // namespace executorch_rknn
