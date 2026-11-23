// float <-> int8 affine quantisation, q = clamp(round_half_even(v / scale) + zp),
// fused with the layout change between ExecuTorch (NCHW) and the NPU.
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

#if defined(__ARM_NEON)
// Transposes an 8x8 block of bytes held as 8 rows.
inline void transpose8x8(int8x8_t r[8]) {
  const int8x8x2_t t01 = vtrn_s8(r[0], r[1]), t23 = vtrn_s8(r[2], r[3]);
  const int8x8x2_t t45 = vtrn_s8(r[4], r[5]), t67 = vtrn_s8(r[6], r[7]);
  const int16x4x2_t u0 = vtrn_s16(vreinterpret_s16_s8(t01.val[0]), vreinterpret_s16_s8(t23.val[0]));
  const int16x4x2_t u1 = vtrn_s16(vreinterpret_s16_s8(t01.val[1]), vreinterpret_s16_s8(t23.val[1]));
  const int16x4x2_t u2 = vtrn_s16(vreinterpret_s16_s8(t45.val[0]), vreinterpret_s16_s8(t67.val[0]));
  const int16x4x2_t u3 = vtrn_s16(vreinterpret_s16_s8(t45.val[1]), vreinterpret_s16_s8(t67.val[1]));
  const int32x2x2_t v0 = vtrn_s32(vreinterpret_s32_s16(u0.val[0]), vreinterpret_s32_s16(u2.val[0]));
  const int32x2x2_t v1 = vtrn_s32(vreinterpret_s32_s16(u1.val[0]), vreinterpret_s32_s16(u3.val[0]));
  const int32x2x2_t v2 = vtrn_s32(vreinterpret_s32_s16(u0.val[1]), vreinterpret_s32_s16(u2.val[1]));
  const int32x2x2_t v3 = vtrn_s32(vreinterpret_s32_s16(u1.val[1]), vreinterpret_s32_s16(u3.val[1]));
  r[0] = vreinterpret_s8_s32(v0.val[0]);
  r[1] = vreinterpret_s8_s32(v1.val[0]);
  r[2] = vreinterpret_s8_s32(v2.val[0]);
  r[3] = vreinterpret_s8_s32(v3.val[0]);
  r[4] = vreinterpret_s8_s32(v0.val[1]);
  r[5] = vreinterpret_s8_s32(v1.val[1]);
  r[6] = vreinterpret_s8_s32(v2.val[1]);
  r[7] = vreinterpret_s8_s32(v3.val[1]);
}
#endif

// tile[l * W + w] -> out[w * C2 + l] for lanes l < lanes.
inline void interleave_lanes(const int8_t* tile, int8_t* out, uint32_t lanes, uint32_t W, uint32_t C2) {
  uint32_t l = 0;
#if defined(__ARM_NEON)
  for (; l + 8 <= lanes; l += 8) {
    uint32_t w = 0;
    for (; w + 8 <= W; w += 8) {
      int8x8_t r[8];
      for (int i = 0; i < 8; ++i) r[i] = vld1_s8(tile + (l + i) * W + w);
      transpose8x8(r);
      for (int i = 0; i < 8; ++i) vst1_s8(out + (w + i) * C2 + l, r[i]);
    }
    for (; w < W; ++w)
      for (uint32_t i = 0; i < 8; ++i) out[w * C2 + l + i] = tile[(l + i) * W + w];
  }
#endif
  for (; l < lanes; ++l)
    for (uint32_t w = 0; w < W; ++w) out[w * C2 + l] = tile[l * W + w];
}

// in[w * C2 + l] -> tile[l * W + w] for lanes l < lanes.
inline void deinterleave_lanes(const int8_t* in, int8_t* tile, uint32_t lanes, uint32_t W, uint32_t C2) {
  uint32_t l = 0;
#if defined(__ARM_NEON)
  for (; l + 8 <= lanes; l += 8) {
    uint32_t w = 0;
    for (; w + 8 <= W; w += 8) {
      int8x8_t r[8];
      for (int i = 0; i < 8; ++i) r[i] = vld1_s8(in + (w + i) * C2 + l);
      transpose8x8(r);
      for (int i = 0; i < 8; ++i) vst1_s8(tile + (l + i) * W + w, r[i]);
    }
    for (; w < W; ++w)
      for (uint32_t i = 0; i < 8; ++i) tile[(l + i) * W + w] = in[w * C2 + l + i];
  }
#endif
  for (; l < lanes; ++l)
    for (uint32_t w = 0; w < W; ++w) tile[l * W + w] = in[w * C2 + l];
}

// NCHW float -> NC1HWC2 int8: channel c at block c / C2, lane c % C2; each
// block holds `plane` positions with rows w_stride apart. tile holds C2 * W
// bytes: one row of a block is quantised into it channel by channel, then
// interleaved into the block. Padding is left as is.
inline void quantize_nchw_to_nc1hwc2_s8(const float* src, int8_t* dst, uint32_t N, uint32_t C, uint32_t H,
                                        uint32_t W, uint32_t C1, uint32_t C2, size_t plane, size_t w_stride,
                                        float inv_scale, int32_t zp, int8_t* tile) {
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c1 = 0; c1 < C1 && c1 * C2 < C; ++c1) {
      const uint32_t lanes = std::min(C2, C - c1 * C2);
      for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t l = 0; l < lanes; ++l)
          quantize_row_s8(src + ((static_cast<size_t>(n) * C + c1 * C2 + l) * H + h) * W, 0, tile + l * W, 1, W,
                          inv_scale, zp);
        interleave_lanes(tile, dst + ((static_cast<size_t>(n) * C1 + c1) * plane + h * w_stride) * C2, lanes, W,
                         C2);
      }
    }
}

// One row of int8 -> float: (q - zp) * scale.
inline void dequantize_row_s8(const int8_t* src, float* dst, uint32_t W, float scale, int32_t zp) {
  uint32_t w = 0;
#if defined(__ARM_NEON)
  const float32x4_t s = vdupq_n_f32(scale);
  const int16x8_t z = vdupq_n_s16(static_cast<int16_t>(zp));
  for (; w + 8 <= W; w += 8) {
    const int16x8_t q = vsubq_s16(vmovl_s8(vld1_s8(src + w)), z);
    vst1q_f32(dst + w, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q))), s));
    vst1q_f32(dst + w + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q))), s));
  }
#endif
  for (; w < W; ++w) dst[w] = static_cast<float>(src[w] - zp) * scale;
}

// NC1HWC2 int8 -> NCHW float; the inverse of quantize_nchw_to_nc1hwc2_s8.
inline void dequantize_nc1hwc2_to_nchw_s8(const int8_t* src, float* dst, uint32_t N, uint32_t C, uint32_t H,
                                          uint32_t W, uint32_t C1, uint32_t C2, size_t plane, size_t w_stride,
                                          float scale, int32_t zp, int8_t* tile) {
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c1 = 0; c1 < C1 && c1 * C2 < C; ++c1) {
      const uint32_t lanes = std::min(C2, C - c1 * C2);
      for (uint32_t h = 0; h < H; ++h) {
        deinterleave_lanes(src + ((static_cast<size_t>(n) * C1 + c1) * plane + h * w_stride) * C2, tile, lanes, W,
                           C2);
        for (uint32_t l = 0; l < lanes; ++l)
          dequantize_row_s8(tile + l * W, dst + ((static_cast<size_t>(n) * C + c1 * C2 + l) * H + h) * W, W,
                            scale, zp);
      }
    }
}

}  // namespace executorch_rknn
