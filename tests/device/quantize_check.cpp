// Checks quantize_nchw_to_nhwc_s8 (RknnQuantize.h) against a scalar reference:
// random values, exact .5 ties, and values far outside the int8 range.
// Runs on the host (scalar path) and on the board (NEON path); exits non-zero
// on any mismatch.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "RknnQuantize.h"

int main() {
  // Odd width with row padding, so both the vector loop and the tail run.
  const uint32_t C = 3, H = 31, W = 29;
  const size_t w_stride = 32;
  const float scale = 0.0308209f, inv_scale = 1.0f / scale;
  const int32_t zp = 7;

  std::mt19937 gen(0);
  std::normal_distribution<float> normal(0.0f, 1.5f);
  std::vector<float> src(C * H * W);
  for (auto& v : src) v = normal(gen);
  for (size_t i = 0; i < 1500; ++i) src[i] = (static_cast<int>(i) - 750) * 0.5f * scale;
  for (size_t i = 1500; i < 1600; ++i) src[i] = (i % 2 ? 1.0f : -1.0f) * (i % 3 ? 1e9f : (i % 5 ? 3e5f : 1e38f));

  std::vector<int8_t> got(H * w_stride * C, 0), want(got.size(), 0);
  executorch_rknn::quantize_nchw_to_nhwc_s8(src.data(), got.data(), 1, C, H, W, w_stride, inv_scale, zp);
  for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w) {
        const float q = std::nearbyint(src[(c * H + h) * W + w] * inv_scale) + zp;
        want[(h * w_stride + w) * C + c] = static_cast<int8_t>(std::fmin(std::fmax(q, -128.0f), 127.0f));
      }

  int bad = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != want[i] && bad++ < 5) std::printf("index %zu: got %d, want %d\n", i, got[i], want[i]);
  std::printf("%d mismatches of %zu\n", bad, got.size());
  return bad ? 1 : 0;
}
