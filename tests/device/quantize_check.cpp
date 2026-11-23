// Checks the layout-changing (de)quantisers in RknnQuantize.h against scalar
// references: random values, exact .5 ties, and values far outside the int8
// range. Runs on the host (scalar path) and on the board (NEON path); exits
// non-zero on any mismatch.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "RknnQuantize.h"

namespace {

using executorch_rknn::dequantize_nc1hwc2_to_nchw_s8;
using executorch_rknn::quantize_nchw_to_nc1hwc2_s8;
using executorch_rknn::quantize_nchw_to_nhwc_s8;

const float kScale = 0.0308209f, kInvScale = 1.0f / kScale;
const int32_t kZp = 7;

std::vector<float> test_values(size_t n) {
  std::mt19937 gen(0);
  std::normal_distribution<float> normal(0.0f, 1.5f);
  std::vector<float> v(n);
  for (auto& x : v) x = normal(gen);
  for (size_t i = 0; i < 1500; ++i) v[i] = (static_cast<int>(i) - 750) * 0.5f * kScale;
  for (size_t i = 1500; i < 1600; ++i) v[i] = (i % 2 ? 1.0f : -1.0f) * (i % 3 ? 1e9f : (i % 5 ? 3e5f : 1e38f));
  return v;
}

int8_t reference(float v) {
  const float q = std::nearbyint(v * kInvScale) + kZp;
  return static_cast<int8_t>(std::fmin(std::fmax(q, -128.0f), 127.0f));
}

template <typename T>
int report(const char* what, const std::vector<T>& got, const std::vector<T>& want) {
  int bad = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != want[i] && bad++ < 5)
      std::printf("%s index %zu: got %g, want %g\n", what, i, double(got[i]), double(want[i]));
  std::printf("%s: %d mismatches of %zu\n", what, bad, got.size());
  return bad;
}

// Odd width with row padding, so both the vector loop and the tail run.
int check_nhwc() {
  const uint32_t C = 3, H = 31, W = 29;
  const size_t w_stride = 32;
  const auto src = test_values(C * H * W);
  std::vector<int8_t> got(H * w_stride * C, 0), want(got.size(), 0);
  quantize_nchw_to_nhwc_s8(src.data(), got.data(), 1, C, H, W, w_stride, kInvScale, kZp);
  for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w) want[(h * w_stride + w) * C + c] = reference(src[(c * H + h) * W + w]);
  return report("NHWC quantise", got, want);
}

// 20 channels: one full block of 16 and one with 4 lanes; odd width, padded rows.
int check_nc1hwc2() {
  const uint32_t N = 1, C = 20, H = 7, W = 19, C2 = 16, C1 = 2;
  const size_t w_stride = 24, plane = H * w_stride;
  const auto src = test_values(N * C * H * W);
  auto at = [&](uint32_t c, uint32_t h, uint32_t w) { return ((c / C2) * plane + h * w_stride + w) * C2 + c % C2; };

  std::vector<int8_t> tile(C2 * W), got(C1 * plane * C2, 0), want(got.size(), 0);
  quantize_nchw_to_nc1hwc2_s8(src.data(), got.data(), N, C, H, W, C1, C2, plane, w_stride, kInvScale, kZp,
                              tile.data());
  for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w) want[at(c, h, w)] = reference(src[(c * H + h) * W + w]);
  int bad = report("NC1HWC2 quantise", got, want);

  std::vector<float> back(src.size()), back_want(src.size());
  dequantize_nc1hwc2_to_nchw_s8(want.data(), back.data(), N, C, H, W, C1, C2, plane, w_stride, kScale, kZp,
                                tile.data());
  for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        back_want[(c * H + h) * W + w] = static_cast<float>(want[at(c, h, w)] - kZp) * kScale;
  return bad + report("NC1HWC2 dequantise", back, back_want);
}

}  // namespace

int main() { return check_nhwc() + check_nc1hwc2() ? 1 : 0; }
