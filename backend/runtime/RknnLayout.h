// Maps between ExecuTorch's layout (NCHW for 4-D, row-major otherwise) and the
// layout of an RKNN tensor in NPU memory.
//
// Layouts seen on RV1106:
//   inputs   NHWC, rows padded to w_stride (width 29 -> 32 with 3 channels)
//   outputs  NC1HWC2 for 4-D: channels in blocks of C2, padded up; H and W may
//            be reported flattened and padded, e.g. 15x14 as [1, 212]
//            (dims [N, C1, 1, 212, C2]); rows are w_stride apart
#pragma once

#include <cstddef>
#include <cstdint>

#include "rknn_api.h"

namespace executorch_rknn {

// Calls fn(logical_index, npu_index) for every element of a tensor with the
// given logical (ExecuTorch) dims. Returns false if the layout is not handled.
template <typename Fn>
bool for_each_element(const rknn_tensor_attr& a, const uint32_t* dims, uint32_t ndim, Fn fn) {
  if (ndim == 4 && (a.fmt == RKNN_TENSOR_NHWC || a.fmt == RKNN_TENSOR_NC1HWC2)) {
    const uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];
    const size_t ws = a.w_stride ? a.w_stride : W;
    size_t i = 0;
    if (a.fmt == RKNN_TENSOR_NHWC) {
      for (uint32_t n = 0; n < N; ++n)
        for (uint32_t c = 0; c < C; ++c)
          for (uint32_t h = 0; h < H; ++h)
            for (uint32_t w = 0; w < W; ++w, ++i) fn(i, ((n * H + h) * ws + w) * C + c);
      return true;
    }
    if (a.n_dims != 5) return false;
    const size_t C1 = a.dims[1], plane = static_cast<size_t>(a.dims[2]) * a.dims[3], C2 = a.dims[4];
    if (C1 * C2 < C || plane < (H - 1) * ws + W) return false;
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t c = 0; c < C; ++c)
        for (uint32_t h = 0; h < H; ++h)
          for (uint32_t w = 0; w < W; ++w, ++i)
            fn(i, ((n * C1 + c / C2) * plane + h * ws + w) * C2 + c % C2);
    return true;
  }
  if (ndim >= 4 && a.fmt != RKNN_TENSOR_UNDEFINED && a.fmt != RKNN_TENSOR_NCHW) return false;
  size_t numel = 1;
  for (uint32_t d = 0; d < ndim; ++d) numel *= dims[d];
  for (size_t i = 0; i < numel; ++i) fn(i, i);
  return true;
}

}  // namespace executorch_rknn
