// Run any .rknn model on RV1103/RV1106 with float32 input files.
//
// usage: rknn_bench model.rknn loops input-0.bin [input-1.bin ...]
//
// Inputs are float32 in ExecuTorch's layout (NCHW for 4-D, transposed here to
// the NHWC the model reports); they are quantised with the scale and zero point
// the model reports. Outputs are dequantised and written to
// output-<i>.bin as float32. Prints tensor attributes, NPU memory and timing:
// `run` is rknn_run alone; `float-in` adds quantising the float input each
// time, as an application with float input would (the delegate's work).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "RknnLayout.h"
#include "RknnQuantize.h"
#include "rknn_api.h"

namespace {

std::vector<char> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

uint16_t to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
  int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | ((mant + 0x1000) >> 13);
}

float from_fp16(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, x;
  if (exp == 0) {
    x = sign;
  } else if (exp == 31) {
    x = sign | 0x7f800000 | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// The tensor's dims as ExecuTorch orders them: NCHW for a 4-D NHWC attribute
// or a 5-D NC1HWC2 one (whose n_elems counts the real channels).
std::vector<uint32_t> executorch_dims(const rknn_tensor_attr& a) {
  if (a.n_dims == 4 && a.fmt == RKNN_TENSOR_NHWC) return {a.dims[0], a.dims[3], a.dims[1], a.dims[2]};
  if (a.n_dims == 5 && a.fmt == RKNN_TENSOR_NC1HWC2) {
    const uint32_t N = a.dims[0], H = a.dims[2], W = a.dims[3];
    return {N, a.n_elems / (N * H * W), H, W};
  }
  return std::vector<uint32_t>(a.dims, a.dims + a.n_dims);
}

template <typename T>
T quantize(float v, const rknn_tensor_attr& a) {
  float q = std::nearbyint(v / a.scale) + a.zp;
  q = std::min<float>(std::max<float>(q, std::numeric_limits<T>::min()), std::numeric_limits<T>::max());
  return static_cast<T>(q);
}

void write_input(const rknn_tensor_attr& a, float v, void* dst, size_t j) {
  switch (a.type) {
    case RKNN_TENSOR_INT8: static_cast<int8_t*>(dst)[j] = quantize<int8_t>(v, a); break;
    case RKNN_TENSOR_UINT8: static_cast<uint8_t*>(dst)[j] = quantize<uint8_t>(v, a); break;
    case RKNN_TENSOR_INT16: static_cast<int16_t*>(dst)[j] = quantize<int16_t>(v, a); break;
    case RKNN_TENSOR_FLOAT16: static_cast<uint16_t*>(dst)[j] = to_fp16(v); break;
    case RKNN_TENSOR_FLOAT32: static_cast<float*>(dst)[j] = v; break;
    default: break;
  }
}

float read_output(const rknn_tensor_attr& a, const void* src, size_t i) {
  switch (a.type) {
    case RKNN_TENSOR_INT8: return (static_cast<const int8_t*>(src)[i] - a.zp) * a.scale;
    case RKNN_TENSOR_UINT8: return (static_cast<const uint8_t*>(src)[i] - a.zp) * a.scale;
    case RKNN_TENSOR_INT16: return (static_cast<const int16_t*>(src)[i] - a.zp) * a.scale;
    case RKNN_TENSOR_FLOAT16: return from_fp16(static_cast<const uint16_t*>(src)[i]);
    case RKNN_TENSOR_FLOAT32: return static_cast<const float*>(src)[i];
    default: return NAN;
  }
}

bool write_inputs(const rknn_tensor_attr& a, const std::vector<float>& values, void* dst) {
  const auto dims = executorch_dims(a);
  if (a.type == RKNN_TENSOR_INT8 && a.fmt == RKNN_TENSOR_NHWC && dims.size() == 4) {
    executorch_rknn::quantize_nchw_to_nhwc_s8(values.data(), static_cast<int8_t*>(dst), dims[0], dims[1], dims[2],
                                              dims[3], a.w_stride ? a.w_stride : dims[3], 1.0f / a.scale, a.zp);
    return true;
  }
  return executorch_rknn::for_each_element(a, dims.data(), dims.size(),
                                           [&](size_t k, size_t j) { write_input(a, values[k], dst, j); });
}

double mean(const std::vector<double>& v) {
  double sum = 0;
  for (double t : v) sum += t;
  return v.empty() ? 0 : sum / v.size();
}

void print_attr(const char* kind, const rknn_tensor_attr& a) {
  std::string dims;
  for (uint32_t d = 0; d < a.n_dims; ++d) dims += (d ? "x" : "") + std::to_string(a.dims[d]);
  std::printf("%s %u %s [%s] %s %s %s zp=%d scale=%g fl=%d size=%u stride_size=%u w_stride=%u\n", kind,
              a.index, a.name, dims.c_str(), get_format_string(a.fmt), get_type_string(a.type),
              get_qnt_type_string(a.qnt_type), a.zp, a.scale, a.fl, a.size, a.size_with_stride, a.w_stride);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s model.rknn loops input-0.bin [...]\n", argv[0]);
    return 2;
  }
  const int loops = std::atoi(argv[2]);
  auto model = read_file(argv[1]);
  rknn_context ctx;
  if (int r = rknn_init(&ctx, model.data(), model.size(), 0, nullptr); r != RKNN_SUCC) {
    std::fprintf(stderr, "rknn_init failed: %d\n", r);
    return 1;
  }

  rknn_input_output_num io;
  rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  if (static_cast<uint32_t>(argc - 3) != io.n_input) {
    std::fprintf(stderr, "model has %u inputs, got %d files\n", io.n_input, argc - 3);
    return 2;
  }
  rknn_mem_size mem{};
  rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem, sizeof(mem));
  std::printf("mem weight=%u internal=%u dma_total=%llu (bytes)\n", mem.total_weight_size,
              mem.total_internal_size, static_cast<unsigned long long>(mem.total_dma_allocated_size));

  std::vector<rknn_tensor_attr> in(io.n_input), out(io.n_output), native_out(io.n_output);
  std::vector<rknn_tensor_mem*> in_mem(io.n_input), out_mem(io.n_output);
  std::vector<std::vector<float>> in_values(io.n_input);
  for (uint32_t i = 0; i < io.n_input; ++i) {
    in[i].index = i;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in[i], sizeof(in[i]));
    print_attr("input", in[i]);
    rknn_tensor_attr native{};
    native.index = i;
    rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &native, sizeof(native));
    print_attr("native-input", native);
    auto data = read_file(argv[3 + i]);
    if (data.size() != in[i].n_elems * sizeof(float)) {
      std::fprintf(stderr, "input %u: expected %u floats, file has %zu bytes\n", i, in[i].n_elems, data.size());
      return 2;
    }
    std::vector<float>& values = in_values[i];
    values.resize(in[i].n_elems);
    std::memcpy(values.data(), data.data(), data.size());
    in_mem[i] = rknn_create_mem(ctx, in[i].size_with_stride);
    std::memset(in_mem[i]->virt_addr, 0, in[i].size_with_stride);
    // The data is already in the model's input type; without pass_through the
    // runtime converts it again.
    in[i].pass_through = 1;
    if (!write_inputs(in[i], values, in_mem[i]->virt_addr)) {
      std::fprintf(stderr, "input %u: layout not handled\n", i);
      return 2;
    }
    if (int r = rknn_set_io_mem(ctx, in_mem[i], &in[i]); r != RKNN_SUCC) {
      std::fprintf(stderr, "set input %u failed: %d\n", i, r);
      return 1;
    }
  }
  for (uint32_t i = 0; i < io.n_output; ++i) {
    out[i].index = i;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out[i], sizeof(out[i]));
    print_attr("output", out[i]);
    // Outputs are bound in their native layout; the runtime does not convert 4-D ones.
    native_out[i].index = i;
    rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native_out[i], sizeof(native_out[i]));
    print_attr("native-output", native_out[i]);
    out_mem[i] = rknn_create_mem(ctx, native_out[i].size_with_stride);
    if (int r = rknn_set_io_mem(ctx, out_mem[i], &native_out[i]); r != RKNN_SUCC) {
      std::fprintf(stderr, "set output %u failed: %d\n", i, r);
      return 1;
    }
  }

  for (uint32_t i = 0; i < io.n_input; ++i) rknn_mem_sync(ctx, in_mem[i], RKNN_MEMORY_SYNC_TO_DEVICE);
  std::vector<double> run_ms, float_in_ms;
  for (int l = 0; l <= loops; ++l) {
    auto t0 = std::chrono::steady_clock::now();
    if (int r = rknn_run(ctx, nullptr); r != RKNN_SUCC) {
      std::fprintf(stderr, "rknn_run failed: %d\n", r);
      return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < io.n_input; ++i) {
      write_inputs(in[i], in_values[i], in_mem[i]->virt_addr);
      rknn_mem_sync(ctx, in_mem[i], RKNN_MEMORY_SYNC_TO_DEVICE);
    }
    rknn_run(ctx, nullptr);
    auto t2 = std::chrono::steady_clock::now();
    if (l > 0) {
      run_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      float_in_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
  }
  if (!run_ms.empty()) {
    std::printf("run mean=%.3f ms min=%.3f ms over %zu (after 1 warm-up)\n", mean(run_ms),
                *std::min_element(run_ms.begin(), run_ms.end()), run_ms.size());
    std::printf("float-in mean=%.3f ms min=%.3f ms over %zu (after 1 warm-up)\n", mean(float_in_ms),
                *std::min_element(float_in_ms.begin(), float_in_ms.end()), float_in_ms.size());
  }

  for (uint32_t i = 0; i < io.n_output; ++i) {
    std::vector<float> values(out[i].n_elems);
    const auto dims = executorch_dims(out[i]);
    executorch_rknn::for_each_element(native_out[i], dims.data(), dims.size(), [&](size_t k, size_t j) {
      values[k] = read_output(native_out[i], out_mem[i]->virt_addr, j);
    });
    std::ofstream f("output-" + std::to_string(i) + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  }
  for (auto* m : in_mem) rknn_destroy_mem(ctx, m);
  for (auto* m : out_mem) rknn_destroy_mem(ctx, m);
  rknn_destroy(ctx);
  return 0;
}
