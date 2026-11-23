// Run any .rknn model on RV1103/RV1106 with float32 input files.
//
// usage: rknn_bench model.rknn loops input-0.bin [input-1.bin ...]
//
// Inputs are float32 in the model's logical layout; they are quantised with the
// scale and zero point the model reports. Outputs are dequantised and written to
// output-<i>.bin as float32. Prints tensor attributes, NPU memory and timing.

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

template <typename T>
T quantize(float v, const rknn_tensor_attr& a) {
  float q = std::nearbyint(v / a.scale) + a.zp;
  q = std::min<float>(std::max<float>(q, std::numeric_limits<T>::min()), std::numeric_limits<T>::max());
  return static_cast<T>(q);
}

bool write_input(const rknn_tensor_attr& a, const std::vector<float>& src, void* dst) {
  for (uint32_t i = 0; i < a.n_elems; ++i) {
    switch (a.type) {
      case RKNN_TENSOR_INT8: static_cast<int8_t*>(dst)[i] = quantize<int8_t>(src[i], a); break;
      case RKNN_TENSOR_UINT8: static_cast<uint8_t*>(dst)[i] = quantize<uint8_t>(src[i], a); break;
      case RKNN_TENSOR_INT16: static_cast<int16_t*>(dst)[i] = quantize<int16_t>(src[i], a); break;
      case RKNN_TENSOR_FLOAT16: static_cast<uint16_t*>(dst)[i] = to_fp16(src[i]); break;
      case RKNN_TENSOR_FLOAT32: static_cast<float*>(dst)[i] = src[i]; break;
      default: return false;
    }
  }
  return true;
}

float read_output(const rknn_tensor_attr& a, const void* src, uint32_t i) {
  switch (a.type) {
    case RKNN_TENSOR_INT8: return (static_cast<const int8_t*>(src)[i] - a.zp) * a.scale;
    case RKNN_TENSOR_UINT8: return (static_cast<const uint8_t*>(src)[i] - a.zp) * a.scale;
    case RKNN_TENSOR_INT16: return (static_cast<const int16_t*>(src)[i] - a.zp) * a.scale;
    case RKNN_TENSOR_FLOAT16: return from_fp16(static_cast<const uint16_t*>(src)[i]);
    case RKNN_TENSOR_FLOAT32: return static_cast<const float*>(src)[i];
    default: return NAN;
  }
}

void print_attr(const char* kind, const rknn_tensor_attr& a) {
  std::string dims;
  for (uint32_t d = 0; d < a.n_dims; ++d) dims += (d ? "x" : "") + std::to_string(a.dims[d]);
  std::printf("%s %u %s [%s] %s %s %s zp=%d scale=%g fl=%d size=%u stride_size=%u\n", kind, a.index,
              a.name, dims.c_str(), get_format_string(a.fmt), get_type_string(a.type),
              get_qnt_type_string(a.qnt_type), a.zp, a.scale, a.fl, a.size, a.size_with_stride);
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

  std::vector<rknn_tensor_attr> in(io.n_input), out(io.n_output);
  std::vector<rknn_tensor_mem*> in_mem(io.n_input), out_mem(io.n_output);
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
    if (in[i].size_with_stride != in[i].size) {
      std::fprintf(stderr, "input %u: strided layout not handled\n", i);
      return 2;
    }
    std::vector<float> values(in[i].n_elems);
    std::memcpy(values.data(), data.data(), data.size());
    in_mem[i] = rknn_create_mem(ctx, in[i].size_with_stride);
    // The data is already in the model's input type; without pass_through the
    // runtime converts it again.
    in[i].pass_through = 1;
    if (!write_input(in[i], values, in_mem[i]->virt_addr)) {
      std::fprintf(stderr, "input %u: unsupported type\n", i);
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
    rknn_tensor_attr native{};
    native.index = i;
    rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native, sizeof(native));
    print_attr("native-output", native);
    out_mem[i] = rknn_create_mem(ctx, out[i].size_with_stride);
    if (int r = rknn_set_io_mem(ctx, out_mem[i], &out[i]); r != RKNN_SUCC) {
      std::fprintf(stderr, "set output %u failed: %d\n", i, r);
      return 1;
    }
  }

  std::vector<double> times;
  for (int l = 0; l <= loops; ++l) {
    auto t0 = std::chrono::steady_clock::now();
    if (int r = rknn_run(ctx, nullptr); r != RKNN_SUCC) {
      std::fprintf(stderr, "rknn_run failed: %d\n", r);
      return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (l > 0) times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  if (!times.empty()) {
    double sum = 0;
    for (double t : times) sum += t;
    std::printf("run mean=%.3f ms min=%.3f ms over %zu (after 1 warm-up)\n", sum / times.size(),
                *std::min_element(times.begin(), times.end()), times.size());
  }

  for (uint32_t i = 0; i < io.n_output; ++i) {
    std::vector<float> values(out[i].n_elems);
    for (uint32_t j = 0; j < out[i].n_elems; ++j) values[j] = read_output(out[i], out_mem[i]->virt_addr, j);
    std::ofstream f("output-" + std::to_string(i) + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  }
  for (auto* m : in_mem) rknn_destroy_mem(ctx, m);
  for (auto* m : out_mem) rknn_destroy_mem(ctx, m);
  rknn_destroy(ctx);
  return 0;
}
