// Reads the delegate blob written by executorch_rknn/blob.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace executorch_rknn {

constexpr uint32_t kMaxDims = 8;
constexpr uint32_t kMaxTensors = 16;

enum class BlobDtype : uint8_t { Float32 = 0, Int8 = 1, UInt8 = 2 };

struct BlobTensor {
  BlobDtype dtype;
  uint32_t source;  // outputs: the .rknn output this one reads
  uint32_t ndim;
  uint32_t dims[kMaxDims];

  size_t numel() const {
    size_t n = 1;
    for (uint32_t i = 0; i < ndim; ++i) n *= dims[i];
    return n;
  }
};

struct Blob {
  uint32_t n_inputs;
  uint32_t n_outputs;
  BlobTensor tensors[kMaxTensors];  // inputs, then outputs
  const uint8_t* rknn;
  uint64_t rknn_size;

  const BlobTensor& input(uint32_t i) const { return tensors[i]; }
  const BlobTensor& output(uint32_t i) const { return tensors[n_inputs + i]; }
};

// Returns nullptr on success, or what is wrong with the data.
inline const char* parse_blob(const void* data, size_t size, Blob* out) {
  const auto* p = static_cast<const uint8_t*>(data);
  auto read = [&](size_t& offset, void* dst, size_t n) {
    if (offset + n > size) return false;
    std::memcpy(dst, p + offset, n);
    offset += n;
    return true;
  };
  size_t offset = 0;
  char magic[4];
  uint16_t version, n_in, n_out, reserved;
  if (!read(offset, magic, 4) || std::memcmp(magic, "RKNB", 4) != 0) return "bad magic";
  if (!read(offset, &version, 2) || version != 2) return "unsupported version";
  if (!read(offset, &n_in, 2) || !read(offset, &n_out, 2) || !read(offset, &reserved, 2)) return "truncated header";
  if (n_in + n_out > kMaxTensors) return "too many tensors";
  out->n_inputs = n_in;
  out->n_outputs = n_out;
  for (uint32_t t = 0; t < static_cast<uint32_t>(n_in + n_out); ++t) {
    uint8_t dtype, ndim;
    uint16_t source;
    if (!read(offset, &dtype, 1) || !read(offset, &ndim, 1) || !read(offset, &source, 2)) return "truncated tensor";
    if (dtype > 2) return "unknown dtype";
    if (ndim > kMaxDims) return "too many dims";
    out->tensors[t].dtype = static_cast<BlobDtype>(dtype);
    out->tensors[t].source = source;
    out->tensors[t].ndim = ndim;
    if (!read(offset, out->tensors[t].dims, 4u * ndim)) return "truncated dims";
  }
  if (!read(offset, &out->rknn_size, 8)) return "truncated size";
  if (offset + out->rknn_size != size) return "size mismatch";
  out->rknn = p + offset;
  return nullptr;
}

}  // namespace executorch_rknn
