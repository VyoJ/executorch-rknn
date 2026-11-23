// ExecuTorch backend that runs a delegate blob's .rknn model on a Rockchip NPU
// through RKNN's zero-copy API (the only API on RV1103/RV1106).
//
// The delegate's inputs and outputs are float32 in ExecuTorch's layout (NCHW
// for 4-D). execute() quantises inputs into the NPU's own layout and type, and
// dequantises outputs back.

#include <cstdint>
#include <cstring>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

#include "RknnBlob.h"
#include "RknnLayout.h"
#include "RknnQuantize.h"
#include "rknn_api.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::ET_RUNTIME_NAMESPACE::Backend;
using executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext;
using executorch::ET_RUNTIME_NAMESPACE::BackendInitContext;
using executorch::ET_RUNTIME_NAMESPACE::BackendInterface;
using executorch::ET_RUNTIME_NAMESPACE::CompileSpec;
using executorch::ET_RUNTIME_NAMESPACE::DelegateHandle;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace executorch_rknn {
namespace {

struct IoTensor {
  BlobTensor logical;      // what ExecuTorch sees (NCHW for 4-D)
  rknn_tensor_attr attr;   // what the NPU memory holds
  rknn_tensor_mem* mem;
};

struct RknnHandle {
  rknn_context ctx;
  uint32_t n_inputs;
  uint32_t n_outputs;
  IoTensor* inputs;
  IoTensor* outputs;
};

// Calls fn(logical_index, npu_index) for every element of t.
template <typename Fn>
Error for_each_element(const IoTensor& t, Fn fn) {
  ET_CHECK_OR_RETURN_ERROR(executorch_rknn::for_each_element(t.attr, t.logical.dims, t.logical.ndim, fn),
                           NotSupported, "RknnBackend: tensor %s has unsupported layout %s %s", t.attr.name,
                           get_format_string(t.attr.fmt), get_type_string(t.attr.type));
  return Error::Ok;
}

Error write_input(const IoTensor& t, const float* src) {
  void* dst = t.mem->virt_addr;
  const float inv_scale = 1.0f / t.attr.scale;
  const int32_t zp = t.attr.zp;
  const BlobTensor& l = t.logical;
  if (t.attr.type == RKNN_TENSOR_INT8 && t.attr.fmt == RKNN_TENSOR_NHWC && l.ndim == 4) {
    quantize_nchw_to_nhwc_s8(src, static_cast<int8_t*>(dst), l.dims[0], l.dims[1], l.dims[2], l.dims[3],
                             t.attr.w_stride ? t.attr.w_stride : l.dims[3], inv_scale, zp);
    return Error::Ok;
  }
  switch (t.attr.type) {
    case RKNN_TENSOR_INT8:
      return for_each_element(t, [&](size_t i, size_t j) {
        static_cast<int8_t*>(dst)[j] = quantize<int8_t>(src[i], inv_scale, zp);
      });
    case RKNN_TENSOR_UINT8:
      return for_each_element(t, [&](size_t i, size_t j) {
        static_cast<uint8_t*>(dst)[j] = quantize<uint8_t>(src[i], inv_scale, zp);
      });
    case RKNN_TENSOR_FLOAT32:
      return for_each_element(t, [&](size_t i, size_t j) { static_cast<float*>(dst)[j] = src[i]; });
    default:
      ET_LOG(Error, "RknnBackend: input %s has unsupported type %s", t.attr.name, get_type_string(t.attr.type));
      return Error::NotSupported;
  }
}

Error read_output(const IoTensor& t, float* dst) {
  const void* src = t.mem->virt_addr;
  const float scale = t.attr.scale;
  const int32_t zp = t.attr.zp;
  switch (t.attr.type) {
    case RKNN_TENSOR_INT8:
      return for_each_element(t, [&](size_t i, size_t j) {
        dst[i] = (static_cast<const int8_t*>(src)[j] - zp) * scale;
      });
    case RKNN_TENSOR_UINT8:
      return for_each_element(t, [&](size_t i, size_t j) {
        dst[i] = (static_cast<const uint8_t*>(src)[j] - zp) * scale;
      });
    case RKNN_TENSOR_FLOAT32:
      return for_each_element(t, [&](size_t i, size_t j) { dst[i] = static_cast<const float*>(src)[j]; });
    default:
      ET_LOG(Error, "RknnBackend: output %s has unsupported type %s", t.attr.name, get_type_string(t.attr.type));
      return Error::NotSupported;
  }
}

void release(RknnHandle* h) {
  for (uint32_t i = 0; i < h->n_inputs; ++i)
    if (h->inputs[i].mem) rknn_destroy_mem(h->ctx, h->inputs[i].mem);
  for (uint32_t i = 0; i < h->n_outputs; ++i)
    if (h->outputs[i].mem) rknn_destroy_mem(h->ctx, h->outputs[i].mem);
  if (h->ctx) rknn_destroy(h->ctx);
  h->ctx = 0;
}

Error bind(RknnHandle* h, IoTensor& t, uint32_t index, bool is_input) {
  std::memset(&t.attr, 0, sizeof(t.attr));
  t.attr.index = index;
  // Inputs use the logical attributes, which RV1106 reports as NHWC for 4-D.
  // Outputs use the native ones; the runtime does not convert outputs.
  const rknn_query_cmd cmd = is_input ? RKNN_QUERY_INPUT_ATTR : RKNN_QUERY_NATIVE_OUTPUT_ATTR;
  ET_CHECK_OR_RETURN_ERROR(rknn_query(h->ctx, cmd, &t.attr, sizeof(t.attr)) == RKNN_SUCC, Internal,
                           "RknnBackend: querying %s %u failed", is_input ? "input" : "output", index);
  // Native attributes count padding (NC1HWC2 rounds channels up), so compare
  // the logical ones with the program.
  rknn_tensor_attr logical;
  std::memset(&logical, 0, sizeof(logical));
  logical.index = index;
  const rknn_query_cmd logical_cmd = is_input ? RKNN_QUERY_INPUT_ATTR : RKNN_QUERY_OUTPUT_ATTR;
  ET_CHECK_OR_RETURN_ERROR(rknn_query(h->ctx, logical_cmd, &logical, sizeof(logical)) == RKNN_SUCC, Internal,
                           "RknnBackend: querying %s %u failed", is_input ? "input" : "output", index);
  ET_CHECK_OR_RETURN_ERROR(logical.n_elems == t.logical.numel(), InvalidProgram,
                           "RknnBackend: %s %u has %u elements in the model, %zu in the program",
                           is_input ? "input" : "output", index, logical.n_elems, t.logical.numel());
  t.mem = rknn_create_mem(h->ctx, t.attr.size_with_stride);
  ET_CHECK_OR_RETURN_ERROR(t.mem != nullptr, MemoryAllocationFailed, "RknnBackend: rknn_create_mem failed");
  // Zero the input's row padding once; execute() flushes inputs after writing
  // them. Do not write output memory from the CPU: zeroing it here corrupted
  // the NPU's results, even with the sync after rknn_run (cached lines).
  if (is_input) std::memset(t.mem->virt_addr, 0, t.attr.size_with_stride);
  // execute() writes data already in the model's type and layout; without
  // pass_through the runtime converts it again.
  if (is_input) t.attr.pass_through = 1;
  ET_CHECK_OR_RETURN_ERROR(rknn_set_io_mem(h->ctx, t.mem, &t.attr) == RKNN_SUCC, Internal,
                           "RknnBackend: rknn_set_io_mem for %s %u failed", is_input ? "input" : "output", index);
  return Error::Ok;
}

class RknnBackend final : public BackendInterface {
 public:
  bool is_available() const override { return true; }

  Result<DelegateHandle*> init(BackendInitContext& context, FreeableBuffer* processed,
                               ArrayRef<CompileSpec> compile_specs) const override {
    (void)compile_specs;
    Blob blob;
    if (const char* err = parse_blob(processed->data(), processed->size(), &blob)) {
      ET_LOG(Error, "RknnBackend: invalid delegate blob: %s", err);
      return Error::InvalidProgram;
    }
    for (uint32_t i = 0; i < blob.n_inputs + blob.n_outputs; ++i)
      ET_CHECK_OR_RETURN_ERROR(blob.tensors[i].dtype == BlobDtype::Float32, NotSupported,
                               "RknnBackend: only float32 delegate inputs and outputs are supported");

    auto* allocator = context.get_runtime_allocator();
    auto* h = allocator->allocateInstance<RknnHandle>();
    ET_CHECK_OR_RETURN_ERROR(h != nullptr, MemoryAllocationFailed, "RknnBackend: handle");
    std::memset(h, 0, sizeof(*h));
    h->inputs = allocator->allocateList<IoTensor>(blob.n_inputs);
    h->outputs = allocator->allocateList<IoTensor>(blob.n_outputs);
    ET_CHECK_OR_RETURN_ERROR((h->inputs || !blob.n_inputs) && (h->outputs || !blob.n_outputs),
                             MemoryAllocationFailed, "RknnBackend: tensor list");
    std::memset(h->inputs, 0, sizeof(IoTensor) * blob.n_inputs);
    std::memset(h->outputs, 0, sizeof(IoTensor) * blob.n_outputs);

    // rknn_init copies the model, so the blob can be freed afterwards.
    int r = rknn_init(&h->ctx, const_cast<uint8_t*>(blob.rknn), static_cast<uint32_t>(blob.rknn_size), 0, nullptr);
    processed->Free();
    if (r != RKNN_SUCC) {
      ET_LOG(Error, "RknnBackend: rknn_init failed: %d", r);
      h->ctx = 0;
      return Error::InvalidProgram;
    }

    rknn_input_output_num io;
    Error err = Error::Ok;
    if (rknn_query(h->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC ||
        io.n_input != blob.n_inputs || io.n_output != blob.n_outputs) {
      ET_LOG(Error, "RknnBackend: model and program disagree on the number of inputs or outputs");
      err = Error::InvalidProgram;
    }
    for (uint32_t i = 0; err == Error::Ok && i < blob.n_inputs; ++i) {
      h->inputs[i].logical = blob.input(i);
      h->n_inputs = i + 1;
      err = bind(h, h->inputs[i], i, true);
    }
    for (uint32_t i = 0; err == Error::Ok && i < blob.n_outputs; ++i) {
      h->outputs[i].logical = blob.output(i);
      h->n_outputs = i + 1;
      err = bind(h, h->outputs[i], i, false);
    }
    if (err != Error::Ok) {
      release(h);
      return err;
    }
    return h;
  }

  // args holds the delegate's inputs followed by its outputs.
  Error execute(BackendExecutionContext& context, DelegateHandle* handle, Span<EValue*> args) const override {
    (void)context;
    auto* h = static_cast<RknnHandle*>(handle);
    ET_CHECK_OR_RETURN_ERROR(args.size() == h->n_inputs + h->n_outputs, InvalidArgument,
                             "RknnBackend: expected %u arguments, got %zu", h->n_inputs + h->n_outputs,
                             args.size());
    for (uint32_t i = 0; i < h->n_inputs + h->n_outputs; ++i) {
      const IoTensor& t = i < h->n_inputs ? h->inputs[i] : h->outputs[i - h->n_inputs];
      const Tensor& tensor = args[i]->toTensor();
      ET_CHECK_OR_RETURN_ERROR(tensor.scalar_type() == ScalarType::Float &&
                                   static_cast<size_t>(tensor.numel()) == t.logical.numel(),
                               InvalidArgument, "RknnBackend: argument %u does not match the model", i);
    }

    for (uint32_t i = 0; i < h->n_inputs; ++i) {
      ET_CHECK_OK_OR_RETURN_ERROR(write_input(h->inputs[i], args[i]->toTensor().const_data_ptr<float>()));
      rknn_mem_sync(h->ctx, h->inputs[i].mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    }
    int r = rknn_run(h->ctx, nullptr);
    ET_CHECK_OR_RETURN_ERROR(r == RKNN_SUCC, Internal, "RknnBackend: rknn_run failed: %d", r);
    for (uint32_t i = 0; i < h->n_outputs; ++i) {
      rknn_mem_sync(h->ctx, h->outputs[i].mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
      ET_CHECK_OK_OR_RETURN_ERROR(
          read_output(h->outputs[i], args[h->n_inputs + i]->toTensor().mutable_data_ptr<float>()));
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle) release(static_cast<RknnHandle*>(handle));
  }
};

RknnBackend backend_instance;
Backend backend{"RknnBackend", &backend_instance};
const auto registered = executorch::ET_RUNTIME_NAMESPACE::register_backend(backend);

}  // namespace
}  // namespace executorch_rknn
