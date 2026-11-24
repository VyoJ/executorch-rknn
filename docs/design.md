# How it works

## Ahead of time (Python, `aot` container)

1. Quantise the model with PT2E. RKNN reads the resulting QuantizeLinear and
   DequantizeLinear nodes as a QAT model and keeps their scales and zero points,
   so the host-side PT2E model is the reference for what the board computes.

2. `to_edge_transform_and_lower` with `transform_passes()` and
   `RknnPartitioner`. The passes run first, because a partitioner only sees fake
   weights and may not change the graph:
   - `DuplicateQuantizePass` gives each dequantize its own quantize, so a value
     feeding two delegates is quantised inside each of them.
   - `RemoveNoopExpandPass` drops the `expand_copy` nodes `matmul` leaves when
     nothing broadcasts.
   - `ScaleBeforeBmmPass` turns `bmm(q, k) / c` into `bmm(q / c, k)`, which is
     much cheaper for attention scores on the CPU.

3. `RknnPartitioner` groups connected operators that have a converter and pass
   its checks, then adjusts each group so only float tensors cross its
   boundary: every int8 tensor runs from a quantize to a dequantize inside one
   group. Groups whose operators would end up in float16 on the NPU (an input
   that isn't quantised straight away, an output not produced by a dequantize)
   are trimmed until they don't. `min_partition_ops` keeps tiny groups on the
   CPU. Each group becomes its own `.rknn` and RKNN context.

4. `RknnBackend.preprocess` converts the group's Edge graph to ONNX with
   hand-written converters, one per operator family
   (`backend/python/executorch_rknn/onnx_converter/`), and runs rknn-toolkit2
   on it in a subprocess. The toolkit pins old onnx and protobuf, so it lives
   in its own venv (`/opt/rknn`). The result is packed into a small blob: a
   header with the delegate's input and output shapes, then the `.rknn`.

Supported operators: convolution, batch norm, relu, hardtanh, add, mul,
sigmoid, max pool, mean over H and W, addmm, view, permute, cat, contiguous
clone, and PT2E's int8 quantize/dequantize (per-tensor and per-channel).
Each converter has a check for the cases the board gets wrong.

A float graph can still be delegated: pass `calibration_inputs` and
`calibration_program` to `RknnPartitioner`, and it records the activations
reaching each group and lets RKNN calibrate on them.

`RknnOptions`: `target_platform` (`rv1103` or `rv1106`), `optimization_level`
(default 3), `mean_values`/`std_values` to fold input normalisation into the
model, `calibration_dataset` and `quantized_algorithm` for RKNN's own
calibration, and `dump_dir` to keep each delegate's `model.onnx` and
`model.rknn`.

## On the board (C++, `cross` container)

`RknnBackend` registers with ExecuTorch under the same name as the Python
side and links into `executor_runner` (`examples/runner/`).

- `init` parses the blob, calls `rknn_init`, queries every input and output,
  and binds zero-copy memory with `rknn_set_io_mem`. Inputs are bound with
  `pass_through = 1` because the backend writes them already quantised;
  outputs are bound in their native layout.
- `execute` quantises each float NCHW input into the NPU's layout (NHWC with
  padded rows, or NC1HWC2 in blocks of 16 channels) with NEON, runs
  `rknn_run`, and dequantises the outputs back to NCHW float.

Delegate inputs and outputs are always float32. For a whole-graph model this
costs little (1.6 ms of MobileNetV2's 10.6 ms); for a model split into many
partitions the conversions at each boundary add up.

Build the runner with `scripts/build_rknn_runner.sh` in the `cross` container.
`XNNPACK=1` adds the XNNPACK backend; `PROFILE=1` adds ExecuTorch's event
tracer, so `executor_runner --etdump_path=...` records the time of every
delegate call, XNNPACK operator and portable kernel.

## Debugging

- `RKNN_PROFILE=1`: the runtime logs the total time spent converting inputs,
  in `rknn_run`, and converting outputs, over all delegates.
- `RKNN_DUMP=<dir>`: every delegate call writes its float inputs and outputs
  to `<dir>`. Running each delegate's `model.onnx` (kept with `dump_dir`) on
  those inputs with onnxruntime shows which partition drifts from the host.
- `tests/device/rknn_bench.cpp`: runs any `.rknn` on the board with float
  inputs, using the same layout code as the delegate.
