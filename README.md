# executorch-rknn

An [ExecuTorch](https://github.com/pytorch/executorch) backend for the Rockchip
RKNN NPU in the RV1103 and RV1106 (Luckfox Pico boards). Parts of a model the NPU
can run go to RKNN; everything else stays on the CPU, through ExecuTorch's
portable kernels or XNNPACK.

That fallback is the point. On these chips RKNN has no CPU fallback of its own,
so a single unsupported operator stops a whole model. MobileViT-XS, for example,
asks plain RKNN for 80 MB of NPU memory and never loads on a Pico Pro Max; through
this backend its convolutions run on the NPU, its LayerNorms and attention on the
CPU, and it classifies at 519 ms per image.

| Model (Pico Pro Max, RV1106) | Per image | Top-1 on Imagenette, board |
|---|---|---|
| MobileNetV2, whole graph on the NPU | 10.6 ms | 0.600 (float 0.640) |
| MobileViT-XS, NPU with the CPU on XNNPACK | 519 ms | 0.744 (float 0.712) |

Same `.pte` files and binaries run on the Pico Mini B (RV1103, 64 MB RAM).
More numbers in [docs/results.md](docs/results.md).

## Requirements

- Docker, git and adb on the host. Everything else builds in two containers
  (`linux/amd64`; works under emulation on Apple Silicon, slowly).
- A Luckfox Pico with the stock firmware: `librknnmrt.so` 1.6.0 in
  `/oem/usr/lib`. The toolkit version in the image must match it.

Pinned versions are in [docker/VERSIONS.md](docker/VERSIONS.md).

## Quick start

```
scripts/bootstrap.sh                                  # submodules
scripts/dk aot python -m pytest -q tests/aot          # host tests
```

`scripts/dk aot|cross <command>` runs a command in the export or cross-compile
container, building the image on first use.

MobileNetV2, quantised with PT2E, on the board:

```
scripts/dk aot python examples/export_mobilenet_v2.py --check-per-class 10
scripts/dk cross scripts/build_rknn_runner.sh
tests/device/run_rknn_pte.sh out/mv2_rknn.pte out/mv2_rknn
scripts/dk aot python tests/device/compare_rknn_outputs.py out/mv2_rknn
```

MobileViT-XS with the CPU part on XNNPACK (quantisation-aware training takes
about 20 minutes in the emulated container):

```
scripts/dk aot python examples/mobilevit/export.py --qat-steps 300 --xnnpack \
    --pte out/mobilevit.pte --work out/mobilevit
scripts/dk cross env XNNPACK=1 scripts/build_rknn_runner.sh
RUNNER=out/rknn-runner-xnnpack/executor_runner.stripped \
    tests/device/run_rknn_pte.sh out/mobilevit.pte out/mobilevit
scripts/dk aot python tests/device/compare_rknn_outputs.py out/mobilevit
```

The `tests/device/` scripts push files over adb, run them, and pull the results
back. They stop the camera app (`rkipc`) first. On the Mini B, pass
`REMOTE=/root/rknn-pte` so the files are staged on flash rather than in RAM.

## Using it in your own export

```python
from executorch.exir import to_edge_transform_and_lower
from executorch_rknn import RknnOptions, RknnPartitioner, transform_passes

program = torch.export.export(quantized_model, example_inputs)  # PT2E int8
edge = to_edge_transform_and_lower(
    program,
    transform_passes=transform_passes(),
    partitioner=[RknnPartitioner(RknnOptions(target_platform="rv1106"))],
)
open("model.pte", "wb").write(edge.to_executorch().buffer)
```

Add `XnnpackPartitioner()` after `RknnPartitioner` to run the rest on XNNPACK.
The NPU is int8-only, so quantise with PT2E first (the examples use
`XNNPACKQuantizer` with per-channel weights). How the pieces fit together is in
[docs/design.md](docs/design.md); what the hardware can and cannot do is in
[docs/rv1106.md](docs/rv1106.md).

## Layout

```
backend/python/executorch_rknn/   partitioner, passes, Edge-to-ONNX converters, preprocess
backend/runtime/                  C++ delegate: RknnBackend, blob reader, layout and quantisation
examples/                         MobileNetV2 and MobileViT-XS exports, executor_runner build
tests/aot/                        pytest, in the aot container
tests/device/                     board scripts (adb) and output comparison
patches/                          fixes to ExecuTorch, XNNPACK and cpuinfo for GCC 8 and uClibc
docker/                           the two images and the Luckfox toolchain file
```

ExecuTorch is a submodule pinned at v1.6.0; `scripts/apply_patches.sh` applies
`patches/` before every build.

## Model licences

The images download MobileNetV2 (torchvision), Imagenette, and Apple's
MobileViT-XS, whose weights are for research use only. None of them are
redistributed here.
