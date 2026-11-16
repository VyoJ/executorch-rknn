# Pinned versions

| Item | Version | Where it is set |
|---|---|---|
| Base image (aot, cross) | `debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc` (linux/amd64) | `docker/cross.Dockerfile` |
| Luckfox Pico SDK | `824b817f889c2cbff1d48fcdb18ab494a68f69d1` (main, 2026-11-17) | `docker/cross.Dockerfile` `LUCKFOX_SDK_COMMIT` |
| Cross toolchain | `arm-rockchip830-linux-uclibcgnueabihf`, GCC 8.3.0 | from the SDK above |
| CMake (cross) | 3.25.1 (Debian bookworm) | apt |
| Board RKNN runtime | `librknnmrt.so` 1.6.0 (2de554906, 2024-01-17), in `/oem/usr/lib` | on the board's firmware |
| rknn-toolkit2 | 1.6.0+81f21f4d, cp311 wheel from `airockchip/rknn-toolkit2` at `b25dadacc24b88eb7dfcaa47c9c525ecca89b319` (tag v1.6.0), sha256 `f1542483…02d3d6` | `docker/aot.Dockerfile` (`/opt/rknn` venv) |
| rknn-toolkit2 deps | onnx 1.14.1, onnxoptimizer 0.3.8, onnxruntime 1.16.0, protobuf 3.20.3, numpy<2 | `docker/aot.Dockerfile` |
| RKNN runtime, header, demo (cross) | same commit as the toolkit | `docker/cross.Dockerfile` |
| onnx / onnxscript (ExecuTorch venv) | 1.23.2 / 0.7.2 | `docker/aot.Dockerfile` |
| MobileNetV2 weights | `mobilenet_v2-b0353104.pth`, sha256 `b0353104…ee9889` | `docker/aot.Dockerfile` |
| Imagenette | `imagenette2-160.tgz`, sha256 `64d0c485…db02fe5` | `docker/aot.Dockerfile` |
| ExecuTorch | v1.6.0 (`4776f71b6f0daa5dfc7d758d5836fc1cc0d40154`) | `third_party/executorch` submodule |
| torch | 2.15.0+cpu, the version ExecuTorch v1.6.0 pins in `torch_pin.py`. In `cross` it is used only by the kernel codegen. | `TORCH_VERSION` in both Dockerfiles |
| executorch (aot) | 1.6.0 wheel from PyPI, cp311, same commit as the submodule | `docker/aot.Dockerfile` |
| torchvision / torchao (aot) | 0.29.2+cpu / 0.18.0+cpu | `docker/aot.Dockerfile` |
| Python | 3.11 (Debian bookworm) | base image |
