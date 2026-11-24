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
| transformers (aot) | 5.0.0rc1, the version optimum-executorch 1.1.0 pins | `docker/aot.Dockerfile` |
| optimum-executorch / optimum / accelerate (aot) | 1.1.0 / 2.0.0 / 1.16.0, installed `--no-deps` | `docker/aot.Dockerfile` |
| soundfile (aot) | 0.14.0 | `docker/aot.Dockerfile` |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` at `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, safetensors sha256 `80521b40…d55da1` | `docker/aot.Dockerfile` |
| whisper-tiny | `openai/whisper-tiny` at `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, safetensors sha256 `7ebd0e69…2ea2395` | `docker/aot.Dockerfile` |
| jfk.flac | `openai/whisper` at `86098128c0b4f24f0e2aa2994de830614b474227`, sha256 `63a4b1e4…949715` | `docker/aot.Dockerfile` |
| MobileViT-XS | `apple/mobilevit-x-small` at `0f4f298e136829c566224228d82066548a9082de`, `pytorch_model.bin` sha256 `09b264fe…e63e8`; Apple research-only model licence | `docker/aot.Dockerfile` |
| torch (RKNN venv) | 2.1.0+cpu, the newest the toolkit's requirements allow; used only by the toolkit's own graph rewrites | `docker/aot.Dockerfile` |
