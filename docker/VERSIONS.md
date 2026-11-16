# Pinned versions

| Item | Version | Where it is set |
|---|---|---|
| Base image (cross) | `debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc` (linux/amd64) | `docker/cross.Dockerfile` |
| Luckfox Pico SDK | `824b817f889c2cbff1d48fcdb18ab494a68f69d1` (main, 2026-11-17) | `docker/cross.Dockerfile` `LUCKFOX_SDK_COMMIT` |
| Cross toolchain | `arm-rockchip830-linux-uclibcgnueabihf`, GCC 8.3.0 | from the SDK above |
| CMake (cross) | 3.25.1 (Debian bookworm) | apt |
| Board RKNN runtime | `librknnmrt.so` 1.6.0 (2de554906, 2024-01-17), in `/oem/usr/lib` | on the board's firmware |
| rknn-toolkit2 | **1.6.0** (must match the board runtime) | not yet: `aot` image |
| ExecuTorch | v1.6.0 (`4776f71b6f0daa5dfc7d758d5836fc1cc0d40154`) | `third_party/executorch` submodule |
| torch (cross, build-time codegen only) | 2.15.0+cpu, the version ExecuTorch v1.6.0 pins in `torch_pin.py` | `docker/cross.Dockerfile` `TORCH_VERSION` |
