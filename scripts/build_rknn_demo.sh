#!/bin/sh
# Build Rockchip's unmodified RV1106 MobileNet demo (plain RKNN, no ExecuTorch).
# Run inside the cross container: scripts/dk cross scripts/build_rknn_demo.sh
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
build="$root/out/rknn-demo"

cmake -S "$RKNPU2/examples/RV1106_RV1103/rknn_mobilenet_demo" -B "$build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$root/docker/luckfox-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target rknn_mobilenet_demo
ls -l "$build/rknn_mobilenet_demo"
