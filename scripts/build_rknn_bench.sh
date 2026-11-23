#!/bin/sh
# Build rknn_bench, a generic .rknn runner for the board (tests/device/rknn_bench.cpp).
# Run inside the cross container: scripts/dk cross scripts/build_rknn_bench.sh
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/out/rknn_bench"
"$LUCKFOX_TRIPLE-g++" -O2 -std=c++17 -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -Wno-psabi \
  -I"$RKNPU2/runtime/Linux/librknn_api/include" -I"$root/backend/runtime" "$root/tests/device/rknn_bench.cpp" \
  -o "$root/out/rknn_bench/rknn_bench" \
  -L"$RKNPU2/runtime/Linux/librknn_api/armhf-uclibc" -lrknnmrt
ls -l "$root/out/rknn_bench/rknn_bench"
