#!/bin/sh
# Cross-compile executor_runner with RknnBackend (examples/runner) for the board.
# Run inside the cross container: scripts/dk cross scripts/build_rknn_runner.sh
# With XNNPACK=1, also link the XNNPACK backend (built in out/rknn-runner-xnnpack).
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${XNNPACK:-0}" = 1 ]; then
  build="${BUILD_DIR:-$root/out/rknn-runner-xnnpack}"
  # The ARMv6 SIMD32 kernels need ACLE types GCC 8 lacks and only serve cores
  # without NEON. uClibc declares struct timespec only with _GNU_SOURCE.
  xnnpack="-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_PTHREADPOOL=ON -DEXECUTORCH_BUILD_CPUINFO=ON
    -DXNNPACK_ENABLE_ARM_SIMD32=OFF -DXNNPACK_ENABLE_ARM_BF16=OFF -DUSE_GNU_SOURCE=ON"
else
  build="${BUILD_DIR:-$root/out/rknn-runner}"
  xnnpack="-DEXECUTORCH_BUILD_XNNPACK=OFF -DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF"
fi

"$root/scripts/apply_patches.sh"

# shellcheck disable=SC2086
cmake -S "$root/examples/runner" -B "$build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$root/docker/luckfox-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  $xnnpack \
  -DEXECUTORCH_BUILD_DEVTOOLS=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_ENABLE_LOGGING=ON

cmake --build "$build" --target executor_runner "$@"
"$LUCKFOX_TRIPLE-strip" -o "$build/executor_runner.stripped" "$build/executorch/executor_runner"
ls -l "$build/executorch/executor_runner" "$build/executor_runner.stripped"
