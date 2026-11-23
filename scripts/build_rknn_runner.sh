#!/bin/sh
# Cross-compile executor_runner with RknnBackend (examples/runner) for the board.
# Run inside the cross container: scripts/dk cross scripts/build_rknn_runner.sh
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
build="${BUILD_DIR:-$root/out/rknn-runner}"

"$root/scripts/apply_patches.sh"

cmake -S "$root/examples/runner" -B "$build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$root/docker/luckfox-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_CPUINFO=OFF \
  -DEXECUTORCH_BUILD_DEVTOOLS=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_ENABLE_LOGGING=ON

cmake --build "$build" --target executor_runner "$@"
"$LUCKFOX_TRIPLE-strip" -o "$build/executor_runner.stripped" "$build/executorch/executor_runner"
ls -l "$build/executorch/executor_runner" "$build/executor_runner.stripped"
