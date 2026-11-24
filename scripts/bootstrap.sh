#!/bin/sh
# Fetch the ExecuTorch submodule and only the nested submodules the build uses.
# Host-side, needs git only. Run once after cloning.
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"

git -C "$root" submodule update --init third_party/executorch
git -C "$root/third_party/executorch" submodule update --init \
  third-party/flatbuffers \
  third-party/flatcc \
  third-party/gflags \
  third-party/json \
  backends/xnnpack/third-party/FXdiv \
  backends/xnnpack/third-party/FP16 \
  backends/xnnpack/third-party/XNNPACK \
  backends/xnnpack/third-party/cpuinfo \
  backends/xnnpack/third-party/pthreadpool
