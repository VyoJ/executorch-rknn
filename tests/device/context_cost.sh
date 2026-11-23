#!/bin/sh
# Host-side: for each out/rknn_contexts/n<k> (tests/device/export_context_cost.py),
# print the NPU memory of each .rknn context (rknn_bench) and run the .pte
# with run_rknn_pte.sh for latency and peak RSS.
# Build rknn_bench and the runner first.
# usage: tests/device/context_cost.sh
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
remote=/tmp/rknn-contexts

adb get-state >/dev/null
adb shell "rm -rf $remote && mkdir -p $remote"
trap 'adb shell "rm -rf $remote"' EXIT
adb push -q "$root/out/rknn_bench/rknn_bench" "$remote/"
# rknn_bench prints the memory line before it checks the input file.
adb shell "head -c 4 /dev/zero > $remote/dummy.bin"

for dir in "$root"/out/rknn_contexts/n*; do
  echo "== $(basename "$dir")"
  for rknn in "$dir"/rknn_*/model.rknn; do
    adb push -q "$rknn" "$remote/model.rknn"
    adb shell "cd $remote && ./rknn_bench model.rknn 0 dummy.bin 2>/dev/null" | tr -d '\r' | grep '^mem '
  done
  "$root/tests/device/run_rknn_pte.sh" "$dir/model.pte" "$dir" | grep -E 'executed successfully|Maximum resident'
  tr -d '\r' < "$dir/board-pte/000.log" | grep -o 'Model loaded in [0-9.]* ms'
done
