#!/bin/sh
# Host-side: run <dir>/model.rknn on the board with rknn_bench, once per
# <dir>/inputs/*.bin, pulling each output to <dir>/board/*.bin; then a
# 100-iteration timing run on the first input. Logs go to <dir>/board/.
# Files are staged in /tmp (RAM) and removed afterwards.
# Build rknn_bench first: scripts/dk cross scripts/build_rknn_bench.sh
# usage: tests/device/run_rknn_model.sh out/mv2_rknn
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
dir="$(cd "$1" && pwd)"
remote=/tmp/rknn-model

adb get-state >/dev/null
adb shell "/oem/usr/bin/RkLunch-stop.sh >/dev/null 2>&1; rm -rf $remote && mkdir -p $remote"
trap 'adb shell "rm -rf $remote"' EXIT
adb push -q "$root/out/rknn_bench/rknn_bench" "$dir/model.rknn" "$remote/"
rm -rf "$dir/board" && mkdir -p "$dir/board"

for input in "$dir"/inputs/*.bin; do
  name="$(basename "$input")"
  adb push -q "$input" "$remote/input-0.bin"
  adb shell "cd $remote && rm -f output-0.bin && ./rknn_bench model.rknn 0 input-0.bin" \
    > "$dir/board/${name%.bin}.log" 2>&1
  # adbd on the board does not return the remote exit status, so check for the output.
  if [ "$(adb shell "[ -f $remote/output-0.bin ] && echo yes" | tr -d '\r')" = "yes" ]; then
    adb pull -q "$remote/output-0.bin" "$dir/board/$name"
  else
    echo "$name: no output, see $dir/board/${name%.bin}.log" >&2
  fi
done

first="$(ls "$dir"/inputs/*.bin | head -1)"
adb push -q "$first" "$remote/input-0.bin"
adb shell "cd $remote && ./rknn_bench model.rknn 100 input-0.bin" > "$dir/board/timing.log" 2>&1
grep -E '^(input|output|mem|run|float-in) ' "$dir/board/timing.log" | tr -d '\r'
