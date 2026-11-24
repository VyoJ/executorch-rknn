#!/bin/sh
# Host-side: run a .pte with executor_runner + RknnBackend on the board, once per
# <dir>/inputs/*.bin, pulling each output to <dir>/board-pte/*.bin; then a
# 100-execution timing run on the first input. Logs go to <dir>/board-pte/.
# Files are staged in /tmp (RAM) and removed afterwards.
# Build the runner first: scripts/dk cross scripts/build_rknn_runner.sh
# (RUNNER=<path> picks another build, e.g. out/rknn-runner-xnnpack/executor_runner.stripped)
# usage: tests/device/run_rknn_pte.sh out/mv2_rknn.pte out/mv2_rknn
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
pte="$1"
dir="$(cd "$2" && pwd)"
runner="${RUNNER:-$root/out/rknn-runner/executor_runner.stripped}"
remote=/tmp/rknn-pte
out="$dir/board-pte"

[ -f "$runner" ] || { echo "missing $runner; run scripts/build_rknn_runner.sh" >&2; exit 1; }
adb get-state >/dev/null
adb shell "/oem/usr/bin/RkLunch-stop.sh >/dev/null 2>&1; rm -rf $remote && mkdir -p $remote"
trap 'adb shell "rm -rf $remote"' EXIT
adb push -q "$runner" "$remote/executor_runner"
adb push -q "$pte" "$remote/model.pte"
rm -rf "$out" && mkdir -p "$out"

run() {
  adb shell "cd $remote && rm -f out-0.bin && ./executor_runner --model_path=model.pte \
    --inputs=input.bin --output_file=out --print_output=none $*; echo exit=\$?"
}

for input in "$dir"/inputs/*.bin; do
  name="$(basename "$input")"
  adb push -q "$input" "$remote/input.bin"
  run > "$out/${name%.bin}.log" 2>&1
  # adbd on the board does not return the remote exit status, so check for the output.
  if [ "$(adb shell "[ -f $remote/out-0.bin ] && echo yes" | tr -d '\r')" = "yes" ]; then
    adb pull -q "$remote/out-0.bin" "$out/$name"
  else
    echo "$name: no output, see $out/${name%.bin}.log" >&2
  fi
done

adb push -q "$(ls "$dir"/inputs/*.bin | head -1)" "$remote/input.bin"
adb shell "cd $remote && RKNN_PROFILE=1 /usr/bin/time -v ./executor_runner --model_path=model.pte \
  --inputs=input.bin --print_output=none --num_executions=100; echo exit=\$?" > "$out/timing.log" 2>&1
tr -d '\r' < "$out/timing.log" | grep -E 'exit=|executed successfully|RknnBackend profile|Maximum resident'
