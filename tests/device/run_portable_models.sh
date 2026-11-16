#!/bin/sh
# Host-side: run each exported model in out/models on the board with
# executor_runner and pull the outputs back next to the expected ones.
# usage: tests/device/run_portable_models.sh [num_executions]
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
runs="${1:-20}"
remote=/root/et
runner="$root/out/et-cross/executor_runner"

[ -x "$runner" ] || { echo "missing $runner; run scripts/build_runtime.sh" >&2; exit 1; }
adb get-state >/dev/null
adb shell "mkdir -p $remote"
adb push -q "$runner" "$remote/"

for dir in "$root"/out/models/*/; do
  name="$(basename "$dir")"
  inputs="$(cd "$dir" && ls input-*.bin | sort -V | paste -sd, -)"
  adb shell "rm -rf $remote/$name && mkdir -p $remote/$name"
  adb push -q "$dir/model.pte" "$dir"/input-*.bin "$remote/$name/"

  adb shell "cd $remote/$name && /usr/bin/time -v ../executor_runner \
      --model_path=model.pte --inputs=$inputs --output_file=board \
      --print_output=none --num_executions=$runs; echo exit=\$?" \
    > "$dir/board.log" 2>&1
  rm -f "$dir"/board-*.bin
  for f in $(adb shell "ls --color=never $remote/$name/board-*.bin" | tr -d '\r'); do
    adb pull -q "$f" "$dir/"
  done

  printf '%s: %s; %s; %s\n' "$name" \
    "$(grep -o 'exit=[0-9]*' "$dir/board.log")" \
    "$(grep -oE 'executed successfully .* ms' "$dir/board.log")" \
    "$(grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' "$dir/board.log")"
done
