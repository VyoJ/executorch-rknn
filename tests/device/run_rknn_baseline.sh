#!/bin/sh
# Host-side: run Rockchip's MobileNet demo on the board over the images listed
# in out/baseline/board.txt, then a 100-iteration timing run.
# Images are staged in /tmp (RAM) and removed afterwards.
# Logs go to out/baseline/board-*.log.
# usage: tests/device/run_rknn_baseline.sh
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
local_dir="$root/out/baseline"
remote=/root/et/baseline
images=/tmp/rknn-baseline-eval

adb get-state >/dev/null
adb shell "mkdir -p $remote && rm -rf $images && mkdir -p $images"
trap 'adb shell "rm -rf $images"' EXIT
adb push -q "$root/out/rknn-demo/rknn_mobilenet_demo" "$local_dir/mobilenet_v2.rknn" "$remote/"
(cd "$local_dir/eval" && adb push -q $(cat "$local_dir/board.txt") "$images/")

adb shell "cd $remote && for f in $images/*.png; do \
    echo \"IMAGE \${f##*/}\"; ./rknn_mobilenet_demo mobilenet_v2.rknn \$f 1 | sed -n '/Top5/,\$p'; \
    sleep 0.2; \
  done" > "$local_dir/board-eval.log"

first="$(head -1 "$local_dir/board.txt")"
adb shell "cd $remote && /usr/bin/time -v ./rknn_mobilenet_demo mobilenet_v2.rknn $images/$first 100" \
  > "$local_dir/board-timing.log" 2>&1

awk '/Elapse Time/ {sub("ms,", "", $5); t += $5; n++} END {printf "mean %.2f ms over %d runs\n", t / n, n}' \
  "$local_dir/board-timing.log"
grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' "$local_dir/board-timing.log"
