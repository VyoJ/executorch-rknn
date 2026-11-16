#!/bin/sh
# Host-side only: copy files to the board over adb and optionally run a command.
# Builds nothing.
# usage: scripts/deploy.sh [-c "command to run on board"] file [file...]
#   files land in $REMOTE_DIR (default /root/et)
set -e
REMOTE_DIR="${REMOTE_DIR:-/root/et}"
cmd=""
if [ "$1" = "-c" ]; then cmd="$2"; shift 2; fi
[ $# -ge 1 ] || { echo "usage: $0 [-c cmd] file..." >&2; exit 2; }

adb get-state >/dev/null
adb shell "mkdir -p '$REMOTE_DIR'"
for f in "$@"; do
  adb push "$f" "$REMOTE_DIR/" >/dev/null
  adb shell "chmod +x '$REMOTE_DIR/$(basename "$f")'" 2>/dev/null || true
  echo "pushed $f -> $REMOTE_DIR/"
done
if [ -n "$cmd" ]; then
  # adb shell does not propagate the remote exit code reliably; echo it explicitly.
  adb shell "cd '$REMOTE_DIR' && $cmd; echo \"[exit \$?]\""
fi
