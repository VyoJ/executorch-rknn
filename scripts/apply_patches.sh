#!/bin/sh
# Apply patches/<path>/*.patch to third_party/<path>, where <path> is a
# submodule, possibly nested (executorch/backends/xnnpack/third-party/XNNPACK).
# Safe to re-run.
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"

find "$root/patches" -name '*.patch' | sort | while read -r p; do
  rel="$(dirname "${p#"$root/patches/"}")"
  repo="$root/third_party/$rel"
  if git -C "$repo" apply --reverse --check "$p" 2>/dev/null; then
    continue
  fi
  git -C "$repo" apply "$p"
  echo "applied $rel/$(basename "$p")"
done
