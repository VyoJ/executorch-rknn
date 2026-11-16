#!/bin/sh
# Apply patches/<submodule>/*.patch to third_party/<submodule>. Safe to re-run.
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"

for dir in "$root"/patches/*/; do
  [ -d "$dir" ] || continue
  name="$(basename "$dir")"
  repo="$root/third_party/$name"
  for p in "$dir"*.patch; do
    [ -f "$p" ] || continue
    if git -C "$repo" apply --reverse --check "$p" 2>/dev/null; then
      continue
    fi
    git -C "$repo" apply "$p"
    echo "applied $name/$(basename "$p")"
  done
done
