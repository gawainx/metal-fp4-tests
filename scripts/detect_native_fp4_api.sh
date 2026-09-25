#!/bin/zsh
set -euo pipefail

readonly module_cache="${TMPDIR:-/tmp}/metal-fp4-module-cache"
readonly import_source='@import Metal; int main(void) { return 0; }'
readonly probe_source='@import Metal; int main(void) { MTLTensorDataType type = MTLTensorDataTypeMetalFloat4E2M1; return (int)type; }'

if ! print -r -- "$import_source" | xcrun -sdk macosx clang -fmodules -fmodules-cache-path="$module_cache" -x objective-c -fsyntax-only -; then
  print -u2 -- "[fp4] cannot compile the Metal SDK import probe"
  exit 1
fi

if print -r -- "$probe_source" | xcrun -sdk macosx clang -fmodules -fmodules-cache-path="$module_cache" -x objective-c -fsyntax-only - 2>/dev/null; then
  print -- "available"
else
  print -- "unavailable"
fi
