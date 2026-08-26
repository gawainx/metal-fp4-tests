#!/bin/zsh
set -euo pipefail

readonly probe_source='@import Metal; int main(void) { MTLTensorDataType type = MTLTensorDataTypeMetalFloat4E2M1; return (int)type; }'

if print -r -- "$probe_source" | xcrun -sdk macosx clang -fmodules -x objective-c -fsyntax-only - 2>/dev/null; then
  print -- "available"
else
  print -- "unavailable"
fi
