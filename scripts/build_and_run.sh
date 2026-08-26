#!/bin/zsh
set -euo pipefail

# Builds the benchmark, detects the SDK's native FP4 tensor declaration, and runs it.
# Arguments: forwards all arguments to the benchmark executable.
# Returns: exits with the benchmark's status.
make all
readonly native_fp4_api="$(./scripts/detect_native_fp4_api.sh)"
print -- "[fp4] native FP4 tensor API in active SDK: ${native_fp4_api}"
./build/fp4_transformer_bench --native-fp4-sdk "${native_fp4_api}" "$@"
