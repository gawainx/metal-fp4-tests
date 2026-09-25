#!/bin/zsh
set -euo pipefail

readonly package_dir="${0:A:h}"
cd "$package_dir"
exec ./build/compare_precisions
