#!/bin/zsh
set -euo pipefail

readonly package_name="metal-fp4-compare-macos27-arm64"
readonly package_dir="dist/${package_name}"
readonly archive="dist/${package_name}.tar.gz"

if [[ "$(lipo -archs build/compare_fp4)" != "arm64" ]]; then
  print -u2 -- "compare_fp4 must be an arm64 executable"
  exit 1
fi

mkdir -p "${package_dir}/build"
cp build/compare_fp4 build/fp4_native.metallib build/fp4_software_decode.metallib "${package_dir}/build/"
cp scripts/run_release.sh "${package_dir}/run.sh"
cp RELEASE.md "${package_dir}/README.md"
chmod +x "${package_dir}/run.sh"
tar -czf "$archive" -C dist "$package_name"
shasum -a 256 "$archive" > "${archive}.sha256"
print -- "$archive"
