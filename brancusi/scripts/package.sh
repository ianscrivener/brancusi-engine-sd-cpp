#!/usr/bin/env bash
# Package brancusi-engine for one variant.
# Usage: brancusi/scripts/package.sh <variant> <build-dir> <out-dir>
# Writes <out-dir>/brancusi-engine-<version>-<variant>.zip and prints one JSON line:
# {"variant":"...","file":"...","size":N,"sha256":"..."}
set -euo pipefail

variant="$1"
build_dir="$2"
out_dir="$3"

root="$(cd "$(dirname "$0")/../.." && pwd)"
version="$(git -C "$root" describe --tags --always)"
name="brancusi-engine-${version}-${variant}"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/$name" "$out_dir"
exe="brancusi-engine"
[[ "$variant" == win-* ]] && exe="brancusi-engine.exe"

# The binary is in bin/ (Unix generators) or bin/Release/ (Visual Studio).
src="$build_dir/bin/$exe"
[[ -f "$src" ]] || src="$build_dir/bin/Release/$exe"
cp "$src" "$stage/$name/"

# Shared libraries next to the binary (for example CUDA runtime DLLs on Windows).
for lib in "$(dirname "$src")"/*.dll "$(dirname "$src")"/*.so* "$(dirname "$src")"/*.dylib; do
  [[ -f "$lib" ]] && cp "$lib" "$stage/$name/"
done
cp "$root/LICENSE" "$stage/$name/LICENSE-stable-diffusion.cpp.txt"

zip_path="$(cd "$out_dir" && pwd)/$name.zip"
rm -f "$zip_path"
(cd "$stage" && if command -v zip > /dev/null; then zip -qr "$zip_path" "$name"; else 7z a -tzip "$zip_path" "$name" > /dev/null; fi)

if command -v sha256sum > /dev/null; then
  sha="$(sha256sum "$zip_path" | cut -d' ' -f1)"
else
  sha="$(shasum -a 256 "$zip_path" | cut -d' ' -f1)"
fi
size="$(wc -c < "$zip_path" | tr -d ' ')"
printf '{"variant":"%s","file":"%s","size":%s,"sha256":"%s"}\n' "$variant" "$name.zip" "$size" "$sha"
