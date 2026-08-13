#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$ROOT/src/manipulator/altro"
BUILD="$ROOT/build/manipulator_altro"
OUTPUT="${MANIPULATOR_BENCHMARK_OUTPUT:-$ROOT/result/manipulator/altro}"

MODE="${1:-}"
if [[ "$MODE" != "--no-build" ]]; then
  cmake -S "$SOURCE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD" --parallel
fi

if [[ "$MODE" == "--build-only" ]]; then
  exit 0
fi

mkdir -p "$OUTPUT"
MANIPULATOR_BENCHMARK_OUTPUT="$OUTPUT" \
MANIPULATOR_BENCHMARK_SAVE_EE="${MANIPULATOR_BENCHMARK_SAVE_EE:-0}" \
MANIPULATOR_BENCHMARK_SAVE_STATE="${MANIPULATOR_BENCHMARK_SAVE_STATE:-0}" \
  "$BUILD/manipulator_altro_random_pose_benchmark"
