#!/usr/bin/env bash
set -euo pipefail

# Paired sequence around the current 64.35 image. Repeating both arms is more
# useful than a wide flag sweep because the expected gain is sub-millisecond.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"
control_image="${CONTROL_IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-fused@sha256:53d1892ca842ffa2f5e3113f0f775450701bacb7c014d8f497bd63e6ad61d401}"
candidate_image="${CANDIDATE_IMAGE:-misokaio/ghfjdk@sha256:bbda70fede826b43dbd8b92bb03fb880009c9c55162df4ba8a98f0325e9be2f4}"
base_result_dir="${RESULT_DIR:-/home/zeus/content/results/lfm25_silu_fp8}"

run_case() {
  local label="$1"
  local image="$2"
  echo "LFM25_SILU_FP8_CASE=$label IMAGE=$image"
  RESULT_DIR="$base_result_dir/$label" \
    IMAGE="$image" RATE_SCALE="$rate_scale" WARMUP="$warmup" \
    bash "$runner" cold_control
}

run_case control_start "$control_image"
run_case candidate_first "$candidate_image"
run_case control_middle "$control_image"
run_case candidate_second "$candidate_image"
run_case control_end "$control_image"
