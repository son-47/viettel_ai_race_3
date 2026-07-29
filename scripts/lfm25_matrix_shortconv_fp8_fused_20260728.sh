#!/usr/bin/env bash
set -euo pipefail

runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"
control_image="${CONTROL_IMAGE:-misokaio/ghfjdk@sha256:bbda70fede826b43dbd8b92bb03fb880009c9c55162df4ba8a98f0325e9be2f4}"
candidate_image="${CANDIDATE_IMAGE:-misokaio/ghfjdk@sha256:fdc694b7282a591428debbcbb9ae2424bfb5c2905d7950f536c13495a04ac829}"
base_result_dir="${RESULT_DIR:-/home/zeus/content/results/lfm25_shortconv_fp8_fused}"

# Bracket each candidate with the exact 65.71 scheduler/fusion control.
run_control() {
  local label="$1"
  RESULT_DIR="$base_result_dir/$label" \
  IMAGE="$control_image" \
  RATE_SCALE="$rate_scale" \
  WARMUP="$warmup" \
    bash "$runner" fused_silu_on
}

run_candidate() {
  local label="$1"
  RESULT_DIR="$base_result_dir/$label" \
  SHORTCONV_FP8_FUSED_IMAGE="$candidate_image" \
  RATE_SCALE="$rate_scale" \
  WARMUP="$warmup" \
    bash "$runner" shortconv_fp8_fused
}

run_control control_start
run_candidate candidate_first
run_control control_middle
run_candidate candidate_second
run_control control_end
