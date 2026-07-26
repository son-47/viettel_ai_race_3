#!/usr/bin/env bash
set -euo pipefail

# Compare five tags built from the same Dockerfile/source.  Only the image ENV
# defaults differ, so patched-image controls expose both code and clock drift.
# The existing remote runner remains untouched because it may be edited by a
# concurrent workspace task.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"
stock_image="${STOCK_IMAGE:-misokaio/ghfjdk:v0.25.1}"
patched_control_image="${LFM25_CONTROL_IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-control}"
nostack_image="${LFM25_NOSTACK_IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-nostack}"
shortconv_image="${LFM25_SHORTCONV_IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-shortconv}"
qk_image="${LFM25_QK_IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-qk}"
fused_image="${LFM25_FUSED_IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-fused}"
base_result_dir="${RESULT_DIR:-/home/zeus/content/results/lfm25_kernel_fusions}"

run_case() {
  local label="$1"
  local image="$2"
  echo "LFM25_FUSION_CASE=$label IMAGE=$image"
  RESULT_DIR="$base_result_dir/$label" \
    IMAGE="$image" RATE_SCALE="$rate_scale" WARMUP="$warmup" \
    bash "$runner" cold_control
}

run_case stock_control_start "$stock_image"
run_case patched_image_control_start "$patched_control_image"
run_case bypass_single_vstack "$nostack_image"
run_case fused_shortconv "$shortconv_image"
run_case fused_qk_norm_rope "$qk_image"
run_case fused_all "$fused_image"
run_case patched_image_control_end "$patched_control_image"
run_case stock_control_end "$stock_image"
