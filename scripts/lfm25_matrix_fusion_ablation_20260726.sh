#!/usr/bin/env bash
set -euo pipefail

# Isolate the afternoon SiLU/FP8 fusion from the scheduler changes in the
# scored 65.71 compose.  Both arms use the same combined image and the same
# 8192/4096/32 scheduling parameters; only VLLM_LFM25_FUSED_SILU_FP8 changes.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"
image="${IMAGE:-misokaio/ghfjdk@sha256:bbda70fede826b43dbd8b92bb03fb880009c9c55162df4ba8a98f0325e9be2f4}"
base_result_dir="${RESULT_DIR:-/home/zeus/content/results/lfm25_fusion_ablation}"

run_case() {
  local label="$1"
  local runner_case="$2"
  echo "LFM25_FUSION_ABLATION_CASE=$label IMAGE=$image"
  RESULT_DIR="$base_result_dir/$label" IMAGE="$image" \
    RATE_SCALE="$rate_scale" WARMUP="$warmup" \
    bash "$runner" "$runner_case"
}

run_case silu_off_start fused_silu_off
run_case silu_on_first fused_silu_on
run_case silu_off_middle fused_silu_off
run_case silu_on_second fused_silu_on
run_case silu_off_end fused_silu_off
