#!/usr/bin/env bash
set -euo pipefail

# Same-image paired A/B: only the new RMSNorm/FP8 flag changes.  Both cases
# use the exact scheduling parameters from the 65.71 portal compose.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"
image="${IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-rmsnorm-fp8}"
base_result_dir="${RESULT_DIR:-/home/zeus/content/results/lfm25_rmsnorm_fp8}"

run_case() {
  local label="$1"
  local runner_case="$2"
  echo "LFM25_RMSNORM_FP8_CASE=$label IMAGE=$image"
  RESULT_DIR="$base_result_dir/$label" IMAGE="$image" \
    RATE_SCALE="$rate_scale" WARMUP="$warmup" \
    bash "$runner" "$runner_case"
}

run_case control_start fused_rmsnorm_off
run_case candidate_first fused_rmsnorm_on
run_case control_middle fused_rmsnorm_off
run_case candidate_second fused_rmsnorm_on
run_case control_end fused_rmsnorm_off
