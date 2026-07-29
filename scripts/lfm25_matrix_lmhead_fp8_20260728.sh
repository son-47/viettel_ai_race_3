#!/usr/bin/env bash
set -euo pipefail

# Same-image paired A/B.  Only VLLM_LFM25_FP8_LM_HEAD changes.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"

cases=(
  lmhead_fp8_control
  lmhead_fp8
  lmhead_fp8_control
  lmhead_fp8
  lmhead_fp8_control
)

for case_name in "${cases[@]}"; do
  RATE_SCALE="$rate_scale" WARMUP="$warmup" \
    bash "$runner" "$case_name"
done
