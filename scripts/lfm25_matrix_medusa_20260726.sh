#!/usr/bin/env bash
set -euo pipefail

# Run from the remote workspace that contains lfm25_remote_ab.sh and the local
# model directory. Controls bracket candidates to expose clock/load drift.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"

cases=(
  cold_control
  medusa_image_control
  medusa1
  medusa2
  medusa3
  medusa_image_control
  cold_control
)

if [[ "${INCLUDE_FUSED:-0}" == 1 ]]; then
  cases+=(
    medusa_fused_control
    medusa_fused1
    medusa_fused2
    medusa_fused3
    medusa_fused_control
    cold_control
  )
fi

if [[ "${INCLUDE_NON_SPEC:-0}" == 1 ]]; then
  cases+=(shortconv_fp8 lean_decode cold_control)
fi

for case_name in "${cases[@]}"; do
  RATE_SCALE="$rate_scale" WARMUP="$warmup" bash "$runner" "$case_name"
done
