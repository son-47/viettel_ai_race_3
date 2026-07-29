#!/usr/bin/env bash
set -euo pipefail

# Paired, one-variable A/B around the exact portal-65.71 configuration.
# The controls are interleaved to expose GPU clock or host-load drift.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"
repeats="${REPEATS:-1}"

cases=(
  best_control
  best_stream2
  best_control
  best_stream4
  best_control
)

# MRV2 is deliberately opt-in: vLLM v0.25.1 does not select it by default
# for hybrid models, although its V2 model-state implementation supports them.
if [[ "${INCLUDE_MRV2:-0}" == 1 ]]; then
  cases+=(best_mrv2 best_control)
fi

for repeat in $(seq 1 "$repeats"); do
  echo "LFM25_CONTROL_PLANE_REPEAT=$repeat/$repeats"
  for case_name in "${cases[@]}"; do
    RATE_SCALE="$rate_scale" WARMUP="$warmup" \
      bash "$runner" "$case_name"
  done
done
