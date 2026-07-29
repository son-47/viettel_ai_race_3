#!/usr/bin/env bash
set -euo pipefail

# Paired ordering exposes clock/load drift. The paper-fidelity k=15 run is
# last because small LFM targets often lose badly to verification overhead.
runner="${RUNNER:-/home/zeus/content/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-1}"
warmup="${WARMUP:-0}"
collect_spec_metrics="${COLLECT_SPEC_METRICS:-0}"

cases=(
  pld_safe_control
  pld_safe1
  pld_safe2
  pld_safe3
  pld_safe_control
)

if [[ "${INCLUDE_PAPER_K15:-0}" == 1 ]]; then
  cases+=(pld_safe15 pld_safe_control)
fi

for case_name in "${cases[@]}"; do
  RATE_SCALE="$rate_scale" \
  WARMUP="$warmup" \
  COLLECT_SPEC_METRICS="$collect_spec_metrics" \
    bash "$runner" "$case_name"
done
