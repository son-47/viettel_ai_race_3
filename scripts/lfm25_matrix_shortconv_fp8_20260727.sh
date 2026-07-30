#!/usr/bin/env bash
set -euo pipefail

# Paired same-image A/B. Alternate order to reduce warm-cache and temporal bias.
# The OFF arm disables both the new out_proj quantization and its fusion because
# the environment is read before model construction.
image="${IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-shortconv-fp8}"
repeats="${REPEATS:-3}"

for repeat in $(seq 1 "$repeats"); do
  if (( repeat % 2 == 1 )); then
    cases=(shortconv_fp8_fused_off shortconv_fp8_fused_on)
  else
    cases=(shortconv_fp8_fused_on shortconv_fp8_fused_off)
  fi
  for case_name in "${cases[@]}"; do
    echo "PAIR_REPEAT=$repeat CASE=$case_name"
    IMAGE="$image" scripts/lfm25_remote_ab.sh "$case_name"
  done
done

