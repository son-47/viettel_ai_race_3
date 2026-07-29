#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-shortconv-fp8-out-fused}"
iterations="${ITERATIONS:-300}"

docker run --rm --gpus all \
  "$image" \
  python3 /opt/lfm25/test_lfm25_fused_shortconv_fp8_out.py \
    --iterations "$iterations" \
    --batches 1 2 4 8 16 32
