#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-fused}"
iterations="${ITERATIONS:-500}"

docker run --rm --gpus all \
  --entrypoint python3 \
  "$image" \
  /opt/lfm25/test_lfm25_fused_short_conv.py \
  --iterations "$iterations"

docker run --rm --gpus all \
  --entrypoint python3 \
  "$image" \
  /opt/lfm25/test_lfm25_fused_qk_norm_rope.py \
  --iterations "$iterations"
