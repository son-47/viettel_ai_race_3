#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-silu-fp8}"
iterations="${ITERATIONS:-500}"

# Regression-test the two fusions inherited from the 64.35 image first.
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

# On SM90 this runs correctness plus latency. On older GPUs it validates the
# activation/scale math and separately compiles the production E4M3 kernel for
# sm90, because Triton does not support E4M3 stores on the local GTX 1650.
docker run --rm --gpus all \
  --entrypoint python3 \
  "$image" \
  /opt/lfm25/test_lfm25_fused_silu_fp8.py \
  --iterations "$iterations"
