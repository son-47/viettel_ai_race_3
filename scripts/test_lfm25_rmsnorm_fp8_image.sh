#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-misokaio/ghfjdk:v0.25.1-lfm25-rmsnorm-fp8}"
iterations="${ITERATIONS:-500}"

# The candidate inherits every earlier fusion; regress them before measuring
# the newly wired upstream fused RMSNorm/FP8 kernel.
docker run --rm --gpus all --entrypoint python3 "$image" \
  /opt/lfm25/test_lfm25_fused_short_conv.py --iterations "$iterations"
docker run --rm --gpus all --entrypoint python3 "$image" \
  /opt/lfm25/test_lfm25_fused_qk_norm_rope.py --iterations "$iterations"
docker run --rm --gpus all --entrypoint python3 "$image" \
  /opt/lfm25/test_lfm25_fused_silu_fp8.py --iterations "$iterations"
docker run --rm --gpus all --entrypoint python3 "$image" \
  /opt/lfm25/test_lfm25_fused_rmsnorm_fp8.py --iterations "$iterations"
