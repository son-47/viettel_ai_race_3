#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-misokaio/ghfjdk:r2-v027-hybrid-fusions}"
iterations="${ITERATIONS:-500}"

docker run --rm --gpus all --entrypoint python3 \
  "$image" \
  /opt/lfm25/test_r2_lfm25_fused_short_conv.py \
  --iterations "$iterations" \
  --batches 1 2 4 8 16 32

docker run --rm --gpus all --entrypoint python3 \
  "$image" \
  /opt/lfm25/test_r2_lfm25_qk_norm_rope.py \
  --iterations "$iterations" \
  --tokens 1 2 4 8 16 32
