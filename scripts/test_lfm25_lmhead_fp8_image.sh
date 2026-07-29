#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-misokaio/ghfjdk@sha256:2be86725b37a2853d601a7acd55d3fb37906d50ef787a0b7367761fe8e27647e}"

docker run --rm --gpus all \
  --entrypoint python3 \
  "$image" \
  /opt/lfm25/test_lfm25_fp8_lm_head.py \
  --iterations "${ITERATIONS:-200}" \
  --min-decode-speedup "${MIN_DECODE_SPEEDUP:-1.05}"
