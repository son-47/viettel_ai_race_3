#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
dockerfile="${DOCKERFILE:-submission/Dockerfile.lmhead-fp8}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-v0.25.1-lfm25-silu-fp8-lmhead-fp8}"
image="$repository:$tag"

docker build -f "$dockerfile" -t "$image" "$context"

if [[ "${TEST:-0}" == 1 ]]; then
  docker run --rm --gpus all \
    --entrypoint python3 \
    "$image" \
    /opt/lfm25/test_lfm25_fp8_lm_head.py \
    --iterations "${ITERATIONS:-200}" \
    --min-decode-speedup "${MIN_DECODE_SPEEDUP:-1.05}"
fi

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "LMHEAD_FP8_IMAGE=$image"
