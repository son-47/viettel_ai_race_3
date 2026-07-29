#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-v0.25.1-lfm25-shortconv-fp8-out-fused}"
image="$repository:$tag"

docker build \
  -f submission/Dockerfile.shortconv-fp8-out-fused \
  -t "$image" \
  "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "LFM25_SHORTCONV_FP8_OUT_IMAGE=$image"
