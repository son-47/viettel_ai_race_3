#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-v0.25.1-lfm25-silu-fp8-bt8192-20260729}"
image="$repository:$tag"

docker build \
  -f submission/Dockerfile.silu-fp8-bt8192 \
  -t "$image" \
  "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "LFM25_BT8192_IMAGE=$image"
