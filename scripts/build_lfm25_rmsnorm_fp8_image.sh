#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
dockerfile="${DOCKERFILE:-submission/Dockerfile.rmsnorm-fp8-fused}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-v0.25.1-lfm25-rmsnorm-fp8}"
image="$repository:$tag"

docker build -f "$dockerfile" -t "$image" "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "IMAGE=$image"
