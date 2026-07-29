#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-r2-v028-marlin-dspark-k3-s32-fastokens}"
image="$repository:$tag"

docker build \
  -f submission/Dockerfile.r2-hybrid-fusions-fastokens \
  -t "$image" \
  "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "R2_FASTOKENS_HYBRID_IMAGE=$image"
