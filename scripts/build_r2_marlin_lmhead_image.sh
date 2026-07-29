#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-r2-v026-marlin-lmhead}"
image="$repository:$tag"

docker build \
  -f submission/Dockerfile.r2-marlin-lmhead \
  -t "$image" \
  "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "R2_MARLIN_LMHEAD_IMAGE=$image"
