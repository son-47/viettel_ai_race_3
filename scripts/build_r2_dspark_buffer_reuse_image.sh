#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-r2-v026-dspark-buffer-reuse}"
image="$repository:$tag"

docker build \
  -f submission/Dockerfile.r2-dspark-buffer-reuse \
  -t "$image" \
  "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "R2_DSPARK_BUFFER_REUSE_IMAGE=$image"
