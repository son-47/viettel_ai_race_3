#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
tag="${TAG:-r2-v027-hybrid-fusions}"
image="$repository:$tag"

docker build \
  -f submission/Dockerfile.r2-hybrid-fusions \
  -t "$image" \
  "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$image"
fi

echo "R2_HYBRID_FUSIONS_IMAGE=$image"
