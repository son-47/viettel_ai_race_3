#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
repository="${REPOSITORY:-misokaio/ghfjdk}"
fp8_tag="${FP8_TAG:-v0.25.1-lfm25-silu-fp8-shortconv-fp8}"
pld_tag="${PLD_TAG:-v0.25.1-lfm25-silu-fp8-pld-safe}"

docker build \
  -f submission/Dockerfile.shortconv-fp8-online \
  -t "$repository:$fp8_tag" \
  "$context"

docker build \
  -f submission/Dockerfile.pld-safe \
  -t "$repository:$pld_tag" \
  "$context"

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$repository:$fp8_tag"
  docker push "$repository:$pld_tag"
fi

echo "SHORTCONV_FP8_IMAGE=$repository:$fp8_tag"
echo "PLD_SAFE_IMAGE=$repository:$pld_tag"
