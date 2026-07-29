#!/usr/bin/env bash
set -euo pipefail

context="${CONTEXT:-submission}"
dockerfile="${DOCKERFILE:-submission/Dockerfile.shortconv-fused}"
repository="${REPOSITORY:-misokaio/ghfjdk}"

build_variant() {
  local tag="$1"
  local fused="$2"
  local no_stack="$3"
  local qk_fused="$4"
  docker build \
    -f "$dockerfile" \
    --build-arg "LFM25_FUSED_SHORTCONV=$fused" \
    --build-arg "LFM25_BYPASS_SINGLE_VSTACK=$no_stack" \
    --build-arg "LFM25_FUSED_QK_NORM_ROPE=$qk_fused" \
    -t "$repository:$tag" \
    "$context"
}

build_variant v0.25.1-lfm25-control 0 0 0
build_variant v0.25.1-lfm25-nostack 0 1 0
build_variant v0.25.1-lfm25-shortconv 1 1 0
build_variant v0.25.1-lfm25-qk 0 0 1
build_variant v0.25.1-lfm25-fused 1 1 1

if [[ "${PUSH:-0}" == 1 ]]; then
  docker push "$repository:v0.25.1-lfm25-control"
  docker push "$repository:v0.25.1-lfm25-nostack"
  docker push "$repository:v0.25.1-lfm25-shortconv"
  docker push "$repository:v0.25.1-lfm25-qk"
  docker push "$repository:v0.25.1-lfm25-fused"
fi
