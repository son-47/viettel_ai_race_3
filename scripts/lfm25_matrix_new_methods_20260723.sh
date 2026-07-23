#!/usr/bin/env bash
set -uo pipefail

# New-only matrix. Do not add cases from the prior LFM25 research reports:
# those are intentionally excluded so each result represents a new method.
workspace="${WORKSPACE:-/home/zeus/content}"
runner="${RUNNER:-$workspace/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-4}"
warmup="${WARMUP:-0}"
weight_mode="${WEIGHT_MODE:-fp8}"
result_dir="${RESULT_DIR:-$workspace/results/lfm25_new_methods_20260723}"

default_cases=(
  suffix8
  suffix16
  suffix32
  suffix16_loose
  bnb4
  bnb4_suffix16
  draft350m2
  draft350m4
  draft350m6
  draft350m8
)

if [[ -n "${CASES:-}" ]]; then
  case_list="${CASES//,/ }"
  read -r -a cases <<<"$case_list"
else
  cases=("${default_cases[@]}")
fi

failed=()
for case_name in "${cases[@]}"; do
  echo "MATRIX_CASE_START=$case_name"
  if RESULT_DIR="$result_dir" WEIGHT_MODE="$weight_mode" \
      RATE_SCALE="$rate_scale" WARMUP="$warmup" \
      DOWNLOAD_DRAFT="${DOWNLOAD_DRAFT:-0}" \
      DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-LiquidAI/LFM2.5-350M}" \
      bash "$runner" "$case_name"; then
    echo "MATRIX_CASE_DONE=$case_name"
  else
    failed+=("$case_name")
    echo "MATRIX_CASE_FAILED=$case_name" >&2
  fi
done

if ((${#failed[@]})); then
  printf 'MATRIX_FAILED_CASES=%s\n' "${failed[*]}" >&2
fi

echo "MATRIX_COMPLETE=1"
