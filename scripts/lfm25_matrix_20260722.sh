#!/usr/bin/env bash
set -uo pipefail

# One-variable screening around the verified 8192/32 FP8 control.  Each case
# starts from an empty engine/prefix cache; failures are logged and do not stop
# the rest of the matrix.
workspace="${WORKSPACE:-/home/zeus/content}"
runner="${RUNNER:-$workspace/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-4}"
warmup="${WARMUP:-0}"

cases=(
  maxlen8192
  b7168
  b9216
  s40
  block8
  mamba8
  output256
  frontend_inproc
  gpu92
  o2
  exact_graphs
  dbo
  pod_attention
)

failed=()
for case_name in "${cases[@]}"; do
  echo "MATRIX_CASE_START=$case_name"
  if RATE_SCALE="$rate_scale" WARMUP="$warmup" bash "$runner" "$case_name"; then
    echo "MATRIX_CASE_DONE=$case_name"
  else
    failed+=("$case_name")
    echo "MATRIX_CASE_FAILED=$case_name" >&2
  fi
done

if ((${#failed[@]})); then
  printf 'MATRIX_FAILED_CASES=%s\n' "${failed[*]}" >&2
  exit 1
fi

echo "MATRIX_COMPLETE=1"
