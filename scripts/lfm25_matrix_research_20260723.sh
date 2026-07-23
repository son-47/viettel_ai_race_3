#!/usr/bin/env bash
set -uo pipefail

# New-only screening matrix derived from LLM-inference-optimization-paper.
# Cases already measured in LFM25_OPTIMIZATION_20260722.md are deliberately
# absent. Every case starts with a cold engine and empty prefix cache.
workspace="${WORKSPACE:-/home/zeus/content}"
runner="${RUNNER:-$workspace/lfm25_remote_ab.sh}"
rate_scale="${RATE_SCALE:-4}"
warmup="${WARMUP:-0}"
weight_mode="${WEIGHT_MODE:-fp8}"
result_dir="${RESULT_DIR:-$workspace/results/lfm25_research_20260723}"

cases=(
  cold_control
  cascade
  v2_runner
  rust_frontend
  renderer2
  ssm_ds
  mamba32
  mamba64
  hybrid_manager_off
  reserve_off
  kvint8
  kvturbo4
  fp8_per_tensor
  fp8_per_block
  fp8_per_channel
  int8_weight_only
  mxfp8_weight
  mamba_fp16
  mamba_flashinfer
  mamba_fp16_sr_flashinfer
  gptq_w4
  awq_w4
  cold_control
)

failed=()
for case_name in "${cases[@]}"; do
  echo "MATRIX_CASE_START=$case_name"
  if RESULT_DIR="$result_dir" WEIGHT_MODE="$weight_mode" \
      RATE_SCALE="$rate_scale" WARMUP="$warmup" \
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
