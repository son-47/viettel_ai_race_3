#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/home/zeus/content}"
model_dir="${MODEL_DIR:-$workspace/model-lfm25}"
control_image="${CONTROL_IMAGE:-misokaio/ghfjdk@sha256:bbda70fede826b43dbd8b92bb03fb880009c9c55162df4ba8a98f0325e9be2f4}"
candidate_image="${PLD_SAFE_IMAGE:-misokaio/ghfjdk@sha256:d1a4d9bab96cfcaaffbfb531bf7935abcd97ab70787fc4a08eafdc593494eff1}"
python="${PYTHON:-python3}"
parity_script="${PARITY_SCRIPT:-$workspace/harness/check_speculative_parity.py}"
metrics_summarizer="${METRICS_SUMMARIZER:-$workspace/scripts/summarize_spec_metrics.py}"
server_name="lfm25-spec-parity"
reference="${REFERENCE:-$workspace/results/speculative_parity_reference.json}"

cleanup() {
  docker rm -f "$server_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

start_server() {
  local image="$1"
  shift
  cleanup
  docker run -d \
    --name "$server_name" \
    --gpus all \
    --cpus=3 \
    --memory=8g \
    --shm-size=2g \
    -p 8000:8000 \
    -v "$model_dir:/model:ro" \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env VLLM_USE_FASTOKENS=1 \
    --env VLLM_CONFIGURE_LOGGING=0 \
    --env VLLM_LFM25_FUSED_SHORTCONV=1 \
    --env VLLM_LFM25_BYPASS_SINGLE_VSTACK=1 \
    --env VLLM_LFM25_FUSED_QK_NORM_ROPE=1 \
    --env VLLM_LFM25_FUSED_SILU_FP8=1 \
    --entrypoint python3 \
    "$image" \
    -m vllm.entrypoints.openai.api_server \
    --model=/model \
    --quantization=fp8 \
    --served-model-name=LFM2.5-1.2B-Instruct \
    --host=0.0.0.0 \
    --port=8000 \
    --optimization-level=3 \
    --gpu-memory-utilization=0.85 \
    --tensor-parallel-size=1 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len=8192 \
    --max-num-batched-tokens=4096 \
    --max-num-seqs=32 \
    --disable-uvicorn-access-log \
    --language-model-only \
    --skip-mm-profiling \
    --no-enable-log-requests \
    "$@" >/dev/null

  for attempt in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      return
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$server_name" 2>/dev/null || true)" != true ]]; then
      docker logs "$server_name"
      exit 1
    fi
    if [[ "$attempt" == 180 ]]; then
      docker logs "$server_name"
      exit 1
    fi
    sleep 2
  done
}

mkdir -p "$(dirname "$reference")"

echo "Recording non-speculative reference..."
start_server "$control_image"
"$python" "$parity_script" \
  --mode record \
  --reference "$reference"

echo "Comparing rollback-patched PLD candidate..."
start_server "$candidate_image" \
  '--speculative-config={"method":"ngram","num_speculative_tokens":2,"prompt_lookup_min":3,"prompt_lookup_max":3}'
"$python" "$parity_script" \
  --mode compare \
  --reference "$reference"

metrics_file="${reference%.json}.metrics"
curl -fsS http://127.0.0.1:8000/metrics >"$metrics_file"
summary="$("$python" "$metrics_summarizer" "$metrics_file")"
echo "$summary"
drafts="$(echo "$summary" | awk 'NR == 2 {print $2}')"
if [[ -z "$drafts" || "$drafts" == 0 ]]; then
  echo "SPECULATIVE_PARITY_GATE=FAIL_NO_DRAFTS" >&2
  exit 1
fi

echo "SPECULATIVE_PARITY_GATE=PASS"
