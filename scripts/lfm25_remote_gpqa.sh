#!/usr/bin/env bash
set -euo pipefail

# Paired GPQA-Diamond mirror guardrail for the Lightning L4 testbed.
# This is not the organizer's gated/full GPQA run; it is used to reject
# serving changes that buy latency by causing an obvious quality regression.

case_name="${1:-control}"
image="${IMAGE:-misokaio/ghfjdk:v0.25.1}"
workspace="${WORKSPACE:-/home/zeus/content}"
model_dir="${MODEL_DIR:-$workspace/model-lfm25}"
server_name="lfm25-gpqa-server"
result_dir="$workspace/results/gpqa"
extra_args=()
quant_args=(--quantization=fp8)

case "$case_name" in
  bf16) quant_args=() ;;
  control) ;;
  kvfp8) extra_args+=(--kv-cache-dtype=fp8) ;;
  *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac

mkdir -p "$result_dir" "$workspace/hf-cache"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$case_name"
result_file="$result_dir/$run_id.json"
server_log="$result_dir/$run_id.server.log"

cleanup() {
  docker rm -f "$server_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

echo "CASE=$case_name IMAGE=$image"
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
  --entrypoint python3 \
  "$image" \
  -m vllm.entrypoints.openai.api_server \
  --model=/model \
  "${quant_args[@]}" \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 \
  --port=8000 \
  --max-model-len=32768 \
  --optimization-level=3 \
  --gpu-memory-utilization=0.85 \
  --tensor-parallel-size=1 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens=8192 \
  --max-num-seqs=32 \
  --disable-uvicorn-access-log \
  --language-model-only \
  --skip-mm-profiling \
  --no-enable-log-requests \
  --disable-log-stats \
  "${extra_args[@]}" >/dev/null

for attempt in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "SERVER_READY_SECONDS=$((attempt * 2))"
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$server_name" 2>/dev/null || true)" != true ]]; then
    docker logs "$server_name" >"$server_log" 2>&1 || true
    tail -100 "$server_log"
    exit 1
  fi
  if [[ "$attempt" == 180 ]]; then
    docker logs "$server_name" >"$server_log" 2>&1 || true
    tail -100 "$server_log"
    exit 1
  fi
  sleep 2
done

docker run --rm \
  --network host \
  --cpus=2 \
  --memory=4g \
  -v "$workspace:/work" \
  --env PYTHONPATH=/work/evaldeps \
  --env HF_HOME=/work/hf-cache \
  --entrypoint python3 \
  "$image" \
  /work/gpqa_mirror.py \
  --url=http://127.0.0.1:8000 \
  --model=LFM2.5-1.2B-Instruct \
  --n="${N:-100}" \
  --start="${START:-0}" \
  --conc="${CONC:-8}" \
  --max-tokens="${MAX_TOKENS:-1536}" \
  --out="/work/results/gpqa/$run_id.json"

docker logs "$server_name" >"$server_log" 2>&1 || true
echo "RESULT=$result_file"
echo "SERVER_LOG=$server_log"
