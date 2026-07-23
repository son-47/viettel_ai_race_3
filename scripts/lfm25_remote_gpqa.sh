#!/usr/bin/env bash
set -euo pipefail

# Paired GPQA-Diamond mirror guardrail for the Lightning L4 testbed.
# This is not the organizer's gated/full GPQA run; it is used to reject
# serving changes that buy latency by causing an obvious quality regression.

case_name="${1:-control}"
image="${IMAGE:-misokaio/ghfjdk:v0.25.1}"
workspace="${WORKSPACE:-/home/zeus/content}"
model_dir="${MODEL_DIR:-$workspace/model-lfm25}"
draft_model_dir="${DRAFT_MODEL_DIR:-$workspace/model-lfm25-350m}"
server_name="lfm25-gpqa-server"
result_dir="$workspace/results/gpqa"
engine_cache_dir="${ENGINE_CACHE_DIR:-$workspace/vllm-cache}"
extra_args=()
extra_env=()
extra_mount_args=()
extra_deps=()
spec_model=""
spec_tokens=""
quant_args=(--quantization=fp8)

case "$case_name" in
  bf16) quant_args=() ;;
  control) ;;
  fp8_per_tensor) quant_args=(--quantization=fp8_per_tensor) ;;
  fp8_per_block) quant_args=(--quantization=fp8_per_block) ;;
  fp8_per_channel) quant_args=(--quantization=fp8_per_channel) ;;
  int8_weight_only) quant_args=(--quantization=int8_per_channel_weight_only) ;;
  mxfp8_weight) quant_args=(--quantization=mxfp8) ;;
  kvfp8) extra_args+=(--kv-cache-dtype=fp8) ;;
  gptq_w4)
    model_dir="$workspace/quant-models/gptq-w4a16-g128"
    quant_args=()
    ;;
  awq_w4)
    model_dir="$workspace/quant-models/awq-w4a16-asym"
    quant_args=()
    ;;
  suffix8|suffix16|suffix32)
    extra_deps+=(arctic-inference)
    spec_tokens="${case_name#suffix}"
    extra_args+=("--speculative-config={\"method\":\"suffix\",\"num_speculative_tokens\":$spec_tokens,\"suffix_decoding_max_tree_depth\":32,\"suffix_decoding_max_cached_requests\":10000,\"suffix_decoding_max_spec_factor\":1.0,\"suffix_decoding_min_token_prob\":0.1}")
    ;;
  bnb4)
    extra_deps+=("bitsandbytes>=0.49.2")
    quant_args=(--quantization=bitsandbytes)
    ;;
  bnb4_suffix16)
    extra_deps+=("bitsandbytes>=0.49.2" arctic-inference)
    quant_args=(--quantization=bitsandbytes)
    extra_args+=('--speculative-config={"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_tree_depth":32,"suffix_decoding_max_cached_requests":10000,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}')
    ;;
  draft350m2|draft350m4|draft350m6|draft350m8)
    spec_model="/draft-model"
    spec_tokens="${case_name#draft350m}"
    extra_mount_args+=(-v "$draft_model_dir:/draft-model:ro")
    extra_args+=("--speculative-config={\"method\":\"draft_model\",\"model\":\"$spec_model\",\"num_speculative_tokens\":$spec_tokens,\"draft_tensor_parallel_size\":1}")
    ;;
  *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac

if [[ "${#extra_deps[@]}" -gt 0 ]]; then
  deps_dir="$workspace/researchdeps"
  mkdir -p "$deps_dir"
  regular_deps=()
  nodeps_deps=()
  for dep in "${extra_deps[@]}"; do
    case "$dep" in
      arctic-inference)
        [[ -d "$deps_dir/arctic_inference" ]] || regular_deps+=("$dep")
        ;;
      bitsandbytes*)
        [[ -d "$deps_dir/bitsandbytes" ]] || nodeps_deps+=("$dep")
        ;;
      *) regular_deps+=("$dep") ;;
    esac
  done
  if [[ "${#regular_deps[@]}" -gt 0 ]]; then
    docker run --rm --network host \
      -v "$deps_dir:/deps" \
      --entrypoint python3 \
      "$image" \
      -m pip install --disable-pip-version-check --no-cache-dir --target /deps \
      "${regular_deps[@]}"
  fi
  if [[ "${#nodeps_deps[@]}" -gt 0 ]]; then
    docker run --rm --network host \
      -v "$deps_dir:/deps" \
      --entrypoint python3 \
      "$image" \
      -m pip install --disable-pip-version-check --no-cache-dir --no-deps \
      --target /deps "${nodeps_deps[@]}"
  fi
  extra_mount_args+=(-v "$deps_dir:/researchdeps:ro")
  extra_env+=(--env PYTHONPATH=/researchdeps)
fi

if [[ -n "$spec_model" && ! -f "$draft_model_dir/config.json" ]]; then
  if [[ "${DOWNLOAD_DRAFT:-0}" == 1 ]]; then
    draft_model_id="${DRAFT_MODEL_ID:-LiquidAI/LFM2.5-350M}"
    mkdir -p "$draft_model_dir"
    echo "Downloading draft model $draft_model_id to $draft_model_dir"
    docker run --rm --network host \
      -v "$draft_model_dir:/model" \
      --env DRAFT_MODEL_ID="$draft_model_id" \
      --entrypoint python3 \
      "$image" \
      -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["DRAFT_MODEL_ID"], local_dir="/model", local_dir_use_symlinks=False)'
  else
    echo "DRAFT_MODEL_MISSING=$draft_model_dir"
    echo "Set DRAFT_MODEL_DIR to a local LFM2.5-350M checkpoint or DOWNLOAD_DRAFT=1 before running $case_name." >&2
    exit 2
  fi
fi

mkdir -p "$result_dir" "$workspace/hf-cache" "$engine_cache_dir"
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
  -v "$engine_cache_dir:/root/.cache" \
  "${extra_mount_args[@]}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_USE_FASTOKENS=1 \
  --env VLLM_CONFIGURE_LOGGING=0 \
  "${extra_env[@]}" \
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
