#!/usr/bin/env bash
set -euo pipefail

# Controlled LFM2.5 A/B runner for a Lightning proxy GPU. Absolute latency
# does not predict the MiG H200 grader; use paired runs to rank configurations.

case_name="${1:-control}"
image="${IMAGE:-misokaio/ghfjdk:v0.25.1}"
workspace="${WORKSPACE:-/home/zeus/content}"
model_dir="${MODEL_DIR:-$workspace/model-lfm25}"
draft_model_dir="${DRAFT_MODEL_DIR:-$workspace/model-lfm25-350m}"
server_name="lfm25-ab-server"
rate_scale="${RATE_SCALE:-4}"
warmup="${WARMUP:-1}"
result_dir="${RESULT_DIR:-$workspace/results/lfm25_ab}"
weight_mode="${WEIGHT_MODE:-fp8}"
engine_cache_dir="${ENGINE_CACHE_DIR:-$workspace/vllm-cache}"
mkdir -p "$result_dir" "$engine_cache_dir"

max_model_len=32768
max_batched_tokens=8192
max_seqs=32
optimization_level=3
gpu_memory_utilization=0.85
prefix_caching=1
renderer_workers=1
fastokens=1
extra_args=()
extra_env=()
extra_mount_args=()
extra_deps=()
spec_model=""
spec_tokens=""
case "$weight_mode" in
  fp8) quant_args=(--quantization=fp8) ;;
  bf16) quant_args=() ;;
  *)
    echo "Unsupported WEIGHT_MODE=$weight_mode (expected fp8 or bf16)" >&2
    exit 2
    ;;
esac

case "$case_name" in
  control|cold_control) ;;
  fp8_per_tensor) quant_args=(--quantization=fp8_per_tensor) ;;
  fp8_per_block) quant_args=(--quantization=fp8_per_block) ;;
  fp8_per_channel) quant_args=(--quantization=fp8_per_channel) ;;
  int8_weight_only) quant_args=(--quantization=int8_per_channel_weight_only) ;;
  mxfp8_weight) quant_args=(--quantization=mxfp8) ;;
  maxlen8192) max_model_len=8192 ;;
  b6144) max_batched_tokens=6144 ;;
  b7168) max_batched_tokens=7168 ;;
  b9216) max_batched_tokens=9216 ;;
  b10240) max_batched_tokens=10240 ;;
  b12288) max_batched_tokens=12288 ;;
  b16384) max_batched_tokens=16384 ;;
  s40) max_seqs=40 ;;
  s48) max_seqs=48 ;;
  s64) max_seqs=64 ;;
  block8) extra_args+=(--block-size=8) ;;
  block32) extra_args+=(--block-size=32) ;;
  stream2) extra_args+=(--stream-interval=2) ;;
  stream4) extra_args+=(--stream-interval=4) ;;
  throughput) extra_args+=(--performance-mode=throughput) ;;
  api2) extra_args+=(--api-server-count=2) ;;
  frontend_inproc) extra_args+=(--disable-frontend-multiprocessing) ;;
  rust_frontend) extra_env+=(--env VLLM_USE_RUST_FRONTEND=1) ;;
  renderer2) renderer_workers=2 ;;
  renderer3) renderer_workers=3 ;;
  fastokens_off) fastokens=0 ;;
  output64) extra_env+=(--env VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=64) ;;
  output256) extra_env+=(--env VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=256) ;;
  output512) extra_env+=(--env VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=512) ;;
  keepalive600) extra_env+=(--env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600) ;;
  hashxx) extra_args+=(--prefix-caching-hash-algo=xxhash) ;;
  mamba8) extra_args+=(--mamba-block-size=8) ;;
  mamba32) extra_args+=(--mamba-block-size=32) ;;
  mamba64) extra_args+=(--mamba-block-size=64) ;;
  mamba_fp16) extra_args+=(--mamba-cache-dtype=float16) ;;
  mamba_fp16_sr)
    extra_args+=(
      --mamba-cache-dtype=float16
      --mamba-ssm-cache-dtype=float16
      --enable-mamba-cache-stochastic-rounding
    )
    ;;
  mamba_flashinfer) extra_args+=(--mamba-backend=flashinfer) ;;
  mamba_fp16_sr_flashinfer)
    extra_args+=(
      --mamba-backend=flashinfer
      --mamba-cache-dtype=float16
      --mamba-ssm-cache-dtype=float16
      --enable-mamba-cache-stochastic-rounding
    )
    ;;
  o2) optimization_level=2 ;;
  gpu92) gpu_memory_utilization=0.92 ;;
  gpu95) gpu_memory_utilization=0.95 ;;
  pcoff) prefix_caching=0 ;;
  dbo) extra_args+=(--enable-dbo) ;;
  reserve_off) extra_args+=(--no-scheduler-reserve-full-isl) ;;
  cascade) extra_args+=(--no-disable-cascade-attn) ;;
  hybrid_manager_off) extra_args+=(--disable-hybrid-kv-cache-manager) ;;
  v2_runner) extra_env+=(--env VLLM_USE_V2_MODEL_RUNNER=1) ;;
  spinloop) extra_env+=(--env VLLM_USE_SPINLOOP_EXT=1) ;;
  ssm_ds) extra_env+=(--env VLLM_SSM_CONV_STATE_LAYOUT=DS) ;;
  exact_graphs)
    extra_args+=(--cudagraph-capture-sizes)
    for graph_size in $(seq 1 32); do extra_args+=("$graph_size"); done
    ;;
  pod_attention)
    extra_args+=('--attention-config={"use_prefill_decode_attention":true}')
    ;;
  kvfp8) extra_args+=(--kv-cache-dtype=fp8) ;;
  kvint8) extra_args+=(--kv-cache-dtype=int8_per_token_head) ;;
  kvfp8_per_token_head) extra_args+=(--kv-cache-dtype=fp8_per_token_head) ;;
  kvturbo3) extra_args+=(--kv-cache-dtype=turboquant_3bit_nc) ;;
  kvturbo4) extra_args+=(--kv-cache-dtype=turboquant_4bit_nc) ;;
  gptq_w4)
    model_dir="$workspace/quant-models/gptq-w4a16-g128"
    quant_args=()
    ;;
  awq_w4)
    model_dir="$workspace/quant-models/awq-w4a16-asym"
    quant_args=()
    ;;
  ngram4)
    extra_args+=('--speculative-config={"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}')
    ;;
  ngram8)
    extra_args+=('--speculative-config={"method":"ngram","num_speculative_tokens":8,"prompt_lookup_min":2,"prompt_lookup_max":5}')
    ;;
  ngram_gpu2)
    extra_args+=('--speculative-config={"method":"ngram_gpu","num_speculative_tokens":2,"prompt_lookup_min":2,"prompt_lookup_max":3}')
    ;;
  ngram_gpu3)
    extra_args+=('--speculative-config={"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":3}')
    ;;
  suffix8)
    extra_deps+=(arctic-inference)
    extra_args+=('--speculative-config={"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_cached_requests":10000,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}')
    ;;
  suffix16)
    extra_deps+=(arctic-inference)
    extra_args+=('--speculative-config={"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_tree_depth":32,"suffix_decoding_max_cached_requests":10000,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}')
    ;;
  suffix32)
    extra_deps+=(arctic-inference)
    extra_args+=('--speculative-config={"method":"suffix","num_speculative_tokens":32,"suffix_decoding_max_tree_depth":48,"suffix_decoding_max_cached_requests":10000,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}')
    ;;
  suffix16_loose)
    extra_deps+=(arctic-inference)
    extra_args+=('--speculative-config={"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_tree_depth":32,"suffix_decoding_max_cached_requests":10000,"suffix_decoding_max_spec_factor":1.5,"suffix_decoding_min_token_prob":0.05}')
    ;;
  bnb4)
    extra_deps+=("bitsandbytes>=0.49.2")
    quant_args=(--quantization=bitsandbytes)
    ;;
  draft350m2|draft350m4|draft350m6|draft350m8)
    spec_model="/draft-model"
    spec_tokens="${case_name#draft350m}"
    extra_mount_args+=(-v "$draft_model_dir:/draft-model:ro")
    extra_args+=("--speculative-config={\"method\":\"draft_model\",\"model\":\"$spec_model\",\"num_speculative_tokens\":$spec_tokens,\"draft_tensor_parallel_size\":1}")
    ;;
  bnb4_suffix16)
    extra_deps+=("bitsandbytes>=0.49.2" arctic-inference)
    quant_args=(--quantization=bitsandbytes)
    extra_args+=('--speculative-config={"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_tree_depth":32,"suffix_decoding_max_cached_requests":10000,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}')
    ;;
  partial2)
    extra_args+=(--max-num-partial-prefills=2 --max-long-partial-prefills=1 --long-prefill-token-threshold=512)
    ;;
  flashinfer) extra_env+=(--env VLLM_ATTENTION_BACKEND=FLASHINFER) ;;
  *)
    echo "Unknown case: $case_name" >&2
    exit 2
    ;;
esac

if [[ "$case_name" == cold_control ]]; then
  warmup=0
fi

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
  # Install optional research-only dependencies into a mounted directory so
  # the pinned vLLM image remains unchanged and reproducible.
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

cleanup() {
  docker rm -f "$server_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

prefix_args=(--enable-prefix-caching)
if [[ "$prefix_caching" == 0 ]]; then
  prefix_args=(--no-enable-prefix-caching)
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$case_name-r${rate_scale}"
result_file="$result_dir/$run_id.json"
server_log="$result_dir/$run_id.server.log"

echo "CASE=$case_name IMAGE=$image WEIGHT_MODE=$weight_mode WARMUP=$warmup RATE_SCALE=$rate_scale"
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
  --env VLLM_USE_FASTOKENS="$fastokens" \
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
  --max-model-len="$max_model_len" \
  --optimization-level="$optimization_level" \
  --gpu-memory-utilization="$gpu_memory_utilization" \
  --tensor-parallel-size=1 \
  --renderer-num-workers="$renderer_workers" \
  "${prefix_args[@]}" \
  --enable-chunked-prefill \
  --max-num-batched-tokens="$max_batched_tokens" \
  --max-num-seqs="$max_seqs" \
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

run_client() {
  docker run --rm \
    --network host \
    --cpus=2 \
    -v "$model_dir:/model:ro" \
    -v "$workspace:/work:ro" \
    -v "$result_dir:/results" \
    --entrypoint python3 \
    "$image" \
    /work/benchmark_grading_spec.py \
    --tokenizer=/model \
    --base-url=http://127.0.0.1:8000 \
    --model=LFM2.5-1.2B-Instruct \
    "$@"
}

if [[ "$warmup" == 1 ]]; then
  # Prime concurrent prefill/decode shapes with unrelated natural-language
  # content. It cannot prefix-hit the token-exact measured workload.
  docker run --rm \
    --network host \
    --cpus=2 \
    -v "$workspace:/work:ro" \
    -v "$result_dir:/results" \
    --entrypoint python3 \
    "$image" \
    /work/benchmark_ngram_ab.py \
    --base-url=http://127.0.0.1:8000 \
    --model=LFM2.5-1.2B-Instruct \
    --out=/results/$run_id.warmup.json \
    --conversations=32 \
    --warmup-conversations=0 \
    --turns=1 \
    --output-tokens=300 \
    --arrival-rate=1000 \
    --think-time=0 \
    --timeout=120 >/dev/null
  echo "WARMUP_DONE"
fi

run_client \
  --out="/results/$run_id.json" \
  --num-conversations=70 \
  --turns=6 \
  --conversation-rate=0.228 \
  --rate-scale="$rate_scale" \
  --timeout=180

curl -fsS http://127.0.0.1:8000/metrics >"$result_dir/$run_id.metrics" || true
docker logs "$server_name" >"$server_log" 2>&1 || true
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader
echo "RESULT=$result_file"
echo "SERVER_LOG=$server_log"
