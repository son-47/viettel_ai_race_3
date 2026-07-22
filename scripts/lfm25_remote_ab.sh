#!/usr/bin/env bash
set -euo pipefail

# Controlled LFM2.5 A/B runner for the Lightning L4 testbed. Absolute latency
# does not predict the MiG H200 grader; use paired runs to rank configurations.

case_name="${1:-control}"
image="${IMAGE:-misokaio/ghfjdk:v0.25.1}"
workspace="${WORKSPACE:-/home/zeus/content}"
model_dir="${MODEL_DIR:-$workspace/model-lfm25}"
server_name="lfm25-ab-server"
rate_scale="${RATE_SCALE:-4}"
warmup="${WARMUP:-1}"
result_dir="$workspace/results/lfm25_ab"
mkdir -p "$result_dir"

max_model_len=32768
max_batched_tokens=8192
max_seqs=32
optimization_level=3
gpu_memory_utilization=0.85
prefix_caching=1
extra_args=()
extra_env=()

case "$case_name" in
  control|cold_control) ;;
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
  output64) extra_env+=(--env VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=64) ;;
  output256) extra_env+=(--env VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=256) ;;
  output512) extra_env+=(--env VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=512) ;;
  keepalive600) extra_env+=(--env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600) ;;
  hashxx) extra_args+=(--prefix-caching-hash-algo=xxhash) ;;
  mamba8) extra_args+=(--mamba-block-size=8) ;;
  mamba32) extra_args+=(--mamba-block-size=32) ;;
  o2) optimization_level=2 ;;
  gpu92) gpu_memory_utilization=0.92 ;;
  gpu95) gpu_memory_utilization=0.95 ;;
  pcoff) prefix_caching=0 ;;
  dbo) extra_args+=(--enable-dbo) ;;
  reserve_off) extra_args+=(--no-scheduler-reserve-full-isl) ;;
  exact_graphs)
    extra_args+=(--cudagraph-capture-sizes)
    for graph_size in $(seq 1 32); do extra_args+=("$graph_size"); done
    ;;
  pod_attention)
    extra_args+=('--attention-config={"use_prefill_decode_attention":true}')
    ;;
  kvfp8) extra_args+=(--kv-cache-dtype=fp8) ;;
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

echo "CASE=$case_name IMAGE=$image WARMUP=$warmup RATE_SCALE=$rate_scale"
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
  "${extra_env[@]}" \
  --entrypoint python3 \
  "$image" \
  -m vllm.entrypoints.openai.api_server \
  --model=/model \
  --quantization=fp8 \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 \
  --port=8000 \
  --max-model-len="$max_model_len" \
  --optimization-level="$optimization_level" \
  --gpu-memory-utilization="$gpu_memory_utilization" \
  --tensor-parallel-size=1 \
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
