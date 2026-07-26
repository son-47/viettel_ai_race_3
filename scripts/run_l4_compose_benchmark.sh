#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/home/zeus/content}"
compose_dir="$workspace/compose"
override="$workspace/docker-compose_l4_override.yml"
result_dir="$workspace/results_l4"
rate_scale="${RATE_SCALE:-1}"
timeout_s="${TIMEOUT_S:-180}"

mkdir -p "$result_dir"

cases=(
  "base_63_94:docker-compose_63.94.yml"
  "shortconv_64_35:docker-compose_shortconv_fused_64.35.yml"
  "silu_fp8_candidate:docker-compose_silu_fp8_candidate.yml"
  "processing:docker-compose_processing.yml"
)

cleanup_case() {
  local project="$1"
  docker compose -p "$project" -f "$compose_dir/${2}" -f "$override" down --remove-orphans >/dev/null 2>&1 || true
}

for item in "${cases[@]}"; do
  label="${item%%:*}"
  file="${item#*:}"
  project="l4_${label}"
  compose=(docker compose -p "$project" -f "$compose_dir/$file" -f "$override")
  echo "L4_CASE=$label COMPOSE=$file"
  "${compose[@]}" down --remove-orphans >/dev/null 2>&1 || true
  "${compose[@]}" pull model
  "${compose[@]}" up -d model

  ready=0
  for attempt in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "SERVER_READY_SECONDS=$((attempt * 2))"
      ready=1
      break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${project}-model-1" 2>/dev/null || true)" != true ]]; then
      "${compose[@]}" logs --no-color model | tail -120 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    "${compose[@]}" logs --no-color model | tail -120 || true
    exit 1
  fi

  "${compose[@]}" run --rm --no-deps --entrypoint python3 model \
    /work/harness/benchmark_grading_spec.py \
    --tokenizer=/model \
    --base-url=http://model:8000 \
    --model=LFM2.5-1.2B-Instruct \
    --out=/results/${label}.json \
    --num-conversations=70 \
    --turns=6 \
    --rate-scale="$rate_scale" \
    --timeout="$timeout_s"

  "${compose[@]}" logs --no-color model >"$result_dir/${label}.server.log" 2>&1 || true
  "${compose[@]}" down --remove-orphans
done

python3 - "$result_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    summary = data["summary"]
    print(json.dumps({
        "case": path.stem,
        "ers": summary.get("ers"),
        "requests_successful": summary.get("requests_successful"),
        "ttft_p50_ms": summary.get("ttft_ms", {}).get("p50"),
        "ttft_p95_ms": summary.get("ttft_ms", {}).get("p95"),
        "tpot_p50_ms": summary.get("tpot_ms", {}).get("p50"),
        "tpot_p95_ms": summary.get("tpot_ms", {}).get("p95"),
        "duration_s": summary.get("duration_s"),
    }, sort_keys=True))
PY
