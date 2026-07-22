"""Replay the published 420-request Viettel AI Race grading workload.

The private prompts and exact Poisson rate are not published.  This benchmark
reproduces the declared token structure, multi-turn dependency, output length,
and seeded Poisson conversation arrivals.  It is intended for controlled A/B
tests; absolute latency does not predict the MiG H200 grader.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


def latency_score(value_ms: float, floor_ms: float, ceiling_ms: float) -> float:
    normalized = min(
        1.0,
        max(0.0, (ceiling_ms - value_ms) / (ceiling_ms - floor_ms)),
    )
    return normalized**2


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - index)
        + ordered[upper] * (index - lower)
    )


def find_stable_token_unit(tokenizer) -> str:
    for unit in (" x", " the", " context", " data", " z", " 0"):
        if len(tokenizer.encode(unit, add_special_tokens=False)) != 1:
            continue
        if len(tokenizer.encode(unit * 64, add_special_tokens=False)) == 64:
            return unit
    raise RuntimeError("Could not find a stable one-token filler unit")


def exact_token_text(tokenizer, header: str, target_tokens: int, unit: str) -> str:
    header_tokens = len(tokenizer.encode(header, add_special_tokens=False))
    estimate = max(0, target_tokens - header_tokens)
    for count in range(max(0, estimate - 16), estimate + 17):
        text = header + unit * count
        if len(tokenizer.encode(text, add_special_tokens=False)) == target_tokens:
            return text
    raise RuntimeError(
        f"Could not construct {target_tokens} tokens for header {header!r}"
    )


async def request_turn(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    prompt_tokens: int,
    output_tokens_pinned: int,
    timeout_s: float,
) -> tuple[dict, str]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": output_tokens_pinned,
        "min_tokens": output_tokens_pinned,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = None
    server_prompt_tokens = None
    completion_tokens = 0
    response_parts: list[str] = []
    error = None

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as response:
            if response.status != 200:
                error = f"HTTP {response.status}: {(await response.text())[:500]}"
            else:
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8", "ignore")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        event = json.loads(payload)
                        usage = event.get("usage") or {}
                        if usage.get("prompt_tokens") is not None:
                            server_prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            completion_tokens = int(usage["completion_tokens"])
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        content = (choices[0].get("delta") or {}).get("content")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            response_parts.append(content)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ended = time.perf_counter()
    if first_token_at is None and error is None:
        error = "no output token"
    if completion_tokens != output_tokens_pinned and error is None:
        error = f"expected {output_tokens_pinned} output tokens, got {completion_tokens}"

    ttft_ms = (
        (first_token_at - started) * 1000.0
        if first_token_at is not None
        else None
    )
    tpot_ms = (
        (ended - first_token_at) * 1000.0 / (completion_tokens - 1)
        if first_token_at is not None and completion_tokens > 1
        else None
    )
    return (
        {
            "prompt_tokens": server_prompt_tokens or prompt_tokens,
            "locally_counted_prompt_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": (ended - started) * 1000.0,
            "error": error,
        },
        "".join(response_parts),
    )


async def run(args: argparse.Namespace) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    unit = find_stable_token_unit(tokenizer)
    shared_system = exact_token_text(
        tokenizer,
        "Shared production system context:",
        args.shared_system_tokens,
        unit,
    )
    conversation_prefixes = [
        exact_token_text(
            tokenizer,
            f"Private context for conversation {conversation_id:04d}:",
            args.conversation_prefix_tokens,
            unit,
        )
        for conversation_id in range(args.num_conversations)
    ]
    user_turns = [
        exact_token_text(
            tokenizer,
            f"User turn {turn_index + 1}: provide the next detailed response.",
            args.new_user_tokens,
            unit,
        )
        for turn_index in range(args.turns)
    ]

    random_generator = random.Random(args.seed)
    arrival_offsets = [0.0]
    for _ in range(1, args.num_conversations):
        arrival_offsets.append(
            arrival_offsets[-1]
            + random_generator.expovariate(args.conversation_rate)
            / args.rate_scale
        )

    connector = aiohttp.TCPConnector(limit=args.num_conversations)
    results: list[dict] = []
    result_lock = asyncio.Lock()
    run_started = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:

        async def replay_conversation(conversation_id: int) -> None:
            await asyncio.sleep(
                max(
                    0.0,
                    run_started
                    + arrival_offsets[conversation_id]
                    - time.perf_counter(),
                )
            )
            messages = [{"role": "system", "content": shared_system}]
            for turn_index in range(args.turns):
                user_content = user_turns[turn_index]
                if turn_index == 0:
                    user_content = (
                        conversation_prefixes[conversation_id] + user_content
                    )
                messages.append({"role": "user", "content": user_content})
                templated = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                input_ids = (
                    templated["input_ids"]
                    if isinstance(templated, dict)
                    else templated
                )
                if input_ids and isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                prompt_tokens = len(input_ids)
                result, assistant_text = await request_turn(
                    session,
                    args.base_url,
                    args.model,
                    messages,
                    prompt_tokens,
                    args.output_tokens,
                    args.timeout,
                )
                result.update(
                    {
                        "conversation_id": conversation_id,
                        "turn_index": turn_index,
                        "arrival_offset_s": arrival_offsets[conversation_id],
                        "output_sha256": hashlib.sha256(
                            assistant_text.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                async with result_lock:
                    results.append(result)
                if result["error"] is not None:
                    break
                messages.append({"role": "assistant", "content": assistant_text})
                if turn_index + 1 < args.turns and args.think_time > 0:
                    await asyncio.sleep(args.think_time / args.rate_scale)

        await asyncio.gather(
            *(replay_conversation(i) for i in range(args.num_conversations))
        )

    results.sort(key=lambda item: (item["conversation_id"], item["turn_index"]))
    successful = [item for item in results if item["error"] is None]
    ttfts = [float(item["ttft_ms"]) for item in successful]
    tpots = [float(item["tpot_ms"]) for item in successful]
    per_request_scores = []
    ttft_zero_count = 0
    tpot_zero_count = 0
    for item in results:
        if item["error"] is not None:
            per_request_scores.append(0.0)
            continue
        ttft_score = latency_score(float(item["ttft_ms"]), 10.0, 400.0)
        tpot_score = latency_score(float(item["tpot_ms"]), 1.0, 10.0)
        ttft_zero_count += ttft_score == 0.0
        tpot_zero_count += tpot_score == 0.0
        per_request_scores.append(0.5 * (ttft_score + tpot_score))

    expected_requests = args.num_conversations * args.turns
    per_request_scores.extend([0.0] * (expected_requests - len(results)))
    summary = {
        "note": "Synthetic token-exact A/B workload; not a private-grader score.",
        "config": vars(args),
        "duration_s": time.perf_counter() - run_started,
        "requests_expected": expected_requests,
        "requests_completed": len(results),
        "requests_successful": len(successful),
        "requests_zero_total_score": sum(score == 0.0 for score in per_request_scores),
        "ttft_zero_count": ttft_zero_count,
        "tpot_zero_count": tpot_zero_count,
        "ers": statistics.fmean(per_request_scores),
        "ttft_ms": {
            "mean": statistics.fmean(ttfts) if ttfts else None,
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
        },
        "tpot_ms": {
            "mean": statistics.fmean(tpots) if tpots else None,
            "p50": percentile(tpots, 0.50),
            "p95": percentile(tpots, 0.95),
        },
        "prompt_tokens": {
            "min": min((item["prompt_tokens"] for item in results), default=None),
            "mean": (
                statistics.fmean(item["prompt_tokens"] for item in results)
                if results
                else None
            ),
            "max": max((item["prompt_tokens"] for item in results), default=None),
        },
        "errors": [item for item in results if item["error"] is not None][:20],
    }
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-conversations", type=int, default=70)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--shared-system-tokens", type=int, default=1000)
    parser.add_argument("--conversation-prefix-tokens", type=int, default=1000)
    parser.add_argument("--new-user-tokens", type=int, default=150)
    parser.add_argument("--output-tokens", type=int, default=300)
    parser.add_argument("--conversation-rate", type=float, default=0.228)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--think-time", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    output = asyncio.run(run(args))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
