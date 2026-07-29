"""Record/compare deterministic outputs for the speculative-decode gate.

Run once against the non-speculative control with ``--mode record`` and once
against the speculative candidate with ``--mode compare``.  Every comparison
request reuses the exact control-side message history, so an early mismatch
cannot contaminate later prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import aiohttp


SYSTEM_PROMPT = (
    "You are a precise assistant. Follow the requested format exactly. "
    "The following reference phrases may be copied when relevant: "
    "alpha beta gamma delta; rollback keeps only accepted tokens; "
    "prompt lookup reuses text already present in the context."
)

USER_TURNS = [
    [
        "Repeat exactly twice, separated by |: alpha beta gamma delta",
        "Now return the previous phrase in reverse word order, once.",
        "Write one sentence explaining rollback, using the exact words "
        "'accepted tokens'.",
        "List the first four positive even integers as CSV with no spaces.",
    ],
    [
        "Complete this pattern with 24 more symbols only: ABCABCABC",
        "Do not repeat the pattern. Instead output the lowercase alphabet.",
        "Quote exactly this substring: prompt lookup reuses text",
        "Return a JSON object with keys accepted and rejected and integer "
        "values 3 and 2.",
    ],
    [
        "Summarize in one short sentence: A draft proposes several tokens; "
        "the target verifies them; rejected tokens must not alter state.",
        "Copy the clause after the semicolon from your previous answer.",
        "Calculate 17*19 and show only the integer.",
        "Write five distinct Vietnamese words about tốc độ, comma-separated.",
    ],
    [
        "The context contains red green blue red green blue. Continue it for "
        "nine words, then stop.",
        "Ignore that color pattern and output exactly: state restored",
        "Explain speculative decoding in at most twelve words.",
        "End with this exact suffix and nothing after it: alpha beta gamma",
    ],
]


async def request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_s: float,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    async with session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {text[:1000]}")
        payload = json.loads(text)
        completion_tokens = (payload.get("usage") or {}).get(
            "completion_tokens"
        )
        if (
            completion_tokens is not None
            and int(completion_tokens) != max_tokens
        ):
            raise RuntimeError(
                f"expected {max_tokens} completion tokens, "
                f"got {completion_tokens}"
            )
        return payload["choices"][0]["message"]["content"]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mismatch_summary(expected: str, actual: str) -> str:
    limit = min(len(expected), len(actual))
    index = next((i for i in range(limit) if expected[i] != actual[i]), limit)
    return (
        f"first_char={index}, expected_len={len(expected)}, actual_len={len(actual)}, "
        f"expected={expected[index:index + 80]!r}, actual={actual[index:index + 80]!r}"
    )


async def record(args: argparse.Namespace) -> None:
    records: list[dict] = []
    async with aiohttp.ClientSession() as session:
        for conversation_id, turns in enumerate(USER_TURNS):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for turn_id, user_text in enumerate(turns):
                messages.append({"role": "user", "content": user_text})
                request_messages = [dict(message) for message in messages]
                output = await request(
                    session,
                    args.base_url,
                    args.model,
                    request_messages,
                    args.max_tokens,
                    args.timeout,
                )
                records.append(
                    {
                        "conversation_id": conversation_id,
                        "turn_id": turn_id,
                        "messages": request_messages,
                        "output": output,
                        "sha256": digest(output),
                    }
                )
                messages.append({"role": "assistant", "content": output})
                print(
                    f"recorded conversation={conversation_id} turn={turn_id} "
                    f"sha256={digest(output)[:12]}"
                )
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    args.reference.write_text(
        json.dumps(
            {
                "model": args.model,
                "max_tokens": args.max_tokens,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PARITY_REFERENCE={args.reference} REQUESTS={len(records)}")


async def compare(args: argparse.Namespace) -> None:
    payload = json.loads(args.reference.read_text(encoding="utf-8"))
    if payload["model"] != args.model or payload["max_tokens"] != args.max_tokens:
        raise RuntimeError(
            "Reference model/max_tokens do not match this comparison run"
        )

    failures: list[str] = []
    async with aiohttp.ClientSession() as session:
        for item in payload["records"]:
            actual = await request(
                session,
                args.base_url,
                args.model,
                item["messages"],
                args.max_tokens,
                args.timeout,
            )
            label = (
                f"conversation={item['conversation_id']} turn={item['turn_id']}"
            )
            if actual != item["output"]:
                failures.append(
                    f"{label}: {mismatch_summary(item['output'], actual)}"
                )
                print(f"MISMATCH {failures[-1]}")
            else:
                print(f"matched {label} sha256={digest(actual)[:12]}")

    print(
        f"PARITY_REQUESTS={len(payload['records'])} "
        f"PARITY_FAILURES={len(failures)}"
    )
    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("record", "compare"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/speculative_parity_reference.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(record(args) if args.mode == "record" else compare(args))


if __name__ == "__main__":
    main()
