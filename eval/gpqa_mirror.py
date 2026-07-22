"""GPQA-Diamond accuracy via ungated mirror (fingertap/GPQA-Diamond).

Purpose: Δ-gate for quantization (BF16 vs FP8) when the official gated set
(Idavidrein/gpqa) is not yet accessible. Prompts are identical across endpoints,
so the DELTA is trustworthy even if absolute acc differs slightly from the
official task (choice order is the mirror's fixed a-d, not lm-eval's shuffle).

Usage:
  .venv/bin/python eval/gpqa_mirror.py --url http://localhost:8000 --n 100 \
      --out results/gpqa/bf16_mirror.json
"""
from __future__ import annotations
import argparse, asyncio, json, re, os
import aiohttp
from datasets import load_dataset

PROMPT = (
    "{q}\n\n"
    "Think step by step, then finish with exactly one line in the form:\n"
    "The answer is (X)\n"
    "where X is one of A, B, C, D."
)

def extract(text: str) -> str | None:
    if not text:
        return None
    m = list(re.finditer(r"answer\s+is\s*\(?\s*([A-Da-d])\s*\)?", text))
    if m:
        return m[-1].group(1).upper()
    m = list(re.finditer(r"\(([A-Da-d])\)", text))
    if m:
        return m[-1].group(1).upper()
    m = list(re.finditer(r"\b([A-D])\b", text[-200:]))
    if m:
        return m[-1].group(1)
    return None

async def one(sem, session, url, model, row, max_tokens):
    async with sem:
        payload = {"model": model,
                   "messages": [{"role": "user", "content": PROMPT.format(q=row["question"])}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
        try:
            async with session.post(f"{url}/v1/chat/completions", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=900)) as r:
                d = await r.json()
                out = d["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            return {"pred": None, "gold": row["answer"].strip().upper(), "err": repr(e)}
        return {"pred": extract(out), "gold": row["answer"].strip().upper(),
                "len": len(out or "")}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.5-2B")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    if args.start < 0 or args.start >= len(ds):
        raise ValueError(f"--start must be in [0, {len(ds) - 1}]")
    end = min(args.start + args.n, len(ds))
    rows = [ds[i] for i in range(args.start, end)]
    sem = asyncio.Semaphore(args.conc)
    async with aiohttp.ClientSession() as s:
        res = await asyncio.gather(*[one(sem, s, args.url, args.model, r, args.max_tokens)
                                     for r in rows])
    n = len(res)
    correct = sum(1 for r in res if r["pred"] == r["gold"])
    unparsed = sum(1 for r in res if r["pred"] is None)
    errs = sum(1 for r in res if r.get("err"))
    acc = correct / n
    summary = {"start": args.start, "n": n, "correct": correct, "acc": acc,
               "unparsed": unparsed, "request_errors": errs}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "results": res}, open(args.out, "w"), indent=1)
    print(json.dumps(summary))

if __name__ == "__main__":
    asyncio.run(main())
