"""Pre-serve kernel warmup for the fixed AI Race LFM2.5 workload."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time

from vllm import SamplingParams
from vllm.inputs import TokensPrompt


logger = logging.getLogger("vllm.entrypoints.openai.api_server")


async def warmup_engine(engine_client) -> None:
    """Prime concurrent prefill/decode shapes before the HTTP server is ready."""
    setting = os.environ.get("AI_RACE_WARMUP", "32")
    if not setting or setting.lower() == "off":
        logger.info("AI Race pre-serve warmup disabled")
        return

    count = int(setting)
    if count < 1 or count > 64:
        raise ValueError("AI_RACE_WARMUP must be in [1, 64] or 'off'")

    rng = random.Random(2026)
    prompts = []
    for index in range(count):
        prompt_length = 128 + (index * 67) % 2048
        prompts.append(
            [rng.randrange(32, 65_000) for _ in range(prompt_length)]
        )

    async def generate(index: int) -> None:
        output_length = 8 * (1 + index % 4)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=output_length,
            min_tokens=output_length,
            ignore_eos=True,
        )
        async for _ in engine_client.generate(
            TokensPrompt(prompt_token_ids=prompts[index]),
            sampling_params,
            f"ai-race-warmup-{index}",
        ):
            pass

    started = time.monotonic()
    await asyncio.gather(*(generate(index) for index in range(count)))
    cache_reset = await engine_client.reset_prefix_cache()
    logger.info(
        "AI Race pre-serve warmup finished: requests=%d seconds=%.2f "
        "prefix_cache_reset=%s",
        count,
        time.monotonic() - started,
        cache_reset,
    )
