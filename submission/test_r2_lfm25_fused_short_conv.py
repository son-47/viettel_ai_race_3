"""GPU exact-parity test for the guarded R2 ShortConv fusion."""

from __future__ import annotations

import argparse
import json

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.lfm25_fused_short_conv import (
    fused_lfm25_short_conv_decode,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def stock_step(b, c, x, state, weight, indices):
    gated = (b * x).contiguous()
    convolved = causal_conv1d_update(
        gated,
        state,
        weight,
        bias=None,
        activation=None,
        conv_state_indices=indices,
    )
    return c * convolved


def elapsed_us(operation, iterations: int) -> float:
    for _ in range(50):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def run_case(batch: int, state_dtype: torch.dtype, iterations: int) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260726 + batch)
    dim = 2048
    blocks = max(64, batch + 1)
    b = torch.randn(batch, dim, device=device, dtype=torch.bfloat16, generator=generator)
    c = torch.randn_like(b)
    x = torch.randn_like(b)
    weight = torch.randn(
        dim, 3, device=device, dtype=torch.bfloat16, generator=generator
    )
    state = torch.randn(
        blocks, dim, 2, device=device, dtype=state_dtype, generator=generator
    )
    indices = torch.arange(batch, device=device, dtype=torch.int32).unsqueeze(1)
    if batch > 1:
        indices[-1, 0] = NULL_BLOCK_ID

    reference_state = state.clone()
    fused_state = state.clone()
    reference_output = stock_step(b, c, x, reference_state, weight, indices)
    fused_output = fused_lfm25_short_conv_decode(
        b, c, x, fused_state, weight, indices
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)

    stock_state = state.clone()
    kernel_state = state.clone()
    stock_us = elapsed_us(
        lambda: stock_step(b, c, x, stock_state, weight, indices), iterations
    )
    fused_us = elapsed_us(
        lambda: fused_lfm25_short_conv_decode(
            b, c, x, kernel_state, weight, indices
        ),
        iterations,
    )
    return {
        "batch": batch,
        "state_dtype": str(state_dtype),
        "stock_us": stock_us,
        "fused_us": fused_us,
        "speedup": stock_us / fused_us,
        "exact_output_and_state": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = [
        run_case(batch, state_dtype, args.iterations)
        for state_dtype in (torch.bfloat16, torch.float32)
        for batch in args.batches
    ]
    print(json.dumps({"device": torch.cuda.get_device_name(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
