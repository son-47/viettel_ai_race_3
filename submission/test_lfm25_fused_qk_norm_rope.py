"""GPU correctness and latency smoke test for LFM2.5 QK fusion."""

from __future__ import annotations

import argparse
import json

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.lfm25_fused_qk_norm_rope import (
    fused_lfm25_qk_rmsnorm_rope,
)


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM


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


def run_case(token_count: int, dtype: torch.dtype, iterations: int) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260726 + token_count)
    packed = torch.randn(
        token_count,
        Q_SIZE + 2 * KV_SIZE,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    q, k, _ = packed.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
    positions = torch.arange(token_count, device=device, dtype=torch.int64) + 4000

    eps = 1e-5
    q_weight = torch.randn(
        HEAD_DIM, device=device, dtype=dtype, generator=generator
    ).to(device)
    k_weight = torch.randn(
        HEAD_DIM, device=device, dtype=dtype, generator=generator
    ).to(device)

    # Reproduce RotaryEmbedding._compute_cos_sin_cache for full NeoX RoPE.
    inverse_frequency = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    frequencies = torch.einsum(
        "i,j -> ij", torch.arange(8192, dtype=torch.float32), inverse_frequency
    )
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(
        device=device, dtype=dtype
    )

    def stock():
        q_stock = q.view(token_count, NUM_Q_HEADS, HEAD_DIM).contiguous()
        k_stock = k.view(token_count, NUM_KV_HEADS, HEAD_DIM).contiguous()
        q_normalized = torch.empty_like(q_stock)
        k_normalized = torch.empty_like(k_stock)
        ops.rms_norm(q_normalized, q_stock, q_weight, eps)
        ops.rms_norm(k_normalized, k_stock, k_weight, eps)
        ops.rotary_embedding(
            positions,
            q_normalized,
            k_normalized,
            HEAD_DIM,
            cache,
            True,
        )
        return q_normalized.view(token_count, Q_SIZE), k_normalized.view(
            token_count, KV_SIZE
        )

    def fused():
        return fused_lfm25_qk_rmsnorm_rope(
            q,
            k,
            q_weight,
            k_weight,
            cache,
            positions,
            eps,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            HEAD_DIM,
        )

    reference_q, reference_k = stock()
    fused_q, fused_k = fused()
    torch.cuda.synchronize()
    # BF16 has a 2^-7 mantissa.  This bound catches indexing/rotation errors
    # while allowing a one-ULP difference from a different reduction tree.
    atol = 1.5625e-2 if dtype == torch.bfloat16 else 3.0e-3
    torch.testing.assert_close(fused_q, reference_q, rtol=0, atol=atol)
    torch.testing.assert_close(fused_k, reference_k, rtol=0, atol=atol)

    stock_us = elapsed_us(stock, iterations)
    fused_us = elapsed_us(fused, iterations)
    return {
        "tokens": token_count,
        "dtype": str(dtype),
        "stock_us": stock_us,
        "fused_us": fused_us,
        "speedup": stock_us / fused_us,
        "q_max_abs_error": (fused_q.float() - reference_q.float()).abs().max().item(),
        "k_max_abs_error": (fused_k.float() - reference_k.float()).abs().max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for token_count in args.tokens:
            results.append(run_case(token_count, dtype, args.iterations))
    print(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "results": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
