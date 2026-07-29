"""H200 microbenchmark for the stock LFM2.5 w2 + norm/quant boundary.

This establishes the exact baseline that a CUTLASS EVT/PDL implementation must
beat.  It intentionally benchmarks the kernels, not model loading or scheduling.
Run inside the vLLM 0.25.1 CUDA image on SM90.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable

import torch

from vllm import _custom_ops as ops


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
NORM_EPS = 1e-5
FP8_MAX = 448.0


def _measure_us(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    reset: Callable[[], None],
    warmup: int,
    iterations: int,
) -> tuple[float, float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    result = fn()
    for _ in range(iterations):
        reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)

    return statistics.median(samples), statistics.quantiles(samples, n=20)[18], result


def _stats_scale_reference(
    residual_out: torch.Tensor, norm_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_fp32 = residual_out.float()
    sumsq = residual_fp32.square().sum(dim=-1)
    weighted_amax = (residual_fp32 * norm_weight.float()).abs().amax(dim=-1)
    inv_rms = torch.rsqrt(sumsq / HIDDEN_SIZE + NORM_EPS)
    scale = (weighted_amax * inv_rms / FP8_MAX).unsqueeze(-1)
    return sumsq, weighted_amax, scale


def benchmark_batch(
    batch: int,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    activation = (
        torch.randn(
            (batch, INTERMEDIATE_SIZE), device="cuda", dtype=torch.bfloat16
        )
        * 0.05
    )
    activation_q, activation_scale = ops.scaled_fp8_quant(
        activation, use_per_token_if_dynamic=True
    )

    residual_seed = torch.randn(
        (batch, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
    )
    residual = residual_seed.clone()

    def reset() -> None:
        residual.copy_(residual_seed)

    def stock_pair() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w2_out = ops.cutlass_scaled_mm(
            activation_q,
            weight_q,
            scale_a=activation_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        output_q, output_scale = ops.rms_norm_dynamic_per_token_quant(
            w2_out,
            norm_weight,
            NORM_EPS,
            torch.float8_e4m3fn,
            residual=residual,
        )
        return output_q, output_scale, w2_out

    median_us, p95_us, (output_q, output_scale, _) = _measure_us(
        stock_pair, reset, warmup, iterations
    )
    torch.cuda.synchronize()

    _, _, derived_scale = _stats_scale_reference(residual, norm_weight)
    scale_abs_error = (derived_scale - output_scale).abs().max().item()
    scale_rel_error = (
        (derived_scale - output_scale).abs()
        / torch.maximum(output_scale.abs(), torch.tensor(1e-12, device="cuda"))
    ).max().item()

    return {
        "batch": batch,
        "median_us": median_us,
        "p95_us": p95_us,
        "derived_scale_max_abs_error": scale_abs_error,
        "derived_scale_max_rel_error": scale_rel_error,
        "output_checksum": int(output_q.view(torch.uint8).sum(dtype=torch.int64).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 9:
        raise RuntimeError(f"This benchmark requires SM90, got sm{capability[0]}{capability[1]}")

    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)

    weight_bf16 = (
        torch.randn(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    weight_q, weight_scale = ops.scaled_fp8_quant(weight_bf16)
    norm_weight = 1.0 + 0.05 * torch.randn(
        (HIDDEN_SIZE,), device="cuda", dtype=torch.bfloat16
    )

    results = [
        benchmark_batch(
            batch,
            weight_q,
            weight_scale,
            norm_weight,
            args.warmup,
            args.iterations,
        )
        for batch in args.batches
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": capability,
                "shape": {
                    "k": INTERMEDIATE_SIZE,
                    "n": HIDDEN_SIZE,
                    "epsilon": NORM_EPS,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

