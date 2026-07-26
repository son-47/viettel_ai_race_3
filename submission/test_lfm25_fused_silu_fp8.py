"""GPU correctness and latency test for fused LFM2.5 SwiGLU/FP8 quantization."""

from __future__ import annotations

import argparse
import json
import re
import subprocess

import torch
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as triton_compile
from vllm import _custom_ops as ops
from vllm.model_executor.layers.lfm25_fused_silu_fp8 import (
    _fused_lfm25_silu_math_for_test,
    _lfm25_fused_silu_fp8_kernel,
    fused_lfm25_silu_fp8_quant,
)
from vllm.triton_utils import tl

INTERMEDIATE_SIZE = 8192


def compile_sm90() -> dict:
    """Compile the production BF16/E4M3 kernel for Hopper without running it."""
    source = ASTSource(
        _lfm25_fused_silu_fp8_kernel,
        {
            "input_ptr": "*bf16",
            "output_ptr": "*fp8e4nv",
            "scale_ptr": "*fp32",
        },
        {
            "input_stride_token": 2 * INTERMEDIATE_SIZE,
            "output_stride_token": INTERMEDIATE_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "BLOCK_SIZE": INTERMEDIATE_SIZE,
            "INPUT_DTYPE": tl.bfloat16,
            "TEST_MATH_ONLY": False,
        },
    )
    compiled = triton_compile(
        source,
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 8, "num_stages": 2},
    )
    target_match = re.search(r"^\.target\s+(\S+)", compiled.asm["ptx"], re.MULTILINE)
    if target_match is None:
        raise RuntimeError("could not determine PTX target")
    ptxas = subprocess.run(
        [
            "/usr/local/cuda/bin/ptxas",
            "-v",
            f"-arch={target_match.group(1)}",
            "-o",
            "/dev/null",
            "-",
        ],
        input=compiled.asm["ptx"],
        text=True,
        capture_output=True,
        check=True,
    )
    register_match = re.search(r"Used (\d+) registers", ptxas.stderr)
    spill_store_match = re.search(r"(\d+) bytes spill stores", ptxas.stderr)
    spill_load_match = re.search(r"(\d+) bytes spill loads", ptxas.stderr)
    if register_match is None or spill_store_match is None or spill_load_match is None:
        raise RuntimeError(f"unexpected ptxas resource output: {ptxas.stderr}")
    return {
        "target": "sm90",
        "kernel": compiled.metadata.name,
        "shared_memory_bytes": compiled.metadata.shared,
        "registers": int(register_match.group(1)),
        "spill_store_bytes": int(spill_store_match.group(1)),
        "spill_load_bytes": int(spill_load_match.group(1)),
        "hash": compiled.hash,
    }


def run_math_case(token_count: int, dtype: torch.dtype) -> dict:
    """Validate all fused math before the Hopper-only E4M3 output conversion."""
    generator = torch.Generator(device="cuda").manual_seed(20260726 + token_count)
    gate_up = torch.randn(
        token_count,
        2 * INTERMEDIATE_SIZE,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    reference = torch.empty(
        (token_count, INTERMEDIATE_SIZE), device="cuda", dtype=dtype
    )
    torch.ops._C.silu_and_mul(reference, gate_up)
    _, reference_scale = ops.scaled_fp8_quant(
        reference,
        scale=None,
        use_per_token_if_dynamic=True,
    )
    fused, fused_scale = _fused_lfm25_silu_math_for_test(gate_up)
    torch.cuda.synchronize()

    atol = 0.0625 if dtype == torch.bfloat16 else 0.0078125
    torch.testing.assert_close(fused, reference, rtol=0, atol=atol)
    torch.testing.assert_close(fused_scale, reference_scale, rtol=2e-3, atol=1e-7)
    return {
        "tokens": token_count,
        "dtype": str(dtype),
        "max_activation_abs_error": float((fused - reference).abs().max().item()),
        "max_scale_abs_error": float(
            (fused_scale - reference_scale).abs().max().item()
        ),
    }


def elapsed_us(operation, iterations: int) -> float:
    for _ in range(30):
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
    generator = torch.Generator(device="cuda").manual_seed(20260726 + token_count)
    gate_up = torch.randn(
        token_count,
        2 * INTERMEDIATE_SIZE,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )

    def stock():
        activated = torch.empty(
            (token_count, INTERMEDIATE_SIZE), device="cuda", dtype=dtype
        )
        torch.ops._C.silu_and_mul(activated, gate_up)
        return ops.scaled_fp8_quant(
            activated,
            scale=None,
            use_per_token_if_dynamic=True,
        )

    def fused():
        return fused_lfm25_silu_fp8_quant(gate_up)

    reference_q, reference_scale = stock()
    fused_q, fused_scale = fused()
    torch.cuda.synchronize()

    torch.testing.assert_close(fused_scale, reference_scale, rtol=2e-3, atol=1e-7)
    reference_dequant = reference_q.float() * reference_scale
    fused_dequant = fused_q.float() * fused_scale
    torch.testing.assert_close(
        fused_dequant,
        reference_dequant,
        rtol=0,
        atol=0.0625,
    )
    mismatch_rate = float((fused_q != reference_q).float().mean().item())

    stock_us = elapsed_us(stock, iterations)
    fused_us = elapsed_us(fused, iterations)
    return {
        "tokens": token_count,
        "dtype": str(dtype),
        "stock_us": stock_us,
        "fused_us": fused_us,
        "speedup": stock_us / fused_us,
        "fp8_mismatch_rate": mismatch_rate,
        "max_scale_abs_error": float(
            (fused_scale - reference_scale).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--compile-only-sm90", action="store_true")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    args = parser.parse_args()
    compile_result = compile_sm90()
    if args.compile_only_sm90:
        print(json.dumps({"compile": compile_result}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 9:
        math_results = []
        for dtype in (torch.bfloat16, torch.float16):
            for token_count in args.tokens:
                math_results.append(run_math_case(token_count, dtype))
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "compile": compile_result,
                    "math_results": math_results,
                    "runtime_skipped": (
                        "E4M3 stores require SM90+; fused activation/scale math "
                        "passed here, but run the final FP8/latency test on H100/H200"
                    ),
                },
                indent=2,
            )
        )
        return

    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for token_count in args.tokens:
            results.append(run_case(token_count, dtype, args.iterations))
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "compile": compile_result,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
