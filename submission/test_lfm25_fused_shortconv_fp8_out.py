"""SM90 compile, parity, and latency test for ShortConv-to-FP8 fusion."""

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
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.lfm25_fused_shortconv_fp8_out import (
    _fused_lfm25_shortconv_math_for_test,
    _lfm25_fused_shortconv_fp8_out_kernel,
    fused_lfm25_shortconv_fp8_quant,
)
from vllm.triton_utils import tl
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


DIM = 2048


def compile_sm90() -> dict:
    source = ASTSource(
        _lfm25_fused_shortconv_fp8_out_kernel,
        {
            "b_ptr": "*bf16",
            "c_ptr": "*bf16",
            "x_ptr": "*bf16",
            "state_ptr": "*bf16",
            "weight_ptr": "*bf16",
            "state_indices_ptr": "*i32",
            "out_ptr": "*fp8e4nv",
            "scale_ptr": "*fp32",
        },
        {
            "stride_b_token": DIM,
            "stride_c_token": DIM,
            "stride_x_token": DIM,
            "stride_state_block": DIM * 2,
            "stride_state_dim": 2,
            "stride_state_token": 1,
            "stride_weight_dim": 3,
            "stride_indices": 1,
            "stride_out_token": DIM,
            "dim": DIM,
            "null_block_id": -1,
            "BLOCK_SIZE": DIM,
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
        raise RuntimeError(f"unexpected ptxas output: {ptxas.stderr}")
    return {
        "target": "sm90",
        "kernel": compiled.metadata.name,
        "shared_memory_bytes": compiled.metadata.shared,
        "registers": int(register_match.group(1)),
        "spill_store_bytes": int(spill_store_match.group(1)),
        "spill_load_bytes": int(spill_load_match.group(1)),
        "hash": compiled.hash,
    }


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


def run_case(batch: int, state_dtype: torch.dtype, iterations: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20260729 + batch)
    b = torch.randn(batch, DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    c = torch.randn_like(b)
    x = torch.randn_like(b)
    weight = torch.randn(
        DIM, 3, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    state = torch.randn(
        max(64, batch + 1),
        DIM,
        2,
        device="cuda",
        dtype=state_dtype,
        generator=generator,
    )
    indices = torch.arange(batch, device="cuda", dtype=torch.int32).unsqueeze(1)
    if batch > 1:
        indices[-1, 0] = NULL_BLOCK_ID

    reference_state = state.clone()
    fused_state = state.clone()
    reference = stock_step(b, c, x, reference_state, weight, indices)
    fused, scales = _fused_lfm25_shortconv_math_for_test(
        b, c, x, fused_state, weight, indices
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(fused, reference, rtol=0, atol=0)
    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)
    reference_scales = torch.clamp(
        reference.float().abs().amax(dim=1, keepdim=True) / 448.0,
        min=1.0 / (448.0 * 512.0),
    )
    torch.testing.assert_close(scales, reference_scales, rtol=2e-3, atol=1e-7)

    math_us = elapsed_us(
        lambda: _fused_lfm25_shortconv_math_for_test(
            b, c, x, fused_state, weight, indices
        ),
        iterations,
    )
    result = {
        "batch": batch,
        "state_dtype": str(state_dtype),
        "math_only_us": math_us,
        "exact_output_and_state": True,
        "max_scale_abs_error": float((scales - reference_scales).abs().max().item()),
    }
    if torch.cuda.get_device_capability()[0] >= 9:
        reference_quantized, reference_scale = ops.scaled_fp8_quant(
            reference,
            scale=None,
            use_per_token_if_dynamic=True,
        )
        production_state = state.clone()
        fused_quantized, fused_scale = fused_lfm25_shortconv_fp8_quant(
            b, c, x, production_state, weight, indices
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(production_state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(fused_scale, reference_scale, rtol=2e-3, atol=1e-7)
        torch.testing.assert_close(
            fused_quantized.float() * fused_scale,
            reference_quantized.float() * reference_scale,
            rtol=0,
            atol=0.0625,
        )
        stock_state = state.clone()
        fused_bench_state = state.clone()
        stock_us = elapsed_us(
            lambda: ops.scaled_fp8_quant(
                stock_step(b, c, x, stock_state, weight, indices),
                scale=None,
                use_per_token_if_dynamic=True,
            ),
            iterations,
        )
        production_us = elapsed_us(
            lambda: fused_lfm25_shortconv_fp8_quant(
                b, c, x, fused_bench_state, weight, indices
            ),
            iterations,
        )
        result["stock_conv_then_fp8_us"] = stock_us
        result["production_fp8_us"] = production_us
        result["kernel_pair_speedup"] = stock_us / production_us
        result["fp8_mismatch_rate"] = float(
            (fused_quantized != reference_quantized).float().mean().item()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--compile-only-sm90", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = parser.parse_args()
    compile_result = compile_sm90()
    if args.compile_only_sm90:
        print(json.dumps({"compile": compile_result}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = [
        run_case(batch, state_dtype, args.iterations)
        for state_dtype in (torch.bfloat16, torch.float32)
        for batch in args.batches
    ]
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
