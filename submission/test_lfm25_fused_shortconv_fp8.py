"""Correctness, SM90 compile, and latency test for ShortConv + FP8 fusion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as triton_compile


DIM = 2048
COMPILE_NULL_BLOCK_ID = -1
NULL_BLOCK_ID = None
ops = None
fused_lfm25_short_conv_decode = None
_fused_lfm25_shortconv_math_for_test = None
fused_lfm25_shortconv_fp8_quant = None


def load_compile_kernel():
    """Load only the production kernel when Docker has no NVIDIA runtime.

    vLLM's package initialization queries the active platform.  Docker Desktop
    on a CPU host cannot complete that initialization, although Triton can
    still compile an explicit SM90 target.  Temporary minimal package stubs let
    us execute the production module and are removed before any runtime test.
    """
    module_names = [
        "vllm",
        "vllm.platforms",
        "vllm.triton_utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.utils",
    ]
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in module_names}
    try:
        for package_name in (
            "vllm",
            "vllm.v1",
            "vllm.v1.attention",
            "vllm.v1.attention.backends",
        ):
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

        platform_stub = types.ModuleType("vllm.platforms")
        platform_stub.current_platform = types.SimpleNamespace(
            fp8_dtype=lambda: torch.float8_e4m3fn
        )
        sys.modules["vllm.platforms"] = platform_stub

        triton_stub = types.ModuleType("vllm.triton_utils")
        triton_stub.tl = tl
        triton_stub.triton = triton
        sys.modules["vllm.triton_utils"] = triton_stub

        attention_utils_stub = types.ModuleType(
            "vllm.v1.attention.backends.utils"
        )
        attention_utils_stub.NULL_BLOCK_ID = COMPILE_NULL_BLOCK_ID
        sys.modules["vllm.v1.attention.backends.utils"] = attention_utils_stub

        purelib = Path(sysconfig.get_paths()["purelib"])
        module_path = purelib / (
            "vllm/model_executor/layers/mamba/ops/"
            "lfm25_fused_shortconv_fp8.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_lfm25_fused_shortconv_fp8_compile", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load production kernel: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._lfm25_fused_shortconv_fp8_kernel
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def load_runtime_symbols() -> None:
    global ops
    global fused_lfm25_short_conv_decode
    global _fused_lfm25_shortconv_math_for_test
    global fused_lfm25_shortconv_fp8_quant

    from vllm import _custom_ops as runtime_ops
    from vllm.model_executor.layers.mamba.ops.lfm25_fused_short_conv import (
        fused_lfm25_short_conv_decode as runtime_shortconv,
    )
    from vllm.model_executor.layers.mamba.ops.lfm25_fused_shortconv_fp8 import (
        _fused_lfm25_shortconv_math_for_test as runtime_math,
        fused_lfm25_shortconv_fp8_quant as runtime_quant,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID as runtime_null

    global NULL_BLOCK_ID
    NULL_BLOCK_ID = runtime_null
    ops = runtime_ops
    fused_lfm25_short_conv_decode = runtime_shortconv
    _fused_lfm25_shortconv_math_for_test = runtime_math
    fused_lfm25_shortconv_fp8_quant = runtime_quant


def compile_sm90() -> dict:
    """Compile the production BF16/E4M3 kernel for the grader architecture."""
    kernel = load_compile_kernel()
    source = ASTSource(
        kernel,
        {
            "b_ptr": "*bf16",
            "c_ptr": "*bf16",
            "x_ptr": "*bf16",
            "state_ptr": "*bf16",
            "weight_ptr": "*bf16",
            "state_indices_ptr": "*i32",
            "output_ptr": "*fp8e4nv",
            "scale_ptr": "*fp32",
        },
        {
            "stride_b_token": DIM,
            "stride_b_dim": 1,
            "stride_c_token": DIM,
            "stride_c_dim": 1,
            "stride_x_token": DIM,
            "stride_x_dim": 1,
            "stride_state_block": DIM * 2,
            "stride_state_dim": 2,
            "stride_state_token": 1,
            "stride_weight_dim": 3,
            "stride_weight_token": 1,
            "stride_indices": 1,
            "stride_output_token": DIM,
            "dim": DIM,
            "null_block_id": COMPILE_NULL_BLOCK_ID,
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


def make_case(batch: int, state_dtype: torch.dtype):
    generator = torch.Generator(device="cuda").manual_seed(20260727 + batch)
    blocks = max(64, batch + 1)
    b = torch.randn(
        batch, DIM, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    c = torch.randn_like(b)
    x = torch.randn_like(b)
    weight = torch.randn(
        DIM, 3, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    state = torch.randn(
        blocks,
        DIM,
        2,
        device="cuda",
        dtype=state_dtype,
        generator=generator,
    )
    indices = torch.arange(batch, device="cuda", dtype=torch.int32).unsqueeze(1)
    if batch > 1:
        indices[-1, 0] = NULL_BLOCK_ID
    return b, c, x, weight, state, indices


def stock_quant(b, c, x, state, weight, indices):
    hidden = fused_lfm25_short_conv_decode(b, c, x, state, weight, indices)
    return ops.scaled_fp8_quant(
        hidden,
        scale=None,
        use_per_token_if_dynamic=True,
    )


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


def run_math_case(batch: int, state_dtype: torch.dtype) -> dict:
    b, c, x, weight, initial_state, indices = make_case(batch, state_dtype)
    reference_state = initial_state.clone()
    fused_state = initial_state.clone()
    reference_hidden = fused_lfm25_short_conv_decode(
        b, c, x, reference_state, weight, indices
    )
    _, reference_scale = ops.scaled_fp8_quant(
        reference_hidden,
        scale=None,
        use_per_token_if_dynamic=True,
    )
    fused_hidden, fused_scale = _fused_lfm25_shortconv_math_for_test(
        b, c, x, fused_state, weight, indices
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_hidden, reference_hidden, rtol=0, atol=0)
    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)
    torch.testing.assert_close(fused_scale, reference_scale, rtol=2e-3, atol=1e-7)
    return {
        "batch": batch,
        "state_dtype": str(state_dtype),
        "exact_hidden_and_state": True,
        "max_scale_abs_error": float(
            (fused_scale - reference_scale).abs().max().item()
        ),
    }


def run_case(batch: int, state_dtype: torch.dtype, iterations: int) -> dict:
    b, c, x, weight, initial_state, indices = make_case(batch, state_dtype)
    reference_state = initial_state.clone()
    fused_state = initial_state.clone()
    reference_q, reference_scale = stock_quant(
        b, c, x, reference_state, weight, indices
    )
    fused_q, fused_scale = fused_lfm25_shortconv_fp8_quant(
        b, c, x, fused_state, weight, indices
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)
    torch.testing.assert_close(fused_scale, reference_scale, rtol=2e-3, atol=1e-7)
    torch.testing.assert_close(
        fused_q.float() * fused_scale,
        reference_q.float() * reference_scale,
        rtol=0,
        atol=0.0625,
    )

    stock_state = initial_state.clone()
    kernel_state = initial_state.clone()
    stock_us = elapsed_us(
        lambda: stock_quant(b, c, x, stock_state, weight, indices),
        iterations,
    )
    fused_us = elapsed_us(
        lambda: fused_lfm25_shortconv_fp8_quant(
            b, c, x, kernel_state, weight, indices
        ),
        iterations,
    )
    return {
        "batch": batch,
        "state_dtype": str(state_dtype),
        "stock_fused_shortconv_plus_quant_us": stock_us,
        "fused_shortconv_quant_us": fused_us,
        "speedup": stock_us / fused_us,
        "fp8_mismatch_rate": float((fused_q != reference_q).float().mean().item()),
        "exact_state": True,
        "max_scale_abs_error": float(
            (fused_scale - reference_scale).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--compile-only-sm90", action="store_true")
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    args = parser.parse_args()
    compile_result = compile_sm90()
    if args.compile_only_sm90:
        print(json.dumps({"compile": compile_result}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    load_runtime_symbols()

    if torch.cuda.get_device_capability()[0] < 9:
        math_results = [
            run_math_case(batch, state_dtype)
            for state_dtype in (torch.bfloat16, torch.float32)
            for batch in args.batches
        ]
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "compile": compile_result,
                    "math_results": math_results,
                    "runtime_skipped": "E4M3 stores require SM90+",
                },
                indent=2,
            )
        )
        return

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
