"""LFM2.5 SwiGLU plus dynamic per-token FP8 quantization fusion.

Online FP8 serving normally launches ``silu_and_mul`` to materialize an
8,192-wide BF16/FP16 activation and then launches dynamic per-token FP8
quantization before the MLP down projection.  LFM2.5 executes that pair in all
16 decoder layers.  This module computes the rounded SwiGLU value, its rowwise
scale, and the FP8 activation in one Triton launch.

The linear helper is intentionally restricted to the online per-tensor FP8
method backed by CUTLASS on tensor-parallel size one.  Unsupported backends
continue through vLLM's stock activation and linear paths.
"""

from __future__ import annotations

import os

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

FUSED_SILU_FP8_ENABLED = os.getenv("VLLM_LFM25_FUSED_SILU_FP8", "0") == "1"


@triton.jit
def _lfm25_fused_silu_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    input_stride_token: tl.constexpr,
    output_stride_token: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    TEST_MATH_ONLY: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < intermediate_size
    input_base = input_ptr + token * input_stride_token

    gate = tl.load(input_base + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        input_base + intermediate_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # Stock vLLM materializes the SwiGLU output in the projection dtype before
    # dynamic quantization.  Preserve that BF16/FP16 rounding boundary both for
    # the rowwise maximum and for the value that is converted to FP8.
    # tl.exp follows the libdevice path on NVIDIA; tl.sigmoid uses the faster
    # exp2 approximation and measurably changes some BF16 row maxima.
    # vLLM's packed CUDA kernel casts SiLU(gate) back to the projection dtype
    # before multiplying by ``up``; this is a real second rounding boundary.
    silu_gate = (gate / (1.0 + tl.exp(-gate))).to(INPUT_DTYPE).to(tl.float32)
    activated = (silu_gate * up).to(INPUT_DTYPE).to(tl.float32)
    abs_max = tl.max(tl.where(mask, tl.abs(activated), 0.0), axis=0)
    # Match vLLM's dynamic_per_token_scaled_fp8_quant contract for E4M3FN.
    # Keep literals inside the JIT function: Triton 3.6 rejects ordinary
    # module globals even when Python treats them as constants.
    scale = tl.maximum(abs_max * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
    quantized = tl.maximum(tl.minimum(activated / scale, 448.0), -448.0)

    if TEST_MATH_ONLY:
        tl.store(
            output_ptr + token * output_stride_token + offsets,
            activated,
            mask=mask,
        )
    else:
        tl.store(
            output_ptr + token * output_stride_token + offsets,
            quantized,
            mask=mask,
        )
    tl.store(scale_ptr + token, scale)


def fused_lfm25_silu_fp8_quant(
    gate_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stock-compatible per-token FP8 SwiGLU output and FP32 scale."""
    if not gate_up.is_cuda:
        raise ValueError("fused LFM2.5 SwiGLU/FP8 is CUDA-only")
    if gate_up.ndim != 2:
        raise ValueError("gate_up must be a flattened [tokens, 2 * dim] tensor")
    if gate_up.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("gate_up must use BF16 or FP16")
    if gate_up.shape[1] == 0 or gate_up.shape[1] % 2:
        raise ValueError("gate_up's feature dimension must be positive and even")
    if gate_up.stride(1) != 1:
        raise ValueError("gate_up must have unit feature stride")

    token_count = gate_up.shape[0]
    intermediate_size = gate_up.shape[1] // 2
    output = torch.empty(
        (token_count, intermediate_size),
        dtype=current_platform.fp8_dtype(),
        device=gate_up.device,
    )
    scales = torch.empty((token_count, 1), dtype=torch.float32, device=gate_up.device)
    if token_count == 0:
        return output, scales

    block_size = triton.next_power_of_2(intermediate_size)
    if block_size > 65536:
        raise ValueError("intermediate dimension is too large for the fused kernel")
    _lfm25_fused_silu_fp8_kernel[(token_count,)](
        gate_up,
        output,
        scales,
        gate_up.stride(0),
        output.stride(0),
        intermediate_size,
        BLOCK_SIZE=block_size,
        INPUT_DTYPE=(tl.bfloat16 if gate_up.dtype == torch.bfloat16 else tl.float16),
        TEST_MATH_ONLY=False,
        num_warps=8,
        num_stages=2,
    )
    return output, scales


def _fused_lfm25_silu_math_for_test(
    gate_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exercise the fused math on pre-Hopper GPUs without an E4M3 store."""
    if not gate_up.is_cuda or gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("gate_up must be a CUDA [tokens, 2 * dim] tensor")
    intermediate_size = gate_up.shape[1] // 2
    activated = torch.empty(
        (gate_up.shape[0], intermediate_size),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    scales = torch.empty(
        (gate_up.shape[0], 1), dtype=torch.float32, device=gate_up.device
    )
    if gate_up.shape[0] == 0:
        return activated, scales
    block_size = triton.next_power_of_2(intermediate_size)
    _lfm25_fused_silu_fp8_kernel[(gate_up.shape[0],)](
        gate_up,
        activated,
        scales,
        gate_up.stride(0),
        activated.stride(0),
        intermediate_size,
        BLOCK_SIZE=block_size,
        INPUT_DTYPE=(tl.bfloat16 if gate_up.dtype == torch.bfloat16 else tl.float16),
        TEST_MATH_ONLY=True,
        num_warps=8,
        num_stages=2,
    )
    return activated, scales


def supports_fused_lfm25_silu_fp8_linear(linear: torch.nn.Module) -> bool:
    """Whether ``linear`` has the exact online-FP8 CUTLASS contract we use."""
    quant_method = getattr(linear, "quant_method", None)
    fp8_linear = getattr(quant_method, "fp8_linear", None)
    return bool(
        FUSED_SILU_FP8_ENABLED
        and getattr(linear, "tp_size", None) == 1
        and getattr(linear, "bias", None) is None
        and quant_method is not None
        and quant_method.__class__.__name__ == "Fp8PerTensorOnlineLinearMethod"
        and fp8_linear is not None
        and fp8_linear.__class__.__name__ == "CutlassFP8ScaledMMLinearKernel"
    )


def fused_lfm25_silu_fp8_linear(
    gate_up: torch.Tensor,
    linear: torch.nn.Module,
) -> torch.Tensor:
    """Fuse SwiGLU/FP8 quant and invoke the guarded CUTLASS down projection."""
    if not supports_fused_lfm25_silu_fp8_linear(linear):
        raise ValueError("linear layer does not support fused LFM2.5 SwiGLU/FP8")

    quantized, scales = fused_lfm25_silu_fp8_quant(gate_up)
    fp8_linear = linear.quant_method.fp8_linear
    weight = linear.weight
    weight_scale = linear.weight_scale
    output_shape = [*gate_up.shape[:-1], weight.shape[1]]
    return fp8_linear.apply_scaled_mm(
        A=quantized,
        B=weight,
        out_dtype=gate_up.dtype,
        As=scales,
        Bs=weight_scale,
        bias=None,
        output_shape=output_shape,
    )
