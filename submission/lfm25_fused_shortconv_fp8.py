"""Fuse LFM2.5 ShortConv decode with the FP8 input of its output GEMM.

The existing LFM2.5 ShortConv fusion still writes a 2,048-wide BF16 tensor.
When ``out_proj`` uses online FP8, vLLM immediately reads that tensor in a
second kernel to derive a dynamic scale and quantize it.  This module performs
the recurrent update, C gate, stock-compatible BF16 rounding, rowwise scale,
and E4M3 conversion in one Triton launch, then calls the same guarded CUTLASS
scaled-MM path used by vLLM.

Only the ShortConv ``out_proj`` receives the online quantization config.  Its
larger ``in_proj`` stays BF16, limiting the accuracy change and making this an
independent A/B candidate rather than changing the proven 65.71 submission.
"""

from __future__ import annotations

import os

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


FUSED_SHORTCONV_FP8_ENABLED = (
    os.getenv("VLLM_LFM25_FUSED_SHORTCONV_FP8", "0") == "1"
)


@triton.jit
def _lfm25_fused_shortconv_fp8_kernel(
    b_ptr,
    c_ptr,
    x_ptr,
    state_ptr,
    weight_ptr,
    state_indices_ptr,
    output_ptr,
    scale_ptr,
    stride_b_token: tl.constexpr,
    stride_b_dim: tl.constexpr,
    stride_c_token: tl.constexpr,
    stride_c_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_state_block: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_token: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_output_token: tl.constexpr,
    dim: tl.constexpr,
    null_block_id: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    TEST_MATH_ONLY: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    feature_mask = offsets < dim

    b = tl.load(
        b_ptr + token * stride_b_token + offsets * stride_b_dim,
        mask=feature_mask,
        other=0.0,
    )
    c = tl.load(
        c_ptr + token * stride_c_token + offsets * stride_c_dim,
        mask=feature_mask,
        other=0.0,
    )
    x = tl.load(
        x_ptr + token * stride_x_token + offsets * stride_x_dim,
        mask=feature_mask,
        other=0.0,
    )

    # Match stock ShortConv's first materialization and its state-dtype cast.
    gated_input = (b.to(tl.float32) * x.to(tl.float32)).to(INPUT_DTYPE)
    new_state_value = gated_input.to(state_ptr.dtype.element_ty)

    state_index = tl.load(state_indices_ptr + token * stride_indices).to(tl.int64)
    valid_state = state_index != null_block_id
    state_mask = feature_mask & valid_state
    state_base = (
        state_ptr
        + state_index * stride_state_block
        + offsets * stride_state_dim
    )
    state_0 = tl.load(state_base, mask=state_mask, other=0.0)
    state_1 = tl.load(
        state_base + stride_state_token,
        mask=state_mask,
        other=0.0,
    )

    weight_base = weight_ptr + offsets * stride_weight_dim
    weight_0 = tl.load(weight_base, mask=feature_mask, other=0.0)
    weight_1 = tl.load(
        weight_base + stride_weight_token,
        mask=feature_mask,
        other=0.0,
    )
    weight_2 = tl.load(
        weight_base + 2 * stride_weight_token,
        mask=feature_mask,
        other=0.0,
    )

    convolution = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    convolution += state_0 * weight_0
    convolution += state_1 * weight_1
    convolution += new_state_value * weight_2
    rounded_convolution = convolution.to(state_ptr.dtype.element_ty).to(INPUT_DTYPE)

    # CUDA-graph padding uses NULL_BLOCK_ID. causal_conv1d_update leaves the
    # corresponding B*x row unchanged after its state-dtype conversion.
    padded_convolution = new_state_value.to(INPUT_DTYPE)
    selected_convolution = tl.where(
        valid_state,
        rounded_convolution,
        padded_convolution,
    )

    # Stock writes C*conv to BF16/FP16 before dynamic quantization. Preserve
    # this final rounding boundary for both the maximum and E4M3 conversion.
    hidden = (
        c.to(tl.float32) * selected_convolution.to(tl.float32)
    ).to(INPUT_DTYPE).to(tl.float32)
    abs_max = tl.max(tl.where(feature_mask, tl.abs(hidden), 0.0), axis=0)
    scale = tl.maximum(abs_max * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
    quantized = tl.maximum(tl.minimum(hidden / scale, 448.0), -448.0)

    output_base = output_ptr + token * stride_output_token + offsets
    if TEST_MATH_ONLY:
        tl.store(output_base, hidden, mask=feature_mask)
    else:
        tl.store(output_base, quantized, mask=feature_mask)
    tl.store(scale_ptr + token, scale)

    # Width three stores [previous newest value, current B*x]. Padded graph
    # rows are completely masked, so a NULL_BLOCK_ID never touches state[-1].
    tl.store(state_base, state_1, mask=state_mask)
    tl.store(
        state_base + stride_state_token,
        new_state_value,
        mask=state_mask,
    )


def _validate_inputs(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    tensors = (b, c, x, conv_state, weight, state_indices)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused LFM2.5 ShortConv/FP8 is CUDA-only")
    if any(tensor.device != b.device for tensor in tensors[1:]):
        raise ValueError("all fused ShortConv tensors must share one CUDA device")
    if b.ndim != 2 or b.shape != c.shape or b.shape != x.shape:
        raise ValueError("B, C, and x must share the [tokens, dim] shape")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("B, C, and x must use BF16 or FP16")
    if c.dtype != b.dtype or x.dtype != b.dtype or weight.dtype != b.dtype:
        raise ValueError("B, C, x, and weights must share a dtype")
    if b.stride(1) != 1 or c.stride(1) != 1 or x.stride(1) != 1:
        raise ValueError("B, C, and x must have unit feature stride")
    if conv_state.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("conv_state must use BF16, FP16, or FP32")
    if conv_state.ndim != 3 or conv_state.shape[2] < 2:
        raise ValueError("conv_state must have shape [blocks, dim, >=2]")
    if conv_state.shape[1] != b.shape[1]:
        raise ValueError("conv_state and projection dimensions differ")
    if weight.ndim != 2 or weight.shape != (b.shape[1], 3):
        raise ValueError("the fused path requires width-three depthwise weights")
    if weight.stride(1) != 1:
        raise ValueError("the fused path requires contiguous width-three weights")
    if state_indices.ndim not in (1, 2) or state_indices.shape[0] < b.shape[0]:
        raise ValueError("one state-index row is required per decode token")
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state indices must use int32 or int64")


def _launch_shortconv_fp8(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    math_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(b, c, x, conv_state, weight, state_indices)
    output_dtype = b.dtype if math_only else current_platform.fp8_dtype()
    output = torch.empty(b.shape, dtype=output_dtype, device=b.device)
    scales = torch.empty((b.shape[0], 1), dtype=torch.float32, device=b.device)
    if b.shape[0] == 0:
        return output, scales

    block_size = triton.next_power_of_2(b.shape[1])
    if block_size > 65536:
        raise ValueError("ShortConv dimension is too large for the fused kernel")
    _lfm25_fused_shortconv_fp8_kernel[(b.shape[0],)](
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        output,
        scales,
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        x.stride(0),
        x.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        weight.stride(1),
        state_indices.stride(0),
        output.stride(0),
        b.shape[1],
        NULL_BLOCK_ID,
        BLOCK_SIZE=block_size,
        INPUT_DTYPE=(tl.bfloat16 if b.dtype == torch.bfloat16 else tl.float16),
        TEST_MATH_ONLY=math_only,
        num_warps=8,
        num_stages=2,
    )
    return output, scales


def fused_lfm25_shortconv_fp8_quant(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stock-compatible ShortConv output quantized per decode token."""
    return _launch_shortconv_fp8(
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        math_only=False,
    )


def _fused_lfm25_shortconv_math_for_test(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exercise ShortConv/scale math on a GPU without requiring an FP8 store."""
    return _launch_shortconv_fp8(
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        math_only=True,
    )


def supports_fused_lfm25_shortconv_fp8_linear(
    linear: torch.nn.Module,
) -> bool:
    """Whether ``linear`` exposes the online-FP8 CUTLASS contract we call."""
    quant_method = getattr(linear, "quant_method", None)
    fp8_linear = getattr(quant_method, "fp8_linear", None)
    return bool(
        FUSED_SHORTCONV_FP8_ENABLED
        and getattr(linear, "tp_size", None) == 1
        and getattr(linear, "bias", None) is None
        and quant_method is not None
        and quant_method.__class__.__name__ == "Fp8PerTensorOnlineLinearMethod"
        and fp8_linear is not None
        and fp8_linear.__class__.__name__ == "CutlassFP8ScaledMMLinearKernel"
    )


def fused_lfm25_shortconv_fp8_linear(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
    linear: torch.nn.Module,
) -> torch.Tensor:
    """Fuse ShortConv/FP8 quant and invoke the guarded output projection."""
    if not supports_fused_lfm25_shortconv_fp8_linear(linear):
        raise ValueError("linear layer does not support fused ShortConv/FP8")
    quantized, scales = fused_lfm25_shortconv_fp8_quant(
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
    )
    fp8_linear = linear.quant_method.fp8_linear
    fp8_weight = linear.weight
    output_shape = [*b.shape[:-1], fp8_weight.shape[1]]
    return fp8_linear.apply_scaled_mm(
        A=quantized,
        B=fp8_weight,
        out_dtype=b.dtype,
        As=scales,
        Bs=linear.weight_scale,
        bias=None,
        output_shape=output_shape,
    )
