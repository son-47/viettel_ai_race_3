"""LFM2.5 decode-only ShortConv plus dynamic FP8 output fusion.

The existing LFM2.5 fast path fuses ``B*x``, the width-three recurrent
convolution, the state update, and ``C*conv`` but materializes the result in
BF16 before the online-FP8 ``out_proj`` quantizer reads it again.  This module
keeps the stock rounding boundaries while emitting the FP8 activation and its
per-token scale directly from the ShortConv kernel.

Only the decode-only, bias-free, width-three, TP=1 CUTLASS path is enabled.
Every unsupported shape/backend remains on vLLM's existing implementation.
"""

from __future__ import annotations

import os

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


FUSED_SHORTCONV_FP8_OUT_ENABLED = (
    os.getenv("VLLM_LFM25_FUSED_SHORTCONV_FP8_OUT", "0") == "1"
)


@triton.jit
def _lfm25_fused_shortconv_fp8_out_kernel(
    b_ptr,
    c_ptr,
    x_ptr,
    state_ptr,
    weight_ptr,
    state_indices_ptr,
    out_ptr,
    scale_ptr,
    stride_b_token: tl.constexpr,
    stride_c_token: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_state_block: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_out_token: tl.constexpr,
    dim: tl.constexpr,
    null_block_id: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    TEST_MATH_ONLY: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim

    b = tl.load(
        b_ptr + token * stride_b_token + offsets,
        mask=mask,
        other=0.0,
    )
    c = tl.load(
        c_ptr + token * stride_c_token + offsets,
        mask=mask,
        other=0.0,
    )
    x = tl.load(
        x_ptr + token * stride_x_token + offsets,
        mask=mask,
        other=0.0,
    )

    # Stock first writes B*x in the projection dtype and then converts it to
    # the recurrent-state dtype.  Both rounding points are accuracy-visible.
    gated_input = (b.to(tl.float32) * x.to(tl.float32)).to(INPUT_DTYPE)
    new_state = gated_input.to(state_ptr.dtype.element_ty)
    state_index = tl.load(state_indices_ptr + token * stride_indices).to(tl.int64)

    if state_index == null_block_id:
        # Match causal_conv1d_update's CUDA-graph padding behavior: no state
        # access/update and the rounded B*x value passes through to the C gate.
        rounded_convolution = new_state.to(INPUT_DTYPE)
    else:
        state_base = (
            state_ptr
            + state_index * stride_state_block
            + offsets * stride_state_dim
        )
        state_0 = tl.load(state_base, mask=mask, other=0.0)
        state_1 = tl.load(
            state_base + stride_state_token,
            mask=mask,
            other=0.0,
        )
        weight_base = weight_ptr + offsets * stride_weight_dim
        weight_0 = tl.load(weight_base, mask=mask, other=0.0)
        weight_1 = tl.load(weight_base + 1, mask=mask, other=0.0)
        weight_2 = tl.load(weight_base + 2, mask=mask, other=0.0)

        convolution = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        convolution += state_0 * weight_0
        convolution += state_1 * weight_1
        convolution += new_state * weight_2
        rounded_convolution = convolution.to(state_ptr.dtype.element_ty).to(
            INPUT_DTYPE
        )

        # Width three stores the two recurrent values [old_1, current B*x].
        tl.store(state_base, state_1, mask=mask)
        tl.store(state_base + stride_state_token, new_state, mask=mask)

    # Stock writes C*conv to a BF16/FP16 tensor before dynamic quantization.
    # Preserve that materialization boundary in registers before the absmax.
    activated = (c.to(tl.float32) * rounded_convolution.to(tl.float32)).to(
        INPUT_DTYPE
    ).to(tl.float32)
    abs_max = tl.max(tl.where(mask, tl.abs(activated), 0.0), axis=0)
    scale = tl.maximum(abs_max * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
    quantized = tl.maximum(tl.minimum(activated / scale, 448.0), -448.0)

    output_offsets = token * stride_out_token + offsets
    if TEST_MATH_ONLY:
        tl.store(out_ptr + output_offsets, activated, mask=mask)
    else:
        tl.store(out_ptr + output_offsets, quantized, mask=mask)
    tl.store(scale_ptr + token, scale)


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
        raise ValueError("all fused ShortConv/FP8 tensors must share a device")
    if b.ndim != 2 or b.shape != c.shape or b.shape != x.shape:
        raise ValueError("B, C, and x must share a [tokens, dim] shape")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("B, C, and x must use BF16 or FP16")
    if c.dtype != b.dtype or x.dtype != b.dtype or weight.dtype != b.dtype:
        raise ValueError("B, C, x, and convolution weights must share a dtype")
    if any(tensor.stride(1) != 1 for tensor in (b, c, x)):
        raise ValueError("B, C, and x require unit feature stride")
    if conv_state.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("conv_state must use BF16, FP16, or FP32")
    if conv_state.ndim != 3 or conv_state.shape[1:] != (b.shape[1], 2):
        raise ValueError("conv_state must have shape [blocks, dim, 2]")
    if weight.ndim != 2 or weight.shape != (b.shape[1], 3):
        raise ValueError("the fused path requires [dim, 3] weights")
    if weight.stride(1) != 1:
        raise ValueError("width-three weights must be contiguous")
    if state_indices.ndim not in (1, 2) or state_indices.shape[0] < b.shape[0]:
        raise ValueError("one state index is required per decode token")
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state indices must use int32 or int64")


def _launch(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    *,
    test_math_only: bool,
) -> None:
    if b.shape[0] == 0:
        return
    block_size = triton.next_power_of_2(b.shape[1])
    if block_size > 65536:
        raise ValueError("ShortConv dimension is too large for the fused kernel")
    _lfm25_fused_shortconv_fp8_out_kernel[(b.shape[0],)](
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        output,
        scales,
        b.stride(0),
        c.stride(0),
        x.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        state_indices.stride(0),
        output.stride(0),
        b.shape[1],
        NULL_BLOCK_ID,
        BLOCK_SIZE=block_size,
        INPUT_DTYPE=tl.bfloat16 if b.dtype == torch.bfloat16 else tl.float16,
        TEST_MATH_ONLY=test_math_only,
        num_warps=8,
        num_stages=2,
    )


def fused_lfm25_shortconv_fp8_quant(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP8 ShortConv activation and FP32 per-token scales."""
    _validate_inputs(b, c, x, conv_state, weight, state_indices)
    output = torch.empty(
        b.shape,
        dtype=current_platform.fp8_dtype(),
        device=b.device,
    )
    scales = torch.empty((b.shape[0], 1), dtype=torch.float32, device=b.device)
    _launch(
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        output,
        scales,
        test_math_only=False,
    )
    return output, scales


def _fused_lfm25_shortconv_math_for_test(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact pre-quantization math on GPUs without an E4M3 store."""
    _validate_inputs(b, c, x, conv_state, weight, state_indices)
    output = torch.empty_like(b)
    scales = torch.empty((b.shape[0], 1), dtype=torch.float32, device=b.device)
    _launch(
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        output,
        scales,
        test_math_only=True,
    )
    return output, scales


def supports_fused_lfm25_shortconv_fp8_out(linear: torch.nn.Module) -> bool:
    """Whether a ShortConv out projection has the required CUTLASS contract."""
    quant_method = getattr(linear, "quant_method", None)
    fp8_linear = getattr(quant_method, "fp8_linear", None)
    return bool(
        FUSED_SHORTCONV_FP8_OUT_ENABLED
        and getattr(linear, "tp_size", None) == 1
        and getattr(linear, "bias", None) is None
        and quant_method is not None
        and quant_method.__class__.__name__ == "Fp8PerTensorOnlineLinearMethod"
        and fp8_linear is not None
        and fp8_linear.__class__.__name__ == "CutlassFP8ScaledMMLinearKernel"
    )


def fused_lfm25_shortconv_fp8_out_linear(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    state_indices: torch.Tensor,
    linear: torch.nn.Module,
) -> torch.Tensor:
    """Fuse decode ShortConv/quant and invoke its online-FP8 out projection."""
    if not supports_fused_lfm25_shortconv_fp8_out(linear):
        raise ValueError("ShortConv out_proj does not support the fused FP8 path")
    quantized, scales = fused_lfm25_shortconv_fp8_quant(
        b,
        c,
        x,
        conv_state,
        conv_weight,
        state_indices,
    )
    fp8_linear = linear.quant_method.fp8_linear
    weight = linear.weight
    return fp8_linear.apply_scaled_mm(
        A=quantized,
        B=weight,
        out_dtype=b.dtype,
        As=scales,
        Bs=linear.weight_scale,
        bias=None,
        output_shape=[b.shape[0], weight.shape[1]],
    )
