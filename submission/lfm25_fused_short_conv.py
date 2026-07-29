"""Guarded LFM2.5 width-three ShortConv decode fusion for vLLM."""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


FUSED_DECODE_ENABLED = os.getenv("VLLM_LFM25_FUSED_SHORTCONV", "0") == "1"


@triton.jit()
def _lfm25_fused_short_conv_decode_kernel(
    b_ptr,
    c_ptr,
    x_ptr,
    state_ptr,
    weight_ptr,
    state_indices_ptr,
    out_ptr,
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
    stride_out_token: tl.constexpr,
    stride_out_dim: tl.constexpr,
    dim: tl.constexpr,
    null_block_id: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    sequence_index = tl.program_id(0)
    feature_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    feature_mask = feature_offsets < dim

    b = tl.load(
        b_ptr
        + sequence_index * stride_b_token
        + feature_offsets * stride_b_dim,
        mask=feature_mask,
        other=0.0,
    )
    c = tl.load(
        c_ptr
        + sequence_index * stride_c_token
        + feature_offsets * stride_c_dim,
        mask=feature_mask,
        other=0.0,
    )
    x = tl.load(
        x_ptr
        + sequence_index * stride_x_token
        + feature_offsets * stride_x_dim,
        mask=feature_mask,
        other=0.0,
    )

    gated_input = (b.to(tl.float32) * x.to(tl.float32)).to(
        b_ptr.dtype.element_ty
    )
    output_pointer = (
        out_ptr
        + sequence_index * stride_out_token
        + feature_offsets * stride_out_dim
    )
    state_index = tl.load(
        state_indices_ptr + sequence_index * stride_indices
    ).to(tl.int64)
    if state_index == null_block_id:
        padded_input = gated_input.to(state_ptr.dtype.element_ty).to(
            b_ptr.dtype.element_ty
        )
        padded_output = c.to(tl.float32) * padded_input.to(tl.float32)
        tl.store(output_pointer, padded_output, mask=feature_mask)
        return

    state_base = (
        state_ptr
        + state_index * stride_state_block
        + feature_offsets * stride_state_dim
    )
    state_0 = tl.load(state_base, mask=feature_mask, other=0.0)
    state_1 = tl.load(
        state_base + stride_state_token, mask=feature_mask, other=0.0
    )

    weight_base = weight_ptr + feature_offsets * stride_weight_dim
    weight_0 = tl.load(weight_base, mask=feature_mask, other=0.0)
    weight_1 = tl.load(
        weight_base + stride_weight_token, mask=feature_mask, other=0.0
    )
    weight_2 = tl.load(
        weight_base + 2 * stride_weight_token,
        mask=feature_mask,
        other=0.0,
    )

    new_state_value = gated_input.to(state_ptr.dtype.element_ty)
    convolution = tl.zeros((BLOCK_N,), dtype=tl.float32)
    convolution += state_0 * weight_0
    convolution += state_1 * weight_1
    convolution += new_state_value * weight_2
    rounded_convolution = convolution.to(state_ptr.dtype.element_ty).to(
        b_ptr.dtype.element_ty
    )
    output = c.to(tl.float32) * rounded_convolution.to(tl.float32)
    tl.store(output_pointer, output, mask=feature_mask)
    tl.store(state_base, state_1, mask=feature_mask)
    tl.store(
        state_base + stride_state_token,
        new_state_value,
        mask=feature_mask,
    )


def fused_lfm25_short_conv_decode(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    tensors = (b, c, x, conv_state, weight, state_indices)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused LFM2.5 ShortConv is CUDA-only")
    if any(tensor.device != b.device for tensor in tensors[1:]):
        raise ValueError("all fused ShortConv tensors must share one CUDA device")
    if b.ndim != 2 or b.shape != c.shape or b.shape != x.shape:
        raise ValueError("B, C, and x must have the same [tokens, dim] shape")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("B, C, and x must use BF16 or FP16")
    if c.dtype != b.dtype or x.dtype != b.dtype or weight.dtype != b.dtype:
        raise ValueError("B, C, x, and weights must share a dtype")
    if conv_state.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("conv_state must use BF16, FP16, or FP32")
    if conv_state.ndim != 3 or conv_state.shape[2] < 2:
        raise ValueError("conv_state must have shape [blocks, dim, >=2]")
    if weight.ndim != 2 or weight.shape != (b.shape[1], 3):
        raise ValueError("the fused path requires width-three depthwise weights")
    if weight.stride(1) != 1:
        raise ValueError("the fused path requires contiguous width-three weights")
    if conv_state.shape[1] != b.shape[1]:
        raise ValueError("conv_state and projection dimensions differ")
    if state_indices.ndim not in (1, 2) or state_indices.shape[0] < b.shape[0]:
        raise ValueError("one state-index row is required per decode token")
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state indices must use int32 or int64")

    output = torch.empty_like(b)
    block_n = 256
    grid = (b.shape[0], triton.cdiv(b.shape[1], block_n))
    _lfm25_fused_short_conv_decode_kernel[grid](
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        output,
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
        output.stride(1),
        b.shape[1],
        NULL_BLOCK_ID,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return output
