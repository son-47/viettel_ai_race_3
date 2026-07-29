"""Opt-in fused Q/K RMSNorm + RoPE kernel for LFM2.5 attention.

The kernel consumes strided Q/K views from the packed QKV projection and
writes the final contiguous attention inputs in one launch. It preserves the
stock rounding boundary by rounding RMSNorm to the input dtype before RoPE.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton


FUSED_QK_NORM_ROPE_ENABLED = (
    os.getenv("VLLM_LFM25_FUSED_QK_NORM_ROPE", "0") == "1"
)


@triton.jit
def _lfm25_fused_qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        weight_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = q_ptr + token * q_stride_t + local_head * head_dim
        weight_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    head_offsets = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offsets < head_dim
    x = tl.load(in_base + head_offsets, mask=head_mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(variance + eps)

    if HAS_PASS:
        weight = tl.load(
            weight_ptr + head_offsets, mask=head_mask, other=0.0
        ).to(tl.float32)
        normalized = (x * inv_rms * weight).to(INPUT_DTYPE).to(tl.float32)
        pass_mask = head_mask & (head_offsets >= rotary_dim)
        tl.store(out_base + head_offsets, normalized, mask=pass_mask)

    rotary_offsets = tl.arange(0, ROT_HALF_BLOCK)
    rotary_mask = rotary_offsets < half_rotary
    x_1 = tl.load(
        in_base + rotary_offsets, mask=rotary_mask, other=0.0
    ).to(tl.float32)
    x_2 = tl.load(
        in_base + half_rotary + rotary_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    weight_1 = tl.load(
        weight_ptr + rotary_offsets, mask=rotary_mask, other=0.0
    ).to(tl.float32)
    weight_2 = tl.load(
        weight_ptr + half_rotary + rotary_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)

    x_1 = (x_1 * inv_rms * weight_1).to(INPUT_DTYPE).to(tl.float32)
    x_2 = (x_2 * inv_rms * weight_2).to(INPUT_DTYPE).to(tl.float32)

    position = tl.load(positions_ptr + token).to(tl.int64)
    cache_base = position * cache_stride_p
    cosine = tl.load(
        cos_sin_cache_ptr + cache_base + rotary_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    sine = tl.load(
        cos_sin_cache_ptr + cache_base + half_rotary + rotary_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)

    out_1 = x_1 * cosine - x_2 * sine
    out_2 = x_2 * cosine + x_1 * sine
    tl.store(out_base + rotary_offsets, out_1, mask=rotary_mask)
    tl.store(
        out_base + half_rotary + rotary_offsets, out_2, mask=rotary_mask
    )


def fused_lfm25_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = (q, k, q_weight, k_weight, cos_sin_cache, positions)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused LFM2.5 QK norm/RoPE is CUDA-only")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all fused QK norm/RoPE tensors must share a CUDA device")
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("Q and K must be flattened [tokens, heads * head_dim]")
    if q.shape[0] != k.shape[0]:
        raise ValueError("Q and K token counts differ")
    if q.shape[1] != num_q_heads * head_dim:
        raise ValueError("Q shape does not match num_q_heads * head_dim")
    if k.shape[1] != num_kv_heads * head_dim:
        raise ValueError("K shape does not match num_kv_heads * head_dim")
    if q.stride(1) != 1 or k.stride(1) != 1:
        raise ValueError("the fused path requires unit feature stride")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise ValueError("Q and K must share BF16 or FP16 dtype")
    if q_weight.numel() != head_dim or k_weight.numel() != head_dim:
        raise ValueError("RMSNorm weight size must equal head_dim")
    if positions.ndim != 1 or positions.numel() != q.shape[0]:
        raise ValueError("positions must have one flattened entry per token")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must use int32 or int64")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError("cos/sin cache must have shape [positions, rotary_dim]")
    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.stride(1) != 1:
        raise ValueError("cos/sin cache must match Q dtype and be row-contiguous")

    token_count = q.shape[0]
    q_out = torch.empty(
        (token_count, num_q_heads * head_dim), dtype=q.dtype, device=q.device
    )
    k_out = torch.empty(
        (token_count, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    if token_count == 0:
        return q_out, k_out

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rotary_block = triton.next_power_of_2(half_rotary)
    grid = (token_count, num_q_heads + num_kv_heads)
    _lfm25_fused_qk_norm_rope_kernel[grid](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        half_rotary,
        eps,
        INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rotary_block,
        HAS_PASS=rotary_dim < head_dim,
        num_warps=max(1, head_block // 64),
        num_stages=2,
    )
    return q_out, k_out
