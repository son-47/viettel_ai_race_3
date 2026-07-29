"""Install the Marlin/DSpark-safe LFM2.5 fusions into the pinned R2 image.

This ports three optimizations from the 65.71 image: Q/K RMSNorm + NeoX RoPE,
the singleton-vstack bypass, and a narrowly guarded ShortConv decode fusion.
The ShortConv fusion is disabled whenever DSpark supplies accepted-token
metadata, preserving R2's rollback-aware verification path. The CUTLASS-only
SiLU-to-FP8 path is intentionally not installed.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
from pathlib import Path


SHORT_CONV_PATH = Path("vllm/model_executor/layers/mamba/short_conv.py")
SHORT_CONV_KERNEL_PATH = Path(
    "vllm/model_executor/layers/mamba/ops/lfm25_fused_short_conv.py"
)
LFM2_PATH = Path("vllm/model_executor/models/lfm2.py")
QK_KERNEL_PATH = Path("vllm/model_executor/layers/lfm25_fused_qk_norm_rope.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def package_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vLLM is not importable; pass --root explicitly")
    return Path(spec.origin).resolve().parent.parent


def patch_short_conv(text: str) -> str:
    import_old = '''from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
'''
    import_new = import_old + '''from vllm.model_executor.layers.mamba.ops.lfm25_fused_short_conv import (
    FUSED_DECODE_ENABLED,
    fused_lfm25_short_conv_decode,
)
'''
    text = replace_once(
        text,
        import_old,
        import_new,
        "LFM2.5 fused ShortConv import",
    )

    flag_old = '''_CONV_QUANT = os.environ.get("VLLM_CONV_FP8", "0") == "1"
'''
    flag_new = flag_old + '''_BYPASS_SINGLE_VSTACK = (
    os.environ.get("VLLM_LFM25_BYPASS_SINGLE_VSTACK", "0") == "1"
)
'''
    text = replace_once(
        text,
        flag_old,
        flag_new,
        "LFM2.5 singleton-vstack flag",
    )

    decode_old = '''        if has_decode:
            Bx_d = (B_d * x_d).contiguous()
            if num_accepted_tokens is not None:
                # Speculative decoding: verify 1+num_spec draft tokens per
                # request and slide the conv state, rolling back rejected
                # drafts. Mirrors Mamba2 (mamba_mixer2.py). state_indices_tensor_d
                # is [num_reqs, 1+K].
                Bx = causal_conv1d_update(
                    Bx_d,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_tensor_d,
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=query_start_loc_d,
                    max_query_len=state_indices_tensor_d.size(-1),
                )
            else:
                # Non-spec decode: exactly as stock.
                Bx = causal_conv1d_update(
                    Bx_d,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_tensor_d,
                )
            y = C_d * Bx
            conv_output_list.insert(0, y)
'''
    decode_new = '''        if has_decode:
            can_fuse_decode = (
                FUSED_DECODE_ENABLED
                and self.L_cache == 3
                and self.conv.bias is None
                and state_indices_tensor_d is not None
                and num_accepted_tokens is None
                and num_decodes == num_decode_tokens
            )
            if can_fuse_decode:
                y = fused_lfm25_short_conv_decode(
                    B_d,
                    C_d,
                    x_d,
                    conv_state,
                    conv_weights,
                    state_indices_tensor_d,
                )
            else:
                Bx_d = (B_d * x_d).contiguous()
                if num_accepted_tokens is not None:
                    # Preserve the R2 rollback-aware DSpark verification path.
                    Bx = causal_conv1d_update(
                        Bx_d,
                        conv_state,
                        conv_weights,
                        self.conv.bias,
                        activation=None,
                        conv_state_indices=state_indices_tensor_d,
                        num_accepted_tokens=num_accepted_tokens,
                        query_start_loc=query_start_loc_d,
                        max_query_len=state_indices_tensor_d.size(-1),
                    )
                else:
                    Bx = causal_conv1d_update(
                        Bx_d,
                        conv_state,
                        conv_weights,
                        self.conv.bias,
                        activation=None,
                        conv_state_indices=state_indices_tensor_d,
                    )
                y = C_d * Bx
            conv_output_list.insert(0, y)
'''
    text = replace_once(
        text,
        decode_old,
        decode_new,
        "LFM2.5 guarded fused ShortConv decode path",
    )

    stack_old = '''        # Merge prefill and decode outputs before passing to gated MLP
        hidden_states = torch.vstack(conv_output_list)
'''
    stack_new = '''        # Avoid an allocation/copy when an iteration contains only decode
        # or only prefill. Mixed iterations retain the original ordering.
        hidden_states = (
            conv_output_list[0]
            if _BYPASS_SINGLE_VSTACK and len(conv_output_list) == 1
            else torch.vstack(conv_output_list)
        )
'''
    return replace_once(
        text,
        stack_old,
        stack_new,
        "LFM2.5 singleton-vstack bypass",
    )


def patch_lfm2_attention(text: str) -> str:
    import_old = '''from vllm.model_executor.layers.layernorm import RMSNorm
'''
    import_new = import_old + '''from vllm.model_executor.layers.lfm25_fused_qk_norm_rope import (
    FUSED_QK_NORM_ROPE_ENABLED,
    fused_lfm25_qk_rmsnorm_rope,
)
'''
    text = replace_once(
        text,
        import_old,
        import_new,
        "LFM2.5 fused QK norm/RoPE import",
    )

    attention_old = '''        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(n_tokens, self.num_heads, self.head_dim).contiguous()
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim).contiguous()
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(n_tokens, self.num_heads * self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads * self.head_dim)
'''
    attention_new = '''        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if (
            FUSED_QK_NORM_ROPE_ENABLED
            and qkv.is_cuda
            and qkv.dtype in (torch.bfloat16, torch.float16)
            and getattr(self.rotary_emb, "is_neox_style", False)
            and self.rotary_emb.rotary_dim == self.head_dim
            and hasattr(self.rotary_emb, "_match_cos_sin_cache_dtype")
        ):
            cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(q)
            q, k = fused_lfm25_qk_rmsnorm_rope(
                q,
                k,
                self.q_layernorm.weight,
                self.k_layernorm.weight,
                cos_sin_cache,
                positions.flatten(),
                self.q_layernorm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
        else:
            q = q.view(n_tokens, self.num_heads, self.head_dim).contiguous()
            k = k.view(n_tokens, self.num_kv_heads, self.head_dim).contiguous()
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)
            q, k = self.rotary_emb(positions, q, k)
            q = q.view(n_tokens, self.num_heads * self.head_dim)
            k = k.view(n_tokens, self.num_kv_heads * self.head_dim)
'''
    return replace_once(
        text,
        attention_old,
        attention_new,
        "LFM2.5 fused QK norm/RoPE path",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing the vllm package (defaults to site-packages)",
    )
    parser.add_argument("--short-conv-kernel-source", type=Path)
    parser.add_argument("--qk-kernel-source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = package_root(args.root)
    short_conv_path = root / SHORT_CONV_PATH
    short_conv_original = short_conv_path.read_text(encoding="utf-8")
    short_conv_patched = patch_short_conv(short_conv_original)
    ast.parse(short_conv_patched, filename=str(short_conv_path))

    lfm2_path = root / LFM2_PATH
    lfm2_original = lfm2_path.read_text(encoding="utf-8")
    lfm2_patched = patch_lfm2_attention(lfm2_original)
    ast.parse(lfm2_patched, filename=str(lfm2_path))

    if args.short_conv_kernel_source is None or args.qk_kernel_source is None:
        if not args.check:
            raise RuntimeError(
                "both kernel source arguments are required when installing"
            )
    else:
        short_conv_kernel_source = args.short_conv_kernel_source.resolve()
        short_conv_kernel_text = short_conv_kernel_source.read_text(encoding="utf-8")
        ast.parse(short_conv_kernel_text, filename=str(short_conv_kernel_source))
        qk_kernel_source = args.qk_kernel_source.resolve()
        qk_kernel_text = qk_kernel_source.read_text(encoding="utf-8")
        ast.parse(qk_kernel_text, filename=str(qk_kernel_source))
        if not args.check:
            short_conv_destination = root / SHORT_CONV_KERNEL_PATH
            short_conv_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(short_conv_kernel_source, short_conv_destination)
            qk_destination = root / QK_KERNEL_PATH
            qk_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(qk_kernel_source, qk_destination)

    if not args.check:
        short_conv_path.write_text(short_conv_patched, encoding="utf-8")
        lfm2_path.write_text(lfm2_patched, encoding="utf-8")

    action = "checked" if args.check else "patched"
    print(f"{action}: {SHORT_CONV_PATH}")
    print(f"{action}: {LFM2_PATH}")


if __name__ == "__main__":
    main()
