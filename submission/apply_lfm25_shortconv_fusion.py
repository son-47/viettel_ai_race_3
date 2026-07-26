"""Install opt-in LFM2.5 kernel fusions into pinned vLLM.

The source replacements are exact and fail loudly if the base image drifts.
Use ``--check`` to validate a vLLM tree without changing it and ``--self-test``
for the numerical state-transition test that does not require a GPU.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import shutil


SHORT_CONV_PATH = Path("vllm/model_executor/layers/mamba/short_conv.py")
SHORT_CONV_KERNEL_PATH = Path(
    "vllm/model_executor/layers/mamba/ops/lfm25_fused_short_conv.py"
)
LFM2_PATH = Path("vllm/model_executor/models/lfm2.py")
QK_KERNEL_PATH = Path(
    "vllm/model_executor/layers/lfm25_fused_qk_norm_rope.py"
)


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
    BYPASS_SINGLE_VSTACK,
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

    decode_old = '''        if has_decode:
            Bx_d = (B_d * x_d).contiguous()
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

        # Merge prefill and decode outputs before passing to gated MLP
        hidden_states = torch.vstack(conv_output_list)
'''
    decode_new = '''        if has_decode:
            if (
                FUSED_DECODE_ENABLED
                and self.L_cache == 3
                and self.conv.bias is None
                and state_indices_tensor_d is not None
                and attn_metadata.num_accepted_tokens is None
                and num_decodes == attn_metadata.num_decode_tokens
            ):
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

        # torch.vstack([tensor]) still allocates and copies. Decode-only and
        # prefill-only iterations can feed their sole tensor to out_proj.
        hidden_states = (
            conv_output_list[0]
            if BYPASS_SINGLE_VSTACK and len(conv_output_list) == 1
            else torch.vstack(conv_output_list)
        )
'''
    return replace_once(
        text,
        decode_old,
        decode_new,
        "LFM2.5 fused ShortConv decode path",
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
        "LFM2.5 fused QK norm/RoPE attention path",
    )


def self_test() -> None:
    import numpy as np

    generator = np.random.default_rng(20260726)
    blocks, batch, dim = 7, 4, 31
    b = generator.standard_normal((batch, dim)).astype(np.float16)
    c = generator.standard_normal((batch, dim)).astype(np.float16)
    x = generator.standard_normal((batch, dim)).astype(np.float16)
    weight = generator.standard_normal((dim, 3)).astype(np.float16)
    indices = np.array([5, 1, 6, 3], dtype=np.int32)

    for state_dtype in (np.float16, np.float32):
        state = generator.standard_normal((blocks, dim, 2)).astype(state_dtype)
        reference_state = state.copy()
        reference_output = np.empty_like(b)
        for row, state_index in enumerate(indices):
            # Match stock: B*x rounds in projection dtype, then state dtype.
            gated_input = (b[row] * x[row]).astype(np.float16).astype(state_dtype)
            convolution = (
                reference_state[state_index, :, 0].astype(np.float32)
                * weight[:, 0].astype(np.float32)
                + reference_state[state_index, :, 1].astype(np.float32)
                * weight[:, 1].astype(np.float32)
                + gated_input.astype(np.float32) * weight[:, 2].astype(np.float32)
            )
            rounded_convolution = convolution.astype(state_dtype).astype(np.float16)
            reference_output[row] = (
                c[row].astype(np.float32) * rounded_convolution.astype(np.float32)
            ).astype(np.float16)
            reference_state[state_index, :, 0] = reference_state[
                state_index, :, 1
            ]
            reference_state[state_index, :, 1] = gated_input

        fused_state = state.copy()
        gated_input = (b * x).astype(np.float16).astype(state_dtype)
        old_0 = fused_state[indices, :, 0].copy()
        old_1 = fused_state[indices, :, 1].copy()
        convolution = (
            old_0.astype(np.float32) * weight[:, 0].astype(np.float32)
            + old_1.astype(np.float32) * weight[:, 1].astype(np.float32)
            + gated_input.astype(np.float32) * weight[:, 2].astype(np.float32)
        )
        fused_output = (
            c.astype(np.float32)
            * convolution.astype(state_dtype).astype(np.float16).astype(np.float32)
        ).astype(np.float16)
        fused_state[indices, :, 0] = old_1
        fused_state[indices, :, 1] = gated_input

        np.testing.assert_array_equal(reference_output, fused_output)
        np.testing.assert_array_equal(reference_state, fused_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing the vllm package (defaults to site-packages)",
    )
    parser.add_argument(
        "--kernel-source",
        type=Path,
        help="Path to lfm25_fused_short_conv.py (required when installing)",
    )
    parser.add_argument(
        "--qk-kernel-source",
        type=Path,
        help="Path to lfm25_fused_qk_norm_rope.py (required when installing)",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()

    root = package_root(args.root)
    short_conv_path = root / SHORT_CONV_PATH
    original = short_conv_path.read_text(encoding="utf-8")
    patched = patch_short_conv(original)
    ast.parse(patched, filename=str(short_conv_path))

    lfm2_path = root / LFM2_PATH
    lfm2_original = lfm2_path.read_text(encoding="utf-8")
    lfm2_patched = patch_lfm2_attention(lfm2_original)
    ast.parse(lfm2_patched, filename=str(lfm2_path))

    if args.kernel_source is None or args.qk_kernel_source is None:
        if not args.check:
            raise RuntimeError(
                "--kernel-source and --qk-kernel-source are required when installing"
            )
    else:
        kernel_source = args.kernel_source.resolve()
        kernel_text = kernel_source.read_text(encoding="utf-8")
        ast.parse(kernel_text, filename=str(kernel_source))

        qk_kernel_source = args.qk_kernel_source.resolve()
        qk_kernel_text = qk_kernel_source.read_text(encoding="utf-8")
        ast.parse(qk_kernel_text, filename=str(qk_kernel_source))
        if not args.check:
            destination = root / SHORT_CONV_KERNEL_PATH
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(kernel_source, destination)
            qk_destination = root / QK_KERNEL_PATH
            qk_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(qk_kernel_source, qk_destination)

    if not args.check:
        short_conv_path.write_text(patched, encoding="utf-8")
        lfm2_path.write_text(lfm2_patched, encoding="utf-8")
    action = "checked" if args.check else "patched"
    print(f"{action}: {SHORT_CONV_PATH}")
    print(f"{action}: {LFM2_PATH}")


if __name__ == "__main__":
    main()
