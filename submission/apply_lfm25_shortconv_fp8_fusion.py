"""Install the opt-in LFM2.5 ShortConv-decode + FP8 fusion.

This patch is intentionally applied on top of the pinned 65.71 fusion image.
Every replacement is exact so a base-image drift fails the Docker build rather
than silently producing a partially optimized server.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
from pathlib import Path


SHORT_CONV_PATH = Path("vllm/model_executor/layers/mamba/short_conv.py")
LFM2_PATH = Path("vllm/model_executor/models/lfm2.py")
KERNEL_PATH = Path(
    "vllm/model_executor/layers/mamba/ops/lfm25_fused_shortconv_fp8.py"
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
    text = replace_once(
        text,
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n",
        "from vllm.model_executor.layers.quantization import QuantizationConfig\n"
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n",
        "ShortConv QuantizationConfig import",
    )

    existing_fusion_import = '''from vllm.model_executor.layers.mamba.ops.lfm25_fused_short_conv import (
    BYPASS_SINGLE_VSTACK,
    FUSED_DECODE_ENABLED,
    fused_lfm25_short_conv_decode,
)
'''
    new_fusion_import = existing_fusion_import + '''from vllm.model_executor.layers.mamba.ops.lfm25_fused_shortconv_fp8 import (
    FUSED_SHORTCONV_FP8_ENABLED,
    fused_lfm25_shortconv_fp8_linear,
    supports_fused_lfm25_shortconv_fp8_linear,
)
'''
    text = replace_once(
        text,
        existing_fusion_import,
        new_fusion_import,
        "ShortConv/FP8 fusion import",
    )

    text = replace_once(
        text,
        '        cache_config: CacheConfig | None = None,\n'
        '        prefix: str = "",\n',
        '        cache_config: CacheConfig | None = None,\n'
        '        quant_config: QuantizationConfig | None = None,\n'
        '        prefix: str = "",\n',
        "ShortConv quant_config argument",
    )

    out_proj_old = '''        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.out_proj",
        )
'''
    out_proj_new = '''        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            quant_config=(
                quant_config if FUSED_SHORTCONV_FP8_ENABLED else None
            ),
            prefix=f"{prefix}.out_proj",
        )
        self.use_fused_shortconv_fp8 = (
            supports_fused_lfm25_shortconv_fp8_linear(self.out_proj)
        )
'''
    text = replace_once(
        text,
        out_proj_old,
        out_proj_new,
        "ShortConv output-only FP8 quantization",
    )

    decode_split_old = '''        x_d, x_p = torch.split(
            x[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        conv_output_list = []

        if has_prefill:
'''
    decode_split_new = '''        x_d, x_p = torch.split(
            x[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        # Decode-only is the steady state for this workload. Return directly
        # from the fused ShortConv -> dynamic FP8 -> CUTLASS out_proj path so
        # no BF16 hidden tensor or standalone quantization kernel is created.
        if (
            self.use_fused_shortconv_fp8
            and not has_prefill
            and has_decode
            and self.L_cache == 3
            and self.conv.bias is None
            and state_indices_tensor_d is not None
            and attn_metadata.num_accepted_tokens is None
            and num_decodes == num_actual_tokens
        ):
            output[:num_actual_tokens] = fused_lfm25_shortconv_fp8_linear(
                B_d,
                C_d,
                x_d,
                conv_state,
                conv_weights,
                state_indices_tensor_d,
                self.out_proj,
            )
            return

        conv_output_list = []

        if has_prefill:
'''
    return replace_once(
        text,
        decode_split_old,
        decode_split_new,
        "decode-only ShortConv/FP8 path",
    )


def patch_lfm2(text: str) -> str:
    return replace_once(
        text,
        '            cache_config=cache_config,\n'
        '            prefix=f"{prefix}.conv",\n',
        '            cache_config=cache_config,\n'
        '            quant_config=quant_config,\n'
        '            prefix=f"{prefix}.conv",\n',
        "LFM2 ShortConv quant_config wiring",
    )


def self_test() -> None:
    """Check the fused recurrence and its post-BF16 row-scale semantics."""
    import numpy as np

    generator = np.random.default_rng(20260727)
    blocks, tokens, dim = 9, 5, 37
    b = generator.standard_normal((tokens, dim)).astype(np.float16)
    c = generator.standard_normal((tokens, dim)).astype(np.float16)
    x = generator.standard_normal((tokens, dim)).astype(np.float16)
    weight = generator.standard_normal((dim, 3)).astype(np.float16)
    indices = np.array([2, 8, 1, 6, 4], dtype=np.int32)

    for state_dtype in (np.float16, np.float32):
        initial_state = generator.standard_normal((blocks, dim, 2)).astype(
            state_dtype
        )
        reference_state = initial_state.copy()
        reference_hidden = np.empty_like(b)
        for row, state_index in enumerate(indices):
            gated = (b[row] * x[row]).astype(np.float16).astype(state_dtype)
            convolution = (
                reference_state[state_index, :, 0].astype(np.float32)
                * weight[:, 0].astype(np.float32)
                + reference_state[state_index, :, 1].astype(np.float32)
                * weight[:, 1].astype(np.float32)
                + gated.astype(np.float32) * weight[:, 2].astype(np.float32)
            )
            rounded = convolution.astype(state_dtype).astype(np.float16)
            reference_hidden[row] = (
                c[row].astype(np.float32) * rounded.astype(np.float32)
            ).astype(np.float16)
            reference_state[state_index, :, 0] = reference_state[
                state_index, :, 1
            ]
            reference_state[state_index, :, 1] = gated

        fused_state = initial_state.copy()
        gated = (b * x).astype(np.float16).astype(state_dtype)
        old_0 = fused_state[indices, :, 0].copy()
        old_1 = fused_state[indices, :, 1].copy()
        convolution = (
            old_0.astype(np.float32) * weight[:, 0].astype(np.float32)
            + old_1.astype(np.float32) * weight[:, 1].astype(np.float32)
            + gated.astype(np.float32) * weight[:, 2].astype(np.float32)
        )
        fused_hidden = (
            c.astype(np.float32)
            * convolution.astype(state_dtype).astype(np.float16).astype(np.float32)
        ).astype(np.float16)
        fused_state[indices, :, 0] = old_1
        fused_state[indices, :, 1] = gated

        np.testing.assert_array_equal(fused_hidden, reference_hidden)
        np.testing.assert_array_equal(fused_state, reference_state)
        minimum_scale = 1.0 / (448.0 * 512.0)
        expected_scale = np.maximum(
            np.max(np.abs(reference_hidden.astype(np.float32)), axis=1) / 448.0,
            minimum_scale,
        )
        fused_scale = np.maximum(
            np.max(np.abs(fused_hidden.astype(np.float32)), axis=1) / 448.0,
            minimum_scale,
        )
        np.testing.assert_array_equal(fused_scale, expected_scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing the vllm package (defaults to site-packages)",
    )
    parser.add_argument("--kernel-source", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()

    root = package_root(args.root)
    short_conv_path = root / SHORT_CONV_PATH
    short_conv_original = short_conv_path.read_text(encoding="utf-8")
    short_conv_patched = patch_short_conv(short_conv_original)
    ast.parse(short_conv_patched, filename=str(short_conv_path))

    lfm2_path = root / LFM2_PATH
    lfm2_original = lfm2_path.read_text(encoding="utf-8")
    lfm2_patched = patch_lfm2(lfm2_original)
    ast.parse(lfm2_patched, filename=str(lfm2_path))

    if args.kernel_source is None:
        if not args.check:
            raise RuntimeError("--kernel-source is required when installing")
    else:
        kernel_source = args.kernel_source.resolve()
        kernel_text = kernel_source.read_text(encoding="utf-8")
        ast.parse(kernel_text, filename=str(kernel_source))
        if not args.check:
            destination = root / KERNEL_PATH
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(kernel_source, destination)

    if not args.check:
        short_conv_path.write_text(short_conv_patched, encoding="utf-8")
        lfm2_path.write_text(lfm2_patched, encoding="utf-8")
    action = "checked" if args.check else "patched"
    print(f"{action}: {SHORT_CONV_PATH}")
    print(f"{action}: {LFM2_PATH}")


if __name__ == "__main__":
    main()
