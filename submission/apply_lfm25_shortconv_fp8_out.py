"""Install the opt-in LFM2.5 ShortConv-to-FP8 decode fusion."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
from pathlib import Path


SHORT_CONV_PATH = Path("vllm/model_executor/layers/mamba/short_conv.py")
KERNEL_PATH = Path(
    "vllm/model_executor/layers/mamba/ops/lfm25_fused_shortconv_fp8_out.py"
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
    import_old = '''from vllm.model_executor.layers.mamba.ops.lfm25_fused_short_conv import (
    BYPASS_SINGLE_VSTACK,
    FUSED_DECODE_ENABLED,
    fused_lfm25_short_conv_decode,
)
'''
    import_new = import_old + '''from vllm.model_executor.layers.mamba.ops.lfm25_fused_shortconv_fp8_out import (
    fused_lfm25_shortconv_fp8_out_linear,
    supports_fused_lfm25_shortconv_fp8_out,
)
'''
    text = replace_once(
        text,
        import_old,
        import_new,
        "ShortConv/FP8 kernel import",
    )

    # The previous full-ShortConv online-FP8 candidate regressed.  Keep the
    # 3x-larger input projection in BF16 and isolate FP8 to out_proj, where the
    # dynamic quantizer is now fused into the recurrent kernel.
    input_quant_old = '''        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
'''
    input_quant_new = '''        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            prefix=f"{prefix}.in_proj",
        )
'''
    text = replace_once(
        text,
        input_quant_old,
        input_quant_new,
        "disable ShortConv in_proj FP8",
    )

    init_old = '''        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
'''
    init_new = '''        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.use_fused_shortconv_fp8_out = (
            supports_fused_lfm25_shortconv_fp8_out(self.out_proj)
        )

        compilation_config = get_current_vllm_config().compilation_config
'''
    text = replace_once(
        text,
        init_old,
        init_new,
        "ShortConv/FP8 support guard",
    )

    decode_old = '''        if has_decode:
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
'''
    decode_new = '''        if has_decode:
            can_fuse_decode = (
                self.L_cache == 3
                and self.conv.bias is None
                and state_indices_tensor_d is not None
                and attn_metadata.num_accepted_tokens is None
                and num_decodes == attn_metadata.num_decode_tokens
            )
            if (
                not has_prefill
                and self.use_fused_shortconv_fp8_out
                and can_fuse_decode
            ):
                contextualized_states = fused_lfm25_shortconv_fp8_out_linear(
                    B_d,
                    C_d,
                    x_d,
                    conv_state,
                    conv_weights,
                    state_indices_tensor_d,
                    self.out_proj,
                )
                output[:num_actual_tokens].copy_(contextualized_states)
                return
            if FUSED_DECODE_ENABLED and can_fuse_decode:
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
'''
    return replace_once(
        text,
        decode_old,
        decode_new,
        "ShortConv decode-to-FP8 fast path",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--kernel-source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = package_root(args.root)
    source_path = root / SHORT_CONV_PATH
    original = source_path.read_text(encoding="utf-8")
    patched = patch_short_conv(original)
    ast.parse(patched, filename=str(source_path))

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
        source_path.write_text(patched, encoding="utf-8")
    print(f"{'checked' if args.check else 'patched'}: {SHORT_CONV_PATH}")


if __name__ == "__main__":
    main()
