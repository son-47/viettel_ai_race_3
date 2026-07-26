"""Install the opt-in LFM2.5 add-RMSNorm + dynamic FP8 fusion."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
from pathlib import Path


LFM2_PATH = Path("vllm/model_executor/models/lfm2.py")
KERNEL_PATH = Path("vllm/model_executor/layers/lfm25_fused_rmsnorm_fp8.py")


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


def patch_lfm2(text: str) -> str:
    import_old = """from vllm.model_executor.layers.layernorm import RMSNorm
"""
    import_new = (
        import_old
        + """from vllm.model_executor.layers.lfm25_fused_rmsnorm_fp8 import (
    fused_lfm25_add_rmsnorm_fp8_linear,
    supports_fused_lfm25_rmsnorm_fp8_linear,
)
"""
    )
    text = replace_once(
        text,
        import_old,
        import_new,
        "LFM2.5 fused RMSNorm/FP8 import",
    )

    mlp_old = """        self.act_fn = SiluAndMul()
        self.use_fused_silu_fp8 = supports_fused_lfm25_silu_fp8_linear(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        if self.use_fused_silu_fp8:
            return fused_lfm25_silu_fp8_linear(gate_up, self.w2)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x
"""
    mlp_new = """        self.act_fn = SiluAndMul()
        self.use_fused_silu_fp8 = supports_fused_lfm25_silu_fp8_linear(self.w2)
        self.use_fused_rmsnorm_fp8 = supports_fused_lfm25_rmsnorm_fp8_linear(
            self.w13
        )

    def _forward_gate_up(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.use_fused_silu_fp8:
            return fused_lfm25_silu_fp8_linear(gate_up, self.w2)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        return self._forward_gate_up(gate_up)

    def forward_fused_add_rmsnorm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up = fused_lfm25_add_rmsnorm_fp8_linear(
            x, residual, norm, self.w13
        )
        return self._forward_gate_up(gate_up), residual
"""
    text = replace_once(
        text,
        mlp_old,
        mlp_new,
        "LFM2.5 fused FFN RMSNorm/FP8 path",
    )

    attention_init_old = """        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
"""
    attention_init_new = """        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.use_fused_rmsnorm_fp8 = supports_fused_lfm25_rmsnorm_fp8_linear(
            self.qkv_proj
        )

    def _forward_qkv(
        self,
        positions: torch.Tensor,
        qkv: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, _ = qkv.shape
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
"""
    text = replace_once(
        text,
        attention_init_old,
        attention_init_new,
        "LFM2.5 fused QKV RMSNorm/FP8 setup",
    )

    attention_end_old = """        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class Lfm2AttentionDecoderLayer(nn.Module):
"""
    attention_end_new = """        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        return self._forward_qkv(positions, qkv)

    def forward_fused_add_rmsnorm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = fused_lfm25_add_rmsnorm_fp8_linear(
            hidden_states, residual, norm, self.qkv_proj
        )
        return self._forward_qkv(positions, qkv), residual


class Lfm2AttentionDecoderLayer(nn.Module):
"""
    text = replace_once(
        text,
        attention_end_old,
        attention_end_new,
        "LFM2.5 fused QKV RMSNorm/FP8 methods",
    )

    attention_layer_old = """        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        return self.feed_forward(hidden_states), residual
"""
    attention_layer_new = """        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
        elif self.self_attn.use_fused_rmsnorm_fp8:
            hidden_states, residual = self.self_attn.forward_fused_add_rmsnorm(
                positions, hidden_states, residual, self.operator_norm
            )
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
        if self.feed_forward.use_fused_rmsnorm_fp8:
            return self.feed_forward.forward_fused_add_rmsnorm(
                hidden_states, residual, self.ffn_norm
            )
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        return self.feed_forward(hidden_states), residual
"""
    text = replace_once(
        text,
        attention_layer_old,
        attention_layer_new,
        "LFM2.5 attention decoder fused RMSNorm/FP8 path",
    )

    shortconv_layer_old = """        hidden_states, residual = self.ffn_norm(output, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual
"""
    shortconv_layer_new = """        if self.feed_forward.use_fused_rmsnorm_fp8:
            return self.feed_forward.forward_fused_add_rmsnorm(
                output, residual, self.ffn_norm
            )
        hidden_states, residual = self.ffn_norm(output, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual
"""
    return replace_once(
        text,
        shortconv_layer_old,
        shortconv_layer_new,
        "LFM2.5 ShortConv decoder fused RMSNorm/FP8 path",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing the vllm package (defaults to site-packages)",
    )
    parser.add_argument("--kernel-source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = package_root(args.root)
    lfm2_path = root / LFM2_PATH
    original = lfm2_path.read_text(encoding="utf-8")
    patched = patch_lfm2(original)
    ast.parse(patched, filename=str(lfm2_path))

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
        lfm2_path.write_text(patched, encoding="utf-8")
    action = "checked" if args.check else "patched"
    print(f"{action}: {LFM2_PATH}")


if __name__ == "__main__":
    main()
