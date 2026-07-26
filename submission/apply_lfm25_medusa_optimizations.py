"""Patch the pinned vLLM image for LFM2.5 Medusa inference.

The patch deliberately targets exact source fragments and fails when the base
image drifts.  It combines three independent changes:

1. Backport vLLM PR #48917 so online quantization reaches LFM2 ShortConv's
   in/out projections.
2. Fuse the three one-layer Medusa residual heads into one GEMM.
3. Pack the three shared-LM-head vocabulary projections into one GEMM.

Use ``--check`` to validate without modifying files.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path


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
    # Fail loudly unless this is the hybrid-speculative LFM2 base image.
    for marker in (
        "num_accepted_tokens=num_accepted_tokens",
        "num_spec=self.num_spec",
    ):
        if marker not in text:
            raise RuntimeError(f"ShortConv speculative support missing: {marker}")

    text = replace_once(
        text,
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n",
        "from vllm.model_executor.layers.quantization import QuantizationConfig\n"
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n",
        "ShortConv QuantizationConfig import",
    )
    text = replace_once(
        text,
        "        cache_config: CacheConfig | None = None,\n"
        "        prefix: str = \"\",\n",
        "        cache_config: CacheConfig | None = None,\n"
        "        quant_config: QuantizationConfig | None = None,\n"
        "        prefix: str = \"\",\n",
        "ShortConv quant_config argument",
    )
    text = replace_once(
        text,
        "            bias=self.bias,\n"
        "            prefix=f\"{prefix}.in_proj\",\n",
        "            bias=self.bias,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.in_proj\",\n",
        "ShortConv in_proj quantization",
    )
    text = replace_once(
        text,
        "            bias=self.bias,\n"
        "            prefix=f\"{prefix}.out_proj\",\n",
        "            bias=self.bias,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.out_proj\",\n",
        "ShortConv out_proj quantization",
    )
    return text


def patch_lfm2(text: str) -> str:
    if "num_spec=vllm_config.num_speculative_tokens" not in text:
        raise RuntimeError("LFM2 speculative state-shape support is missing")
    return replace_once(
        text,
        "            cache_config=cache_config,\n"
        "            prefix=f\"{prefix}.conv\",\n",
        "            cache_config=cache_config,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.conv\",\n",
        "LFM2 ShortConv quant_config wiring",
    )


def patch_medusa_model(text: str) -> str:
    text = replace_once(
        text,
        "import torch\nimport torch.nn as nn\n",
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
        "Medusa functional import",
    )

    old_forward = '''    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        return [block(hidden_states) for block in self.blocks]
'''
    new_forward = '''    def finalize_fused_heads(self) -> bool:
        \"\"\"Pack one-layer Medusa heads after checkpoint loading.

        The competition checkpoint has three independent, bias-free residual
        heads with one linear layer each.  Concatenating their output weights
        changes three small GEMMs into one larger GEMM.  The original modules
        are released only after all weights have been loaded.
        \"\"\"
        if self.config.num_hidden_layers != 1 or self.config.num_heads <= 1:
            return False

        linears = [block.layers[0] for block in self.blocks]
        biases = [linear.bias for linear in linears]
        if not (all(bias is None for bias in biases) or all(bias is not None for bias in biases)):
            return False

        fused_weight = torch.cat(
            [linear.weight.detach() for linear in linears], dim=0
        ).contiguous()
        fused_bias = (
            None
            if biases[0] is None
            else torch.cat([bias.detach() for bias in biases], dim=0).contiguous()
        )
        self.register_buffer("_fused_head_weight", fused_weight, persistent=False)
        self.register_buffer("_fused_head_bias", fused_bias, persistent=False)
        self.blocks = nn.ModuleList()
        return True

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        fused_weight = self._buffers.get("_fused_head_weight")
        if fused_weight is None:
            return [block(hidden_states) for block in self.blocks]

        projected = F.linear(
            hidden_states,
            fused_weight,
            self._buffers.get("_fused_head_bias"),
        )
        projected = projected.view(
            *hidden_states.shape[:-1],
            self.config.num_heads,
            self.config.hidden_size,
        )
        return [
            hidden_states + F.silu(head_projection)
            for head_projection in projected.unbind(dim=-2)
        ]
'''
    text = replace_once(
        text,
        old_forward,
        new_forward,
        "Medusa residual-head fusion",
    )

    old_logits = '''        logits_lst: list[torch.Tensor] = []

        for hs, lm_head in zip(hidden_states, self.lm_heads):
'''
    new_logits = '''        logits_lst: list[torch.Tensor] = []

        # original_lm_head=True means every Medusa head uses the same output
        # matrix.  Pack head-major hidden states so one vocabulary GEMM replaces
        # three serial GEMMs; split restores the exact public API.
        if (
            getattr(self.config, "original_lm_head", False)
            and self.token_map is None
            and hidden_states
        ):
            token_counts = [hs.shape[0] for hs in hidden_states]
            packed_hidden_states = torch.cat(hidden_states, dim=0)
            packed_logits = self.logits_processor(
                self.lm_head, packed_hidden_states
            )
            if packed_logits is None:
                return []
            return list(torch.split(packed_logits, token_counts, dim=0))

        for hs, lm_head in zip(hidden_states, self.lm_heads):
'''
    return replace_once(
        text,
        old_logits,
        new_logits,
        "Medusa shared LM-head fusion",
    )


def patch_medusa_proposer(text: str) -> str:
    old = '''            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.spec_config.draft_model_config,
            )
        assert not (
'''
    new = '''            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.spec_config.draft_model_config,
            )
            # get_model() has completed checkpoint loading, so it is now safe
            # to pack and release the individual residual-head parameters.
            self.model.finalize_fused_heads()
        assert not (
'''
    return replace_once(text, old, new, "Medusa proposer finalization")


PATCHERS = {
    Path("vllm/model_executor/layers/mamba/short_conv.py"): patch_short_conv,
    Path("vllm/model_executor/models/lfm2.py"): patch_lfm2,
    Path("vllm/model_executor/models/medusa.py"): patch_medusa_model,
    Path("vllm/v1/spec_decode/medusa.py"): patch_medusa_proposer,
}


def self_test() -> None:
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError:
        # The lightweight local validation environment may not ship PyTorch;
        # the final Docker build always exercises the PyTorch branch.
        import numpy as np

        generator = np.random.default_rng(20260726)
        batch, hidden, heads = 7, 16, 3
        x = generator.standard_normal((batch, hidden))
        weights = [generator.standard_normal((hidden, hidden)) for _ in range(heads)]

        def silu(value):
            return value / (1.0 + np.exp(-value))

        reference = [x + silu(x @ weight.T) for weight in weights]
        packed = x @ np.concatenate(weights, axis=0).T
        fused = [x + silu(item) for item in packed.reshape(batch, heads, hidden).transpose(1, 0, 2)]
        for expected, actual in zip(reference, fused):
            np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)

        vocab_weights = generator.standard_normal((31, hidden))
        reference_logits = [item @ vocab_weights.T for item in reference]
        fused_logits = np.concatenate(reference, axis=0) @ vocab_weights.T
        for expected, actual in zip(reference_logits, np.split(fused_logits, heads)):
            np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
        return

    generator = torch.Generator().manual_seed(20260726)
    batch, hidden, heads = 7, 16, 3
    x = torch.randn(batch, hidden, generator=generator, dtype=torch.float64)
    weights = [
        torch.randn(hidden, hidden, generator=generator, dtype=torch.float64)
        for _ in range(heads)
    ]
    reference = [x + functional.silu(functional.linear(x, weight)) for weight in weights]
    packed = functional.linear(x, torch.cat(weights, dim=0))
    packed = packed.view(batch, heads, hidden)
    fused = [x + functional.silu(item) for item in packed.unbind(dim=1)]
    for expected, actual in zip(reference, fused):
        torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)

    vocab_weights = torch.randn(31, hidden, generator=generator, dtype=torch.float64)
    reference_logits = [functional.linear(item, vocab_weights) for item in reference]
    fused_logits = functional.linear(torch.cat(reference, dim=0), vocab_weights)
    split_logits = fused_logits.split([batch] * heads, dim=0)
    for expected, actual in zip(reference_logits, split_logits):
        torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing the vllm package (defaults to site-packages)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate all exact replacements and syntax without writing",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()

    root = package_root(args.root)
    for relative_path, patcher in PATCHERS.items():
        path = root / relative_path
        original = path.read_text(encoding="utf-8")
        patched = patcher(original)
        ast.parse(patched, filename=str(path))
        if not args.check:
            path.write_text(patched, encoding="utf-8")
        print(f"{'checked' if args.check else 'patched'}: {relative_path}")


if __name__ == "__main__":
    main()
