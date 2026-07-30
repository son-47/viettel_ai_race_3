"""Add the EAGLE3 auxiliary-hidden-state interface required by DFlash to LFM2.

vLLM 0.25.1 contains the native DFlash proposer and draft model, but its LFM2
target does not expose intermediate layer states.  DFlash uses those states as
context features.  This narrowly ports the existing LlamaModel/EagleModelMixin
pattern to Lfm2Model without changing weights or the normal non-spec output.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path


LFM2_PATH = Path("vllm/model_executor/models/lfm2.py")


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
    if "class Lfm2Model(nn.Module, EagleModelMixin):" in text:
        return text

    text = replace_once(
        text,
        "from .interfaces import HasInnerState, IsHybrid, SupportsLoRA, SupportsPP, SupportsQuant\n",
        """from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
""",
        "LFM2 interfaces import",
    )

    text = replace_once(
        text,
        "class Lfm2Model(nn.Module):\n",
        "class Lfm2Model(nn.Module, EagleModelMixin):\n",
        "LFM2 EagleModelMixin",
    )

    loop_old = '''        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states
'''
    loop_new = '''        aux_hidden_states = self._maybe_add_hidden_state(
            [], 0, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
'''
    text = replace_once(text, loop_old, loop_new, "LFM2 auxiliary hidden states")

    class_old = '''class Lfm2ForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid, SupportsQuant
):
'''
    class_new = '''class Lfm2ForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsEagle3,
    SupportsQuant,
):
'''
    text = replace_once(text, class_old, class_new, "LFM2 SupportsEagle3")

    ast.parse(text, filename=str(LFM2_PATH))
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing the vllm package (defaults to site-packages)",
    )
    args = parser.parse_args()

    root = package_root(args.root)
    path = root / LFM2_PATH
    original = path.read_text(encoding="utf-8")
    patched = patch_lfm2(original)
    path.write_text(patched, encoding="utf-8")
    print(f"Patched DFlash auxiliary hidden-state support: {path}")


if __name__ == "__main__":
    main()
