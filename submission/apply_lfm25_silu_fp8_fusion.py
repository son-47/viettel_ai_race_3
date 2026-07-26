"""Install the opt-in LFM2.5 SwiGLU + dynamic FP8 fusion into vLLM."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
from pathlib import Path

LFM2_PATH = Path("vllm/model_executor/models/lfm2.py")
KERNEL_PATH = Path("vllm/model_executor/layers/lfm25_fused_silu_fp8.py")


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
        + """from vllm.model_executor.layers.lfm25_fused_silu_fp8 import (
    fused_lfm25_silu_fp8_linear,
    supports_fused_lfm25_silu_fp8_linear,
)
"""
    )
    text = replace_once(
        text,
        import_old,
        import_new,
        "LFM2.5 fused SwiGLU/FP8 import",
    )

    init_old = """        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x
"""
    init_new = """        self.act_fn = SiluAndMul()
        self.use_fused_silu_fp8 = supports_fused_lfm25_silu_fp8_linear(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        if self.use_fused_silu_fp8:
            return fused_lfm25_silu_fp8_linear(gate_up, self.w2)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x
"""
    return replace_once(
        text,
        init_old,
        init_new,
        "LFM2.5 fused SwiGLU/FP8 MLP path",
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
