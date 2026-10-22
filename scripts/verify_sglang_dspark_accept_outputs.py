#!/usr/bin/env python3
"""Fail-closed source verification for SGLang #45663 accept-output fusion."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    accept = (
        python_root / "sglang/kernels/ops/speculative/dspark/dspark_accept.py"
    )
    window = (
        python_root
        / "sglang/kernels/ops/speculative/dspark/dspark_verify_window.py"
    )
    verify_path = (
        python_root
        / "sglang/srt/speculative/dspark_components/dspark_verify.py"
    )
    sources = {path: path.read_text() for path in (accept, window, verify_path)}
    for path, source in sources.items():
        ast.parse(source, filename=str(path))

    accept_source = sources[accept]
    accept_needles = (
        "THROUGH: tl.constexpr",
        "out.correct_len.copy_(correct_len)",
        "tl.store(correct_len_out_ptr + offs",
        "tl.store(\n            bonus_out_ptr + offs",
        "THROUGH=out is not None",
    )
    window_source = sources[window]
    window_needles = (
        "out: Optional[torch.Tensor] = None",
        "return out.copy_(out_tokens)",
        "correct_len_i = correct_len.contiguous()",
        "assert out.shape == (bs, T)",
    )
    integration_source = sources[verify_path]
    integration_needles = (
        "static_out = AcceptOuts(",
        "out=static_out,",
        "out=static_out.out_tokens,",
    )
    for label, source, needles in (
        ("accept kernel", accept_source, accept_needles),
        ("out-token kernel", window_source, window_needles),
        ("graph epilogue", integration_source, integration_needles),
    ):
        missing = [needle for needle in needles if needle not in source]
        if missing:
            raise SystemExit(f"incomplete {label} integration: {missing}")

    start = integration_source.index("    def _accept(")
    end = integration_source.index("    def _commit_inject(", start)
    folded_accept = integration_source[start:end]
    forbidden = (
        "self.correct_len_buf[:bs].copy_",
        "self.bonus_buf[:bs].copy_",
        "self.cap_trim_lens_buf[:bs].copy_",
        "self.commit_lens_buf[:bs].copy_",
        "self.new_seq_lens_buf[:bs].copy_",
        "self.out_tokens_buf[:bs].copy_",
    )
    remaining = [needle for needle in forbidden if needle in folded_accept]
    if remaining:
        raise SystemExit(f"legacy post-kernel copy launch remains: {remaining}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45663 DSpark accept-output fusion: verified")


if __name__ == "__main__":
    main()
