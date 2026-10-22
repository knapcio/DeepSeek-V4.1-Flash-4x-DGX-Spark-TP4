#!/usr/bin/env python3
"""Fail-closed source verifier for the pinned-tree SGLang #45793 port."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    target = python_root / "sglang/kernels/ops/attention/dsv4/dequant_k_cache.py"
    source = target.read_text()
    ast.parse(source, filename=str(target))
    required = (
        "_V41_BLOCK_TOKENS = 8",
        "triton.cdiv(num_tokens, _V41_BLOCK_TOKENS)",
        "num_warps=4",
        "def _v41_token_bases(",
        "mask = token_ids < num_tokens",
        "tl.load(page_table_ptr + token_ids, mask=mask, other=0)",
        "return token_ids, mask[:, None], data_base[:, None], scale_base[:, None]",
        "offs = tl.arange(0, DATA_BYTES)[None, :]",
        "boffs = tl.arange(0, DATA_BYTES)[None, :]",
        "tl.reshape(tl.join(lo, hi), (BLOCK_TOKENS, 2 * DATA_BYTES))",
        "output_ptr + token_ids[:, None] * output_stride_0 + ooffs",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise SystemExit(f"incomplete SGLang #45793 port: {missing}")
    if source.count("mask=mask") < 6:
        raise SystemExit("blocked V4.1 loads/stores are not all tail-masked")
    for kernel in (
        "_dequantize_k_cache_paged_v41_fp8_kernel",
        "_dequantize_k_cache_paged_v41_fp4_kernel",
    ):
        start = source.index(f"def {kernel}(")
        body = source[start : source.find("\n\n@triton.jit", start)]
        if (
            "_v41_token_bases(" not in body
            or "BLOCK_TOKENS: tl.constexpr" not in body
        ):
            raise SystemExit(f"{kernel} does not use the shared blocked/tail-safe mapping")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45793 blocked V4.1 KV dequantization: verified")


if __name__ == "__main__":
    main()
