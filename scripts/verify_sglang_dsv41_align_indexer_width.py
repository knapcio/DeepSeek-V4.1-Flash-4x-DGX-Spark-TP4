#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree SGLang #45794 adaptation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    target = python_root / "sglang/srt/layers/attention/deepseek_v4_backend.py"
    source = target.read_text()
    ast.parse(source, filename=str(target))
    required = (
        "if index_k.is_cuda and index_k.dtype == torch.bfloat16 and lc >= 1024:",
        "score_width = (lc + 63) // 64 * 64",
        "index_k, (0, 0, 0, score_width - lc)",
        "// (q.shape[1] * index_k.shape[0] * 2)",
        "indexer.scores(q[tok_c], index_k, weights[tok_c])[:, :lc]",
        "s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise SystemExit(f"incomplete SGLang #45794 port: {missing}")
    pad = source.index("score_width = (lc + 63) // 64 * 64")
    score = source.index(
        "indexer.scores(q[tok_c], index_k, weights[tok_c])[:, :lc]", pad
    )
    mask = source.index(
        "s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)", score
    )
    if not pad < score < mask:
        raise SystemExit("padded columns are not sliced before causal masking")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45794 aligned V4.1 indexer width: verified")


if __name__ == "__main__":
    main()
