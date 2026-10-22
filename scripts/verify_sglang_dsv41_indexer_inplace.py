#!/usr/bin/env python3
"""Fail-closed verifier for the portable SGLang #45745 indexer change."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    target = python_root / "sglang/srt/layers/attention/dsv4/dsv41_sparse.py"
    source = target.read_text()
    tree = ast.parse(source, filename=str(target))
    indexer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Indexer"
    )
    scores = next(
        node
        for node in indexer.body
        if isinstance(node, ast.FunctionDef) and node.name == "scores"
    )
    rendered = ast.unparse(scores)
    required = (
        "torch.einsum('bhd,nd->bhn', q, k)",
        "s.relu_().mul_(weights.unsqueeze(-1))",
        "return s.sum(dim=1).float()",
    )
    missing = [needle for needle in required if needle not in rendered]
    if missing:
        raise SystemExit(f"incomplete SGLang #45745 port: {missing}")
    forbidden = (
        "s.relu() * weights.unsqueeze(-1)",
        "s = (s.relu()",
    )
    found = [needle for needle in forbidden if needle in source]
    if found:
        raise SystemExit(f"out-of-place score temporaries remain: {found}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45745 in-place V4.1 indexer scoring: verified")


if __name__ == "__main__":
    main()
