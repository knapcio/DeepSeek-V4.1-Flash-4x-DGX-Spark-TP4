#!/usr/bin/env python3
"""Fail-closed verifier for SGLang #45518's DSpark probability clamp."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    target = (
        python_root / "sglang/kernels/ops/speculative/dspark/dspark_accept.py"
    )
    tree = ast.parse(target.read_text(), filename=str(target))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "softmax_temp_flashinfer"
        ),
        None,
    )
    if function is None:
        raise SystemExit("softmax_temp_flashinfer is missing")

    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    flashinfer_assignments = [
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_flashinfer_softmax"
    ]
    clamp_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "clamp_"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "probs"
    ]
    if len(flashinfer_assignments) != 1 or len(clamp_calls) != 1:
        raise SystemExit("FlashInfer output is not clamped exactly once")

    clamp = clamp_calls[0]
    max_kw = next((kw for kw in clamp.keywords if kw.arg == "max"), None)
    if (
        max_kw is None
        or not isinstance(max_kw.value, ast.Constant)
        or max_kw.value.value != 1.0
    ):
        raise SystemExit("FlashInfer output clamp does not use max=1.0")

    if not any(
        isinstance(node, ast.Return) and node.value is clamp for node in function.body
    ):
        raise SystemExit("clamped probabilities are not returned directly")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45518 DSpark FlashInfer probability clamp: verified")


if __name__ == "__main__":
    main()
