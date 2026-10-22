#!/usr/bin/env python3
"""Execute the staged production function for SGLang #45518 semantics."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


class _Tensor:
    def __init__(self, shape):
        self.shape = shape

    def reshape(self, *shape):
        return _Tensor(shape)

    def to(self, _dtype):
        return self

    def contiguous(self):
        return self


class _Probabilities:
    def __init__(self):
        self.values = [1.0000068, 0.5]
        self.max_arg = None

    def clamp_(self, *, max):
        self.max_arg = max
        self.values = [min(value, max) for value in self.values]
        return self


class _Torch:
    Tensor = _Tensor
    float32 = object()

    @staticmethod
    def repeat_interleave(tensor, repeats, dim):
        assert repeats == 2
        assert dim == 0
        return tensor


def load_function(python_root: Path):
    target = (
        python_root / "sglang/kernels/ops/speculative/dspark/dspark_accept.py"
    )
    tree = ast.parse(target.read_text(), filename=str(target))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "softmax_temp_flashinfer"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    calls = []

    def flashinfer_softmax(*, logits, temperature):
        calls.append((logits, temperature))
        return _Probabilities()

    namespace = {
        "torch": _Torch,
        "_flashinfer_softmax": flashinfer_softmax,
    }
    exec(compile(module, str(target), "exec"), namespace)
    return namespace["softmax_temp_flashinfer"], calls


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    function, calls = load_function(Path(sys.argv[1]))
    result = function(
        logits=_Tensor((4, 8)),
        temperatures=_Tensor((2,)),
        rows_per_request=2,
    )
    assert len(calls) == 1
    assert result.max_arg == 1.0
    assert result.values == [1.0, 0.5]
    print("SGLang #45518 DSpark FlashInfer probability clamp: passed")


if __name__ == "__main__":
    main()
