#!/usr/bin/env python3
"""Exercise the staged production method for SGLang #45762 semantics."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


class _EnvValue:
    value = False

    @classmethod
    def get(cls):
        return cls.value


def load_method(python_root: Path):
    target = python_root / "sglang/srt/entrypoints/openai/serving_chat.py"
    tree = ast.parse(target.read_text(), filename=str(target))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_get_reasoning_from_request"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "envs": SimpleNamespace(SGLANG_DEFAULT_THINKING=_EnvValue),
        "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
    }
    exec(compile(module, str(target), "exec"), namespace)
    return namespace["_get_reasoning_from_request"]


def request(kwargs=None):
    return SimpleNamespace(chat_template_kwargs=kwargs, reasoning_effort=None)


def serving(mode: str):
    return SimpleNamespace(
        reasoning_parser="deepseek-v41",
        template_manager=SimpleNamespace(reasoning_config=None),
        _reasoning_detector=SimpleNamespace(reasoning_default=mode),
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    method = load_method(Path(sys.argv[1]))
    dsv41 = serving("explicit_thinking")

    _EnvValue.value = True
    assert method(dsv41, request()) is True
    assert method(dsv41, request({})) is True
    assert method(dsv41, request({"thinking": False})) is False
    assert method(dsv41, request({"thinking": True})) is True

    _EnvValue.value = False
    assert method(dsv41, request()) is False

    mimo = serving("explicit_enable_thinking")
    _EnvValue.value = True
    assert method(mimo, request()) is False
    assert method(mimo, request({"enable_thinking": True})) is True
    print("SGLang #45762 DeepSeek default-thinking semantics: passed")


if __name__ == "__main__":
    main()
