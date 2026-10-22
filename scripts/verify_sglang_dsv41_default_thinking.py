#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree SGLang #45762 adaptation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    target = python_root / "sglang/srt/entrypoints/openai/serving_chat.py"
    source = target.read_text()
    ast.parse(source, filename=str(target))
    required = (
        'if mode in ("explicit_thinking", "explicit_enable_thinking"):',
        "kwargs = request.chat_template_kwargs or {}",
        "if toggle in kwargs:",
        "return kwargs.get(toggle) is True",
        'if toggle == "thinking":',
        "return bool(envs.SGLANG_DEFAULT_THINKING.get())",
        "return False",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise SystemExit(f"incomplete SGLang #45762 port: {missing}")

    start = source.index(
        'if mode in ("explicit_thinking", "explicit_enable_thinking"):'
    )
    explicit = source.index("if toggle in kwargs:", start)
    env_default = source.index('if toggle == "thinking":', explicit)
    fallback = source.index("return False", env_default)
    warning = source.index("logger.warning(", fallback)
    if not start < explicit < env_default < fallback < warning:
        raise SystemExit("default-thinking precedence is not fail-closed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45762 DeepSeek default-thinking parser parity: verified")
