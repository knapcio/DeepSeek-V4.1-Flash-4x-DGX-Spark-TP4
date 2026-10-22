#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree adaptation of SGLang #43549."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(root: Path) -> None:
    serving = root / "sglang/srt/entrypoints/anthropic/serving.py"
    encoder = root / "sglang/srt/entrypoints/openai/encoding_dsv41.py"
    for path in (serving, encoder):
        if not path.is_file():
            raise SystemExit(f"missing dsv41 inline-system source: {path}")

    serving_text = serving.read_text()
    encoder_text = encoder.read_text()
    required_serving = (
        'getattr(openai_serving_chat, "chat_encoding_spec", None) != "dsv41"',
        "and not detect_inline_system_support(self._chat_template())",
    )
    for anchor in required_serving:
        if anchor not in serving_text:
            raise SystemExit(f"Anthropic dsv41 inline-system gate missing: {anchor}")

    required_encoder = (
        'role in ["user", "developer"] or (role == "system" and idx > 0)',
        'if role == "system":',
        "if index > 0:",
        "prompt += SYSTEM_SP_TOKEN",
        'messages[index].get("role") == "system" and index > 0',
    )
    for anchor in required_encoder:
        if anchor not in encoder_text:
            raise SystemExit(f"native dsv41 inline-system support drifted: {anchor}")

    ast.parse(serving_text, filename=str(serving))
    ast.parse(encoder_text, filename=str(encoder))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #43549 dsv41 inline-system preservation verified")
