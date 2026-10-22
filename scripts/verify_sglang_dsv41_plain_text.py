#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree adaptation of SGLang #45457."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path


HELPER_SHA256 = "8e675e59b756678309bc2076c3c71b3525cb02030acdec94e83c0e20d285aa62"


def verify(root: Path, *, helper_sha256: str = HELPER_SHA256) -> None:
    helper = root / "sglang/srt/entrypoints/openai/plain_text_encode.py"
    chat = root / "sglang/srt/entrypoints/openai/chat_encoding.py"
    serving = root / "sglang/srt/entrypoints/openai/serving_chat.py"
    environ = root / "sglang/srt/environ.py"
    for path in (helper, chat, serving, environ):
        if not path.is_file():
            raise SystemExit(f"missing plain-text source: {path}")

    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    if digest != helper_sha256:
        raise SystemExit(f"plain-text helper digest mismatch: {digest}")

    chat_text = chat.read_text()
    serving_text = serving.read_text()
    environ_text = environ.read_text()
    required_chat = (
        "def mark_dsv41_request_text(",
        "def encode_dsv41_prompt(",
        "_PLAIN_TEXT_MARKER not in prompt",
        "return parallel_prompt_encode(tokenizer, prompt)",
        "return plain_text_encode.encode_marked_prompt(tokenizer, prompt)",
        "mark_dsv41_request_text(tokenizer, messages)",
        "from sglang.srt.environ import envs",
    )
    for marker in required_chat:
        if marker not in chat_text:
            raise SystemExit(f"missing chat integration: {marker}")
    required_serving = (
        "messages = chat_encoding.mark_dsv41_request_text(tokenizer, messages)",
        "prompt_ids = chat_encoding.encode_dsv41_prompt(",
    )
    for marker in required_serving:
        if marker not in serving_text:
            raise SystemExit(f"missing serving integration: {marker}")
    task_pos = serving_text.index("encoding_dsv41.attach_task_to_last_user_message(")
    mark_pos = serving_text.index(
        "messages = chat_encoding.mark_dsv41_request_text(tokenizer, messages)"
    )
    if mark_pos < task_pos:
        raise SystemExit("request.task would bypass plain-text marking")
    if "SGLANG_ENABLE_DSV41_PLAIN_TEXT_ENCODE = EnvBool(True)" not in environ_text:
        raise SystemExit("missing default-on environment gate")

    for path in (helper, chat, serving, environ):
        ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45457 dsv41 plain-text encoding verified")
