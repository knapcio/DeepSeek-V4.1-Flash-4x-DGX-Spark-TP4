#!/usr/bin/env python3
"""Fail-closed verifier for the SGLang #41259 + #45410 backport."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path


HELPER_SHA256 = "722cc41b1096977af3bda6c505e7c629b9b8f7308174cda1c7d2c44f9b54c097"


def verify(root: Path, *, helper_sha256: str = HELPER_SHA256) -> None:
    helper = root / "sglang/srt/entrypoints/openai/parallel_prompt_encode.py"
    serving = root / "sglang/srt/entrypoints/openai/serving_chat.py"
    environ = root / "sglang/srt/environ.py"
    for path in (helper, serving, environ):
        if not path.is_file():
            raise SystemExit(f"missing parallel-encode source: {path}")

    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    if digest != helper_sha256:
        raise SystemExit(f"parallel encoder digest mismatch: {digest}")

    serving_text = serving.read_text()
    environ_text = environ.read_text()
    if serving_text.count("parallel_prompt_encode(") < 4:
        raise SystemExit("expected native dsv4/dsv41/dsv32 plus Jinja integration")
    for stale in (
        "tokenizer.encode(real_input)",
        "tokenizer_manager.tokenizer.encode(real_input)",
    ):
        if stale in serving_text:
            raise SystemExit(f"native encoder still uses serial path: {stale}")
    for field in (
        "SGLANG_PARALLEL_PROMPT_ENCODE = EnvBool(True)",
        "SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS = EnvInt(32768)",
    ):
        if field not in environ_text:
            raise SystemExit(f"missing environment gate: {field}")

    for path in (helper, serving, environ):
        ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang parallel native prompt encoder verified")
