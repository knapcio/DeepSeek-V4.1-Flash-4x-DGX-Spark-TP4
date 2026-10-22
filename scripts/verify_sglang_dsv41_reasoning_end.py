#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree adaptation of SGLang #45458."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(root: Path) -> None:
    parser = root / "sglang/srt/parser/reasoning_parser.py"
    chat = root / "sglang/srt/entrypoints/openai/serving_chat.py"
    responses = root / "sglang/srt/entrypoints/openai/serving_responses.py"
    for path in (parser, chat, responses):
        if not path.is_file():
            raise SystemExit(f"missing reasoning-end source: {path}")

    parser_text = parser.read_text()
    chat_text = chat.read_text()
    responses_text = responses.read_text()
    parser_markers = (
        "class DeepSeekV41ReasoningDetector(DeepSeekV4Detector):",
        '"deepseek-v41": DeepSeekV41ReasoningDetector,',
        "def observe_output_ids(self, token_ids: List[int]) -> None:",
        "def observe_stream_output_ids(",
        "self._output_ids.index(self._think_end_token_id, start)",
        "self._spelled_ends_before_token = before.count(self.think_end_token)",
        "def _single_token_id(",
    )
    for marker in parser_markers:
        if marker not in parser_text:
            raise SystemExit(f"missing parser integration: {marker}")
    if '"deepseek-v41": DeepSeekV4Detector,' in parser_text:
        raise SystemExit("V4.1 still aliases the unspaced V4 detector")
    for text, label in ((chat_text, "chat"), (responses_text, "responses")):
        if "observe_stream_output_ids(" not in text:
            raise SystemExit(f"{label} streaming path does not pass output ids")
        if "output_ids=" not in text:
            raise SystemExit(f"{label} non-stream path does not pass output ids")

    for path in (parser, chat, responses):
        ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45458 token-id-aware reasoning end verified")
