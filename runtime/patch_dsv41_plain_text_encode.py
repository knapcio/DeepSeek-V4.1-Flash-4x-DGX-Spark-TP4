#!/usr/bin/env python3
"""Adapt SGLang #45457 to the pinned V4.1 tree without losing #41259.

The upstream change separates template-written control tokens from identical
spellings supplied in request text.  This fail-closed patcher wires the exact,
content-pinned upstream helper into the older serving tree.  Clean prompts
still use ``parallel_prompt_encode``; only prompts carrying escaped request
spellings take the helper's segmented encoding path.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


IMPORT_OLD = "from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv41\n"
IMPORT_NEW = """from sglang.srt.entrypoints.openai import (
    encoding_dsv4,
    encoding_dsv41,
    plain_text_encode,
)
from sglang.srt.entrypoints.openai.parallel_prompt_encode import parallel_prompt_encode
from sglang.srt.environ import envs
"""

FUNCTION_ANCHOR = "\ndef spec_owns_reasoning_history(spec: Optional[str]) -> bool:\n"
FUNCTIONS = r'''

# Grounding tokens are model output; deepseek-recipe encodes them as control
# tokens when they appear in assistant content or reasoning content.
_DSV41_GROUNDING_TOKENS = (
    "<｜ref｜>",
    "<｜/ref｜>",
    "<｜box｜>",
    "<｜/box｜>",
    "<｜point｜>",
    "<｜/point｜>",
    "<｜polygon｜>",
    "<｜/polygon｜>",
)
_ASSISTANT_OUTPUT_FIELDS = ("content", "reasoning_content")
_PLAIN_TEXT_MARKER = "\ufdd0"


def mark_dsv41_request_text(
    tokenizer: Any, messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Mark request control-token spellings before rendering the V4.1 prompt."""
    if not envs.SGLANG_ENABLE_DSV41_PLAIN_TEXT_ENCODE.get():
        return messages
    marked = []
    for message in messages:
        if message.get("role") != "assistant":
            marked.append(plain_text_encode.mark_request_text(tokenizer, message))
            continue
        rest = {k: v for k, v in message.items() if k not in _ASSISTANT_OUTPUT_FIELDS}
        marked_message = plain_text_encode.mark_request_text(tokenizer, rest)
        for field in _ASSISTANT_OUTPUT_FIELDS:
            if field in message:
                marked_message[field] = plain_text_encode.mark_request_text(
                    tokenizer, message[field], keep=_DSV41_GROUNDING_TOKENS
                )
        marked.append(marked_message)
    return marked


def encode_dsv41_prompt(tokenizer: Any, prompt: str) -> List[int]:
    """Encode a marked prompt, retaining parallel encoding for clean prompts."""
    if (
        not envs.SGLANG_ENABLE_DSV41_PLAIN_TEXT_ENCODE.get()
        or _PLAIN_TEXT_MARKER not in prompt
    ):
        return parallel_prompt_encode(tokenizer, prompt)
    return plain_text_encode.encode_marked_prompt(tokenizer, prompt)
'''

SIMPLE_OLD = """        elif spec == "dsv41":
            real_input = encoding_dsv41.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        else:
"""
SIMPLE_NEW = """        elif spec == "dsv41":
            real_input = encoding_dsv41.encode_messages(
                mark_dsv41_request_text(tokenizer, messages),
                thinking_mode=thinking_mode,
            )
            return encode_dsv41_prompt(tokenizer, real_input)
        else:
"""

SERVING_OLD = """            elif is_dsv41:
                if request.task is not None:
                    encoding_dsv41.attach_task_to_last_user_message(
                        messages, request.task
                    )
                real_input, media = encoding_dsv41.encode_messages(
"""
SERVING_NEW = """            elif is_dsv41:
                if request.task is not None:
                    encoding_dsv41.attach_task_to_last_user_message(
                        messages, request.task
                    )
                tokenizer = self.tokenizer_manager.tokenizer
                messages = chat_encoding.mark_dsv41_request_text(tokenizer, messages)
                real_input, media = encoding_dsv41.encode_messages(
"""

ENCODE_OLD = """                prompt_ids = parallel_prompt_encode(
                    self.tokenizer_manager.tokenizer, real_input
                )
            else:
"""
ENCODE_NEW = """                prompt_ids = chat_encoding.encode_dsv41_prompt(
                    tokenizer, real_input
                )
            else:
"""

ENV_OLD = "    SGLANG_DSV41_REASONING_EFFORT = EnvStr(\"high\")\n"
ENV_NEW = ENV_OLD + """    # Encode request control-token spellings as plain text, matching deepseek-recipe.
    SGLANG_ENABLE_DSV41_PLAIN_TEXT_ENCODE = EnvBool(True)
"""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    chat = root / "sglang/srt/entrypoints/openai/chat_encoding.py"
    serving = root / "sglang/srt/entrypoints/openai/serving_chat.py"
    environ = root / "sglang/srt/environ.py"
    helper = root / "sglang/srt/entrypoints/openai/plain_text_encode.py"
    for path in (chat, serving, environ, helper):
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")

    chat_src = chat.read_text()
    chat_src = replace_once(chat_src, IMPORT_OLD, IMPORT_NEW, "chat imports")
    if "def mark_dsv41_request_text(" not in chat_src:
        chat_src = replace_once(
            chat_src, FUNCTION_ANCHOR, FUNCTIONS + FUNCTION_ANCHOR, "chat helpers"
        )
    chat_src = replace_once(chat_src, SIMPLE_OLD, SIMPLE_NEW, "offline dsv41")

    serving_src = serving.read_text()
    serving_src = replace_once(
        serving_src, SERVING_OLD, SERVING_NEW, "serving request marking"
    )
    serving_src = replace_once(
        serving_src, ENCODE_OLD, ENCODE_NEW, "serving marked encoding"
    )

    environ_src = replace_once(environ.read_text(), ENV_OLD, ENV_NEW, "environment gate")

    for path, source in (
        (chat, chat_src),
        (serving, serving_src),
        (environ, environ_src),
    ):
        ast.parse(source, filename=str(path))
        path.write_text(source)
    print("adapted SGLang #45457 with clean-prompt parallel encoding preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
