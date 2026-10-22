#!/usr/bin/env python3
"""Adapt SGLang #45458 to the pinned DeepSeek V4/V4.1 parser tree."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new)


IMPORT_OLD = "from typing import Dict, List, Optional, Tuple, Type\n"
IMPORT_NEW = IMPORT_OLD + "\nfrom transformers import PreTrainedTokenizerBase\n"

BASE_ANCHOR = """    def _maybe_apply_force_nonempty_content(
        self, ret: StreamingParseResult
    ) -> StreamingParseResult:
"""
BASE_METHODS = """    @property
    def wants_output_ids(self) -> bool:
        \"\"\"Whether output token ids can still change how this detector parses.\"\"\"
        return False

    def observe_output_ids(self, token_ids: List[int]) -> None:
        \"\"\"Observe generated token ids before their decoded text is parsed.\"\"\"

    def _think_end_index(self, text: str) -> int:
        \"\"\"Index of the think-end marker that closes reasoning, or -1.\"\"\"
        return text.find(self.think_end_token)

""" + BASE_ANCHOR

NO_END_OLD = """        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):
"""
NO_END_NEW = """        end_idx = self._think_end_index(processed_text)
        if end_idx == -1 and self.think_end_token not in self.previous_content:
"""

EXTRACT_OLD = """        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1]
"""
EXTRACT_NEW = """        if end_idx != -1:
            reasoning_text = processed_text[:end_idx]
            normal_text = processed_text[end_idx + len(self.think_end_token) :]
"""

STREAM_END_OLD = """        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]
"""
STREAM_END_NEW = """        end_idx = self._think_end_index(current_text) if self._in_reasoning else -1
        if end_idx != -1:
            reasoning_text = current_text[:end_idx]
"""

DEEPSEEK_OLD = '''class DeepSeekV4Detector(BaseReasoningFormatDetector):
    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            dsv4_thinking_start_token,
            dsv4_thinking_end_token,
            think_excluded_tokens=[dsv4_eos_token, dsv4_dsml_token],
            # Leading "<" included: has_tool_call() matches on it.
            tool_start_token=f"<{dsv4_dsml_token}",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="explicit_thinking",
            force_nonempty_content=force_nonempty_content,
        )
'''

DEEPSEEK_NEW = '''class DeepSeekV4Detector(BaseReasoningFormatDetector):
    """Only the tokenizer's ``</think>`` token ends reasoning when ids exist."""

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        super().__init__(
            dsv4_thinking_start_token,
            dsv4_thinking_end_token,
            think_excluded_tokens=[dsv4_eos_token, dsv4_dsml_token],
            # Leading "<" included: has_tool_call() matches on it.
            tool_start_token=f"<{dsv4_dsml_token}",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="explicit_thinking",
            force_nonempty_content=force_nonempty_content,
        )
        self._tokenizer = tokenizer
        self._think_end_token_id = _single_token_id(tokenizer, self.think_end_token)
        self._output_ids: Optional[List[int]] = None
        self._spelled_ends_before_token: Optional[int] = None
        self._spelled_ends_emitted = 0

    @property
    def wants_output_ids(self) -> bool:
        return (
            self._think_end_token_id is not None
            and self._spelled_ends_before_token is None
        )

    def observe_output_ids(self, token_ids: List[int]) -> None:
        if not self.wants_output_ids:
            return
        if self._output_ids is None:
            self._output_ids = []
        start = len(self._output_ids)
        self._output_ids.extend(token_ids)
        try:
            end = self._output_ids.index(self._think_end_token_id, start)
        except ValueError:
            return
        before = self._tokenizer.decode(
            self._output_ids[:end], skip_special_tokens=True
        )
        self._spelled_ends_before_token = before.count(self.think_end_token)
        self._output_ids = []

    def _think_end_index(self, text: str) -> int:
        if self._output_ids is None:
            return super()._think_end_index(text)
        if self._spelled_ends_before_token is None:
            return -1
        skip = max(self._spelled_ends_before_token - self._spelled_ends_emitted, 0)
        idx, start = -1, 0
        for _ in range(skip + 1):
            idx = text.find(self.think_end_token, start)
            if idx == -1:
                return -1
            start = idx + len(self.think_end_token)
        return idx

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        ret = super().parse_streaming_increment(new_text)
        if self._in_reasoning:
            self._spelled_ends_emitted += ret.reasoning_text.count(self.think_end_token)
        return ret


class DeepSeekV41ReasoningDetector(DeepSeekV4Detector):
    """V4.1 uses a spaced DSML calls opener."""

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.tool_start_token = f"<{dsv4_dsml_token} calls>"
'''

MAP_OLD = '        "deepseek-v41": DeepSeekV4Detector,\n'
MAP_NEW = '        "deepseek-v41": DeepSeekV41ReasoningDetector,\n'

PARSER_INIT_OLD = "        self.detector = detector_class(**kwargs)\n\n"
PARSER_INIT_NEW = (
    "        self.detector = detector_class(**kwargs)\n"
    "        self._stream_output_ids_seen = 0\n\n"
)

PARSE_NONSTREAM_OLD = """    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        \"\"\"Non-streaming call: one-time parsing\"\"\"
        ret = self.detector.detect_and_parse(full_text)
"""
PARSE_NONSTREAM_NEW = """    def parse_non_stream(
        self, full_text: str, output_ids: Optional[List[int]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        \"\"\"One-time parsing, using generated ids when the detector supports them.\"\"\"
        if output_ids is not None and self.detector.wants_output_ids:
            self.detector.observe_output_ids(output_ids)
        ret = self.detector.detect_and_parse(full_text)
"""

STREAM_METHOD_ANCHOR = """    def parse_stream_end(self) -> Tuple[Optional[str], Optional[str]]:
"""
STREAM_METHOD = """    def observe_stream_output_ids(
        self, output_ids: Optional[List[int]], *, cumulative: bool
    ) -> None:
        \"\"\"Pass only newly generated ids to the detector before decoded text.\"\"\"
        if output_ids is None or not self.detector.wants_output_ids:
            return
        if cumulative:
            output_ids = output_ids[self._stream_output_ids_seen :]
        self._stream_output_ids_seen += len(output_ids)
        self.detector.observe_output_ids(output_ids)

""" + STREAM_METHOD_ANCHOR

SINGLE_TOKEN_HELPER = '''

def _single_token_id(
    tokenizer: Optional[PreTrainedTokenizerBase], token: str
) -> Optional[int]:
    """Return ``token``'s id only when the tokenizer represents it as one token."""
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, int) and tokenizer.convert_ids_to_tokens(token_id) == token:
        return token_id
    return None
'''

CHAT_NONSTREAM_OLD = "                    reasoning_text, text = parser.parse_non_stream(text)\n"
CHAT_NONSTREAM_NEW = """                    reasoning_text, text = parser.parse_non_stream(
                        text, output_ids=ret_item.get("output_ids")
                    )
"""
CHAT_STREAM_OLD = """        reasoning_parser = reasoning_parser_dict[index]
        reasoning_text, normal_text = reasoning_parser.parse_stream_chunk(delta)
"""
CHAT_STREAM_NEW = """        reasoning_parser = reasoning_parser_dict[index]
        reasoning_parser.observe_stream_output_ids(
            content.get("output_ids"),
            cumulative=not get_serving().incremental_streaming_output,
        )
        reasoning_text, normal_text = reasoning_parser.parse_stream_chunk(delta)
"""

RESP_CALL_OLD = """                output_logprobs=output_logprobs,
                require_reasoning=require_reasoning,
"""
RESP_CALL_NEW = """                output_logprobs=output_logprobs,
                output_ids=final_res.get("output_ids"),
                require_reasoning=require_reasoning,
"""
RESP_SIG_OLD = """        output_logprobs: Optional[list] = None,
        *,
        require_reasoning: bool,
"""
RESP_SIG_NEW = """        output_logprobs: Optional[list] = None,
        output_ids: Optional[list[int]] = None,
        *,
        require_reasoning: bool,
"""
RESP_PARSE_OLD = "            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)\n"
RESP_PARSE_NEW = """            reasoning_content, content = reasoning_parser.parse_non_stream(
                final_output, output_ids=output_ids
            )
"""
RESP_STREAM_OLD = """                if not delta and finish_reason is None:
                    continue
"""
RESP_STREAM_NEW = """                if reasoning_parser_obj is not None:
                    reasoning_parser_obj.observe_stream_output_ids(
                        chunk.get("output_ids"), cumulative=not incremental
                    )
                if not delta and finish_reason is None:
                    continue
"""


def patch_parser(source: str) -> str:
    source = replace_once(source, IMPORT_OLD, IMPORT_NEW, "parser import")
    source = replace_once(source, BASE_ANCHOR, BASE_METHODS, "base id hooks")
    source = replace_once(source, NO_END_OLD, NO_END_NEW, "non-stream end lookup")
    source = replace_once(source, EXTRACT_OLD, EXTRACT_NEW, "non-stream split")
    source = replace_once(source, STREAM_END_OLD, STREAM_END_NEW, "stream end lookup")
    source = replace_once(source, DEEPSEEK_OLD, DEEPSEEK_NEW, "DeepSeek detector")
    source = replace_once(source, MAP_OLD, MAP_NEW, "V4.1 detector map")
    source = replace_once(source, PARSER_INIT_OLD, PARSER_INIT_NEW, "parser id state")
    source = replace_once(
        source, PARSE_NONSTREAM_OLD, PARSE_NONSTREAM_NEW, "non-stream id API"
    )
    source = replace_once(
        source, STREAM_METHOD_ANCHOR, STREAM_METHOD, "stream id API"
    )
    if "def _single_token_id(" not in source:
        source += SINGLE_TOKEN_HELPER
    return source


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    parser = root / "sglang/srt/parser/reasoning_parser.py"
    chat = root / "sglang/srt/entrypoints/openai/serving_chat.py"
    responses = root / "sglang/srt/entrypoints/openai/serving_responses.py"
    for path in (parser, chat, responses):
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")

    parser_src = patch_parser(parser.read_text())
    chat_src = replace_once(
        chat.read_text(), CHAT_NONSTREAM_OLD, CHAT_NONSTREAM_NEW, "chat non-stream ids"
    )
    chat_src = replace_once(chat_src, CHAT_STREAM_OLD, CHAT_STREAM_NEW, "chat stream ids")
    responses_src = replace_once(
        responses.read_text(), RESP_CALL_OLD, RESP_CALL_NEW, "responses call ids"
    )
    responses_src = replace_once(
        responses_src, RESP_SIG_OLD, RESP_SIG_NEW, "responses signature ids"
    )
    responses_src = replace_once(
        responses_src, RESP_PARSE_OLD, RESP_PARSE_NEW, "responses non-stream ids"
    )
    responses_src = replace_once(
        responses_src, RESP_STREAM_OLD, RESP_STREAM_NEW, "responses stream ids"
    )

    for path, source in (
        (parser, parser_src),
        (chat, chat_src),
        (responses, responses_src),
    ):
        ast.parse(source, filename=str(path))
        path.write_text(source)
    print("adapted SGLang #45458 token-id-aware reasoning end parsing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
