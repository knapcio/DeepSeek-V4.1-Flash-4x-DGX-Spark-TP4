#!/usr/bin/env python3
"""Run the exact #45458 focused class on the older pinned SGLang layout."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} TEST_REASONING_PARSER.py")
    test_path = Path(sys.argv[1])
    if not test_path.is_file():
        raise SystemExit(f"missing upstream test: {test_path}")

    # The newer upstream test imports a registry module introduced after our
    # pinned source. Its #45458 class does not use that registry; provide the
    # equivalent names solely so pytest can collect the exact content-pinned file.
    try:
        import sglang.srt.parser.reasoning_parser_names  # noqa: F401
    except ModuleNotFoundError:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        module = types.ModuleType("sglang.srt.parser.reasoning_parser_names")
        module.REASONING_PARSER_NAMES = tuple(sorted(ReasoningParser.DetectorMap))
        sys.modules[module.__name__] = module

    return pytest.main(["-q", f"{test_path}::TestDeepSeekV4ThinkEndTokenId"])


if __name__ == "__main__":
    raise SystemExit(main())
