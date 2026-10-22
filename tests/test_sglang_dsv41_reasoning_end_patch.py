#!/usr/bin/env python3
"""Packaging checks for the SGLang #45458 pinned-tree adaptation."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts/verify_sglang_dsv41_reasoning_end.py"
SPEC = importlib.util.spec_from_file_location("verify_dsv41_reasoning_end", VERIFY_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class Dsv41ReasoningEndPackagingTests(unittest.TestCase):
    def test_fetch_pins_upstream_test_and_applies_adaptation(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        patcher = (ROOT / "runtime/patch_dsv41_reasoning_end_token.py").read_text()
        self.assertIn("2592a6d211b56f503707760b158a635510baaf9b", fetch)
        self.assertIn(
            "e1d3bb204617a08abea100c3b5be6de49d8c68483d29de90298c12f10ea204c9",
            fetch,
        )
        self.assertIn("_spelled_ends_before_token", patcher)
        self.assertIn("observe_stream_output_ids", patcher)

    def test_images_verify_and_run_upstream_reasoning_suite(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_reasoning_end.py", source)
                self.assertIn("test_sglang_dsv41_reasoning_end_patch.py", source)
                self.assertIn("upstream-tests/test_reasoning_parser.py", source)


class Dsv41ReasoningEndVerifierTests(unittest.TestCase):
    PARSER = """
class DeepSeekV4Detector: pass
class DeepSeekV41ReasoningDetector(DeepSeekV4Detector): pass
DetectorMap = {"deepseek-v41": DeepSeekV41ReasoningDetector,}
def observe_output_ids(self, token_ids: List[int]) -> None: pass
def observe_stream_output_ids(self): pass
x = self._output_ids.index(self._think_end_token_id, start)
self._spelled_ends_before_token = before.count(self.think_end_token)
def _single_token_id(): pass
"""
    ENDPOINT = "observe_stream_output_ids([])\nparse(output_ids=[])\n"

    def make_tree(self, root: Path, parser_source: str | None = None) -> Path:
        sources = {
            "sglang/srt/parser/reasoning_parser.py": parser_source or self.PARSER,
            "sglang/srt/entrypoints/openai/serving_chat.py": self.ENDPOINT,
            "sglang/srt/entrypoints/openai/serving_responses.py": self.ENDPOINT,
        }
        for relative, source in sources.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return root

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            VERIFY.verify(self.make_tree(Path(tmp)))

    def test_old_v41_alias_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.PARSER.replace(
                '"deepseek-v41": DeepSeekV41ReasoningDetector',
                '"deepseek-v41": DeepSeekV4Detector',
            )
            with self.assertRaises(SystemExit):
                VERIFY.verify(self.make_tree(Path(tmp), source))


if __name__ == "__main__":
    unittest.main()
