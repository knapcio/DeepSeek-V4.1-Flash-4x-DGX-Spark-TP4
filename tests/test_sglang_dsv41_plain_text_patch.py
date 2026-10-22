#!/usr/bin/env python3
"""Packaging checks for the SGLang #45457 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts/verify_sglang_dsv41_plain_text.py"
SPEC = importlib.util.spec_from_file_location("verify_dsv41_plain_text", VERIFY_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class Dsv41PlainTextPackagingTests(unittest.TestCase):
    def test_fetch_pins_exact_upstream_helper_and_test(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        patcher = (ROOT / "runtime/patch_dsv41_plain_text_encode.py").read_text()
        self.assertIn("a0206772e54778229648af1187949b1eb0619234", fetch)
        self.assertIn(VERIFY.HELPER_SHA256, fetch)
        self.assertIn(
            "5702531ee250205f8c1aff916f05c89090cfdec220bf06d467efe1ba8b344020",
            fetch,
        )
        self.assertIn("_PLAIN_TEXT_MARKER not in prompt", patcher)
        self.assertIn("parallel_prompt_encode(tokenizer, prompt)", patcher)

    def test_images_verify_and_run_upstream_suite(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_plain_text.py", source)
                self.assertIn("test_sglang_dsv41_plain_text_patch.py", source)
                self.assertIn("upstream-tests/test_plain_text_encode.py", source)


class Dsv41PlainTextVerifierTests(unittest.TestCase):
    HELPER = "def mark_request_text(*args): return args\ndef encode_marked_prompt(*args): return []\n"
    CHAT = """
from sglang.srt.environ import envs
def mark_dsv41_request_text(tokenizer, messages):
    return mark_dsv41_request_text(tokenizer, messages)
def encode_dsv41_prompt(tokenizer, prompt):
    if _PLAIN_TEXT_MARKER not in prompt:
        return parallel_prompt_encode(tokenizer, prompt)
    return plain_text_encode.encode_marked_prompt(tokenizer, prompt)
def offline(tokenizer, messages):
    return mark_dsv41_request_text(tokenizer, messages)
def attach_task_to_last_user_message(): pass
"""
    SERVING = """
encoding_dsv41.attach_task_to_last_user_message(messages, request.task)
messages = chat_encoding.mark_dsv41_request_text(tokenizer, messages)
prompt_ids = chat_encoding.encode_dsv41_prompt(tokenizer, real_input)
"""
    ENVIRON = "SGLANG_ENABLE_DSV41_PLAIN_TEXT_ENCODE = EnvBool(True)\n"

    def make_tree(self, root: Path) -> tuple[Path, str]:
        sources = {
            "sglang/srt/entrypoints/openai/plain_text_encode.py": self.HELPER,
            "sglang/srt/entrypoints/openai/chat_encoding.py": self.CHAT,
            "sglang/srt/entrypoints/openai/serving_chat.py": self.SERVING,
            "sglang/srt/environ.py": self.ENVIRON,
        }
        for relative, source in sources.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return root, hashlib.sha256(self.HELPER.encode()).hexdigest()

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, digest = self.make_tree(Path(tmp))
            VERIFY.verify(root, helper_sha256=digest)

    def test_helper_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_tree(Path(tmp))
            with self.assertRaises(SystemExit):
                VERIFY.verify(root, helper_sha256="0" * 64)

    def test_task_marked_too_early_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, digest = self.make_tree(Path(tmp))
            serving = root / "sglang/srt/entrypoints/openai/serving_chat.py"
            serving.write_text(self.SERVING.splitlines()[2] + "\n" + self.SERVING.splitlines()[1] + "\n" + self.SERVING.splitlines()[3] + "\n")
            with self.assertRaises(SystemExit):
                VERIFY.verify(root, helper_sha256=digest)


if __name__ == "__main__":
    unittest.main()
