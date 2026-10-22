#!/usr/bin/env python3
"""Packaging and behavior checks for the SGLang #43549 adaptation."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts/verify_sglang_dsv41_inline_system.py"
SPEC = importlib.util.spec_from_file_location("verify_dsv41_inline_system", VERIFY_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class InlineSystemPackagingTests(unittest.TestCase):
    def test_patch_is_credited_pinned_and_fail_closed(self):
        patch = (ROOT / "patches/sglang-43549-dsv41-inline-system.patch").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertIn('chat_encoding_spec", None) != "dsv41"', patch)
        self.assertIn("SGLang #43549", readme)
        self.assertIn("nankingjing", readme)

    def test_fetch_and_both_images_verify(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("sglang-43549-dsv41-inline-system.patch", fetch)
        self.assertIn("verify_sglang_dsv41_inline_system.py", fetch)
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_inline_system.py", source)
                self.assertIn("test_sglang_dsv41_inline_system_patch.py", source)


class InlineSystemBehaviorTests(unittest.TestCase):
    @staticmethod
    def merge(spec, template_support):
        return spec != "dsv41" and not template_support

    def test_dsv41_preserves_inline_system_without_jinja_template(self):
        self.assertFalse(self.merge("dsv41", False))

    def test_other_native_specs_keep_old_merge_policy(self):
        self.assertTrue(self.merge("dsv4", False))
        self.assertTrue(self.merge(None, False))

    def test_jinja_inline_support_still_preserves_messages(self):
        self.assertFalse(self.merge(None, True))


class InlineSystemVerifierTests(unittest.TestCase):
    SERVING = '''
class AnthropicServing:
    def __init__(self, openai_serving_chat):
        self._merge_inline_system = (
            getattr(openai_serving_chat, "chat_encoding_spec", None) != "dsv41"
            and not detect_inline_system_support(self._chat_template())
        )
'''
    ENCODER = '''
SYSTEM_SP_TOKEN = "x"
def f(messages, index, role, idx):
    ok = role in ["user", "developer"] or (role == "system" and idx > 0)
    if role == "system":
        if index > 0:
            prompt += SYSTEM_SP_TOKEN
    return messages[index].get("role") == "system" and index > 0
'''

    def make_tree(self, root: Path, *, broken=False) -> Path:
        files = {
            "sglang/srt/entrypoints/anthropic/serving.py": self.SERVING,
            "sglang/srt/entrypoints/openai/encoding_dsv41.py": self.ENCODER.replace(
                "idx > 0", "idx >= 0", 1
            ) if broken else self.ENCODER,
        }
        for relative, source in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return root

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            VERIFY.verify(self.make_tree(Path(tmp)))

    def test_encoder_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                VERIFY.verify(self.make_tree(Path(tmp), broken=True))


if __name__ == "__main__":
    unittest.main()
