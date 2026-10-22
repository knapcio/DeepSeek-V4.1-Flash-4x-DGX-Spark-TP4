#!/usr/bin/env python3
"""Packaging and fail-closed checks for SGLang #41259 + #45410."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts/verify_sglang_parallel_prompt_encode.py"
SPEC = importlib.util.spec_from_file_location("verify_parallel_prompt_encode", VERIFY_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class ParallelPromptEncodePackagingTests(unittest.TestCase):
    def test_fetch_pins_helper_and_both_upstream_changes(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        patch = (
            ROOT / "patches/sglang-41259-45410-parallel-native-encode.patch"
        ).read_text()
        self.assertIn("fba3ad4b1f4563143f3352beb148c4c28d4be9ed", fetch)
        self.assertIn(VERIFY.HELPER_SHA256, fetch)
        self.assertIn("fa1adf5c9b51fdf25cb7998e963a7de905c58cb5aa35395cf16d6c75422ecf9e", fetch)
        self.assertIn("45410 aeeb943fe71626a7cea183b716b8bb236a285f2e", fetch)
        self.assertIn("SGLANG_PARALLEL_PROMPT_ENCODE = EnvBool(True)", patch)
        self.assertEqual(patch.count("+                prompt_ids = parallel_prompt_encode("), 3)
        self.assertIn("+        prompt_ids = parallel_prompt_encode(", patch)

    def test_both_canary_images_verify_and_test(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_parallel_prompt_encode.py", source)
                self.assertIn("test_sglang_parallel_prompt_encode_patch.py", source)
                self.assertIn("python3 -m pytest -q /opt/dsv41/upstream-tests/test_parallel_prompt_encode.py", source)


class ParallelPromptEncodeVerifierTests(unittest.TestCase):
    HELPER = "def parallel_prompt_encode(tokenizer, text, encode_kwargs=None):\n    return []\n"
    SERVING = """
from sglang.srt.entrypoints.openai.parallel_prompt_encode import parallel_prompt_encode

def native(tokenizer, a, b, c):
    x = parallel_prompt_encode(tokenizer, a)
    y = parallel_prompt_encode(tokenizer, b)
    z = parallel_prompt_encode(tokenizer, c)
    return parallel_prompt_encode(tokenizer, x, {})
"""
    ENVIRON = """
class Envs:
    SGLANG_PARALLEL_PROMPT_ENCODE = EnvBool(True)
    SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS = EnvInt(32768)
"""

    def make_tree(self, root: Path, *, stale=False) -> tuple[Path, str]:
        paths = {
            "sglang/srt/entrypoints/openai/parallel_prompt_encode.py": self.HELPER,
            "sglang/srt/entrypoints/openai/serving_chat.py": self.SERVING
            + ("\ntokenizer.encode(real_input)\n" if stale else ""),
            "sglang/srt/environ.py": self.ENVIRON,
        }
        for relative, source in paths.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        digest = hashlib.sha256(self.HELPER.encode()).hexdigest()
        return root, digest

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, digest = self.make_tree(Path(tmp))
            VERIFY.verify(root, helper_sha256=digest)

    def test_digest_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_tree(Path(tmp))
            with self.assertRaises(SystemExit):
                VERIFY.verify(root, helper_sha256="0" * 64)

    def test_serial_native_call_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, digest = self.make_tree(Path(tmp), stale=True)
            with self.assertRaises(SystemExit):
                VERIFY.verify(root, helper_sha256=digest)


if __name__ == "__main__":
    unittest.main()
