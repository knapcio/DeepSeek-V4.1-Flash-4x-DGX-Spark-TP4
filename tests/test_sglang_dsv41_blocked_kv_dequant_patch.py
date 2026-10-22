#!/usr/bin/env python3
"""Packaging checks for the SGLang #45793 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45793-dsv41-blocked-kv-dequant.patch"


class Dsv41BlockedKvDequantPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("0de6dd957f4da0b0655ef6d41bfd4b0593a08d12", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45793-dsv41-blocked-kv-dequant.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_repeat_verifier_falsifier_and_packaging(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_blocked_kv_dequant.py", source)
                self.assertIn("test_sglang_dsv41_blocked_kv_dequant.py", source)
                self.assertIn("test_sglang_dsv41_blocked_kv_dequant_patch.py", source)


if __name__ == "__main__":
    unittest.main()
