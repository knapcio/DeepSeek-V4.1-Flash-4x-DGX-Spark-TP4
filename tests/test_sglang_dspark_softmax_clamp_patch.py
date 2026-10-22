#!/usr/bin/env python3
"""Packaging checks for the SGLang #45518 DSpark probability clamp."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45518-dspark-softmax-clamp.patch"


class DsparkSoftmaxClampPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("8e85d6cfa9fc9025942c4faf07c362c1ce3d06f4", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45518-dspark-softmax-clamp.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_repeat_verifier_semantics_and_packaging(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dspark_softmax_clamp.py", source)
                self.assertIn("test_sglang_dspark_softmax_clamp.py", source)
                self.assertIn(
                    "test_sglang_dspark_softmax_clamp_patch.py", source
                )


if __name__ == "__main__":
    unittest.main()
