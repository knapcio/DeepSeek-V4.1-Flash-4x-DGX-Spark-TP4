#!/usr/bin/env python3
"""Packaging checks for SGLang #45663 accept-output fusion."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_SHA256 = "b44e8522f386acf02050254307ad71a8f1584af3b5943207faa7c0f0c6ee5fe9"


class DsparkAcceptOutputsPackagingTests(unittest.TestCase):
    def test_exact_patch_is_applied_with_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        patch_path = ROOT / "patches/sglang-45663-dspark-accept-outputs.patch"
        self.assertEqual(hashlib.sha256(patch_path.read_bytes()).hexdigest(), PATCH_SHA256)
        self.assertIn("PATCH_DSPARK_ACCEPT", fetch)
        self.assertIn("patch --dry-run --silent --fuzz=0", fetch)
        self.assertIn("verify_sglang_dspark_accept_outputs.py", fetch)

    def test_images_verify_and_test_the_fusion(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dspark_accept_outputs.py", source)
                self.assertIn("test_sglang_dspark_accept_outputs_patch.py", source)


if __name__ == "__main__":
    unittest.main()
