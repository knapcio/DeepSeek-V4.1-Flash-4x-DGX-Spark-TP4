#!/usr/bin/env python3
"""Packaging checks for the SGLang #45794 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45794-dsv41-align-indexer-width.patch"


class Dsv41AlignIndexerWidthPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("b7344253b3136384c522aded59015f79c9bb7a08", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45794-dsv41-align-indexer-width.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_repeat_verifier_falsifier_and_packaging(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_align_indexer_width.py", source)
                self.assertIn("test_sglang_dsv41_align_indexer_width.py", source)
                self.assertIn("test_sglang_dsv41_align_indexer_width_patch.py", source)


if __name__ == "__main__":
    unittest.main()
