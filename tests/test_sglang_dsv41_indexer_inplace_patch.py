#!/usr/bin/env python3
"""Packaging checks for the SGLang #45745 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45745-dsv41-indexer-inplace.patch"


class Dsv41IndexerInplacePackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("3458c1f1900fbbaffc237d74491f30e6f8cf288c", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45745-dsv41-indexer-inplace.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_repeat_source_and_packaging_checks(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_indexer_inplace.py", source)
                self.assertIn("test_sglang_dsv41_indexer_inplace_patch.py", source)


if __name__ == "__main__":
    unittest.main()
