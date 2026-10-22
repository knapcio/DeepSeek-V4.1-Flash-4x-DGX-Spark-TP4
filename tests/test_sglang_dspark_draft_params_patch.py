#!/usr/bin/env python3
"""Packaging checks for the SGLang #45663 pinned-tree adaptation."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DsparkDraftParamsPackagingTests(unittest.TestCase):
    def test_fetch_content_pins_helper_and_applies_adaptation(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        patcher = (ROOT / "runtime/patch_dspark_draft_params.py").read_text()
        verifier = (ROOT / "scripts/verify_sglang_dspark_draft_params.py").read_text()
        self.assertIn("833e51899fb160b20645a868bf000a98fd3181a4", fetch)
        self.assertIn(
            "993dc3d686f4673b774278110fdbf44a8b1d190af1841c597f9fd7e50d12fbce",
            fetch,
        )
        self.assertIn("fill_draft_params(", patcher)
        self.assertIn("legacy two-launch", verifier)

    def test_images_verify_and_test_the_adaptation(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dspark_draft_params.py", source)
                self.assertIn("test_sglang_dspark_draft_params_patch.py", source)


if __name__ == "__main__":
    unittest.main()
