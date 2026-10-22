#!/usr/bin/env python3
"""Packaging checks for the SGLang #45450 sampling-param restage port."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DsparkParamRestagePackagingTests(unittest.TestCase):
    def test_fetch_applies_and_executes_the_focused_test(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("20189751de186f394abe850b1ad48087df32eb77", fetch)
        self.assertIn(
            "30a3d02028d1db1974456bda9383cdebf7bbaa0c6e1b9e3cc010d3b0d2b9acd9",
            fetch,
        )
        self.assertIn("sglang-45450-dspark-param-restage.patch", fetch)
        self.assertIn("test_sglang_dspark_param_restage.py", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_repeat_source_and_behavior_checks(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dspark_param_restage.py", source)
                self.assertIn("test_sglang_dspark_param_restage.py", source)
                self.assertIn("test_sglang_dspark_param_restage_patch.py", source)


if __name__ == "__main__":
    unittest.main()
