#!/usr/bin/env python3
"""FlashInfer PR #7676 (issue #7675) fence in the b12x SM120/SM121 dense GEMM: the build-time patcher and the boot receipt.

CPU only, stdlib (torch is stubbed when absent, as adapter/mxfp8_b12x.py imports it at the top).

Always:
  - patch_text inserts the one fence line right after the producer loop's producer_acquire, once;
  - it refuses a source where the anchor is missing or repeated, or the fence is already there;
  - the pins map each pristine sha256 to its patched sha256, and no hash is both.
With the real kernel source (the installed flashinfer, or FI7676_FILE=<path>):
  - FI7676_EXPECT=patched (the image battery): the file is the pinned patched one, and removing the fence line gives
    back the pinned pristine file byte for byte (the image differs from FlashInfer 0.6.18 by exactly PR #7676's line);
  - FI7676_EXPECT=pristine: the file is pinned pristine and patching it gives the pinned result;
  - the adapter's boot receipt names the state.
usage: python3 tests/test_fi7676_fence.py
"""
import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime", "flashinfer_fi7676"))
sys.path.insert(0, os.path.join(ROOT, "adapter"))
if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = types.ModuleType("torch")

import apply_fi7676 as fi  # noqa: E402
import mxfp8_b12x  # noqa: E402

LOOP = (
    "        while work_tile.is_valid_tile:\n"
    "                mainloop_producer_state.reset_count()\n"
    "\n"
) + fi.ANCHOR + (
    "\n"
    "                    k_tile_global = k_tile_start + mainloop_producer_state.count\n"
)


def _real_file():
    path = os.environ.get("FI7676_FILE")
    if path:
        return path
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], fi.REL)


class PatchText(unittest.TestCase):
    def test_inserts_once_after_acquire(self):
        out = fi.patch_text(LOOP)
        self.assertEqual(out.count(fi.FENCE), 1)
        self.assertIn(fi.ANCHOR + fi.FENCE + "\n                    k_tile_global", out)
        self.assertEqual(out.replace(fi.FENCE, "", 1), LOOP)

    def test_refuses_missing_repeated_or_present(self):
        with self.assertRaises(ValueError):
            fi.patch_text(LOOP.replace("unroll=2", "unroll=4"))
        with self.assertRaises(ValueError):
            fi.patch_text(LOOP + LOOP)
        with self.assertRaises(ValueError):
            fi.patch_text(fi.patch_text(LOOP))

    def test_pins(self):
        self.assertTrue(fi.PINS)
        self.assertFalse(set(fi.PINS) & set(fi.PINS.values()))
        self.assertEqual(mxfp8_b12x._FI7676_ABSENT in fi.PINS, True)
        self.assertEqual(fi.PINS[mxfp8_b12x._FI7676_ABSENT], mxfp8_b12x._FI7676_PRESENT)
        # v2.7: the receipt knows exactly the patcher's pairs (0.6.18 and the 0.7.2 / 0.7.3 file = the PR base blob)
        self.assertEqual(mxfp8_b12x._FI7676_PAIRS, fi.PINS)
        self.assertEqual(fi.FENCE.strip(), mxfp8_b12x._FI7676_LINE)

    def test_main_refuses_unknown_and_leaves_it(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.py")
            with open(p, "w") as f:
                f.write(LOOP)
            self.assertEqual(fi.main([p]), 2)
            self.assertEqual(fi.main(["--check", p]), 2)
            with open(p) as f:
                self.assertEqual(f.read(), LOOP)

    def test_receipt_unpinned(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.py")
            with open(p, "w") as f:
                f.write(LOOP)
            self.assertIn("unpinned (fence line not found)", mxfp8_b12x.fi7676_receipt(p))
            with open(p, "w") as f:
                f.write(fi.patch_text(LOOP))
            self.assertIn("unpinned (fence line found)", mxfp8_b12x.fi7676_receipt(p))
        self.assertIn("unknown", mxfp8_b12x.fi7676_receipt("/nonexistent/k.py"))


class RealSource(unittest.TestCase):
    def setUp(self):
        self.path = _real_file()
        if not self.path or not os.path.exists(self.path):
            self.skipTest("no FlashInfer b12x kernel source (set FI7676_FILE or install flashinfer)")
        with open(self.path, "rb") as f:
            self.data = f.read()
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.expect = os.environ.get("FI7676_EXPECT", "")

    def test_state(self):
        st, h = fi.state(self.data)
        if self.expect:
            self.assertEqual(st, self.expect, f"{self.path} sha256={h}")
        self.assertIn(st, ("patched", "pristine"), f"{self.path} sha256={h}")
        if st == "patched":
            text = self.data.decode()
            self.assertEqual(text.count(fi.ANCHOR + fi.FENCE), 1)
            back = text.replace(fi.ANCHOR + fi.FENCE, fi.ANCHOR, 1).encode()
            self.assertEqual(hashlib.sha256(back).hexdigest(), fi.POSTS[h])
            self.assertIn(": present sha256=", mxfp8_b12x.fi7676_receipt(self.path))
        else:
            out = fi.patch_text(self.data.decode()).encode()
            self.assertEqual(hashlib.sha256(out).hexdigest(), fi.PINS[h])
            self.assertIn(": ABSENT sha256=", mxfp8_b12x.fi7676_receipt(self.path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
