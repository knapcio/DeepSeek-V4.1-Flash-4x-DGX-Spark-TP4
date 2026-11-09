#!/usr/bin/env python3
"""Every source-pinned adapter must install on the SGLang tree baked into the image.

The adapters hash-pin the engine functions they copy or wrap and refuse to run when one
drifts. At boot that refusal only shows up as "Ignore import error when loading
sglang.srt.models.deepseek_v4" followed by a dead head, so a patch that touches a pinned
function (SGLang #42901 and #45301 both did) passed every other test. This imports the
model the way boot.py does, with every EXTRA_CONTAINER_ENV flag that .env.tp4.example
names switched on, and fails the image build on any drift. Skipped where SGLang is not
installed (the offline CI).
"""
import importlib.util
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Pins that only the off-by-default stages check at install; DSV41_WO_A_INPUT_FP8 (off by default) checks
# the wo_a dispatch it wraps and its install order against wo_a_w8's mid/drop wrapper.
# DSV41_EXTEND_NOSYNC (off by default) pins the backend tail-metadata source it edits.
EXTRA_FLAGS = {"DSV41_PREFILL_SP_FP8_MOE": "1", "DSV41_WO_A_INPUT_FP8": "1", "DSV41_EXTEND_NOSYNC": "1"}

PROBE = """
import sglang.srt.models.deepseek_v4 as v4
import prefill_sp
prefill_sp._load_modules(v4)
prefill_sp._check_tail_engine()
print("adapter engine pins OK")
"""


def profile_flags(text):
    flags = {}
    for line in text.splitlines():
        m = re.match(r'#?EXTRA_CONTAINER_ENV="([^"]*)"', line.strip())
        if m:
            for item in m.group(1).split():
                key, _, value = item.partition("=")
                flags[key] = value
    flags.update(EXTRA_FLAGS)
    return flags


class ProfileFlags(unittest.TestCase):
    def test_union_of_profiles_enables_every_pinned_adapter(self):
        flags = profile_flags((ROOT / ".env.tp4.example").read_text())
        for key in ("DSV41_PREFILL_SP", "DSV41_PREFILL_SP_FP8", "DSV41_HC_FUSED",
                    "DSV41_ROUTER_LIVE", "DSV41_INDEXER_CHUNKED"):
            self.assertNotIn(flags.get(key, "0"), ("", "0", "off", "false"), key)


@unittest.skipUnless(importlib.util.find_spec("sglang"), "needs the image's SGLang")
class EnginePins(unittest.TestCase):
    def test_model_import_installs_every_adapter(self):
        env = dict(os.environ)
        env.update(profile_flags((ROOT / ".env.tp4.example").read_text()))
        env.update(DSV41_SOURCE="/models/DeepSeek-V4.1-Flash", CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, "-c", PROBE], env=env,
                           capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-4000:])
        self.assertIn("adapter engine pins OK", r.stdout, out[-4000:])


if __name__ == "__main__":
    unittest.main()
