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
# DSV41_PREFILL_GEMM_TACTIC (off by default) wraps ModelRunner.load_model for its boot check; its wrapper must unwrap to
# the engine function. The probe also re-runs prefill_sp's drift guard after every install (the l2_prefetch bracket wraps
# the pinned DeepseekV4Model.forward; with the production line's W2 keys its wo_a recorder carries the W2 fork).
# DSV41_PREFILL_MOE_FOLD (production profile since the v2.7.4 candidate) wraps pinned engine functions (forward_normal,
# _forward_shared_experts, maybe_fuse_routed_scale_and_shared_add): installed, and every wrapper unwraps to the engine.
EXTRA_FLAGS = {"DSV41_PREFILL_SP_FP8_MOE": "1", "DSV41_WO_A_INPUT_FP8": "1", "DSV41_EXTEND_NOSYNC": "1",
               "DSV41_PREFILL_GEMM_TACTIC": "1", "DSV41_PREFILL_MOE_FOLD": "1"}

PROBE = """
import inspect
import sglang.srt.models.deepseek_v4 as v4
import prefill_sp
prefill_sp._load_modules(v4)
prefill_sp._check_tail_engine()
import sglang.srt.model_executor.model_runner as mrm
mr = mrm.ModelRunner
assert getattr(mr.load_model, "_dsv41_prefill_gemm_tactic", False), "GEMM tactic boot check not installed"
assert inspect.unwrap(mr.load_model).__qualname__ == "ModelRunner.load_model", "load_model does not unwrap to the engine"
import os
if os.environ.get("DSV41_L2_PREFETCH") == "1":
    # l2_prefetch brackets DeepseekV4Model.forward (pinned by prefill_sp) and, with _WOA, wraps the wo_a dispatch that
    # carries the W2 fork: both installed, both unwrap to the engine (every l2_prefetch wrapper keeps __wrapped__)
    assert getattr(v4.DeepseekV4Model, "_dsv41_l2_prefetch", False), "l2_prefetch bracket not installed"
    assert inspect.unwrap(v4.DeepseekV4Model.forward).__qualname__ == "DeepseekV4Model.forward", "bracket does not unwrap"
    if os.environ.get("DSV41_L2_PREFETCH_WOA") == "1":
        assert getattr(v4._apply_wo_a_bf16_matmul, "_dsv41_l2_prefetch", False), "l2_prefetch wo_a recorder not installed"
        # the inner dispatch is wo_a_w8's / wo_a_input_fp8's own wrapper (no source pin on it): one level is enough
        inner = getattr(v4._apply_wo_a_bf16_matmul, "__wrapped__", None)
        assert inner is not None and not getattr(inner, "_dsv41_l2_prefetch", False), "wo_a recorder without __wrapped__"
if os.environ.get("DSV41_PREFILL_MOE_FOLD") == "1":
    import sglang.srt.models.deepseek_v2 as v2
    import prefill_moe_fold
    assert prefill_moe_fold.STATE["installed"], f"MoE fold not installed: {prefill_moe_fold.STATE['reason']}"
    for fn, name in ((v2.DeepseekV2MoE.forward_normal, "DeepseekV2MoE.forward_normal"),
                     (v2.DeepseekV2MoE._forward_shared_experts, "DeepseekV2MoE._forward_shared_experts"),
                     (v2.maybe_fuse_routed_scale_and_shared_add, "maybe_fuse_routed_scale_and_shared_add")):
        assert getattr(fn, "_dsv41_moe_fold", False), f"MoE fold wrapper missing on {name}"
        assert inspect.unwrap(fn).__qualname__ == name, f"MoE fold wrapper on {name} does not unwrap to the engine"
# prefill_sp's drift guard again AFTER every adapter installed: each pinned engine source still unwraps to the stock one
prefill_sp.check_engine()
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
