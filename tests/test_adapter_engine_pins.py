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
# DSV41_PREFILL_MOE_FOLD (production profile since v2.7.4) wraps pinned engine functions (forward_normal,
# _forward_shared_experts, maybe_fuse_routed_scale_and_shared_add): installed, and every wrapper unwraps to the engine.
# DSV41_L2_PREFETCH_SHPF_MB (l2_prefetch v5; production profile since v2.7.5, 6 MB) wraps DeepseekV2MoE._forward_shared_experts (the fold wraps it
# too: both wrappers must coexist and unwrap to the engine) and DeepseekV4DecoderLayer._hc_mix_stats, both hash-pinned by
# prefill_sp through inspect.unwrap (the 2026-11-10 tier-6 window failed this probe when they did not keep __wrapped__).
# DSV41_MOE_SCALE_CODES / DSV41_MOE_EVICT_FIRST (production profile since v2.7.5) are read by moe_b12x_next at import; the
# production line's values are on through .env.tp4.example.
# DSV41_GREEDY_NOGATHER (production profile since v2.7.6, on through .env.tp4.example) wraps the draft sampler / proposer,
# the graph backend, the verify head, the accept's row argmax and the verify executor; the probe imports those modules (the
# model import alone does not) and checks every hook installed, unwraps to the engine, and every engine attribute the
# adapter calls exists.
EXTRA_FLAGS = {"DSV41_PREFILL_SP_FP8_MOE": "1", "DSV41_WO_A_INPUT_FP8": "1", "DSV41_EXTEND_NOSYNC": "1",
               "DSV41_PREFILL_GEMM_TACTIC": "1", "DSV41_PREFILL_MOE_FOLD": "1", "DSV41_L2_PREFETCH_SHPF_MB": "6"}

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


def _flagged(fn, attr):
    # True when a wrapper along fn's __wrapped__ chain carries attr (two adapters may wrap one engine function)
    for _ in range(16):
        if fn is None:
            return False
        if getattr(fn, attr, False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


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
    # v5 SHPF: the hc-entry and shared-expert wrappers are installed whenever DSV41_L2_PREFETCH=1 (SHPF on or off)
    import sys
    v2 = sys.modules["sglang.srt.models.deepseek_v2"]
    for owner, name in ((v4.DeepseekV4DecoderLayer, "_hc_mix_stats"), (v2.DeepseekV2MoE, "_forward_shared_experts")):
        fn = getattr(owner, name)
        # the fold (v2.7.4, installed after it on deepseek_v2) wraps _forward_shared_experts OUTSIDE the SHPF wrapper:
        # look for the flag along the __wrapped__ chain, not only on the outermost function
        assert _flagged(fn, "_dsv41_l2_prefetch"), f"SHPF wrapper of {name} not installed"
        assert inspect.unwrap(fn).__qualname__ == f"{owner.__name__}.{name}", f"SHPF wrapper of {name} does not unwrap"
if os.environ.get("DSV41_GREEDY_NOGATHER", "").strip().lower() not in ("", "0", "off", "false", "no"):
    import importlib
    S = "sglang.srt.speculative.dspark_components."
    ds = importlib.import_module(S + "dspark_draft_sampler")
    dd = importlib.import_module(S + "dspark_draft")
    dv = importlib.import_module(S + "dspark_verify")
    lp = importlib.import_module("sglang.srt.layers.logits_processor")
    da = importlib.import_module("sglang.kernels.ops.speculative.dspark.dspark_accept")
    fb = importlib.import_module("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend")
    import sglang.kernels.ops.speculative.dspark.sharded_greedy as sg
    import sglang.srt.models.deepseek_v4_dspark as v4d
    import ast

    def chain(fn):
        out = []
        while fn is not None and len(out) < 16:
            out.append(fn)
            fn = getattr(fn, "__wrapped__", None)
        return out

    def engine_args(mod, qual):
        # the engine function's own parameters, from the module source (other adapters' wrappers above ours do not
        # all keep __wrapped__, so inspect.unwrap cannot always reach it)
        tree = ast.parse(inspect.getsource(mod))
        cls, _, name = qual.rpartition(".")
        scope = tree.body
        if cls:
            scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
        fn = next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
        return [x.arg for x in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]

    import greedy_nogather
    P = greedy_nogather._parse(os.environ["DSV41_GREEDY_NOGATHER"])
    # installed (the class / function flags the installs set; a later adapter may wrap outside our wrapper)
    owners = [fb.FullCudaGraphBackend] + ([ds.DsparkDraftSampler, dd.DraftBlockProposer] if "draft" in P else []) + (
        [lp.LogitsProcessor, dv.TargetVerifyExecutor] if "target" in P else [])
    for owner in owners:
        assert getattr(owner, "_dsv41_greedy_nogather", False), f"greedy_nogather not installed on {owner.__name__}"
    if "target" in P:
        assert getattr(da._row_argmax, "_dsv41_greedy_nogather", False), "greedy_nogather not installed on _row_argmax"
    # every greedy_nogather wrapper reachable through __wrapped__ keeps it (tier-6 FIX-20261110); where the chain
    # reaches the engine it ends in the engine's own function
    draft_hooks = [(ds, ds.DsparkDraftSampler, "__call__", "DsparkDraftSampler.__call__"),
                   (ds, ds.DsparkDraftSampler, "__init__", "DsparkDraftSampler.__init__"),
                   (dd, dd.DraftBlockProposer, "propose", "DraftBlockProposer.propose")]
    target_hooks = [(lp, lp.LogitsProcessor, "_get_logits", "LogitsProcessor._get_logits"),
                    (da, da, "_row_argmax", "_row_argmax"),
                    (dv, dv.DsparkVerifyEpilogue, "begin_static_step", "DsparkVerifyEpilogue.begin_static_step"),
                    (dv, dv.TargetVerifyExecutor, "run_non_compact", "TargetVerifyExecutor.run_non_compact")]
    hooks = [(fb, fb.FullCudaGraphBackend, "capture_one", "FullCudaGraphBackend.capture_one"),
             (fb, fb.FullCudaGraphBackend, "replay", "FullCudaGraphBackend.replay")] + (
        draft_hooks if "draft" in P else []) + (target_hooks if "target" in P else [])
    seen = 0
    for mod, owner, name, qual in hooks:
        c = chain(getattr(owner, name))
        mine = [f for f in c if getattr(f, "__module__", "") == "greedy_nogather"]
        seen += bool(mine)
        for f in mine:
            assert getattr(f, "__wrapped__", None) is not None, f"greedy_nogather wrapper of {qual} without __wrapped__"
        end = c[-1]
        if getattr(end, "__module__", "") == mod.__name__:
            assert end.__qualname__ == qual, f"{qual} unwraps to {end.__qualname__}"
    # reachable on the production profile: draft __init__ (spec_sync_free above it keeps no __wrapped__: 0 of 3 are
    # guaranteed), target _row_argmax / begin_static_step (outermost), the backend's two
    assert seen >= 2 + (2 if "target" in P else 0), f"only {seen} greedy_nogather wrappers reachable through __wrapped__"
    # the engine pieces the flavours call by name, and the signatures the wrappers forward
    assert callable(ds.select_draft_hidden_without_anchor)
    assert hasattr(v4d.DeepseekV4ForCausalLMDSpark, "compute_base_logits")
    mh = [c for c in vars(v4d).values() if isinstance(c, type) and "sample_block_greedy_fused" in vars(c)
          and "supports_sharded_greedy" in vars(c)]
    assert mh, "no Markov head with sample_block_greedy_fused + supports_sharded_greedy"
    assert callable(getattr(lp.LogitsProcessor, "_compute_lm_head", None)) and callable(sg._sharded_greedy_finish_kernel)
    assert engine_args(fb, "FullCudaGraphBackend.capture_one")[:5] == [
        "self", "shape_key", "forward_fn", "capture_inputs", "post_warmup_hook"]
    assert engine_args(fb, "FullCudaGraphBackend.replay")[:3] == ["self", "shape_key", "static_forward_batch"]
    assert engine_args(lp, "LogitsProcessor._get_logits") == [
        "self", "hidden_states", "lm_head", "logits_metadata", "embedding_bias", "use_logits_buffer"]
    assert engine_args(dv, "DsparkVerifyEpilogue.begin_static_step") == ["self", "bs", "armed"]
    assert "batch" in engine_args(dv, "TargetVerifyExecutor.run_non_compact")
    assert "sampling_info" in engine_args(dd, "DraftBlockProposer.propose")
    assert engine_args(da, "_row_argmax") == ["logits"]
    src = inspect.getsource(ds.DsparkDraftSampler)
    for attr in ("self.query_token_num", "self.sample_from_anchor", "self.exp_noise", "self.out", "self.confidence_out",
                 "self.confidence_fn", "self.folded_sampling", "self.markov_head", "self.gamma", "self.model"):
        assert attr in src, f"DsparkDraftSampler no longer uses {attr}"
    print("greedy_nogather hooks OK", ",".join(sorted(P)))
if os.environ.get("DSV41_PREFILL_MOE_FOLD") == "1":
    import sglang.srt.models.deepseek_v2 as v2
    import prefill_moe_fold
    assert prefill_moe_fold.STATE["installed"], f"MoE fold not installed: {prefill_moe_fold.STATE['reason']}"
    for fn, name in ((v2.DeepseekV2MoE.forward_normal, "DeepseekV2MoE.forward_normal"),
                     (v2.DeepseekV2MoE._forward_shared_experts, "DeepseekV2MoE._forward_shared_experts"),
                     (v2.maybe_fuse_routed_scale_and_shared_add, "maybe_fuse_routed_scale_and_shared_add")):
        assert _flagged(fn, "_dsv41_moe_fold"), f"MoE fold wrapper missing on {name}"
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
                    "DSV41_ROUTER_LIVE", "DSV41_INDEXER_CHUNKED", "DSV41_GREEDY_NOGATHER"):
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
        self.assertIn("greedy_nogather hooks OK", r.stdout, out[-4000:])


if __name__ == "__main__":
    unittest.main()
