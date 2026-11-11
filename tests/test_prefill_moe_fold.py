"""CPU checks of adapter/prefill_moe_fold.py (DSV41_PREFILL_MOE_FOLD) and its moe_b12x_next / b12x_next wiring.

1. The arithmetic: the fold writes bf16(float(s) + float(r)) with r the stock top-k sum's bf16 value, computed as the
   kernel does it (binary32 RNE add with subnormals, then bf16 round-to-nearest-even of the bits); torch's
   ``s.add_(r, alpha=1.0)`` on bf16 is compared bit for bit over every (s, r) pair of a stride through all 65536 bf16
   patterns (FOLD_TEST_FULL=1: all 2^32 pairs), NaN inputs excluded (both give a NaN).
2. The frame logic on a stand-in of forward_normal's order (shared experts, routed experts, maybe_fuse): the first
   call of a capacity returns the stock result and decides; a fold writes into the shared output and the add site
   returns it without adding; a fold that is not bit-identical leaves the capacity on the stock add; a broken return
   chain turns the fold off; a folded tensor that comes back as other storage raises; gate / rows / capture /
   skip_shared_experts / static reasons keep the stock path; a launch that raises falls back to stock.
3. The b12x_next runtime patch on disk: the accumulate kernel variant, its own custom op, the cache-key suffix, the
   context manager, the refusal in b12x_moe_fp4, and the build script / SOURCE_PATCH bookkeeping.
4. With the image's SGLang (skipped elsewhere): the REAL DeepseekV2MoE.forward_normal and the real
   maybe_fuse_routed_scale_and_shared_add, wrapped by install_v2 (pins checked), driving the real
   moe_b12x_next._fused_b12x_next with a CPU stand-in for the b12x launches: folded output == stock output bit for
   bit, plain and pre-quantized (stage 2b marker) inputs, one counted pre-quantized launch per call.

usage: PYTHONPATH=adapter python3 tests/test_prefill_moe_fold.py
"""
import contextlib
import importlib
import importlib.util
import os
import re
import sys
import types
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "adapter"))
B12X = os.environ.get("B12X_NEXT_ROOT", os.path.join(ROOT, "runtime", "b12x_next"))
if not os.path.isdir(os.path.join(B12X, "b12x_next")) and os.path.isdir("/opt/b12x_next/b12x_next"):
    B12X = "/opt/b12x_next"

import prefill_moe_fold as F  # noqa: E402


# ---------------------------------------------------------------------------------------------------- 1. arithmetic --

def bf16_rne_bits(f32):
    """float32 array -> bf16 bit patterns, round to nearest even (cvt.rn.bf16.f32 for every non-NaN input)."""
    u = f32.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def widen(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def fold_bits(s_bits, r_bits):
    """The accumulate kernel's epilogue: F2F/SHF widenings (exact), FADD (add.rn.f32, no ftz), F2FP RN to bf16."""
    with np.errstate(over="ignore", invalid="ignore"):
        return bf16_rne_bits(widen(s_bits) + widen(r_bits))


class Arithmetic(unittest.TestCase):
    def test_numpy_keeps_subnormals(self):
        tiny = np.array([1e-45], dtype=np.float32)
        self.assertGreater(float(tiny[0] + tiny[0]), 0.0)          # no flush-to-zero on this host

    def test_fold_equals_torch_add(self):
        full = os.environ.get("FOLD_TEST_FULL", "0") == "1"
        allbits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
        finite = allbits[(allbits & 0x7F80) != 0x7F80]             # no NaN / inf patterns as residual sweep rows...
        nonnan = allbits[~(((allbits & 0x7F80) == 0x7F80) & ((allbits & 0x007F) != 0))]
        rows = nonnan if full else nonnan[::61]
        cols = torch.from_numpy(nonnan.view(np.int16).copy()).view(torch.bfloat16)
        n = 0
        for s in rows:
            sb = np.full(nonnan.shape, s, dtype=np.uint16)
            ours = fold_bits(sb, nonnan)
            st = torch.from_numpy(sb.view(np.int16).copy()).view(torch.bfloat16)
            ref = st.add_(cols, alpha=1.0).view(torch.int16).numpy().view(np.uint16)
            nan_ref = ((ref & 0x7F80) == 0x7F80) & ((ref & 0x7F) != 0)
            nan_ours = ((ours & 0x7F80) == 0x7F80) & ((ours & 0x7F) != 0)
            np.testing.assert_array_equal(nan_ours, nan_ref)
            np.testing.assert_array_equal(ours[~nan_ref], ref[~nan_ref])
            n += int((~nan_ref).sum())
        self.assertGreater(n, 10_000_000 if not full else 4_000_000_000)
        self.assertGreater(len(finite), 0)
        print(f"fold == torch bf16 add_ on {n:,} (residual, routed) pairs ({'all' if full else 'strided'})")

    def test_routed_rounding_is_the_stock_store(self):
        # the routed value the fold adds is the bf16 RNE of the fp32 route sum, i.e. exactly what the stock kernel
        # stores and torch then reads back: fold(s, sum) == torch(s + stock_store(sum))
        g = np.random.default_rng(7)
        routes = g.standard_normal((4096, 6)).astype(np.float32)
        rb = bf16_rne_bits(routes)                                   # phase 2 writes bf16 route values
        acc = np.zeros(4096, dtype=np.float32)
        for k in range(6):                                           # route order, fp32, starting from +0
            acc = acc + widen(rb[:, k])
        stock_routed = bf16_rne_bits(acc)
        s = bf16_rne_bits(g.standard_normal(4096).astype(np.float32) * 3)
        ours = fold_bits(s, stock_routed)
        st = torch.from_numpy(s.view(np.int16).copy()).view(torch.bfloat16)
        rt = torch.from_numpy(stock_routed.view(np.int16).copy()).view(torch.bfloat16)
        np.testing.assert_array_equal(ours, st.add_(rt, alpha=1.0).view(torch.int16).numpy().view(np.uint16))


# ---------------------------------------------------------------------------------------------- 2. the frame logic --

class FakeCfg:
    inplace = True


class FakeExperts:
    def __init__(self):
        self.moe_runner_config = FakeCfg()
        self.should_fuse_routed_scaling_factor_in_topk = True
        self.reduce_results = False
        self._dsv41_b12x_next = object()
        self.quant_method = None


class FakeMoE:
    """forward_normal's order: shared experts, routed experts (the fused function), maybe_fuse."""
    K = 64

    def __init__(self, harness):
        self.h = harness
        self.experts = FakeExperts()
        self.shared_experts = object()
        self.num_fused_shared_experts = 0
        self._shared_expert_tp1 = False
        self._fuse_shared_experts_inside_sbo = False
        self._enable_a2a_moe = False

    def forward_normal(self, hidden_states, gemm_output_zero_allocator=None, input_ids=None, input_ids_global=None,
                       skip_shared_experts=False, num_token_non_padded=None):
        shared = None if skip_shared_experts else self._forward_shared_experts(hidden_states)
        routed = self.h.routed_call(self, hidden_states)
        return self.h.mod.maybe_fuse_routed_scale_and_shared_add(self.experts, routed, shared, 2.5)

    def _forward_shared_experts(self, hidden_states, gemm_output_zero_allocator=None, pre_quant_input=None):
        return (hidden_states.float() * 0.75 + 0.01).to(torch.bfloat16)


def stock_maybe_fuse(experts, routed, shared, routed_scaling_factor):
    if shared is not None:
        return shared.add_(routed, alpha=1.0)
    return routed


class Harness:
    """Installs the three wrappers on a fake module / class and plays the routed-experts side of moe_b12x_next."""

    def __init__(self, bad_kernel=False, copy_on_return=False, raise_accumulate=False):
        self.mod = types.SimpleNamespace(maybe_fuse_routed_scale_and_shared_add=stock_maybe_fuse)
        self.cls = type("MoE", (FakeMoE,), {})
        self.cls.forward_normal = F.make_forward_normal(FakeMoE.forward_normal)
        self.cls._forward_shared_experts = F.make_forward_shared(FakeMoE._forward_shared_experts)
        self.mod.maybe_fuse_routed_scale_and_shared_add = F.make_maybe_fuse(stock_maybe_fuse)
        self.bad_kernel, self.copy_on_return, self.raise_accumulate = bad_kernel, copy_on_return, raise_accumulate
        self.local_veto = False
        self.launches = []

    def routed(self, x):
        return (x.float() * -0.31 + x.float().square() * 0.02).to(torch.bfloat16)

    def routed_call(self, moe, x):
        m, k = x.shape

        def launch(o, accumulate, count):
            self.launches.append((accumulate, count))
            r = self.routed(x)
            if accumulate:
                if self.raise_accumulate:
                    raise ValueError("refused before launching")
                v = (o.float() + r.float()).to(torch.bfloat16)
                if self.bad_kernel:
                    v = (v.view(torch.int16) ^ 1).view(torch.bfloat16)
                o.copy_(v)
            else:
                o.copy_(r)

        frame = F.claim(moe.experts._dsv41_b12x_next, m, k, x.device)
        if frame is None:
            out = torch.empty(m, k, dtype=torch.bfloat16)
            launch(out, False, True)
        else:
            out = F.run(frame, 8192 if m > 4096 else 4096, launch, lambda: torch.empty(m, k, dtype=torch.bfloat16),
                        False if self.local_veto else F.frame_ok(frame, m, k, x.device))
        out = out[..., :k].contiguous()                # FusedMoE.forward_impl's combine slice (a view)
        return out.clone() if self.copy_on_return else out

    def moe(self):
        m = self.cls(self)
        m._dsv41_moe_fold_static = ""                  # the static checks are tested separately
        return m


def reset(gate="1"):
    F._VERDICT.clear()
    F._ENABLED[0] = None
    F._LOGGED.clear()
    F.STATE.update(installed=True, chain_ok=True)
    for k in F.STATS:
        F.STATS[k] = 0
    os.environ[F.GATE] = gate
    os.environ[F.MIN_ROWS_ENV] = "2048"
    F.AGREE[0] = False


def stock_result(x):
    h = Harness()
    shared = FakeMoE._forward_shared_experts(None, x)
    return shared.add_(h.routed(x), alpha=1.0)


class FrameLogic(unittest.TestCase):
    def setUp(self):
        reset()
        g = torch.Generator().manual_seed(3)
        self.x = (torch.randn(4096, FakeMoE.K, generator=g) * 4).to(torch.bfloat16)
        self.ref = stock_result(self.x)

    def bits(self, a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    def test_check_then_fold(self):
        h = Harness()
        moe = h.moe()
        out1 = moe.forward_normal(self.x)              # check call: stock result, verdict ON
        self.assertTrue(self.bits(out1, self.ref))
        self.assertEqual(F._VERDICT, {4096: True})
        self.assertEqual(h.launches, [(False, True), (True, False)])
        h.launches.clear()
        out2 = moe.forward_normal(self.x)              # fold
        self.assertTrue(self.bits(out2, self.ref))
        self.assertEqual(h.launches, [(True, True)])
        self.assertEqual((F.STATS["folds"], F.STATS["checks"]), (1, 1))

    def test_bad_kernel_stays_stock(self):
        h = Harness(bad_kernel=True)
        moe = h.moe()
        for _ in range(3):
            self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual(F._VERDICT, {4096: False})
        self.assertEqual(F.STATS["folds"], 0)
        self.assertEqual(F.STATS["stock"], 2)

    def test_broken_return_chain_turns_off(self):
        h = Harness(copy_on_return=True)
        moe = h.moe()
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertFalse(F.STATE["chain_ok"])
        h.launches.clear()
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual(h.launches, [(False, True)])   # no frame any more: the stock path

    def test_fold_returned_as_other_storage_raises(self):
        h = Harness()
        moe = h.moe()
        moe.forward_normal(self.x)
        h.copy_on_return = True
        with self.assertRaises(RuntimeError):
            moe.forward_normal(self.x)

    def test_accumulate_refusal_falls_back(self):
        h = Harness(raise_accumulate=True)
        moe = h.moe()
        F._VERDICT[4096] = True
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual(F.STATS["errors"], 1)
        self.assertEqual(h.launches, [(True, True), (False, True)])

    def test_check_with_raising_accumulate_decides_off(self):
        h = Harness(raise_accumulate=True)
        self.assertTrue(self.bits(h.moe().forward_normal(self.x), self.ref))
        self.assertEqual(F._VERDICT, {4096: False})

    def test_local_veto_still_joins_the_check(self):
        # a rank whose shared output fails the tensor checks still runs the check (and its agreement) with local
        # False, so every rank reaches the same collective; afterwards it runs stock on its own
        h = Harness()
        h.local_veto = True
        moe = h.moe()
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual(F._VERDICT, {4096: False})
        self.assertEqual(F.STATS["checks"], 1)
        reset()
        F._VERDICT[4096] = True
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual((F.STATS["folds"], F.STATS["stock"]), (0, 1))

    def test_stock_paths(self):
        h = Harness()
        moe = h.moe()
        small = self.x[:2047].contiguous()
        self.assertTrue(self.bits(moe.forward_normal(small), stock_result(small)))
        self.assertTrue(self.bits(moe.forward_normal(self.x, skip_shared_experts=True), h.routed(self.x)))
        reset("0")
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual(F._VERDICT, {})
        reset()
        moe._dsv41_moe_fold_static = "a reason"
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual(F._VERDICT, {})
        self.assertEqual(F.STATS["calls"], 0)

    def test_shared_after_routed_is_never_used(self):
        class Deferred(FakeMoE):
            def forward_normal(self, hidden_states, *a, **k):
                routed = self.h.routed_call(self, hidden_states)
                shared = self._forward_shared_experts(hidden_states)
                return self.h.mod.maybe_fuse_routed_scale_and_shared_add(self.experts, routed, shared, 1.0)
        h = Harness()
        Deferred.forward_normal = F.make_forward_normal(Deferred.forward_normal)
        Deferred._forward_shared_experts = F.make_forward_shared(FakeMoE._forward_shared_experts)
        moe = Deferred(h)
        moe._dsv41_moe_fold_static = ""
        self.assertTrue(self.bits(moe.forward_normal(self.x), self.ref))
        self.assertEqual((F.STATS["calls"], F.STATS["checks"], F.STATS["stock"]), (1, 0, 1))   # no check, no fold
        self.assertEqual(F._VERDICT, {})

    def test_wrappers_unwrap(self):
        h = Harness()
        import inspect
        self.assertIs(inspect.unwrap(h.cls.forward_normal), FakeMoE.forward_normal)
        self.assertIs(inspect.unwrap(h.cls._forward_shared_experts), FakeMoE._forward_shared_experts)
        self.assertIs(inspect.unwrap(h.mod.maybe_fuse_routed_scale_and_shared_add), stock_maybe_fuse)

    def test_gate_values(self):
        self.assertTrue(F.gate_on("1"))
        self.assertFalse(F.gate_on("off"))
        with self.assertRaises(ValueError):
            F.gate_on("maybe")
        reset("maybe")
        self.assertFalse(F.enabled())

    def test_static_reasons(self):
        moe = FakeMoE(Harness())
        moe.experts.moe_runner_config.inplace = False
        self.assertIn("deferred", F.static_reason(moe))
        moe = FakeMoE(Harness())
        moe.experts.should_fuse_routed_scaling_factor_in_topk = False
        self.assertIn("alpha", F.static_reason(moe))
        moe = FakeMoE(Harness())
        moe._shared_expert_tp1 = True
        self.assertIn("TP1", F.static_reason(moe))
        moe = FakeMoE(Harness())
        moe.experts._dsv41_b12x_next = None
        self.assertIn("moe_b12x_next", F.static_reason(moe))

    def test_receipt_every_env(self):
        old = os.environ.pop(F.RECEIPT_EVERY_ENV, None)
        try:
            self.assertEqual(F.receipt_every(), 4096)
            os.environ[F.RECEIPT_EVERY_ENV] = "20"
            self.assertEqual(F.receipt_every(), 20)
            os.environ[F.RECEIPT_EVERY_ENV] = "x"
            self.assertEqual(F.receipt_every(), 4096)
            os.environ[F.RECEIPT_EVERY_ENV] = "0"
            self.assertEqual(F.receipt_every(), 1)
        finally:
            os.environ.pop(F.RECEIPT_EVERY_ENV, None)
            if old is not None:
                os.environ[F.RECEIPT_EVERY_ENV] = old


# --------------------------------------------------------------------------------------------- 3. the runtime patch --

class RuntimePatch(unittest.TestCase):
    def read(self, *p):
        with open(os.path.join(*p)) as f:
            return f.read()

    def test_kernel_and_impl(self):
        k = self.read(B12X, "b12x_next", "moe", "_shared", "kernels", "w4a16", "kernel.py")
        self.assertIn('"add.rn.f32 $0, $1, $2;"', k)
        self.assertIn("cache_key = cache_key + (\"accumulate_output\",)", k)
        self.assertIn('"b12x_next::w4a16_topk_sum_accumulate_launch"', k)
        self.assertEqual(k.count('"b12x_next::w4a16_topk_sum_launch"'), 1)      # the stock op is unchanged
        epi = k[k.index("if cutlass.const_expr(self.accumulate_output):\n                # ds41 patch"):]
        epi = epi[:epi.index("@cute.jit")]
        self.assertIn("routed = self._cast_elem(acc)", epi)
        self.assertIn("residual = output_flat[idx].to(cutlass.Float32)", epi)
        self.assertIn("_add_rn_f32_noftz(residual, routed.to(cutlass.Float32))", epi)
        self.assertIn("output_flat[idx] = self._cast_elem(acc)", epi)          # the stock store, else branch
        i = self.read(B12X, "b12x_next", "moe", "fused_moe", "_impl.py")
        self.assertIn("def topk_sum_accumulate():", i)
        self.assertIn("torch.ops.b12x_next.w4a16_topk_sum_accumulate_launch(", i)
        body = i[i.index("def b12x_moe_fp4("):]
        body = body[:body.index('    a = binding.a\n')]
        self.assertIn("if _topk_sum_accumulate_requested():", body)          # refusal before any launch
        self.assertIn('binding.implementation == "dynamic"', body)

    def test_bookkeeping(self):
        import hashlib
        build = self.read(ROOT, "scripts", "build_b12x_next.sh")
        names = re.findall(r'"\$HERE/b12x_next-([a-z0-9-]+)\.patch"', build)
        self.assertEqual(names[-1], "topk-sum-accumulate")
        blob = b""
        for n in names:
            with open(os.path.join(ROOT, "scripts", f"b12x_next-{n}.patch"), "rb") as f:
                blob += f.read()
        sp = self.read(B12X, "b12x_next", "SOURCE_PATCH").strip()
        self.assertEqual(hashlib.sha256(blob).hexdigest()[:16], sp)

    def test_launch_prepared_pins_untouched(self):
        # adapter/moe_launch_prepared.py pins three _impl functions; the patch must not touch them
        spec = importlib.util.find_spec("cutlass")
        if spec is None:
            self.skipTest("needs the image's CuTe DSL to import b12x_next")
        sys.path.insert(0, B12X)
        import moe_launch_prepared as mlp
        fimpl = importlib.import_module("b12x_next.moe.fused_moe._impl")
        self.assertEqual(mlp.check_sources(fimpl), {})
        self.assertTrue(callable(fimpl.topk_sum_accumulate))
        with fimpl.topk_sum_accumulate():
            self.assertTrue(fimpl._topk_sum_accumulate_requested())
        self.assertFalse(fimpl._topk_sum_accumulate_requested())


# ---------------------------------------------------------------------------------- 4. the real engine functions --

@unittest.skipUnless(importlib.util.find_spec("sglang"), "needs the image's SGLang")
class Engine(unittest.TestCase):
    """The real forward_normal / maybe_fuse (wrapped by install_v2) + the real moe_b12x_next fused function."""

    @classmethod
    def setUpClass(cls):
        os.environ[F.GATE] = "1"
        import sglang.srt.models.deepseek_v2 as v2
        cls.v2 = v2
        cls.saved = (v2.DeepseekV2MoE.forward_normal, v2.DeepseekV2MoE._forward_shared_experts,
                     v2.maybe_fuse_routed_scale_and_shared_add, v2.get_exec, v2._is_cuda)
        F.STATE["installed"] = False
        assert F.install_v2(v2), F.STATE["reason"]
        v2.get_exec = lambda: types.SimpleNamespace(moe=types.SimpleNamespace(enable_eplb=False))
        v2._is_cuda = True             # production: CUDA (the CPU branch would scale the routed output in place)
        import moe_b12x_next as mbn
        cls.mbn = mbn
        cls.det = mbn.DETERMINISTIC
        mbn.DETERMINISTIC = True       # production: DSV41_MOE_B12X_NEXT_DETERMINISTIC=1

    @classmethod
    def tearDownClass(cls):
        v2 = cls.v2
        (v2.DeepseekV2MoE.forward_normal, v2.DeepseekV2MoE._forward_shared_experts,
         v2.maybe_fuse_routed_scale_and_shared_add, v2.get_exec, v2._is_cuda) = cls.saved
        cls.mbn.DETERMINISTIC = cls.det

    def make(self, prequant):
        mbn, v2 = self.mbn, self.v2
        K, E, TOPK = 64, 8, 6
        acc = [False]
        launches = []
        store = {}

        def routed_of(xin, ids, w):
            # a deterministic stand-in for the routed experts: depends on the input rows, ids and weights
            r = xin.float() * -0.29 + (ids.float() * w).sum(-1, keepdim=True) * 0.11
            return r.to(torch.bfloat16)

        class Geom:
            chunk = 8192
            key = "test-geom"

            def cap_for(self, m):
                return 4096 if m <= 4096 else 8192

            def prequant_views(self, m):
                return store["views"]

            @staticmethod
            def _compute(x, ids, w, out):
                r = routed_of(x, ids, w)
                out.copy_((out.float() + r.float()).to(torch.bfloat16) if acc[0] else r)
                return out

            def forward(self, impl, x, ids, w, out):
                launches.append(("plain", acc[0], True))
                return self._compute(x, ids, w, out)

            def forward_prequant(self, impl, m, ids, w, out, count=True):
                launches.append(("prequant", acc[0], count))
                if count:
                    mbn.PREQUANT_STATS["calls"] += 1
                return self._compute(store["x"], ids, w, out)

        geom = Geom()
        st = types.SimpleNamespace(geom=geom, impl=object(), hidden=K)
        w13 = torch.zeros(4)
        mbn._LAYERS[w13.data_ptr()] = st

        @contextlib.contextmanager
        def acc_ctx():
            acc[0] = True
            try:
                yield
            finally:
                acc[0] = False

        mbn._B["acc"] = acc_ctx
        mbn.FOLD_STATE["geoms"].add("test-geom")
        mbn._ENG.update(
            TopKOutputChecker=types.SimpleNamespace(format_is_bypassed=lambda t: False),
            StandardCombineInput=lambda hidden_states: types.SimpleNamespace(hidden_states=hidden_states),
            is_allocation_symmetric=lambda: False)
        quant_info = types.SimpleNamespace(w13_weight=w13, padded_hidden=None)
        cut = importlib.import_module("sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe")
        F.STATE["lazy_pins"] = None

        class Experts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_runner_config = types.SimpleNamespace(inplace=True)
                self.should_fuse_routed_scaling_factor_in_topk = True
                self.reduce_results = False
                self._dsv41_b12x_next = st
                self.quant_method = object.__new__(cut.Mxfp4FlashinferCutlassMoEMethod)

            def forward(self, hidden_states, topk_output, pre_quant_input=None):
                pre = None
                if prequant:
                    pre = types.SimpleNamespace(_dsv41_b12x_rows=(geom, hidden_states.shape[0], 1, 2))
                    store["views"] = (types.SimpleNamespace(data_ptr=lambda: 1),
                                      types.SimpleNamespace(data_ptr=lambda: 2))
                d = types.SimpleNamespace(hidden_states=hidden_states, topk_output=topk_output,
                                          hidden_states_pre_quant=pre)
                ci = mbn._fused_b12x_next(d, quant_info, None)
                return ci.hidden_states[..., :K].contiguous()      # FusedMoE.forward_impl's combine slice

        moe = object.__new__(v2.DeepseekV2MoE)
        torch.nn.Module.__init__(moe)
        moe.experts = Experts()
        moe.num_fused_shared_experts = 0
        moe._shared_expert_tp1 = False
        moe._fuse_shared_experts_inside_sbo = False
        moe._enable_a2a_moe = False
        moe.is_nextn = False
        moe.layer_id = 3
        moe.tp_size = 1
        moe.routed_scaling_factor = 2.5
        moe._maybe_quant_moe_input_once = lambda hs: None
        g = torch.Generator().manual_seed(11)

        def gate(hs, alloc=None):
            return torch.zeros(hs.shape[0], E)

        def topk(hs, logits, num_token_non_padded=None, expert_location_dispatch_info=None):
            n = hs.shape[0]
            ids = torch.randint(0, E, (n, TOPK), generator=torch.Generator().manual_seed(n)).to(torch.int32)
            w = torch.rand(n, TOPK, generator=torch.Generator().manual_seed(n + 1))
            return types.SimpleNamespace(topk_ids=ids, topk_weights=w)

        moe.gate = _Callable(gate, e_score_correction_bias_vl=None)
        moe.topk = _Callable(topk)
        moe.shared_experts = _Callable(lambda hs, gemm_output_zero_allocator=None, gateup_pre_quant=None:
                                       (hs.float() * 0.5 + 0.02).to(torch.bfloat16), gate_up_proj=object())
        x = (torch.randn(4096, K, generator=g) * 3).to(torch.bfloat16)
        store["x"] = x
        return moe, x, launches

    def run_case(self, prequant):
        reset()
        moe, x, launches = self.make(prequant)
        F.STATE["installed"] = True
        calls0 = self.mbn.PREQUANT_STATS["calls"]
        os.environ[F.GATE] = "0"
        F._ENABLED[0] = None
        ref = moe.forward_normal(x).clone()                    # the stock path, gate off
        self.assertEqual(launches[-1], ("prequant" if prequant else "plain", False, True))
        os.environ[F.GATE] = "1"
        F._ENABLED[0] = None
        out1 = moe.forward_normal(x).clone()                   # check call
        out2 = moe.forward_normal(x).clone()                   # fold
        self.assertEqual(F._VERDICT, {4096: True})
        self.assertEqual(F.STATS["folds"], 1)
        for o in (out1, out2):
            self.assertTrue(torch.equal(o.view(torch.int16), ref.view(torch.int16)))
        if prequant:
            self.assertEqual(self.mbn.PREQUANT_STATS["calls"] - calls0, 3)   # one counted launch per MoE call
            self.assertEqual([c for c in launches if c[2]], [("prequant", False, True), ("prequant", False, True),
                                                             ("prequant", True, True)])

    def test_real_forward_normal_plain(self):
        self.run_case(False)

    def test_real_forward_normal_prequant(self):
        self.run_case(True)

    def test_pins_and_unwrap(self):
        import inspect
        ok, bad = F.check_pins()
        self.assertTrue(ok, bad)
        ok, bad = F.check_pins(F.LAZY_PINS)
        self.assertTrue(ok, bad)
        v2 = self.v2
        for fn in (v2.DeepseekV2MoE.forward_normal, v2.DeepseekV2MoE._forward_shared_experts,
                   v2.maybe_fuse_routed_scale_and_shared_add):
            self.assertTrue(getattr(fn, "_dsv41_moe_fold", False))
            self.assertFalse(getattr(inspect.unwrap(fn), "_dsv41_moe_fold", False))


class _Callable:
    def __init__(self, fn, **attrs):
        self._fn = fn
        for k, v in attrs.items():
            setattr(self, k, v)

    def __call__(self, *a, **k):
        return self._fn(*a, **k)


if __name__ == "__main__":
    unittest.main(verbosity=1)
