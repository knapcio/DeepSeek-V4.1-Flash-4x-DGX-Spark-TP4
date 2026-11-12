"""CPU checks for the scale-codes retain mode (adapter/moe_b12x_next.py; DSV41_MOE_SCALE_CODES=1 alone, the v2.7.5
production profile, and under the in-boot A/B, adapter/ab_variant.py). No GPU, no b12x_next: the prepared experts, the geometry and the engine helpers are fakes;
moe_b12x_next and ab_variant are the real modules, loaded fresh per scenario with that scenario's environment.

1. default off: without DSV41_MOE_SCALE_CODES (alone, and with an armed A/B whose variants do not use it) the adapter
   neither imports ab_variant for the codes nor sets B12X_NEXT_W4A8_SCALE_CODES, converts nothing, keeps one prepared
   expert set per layer and hands every call (eager, inside a capture, pre-quantized) the native experts.
2. retain: under DSV41_AB_VARIANTS >= 2 with a variant enabling the gate, each layer keeps BOTH prepared experts: the
   native representation, its storage and the layer's parameters untouched (no resize, no ownership walk), the coded
   one a byte-exact round trip; the process-wide switch stays unset (plans and compact launches native); the
   pre-quantized prefill gate does not refuse; the receipts count native kept + codes.
3. selection: coded only inside a capture of a variant whose gate is on, at M >= its DSV41_MOE_SCALE_CODES_MIN_ROWS
   (default 2), on a capacity prepared for codes; eager calls (any runtime variant) and the pre-quantized path native.
4. receipts: the boot line and one capture line per (variant, M, top-k), in the formats the kit greps; refusals say
   "refused" (A/B not armed, malformed row floor).
5. _Geometry.prepare_coded: graph capacities minus compact micro plans, each warmed once with the coded experts.
6. production retain (v2.7.5: DSV41_MOE_SCALE_CODES=1 DSV41_MOE_SCALE_CODES_MIN_ROWS=2, no A/B, the stage-2b profile on):
   both experts kept, no process switch, no PREQUANT refusal; coded only while the current stream captures, at M >= 2,
   on a prepared capacity; eager / pre-quantized / oversized calls native; receipts "scale codes: retain: ..." and
   "capture variant=0 M=<m> coded|native"; a malformed floor refuses at load; the load warm-up is required.
7. DSV41_MOE_SCALE_CODES=release keeps the old codes-only path (process switch set, PREQUANT refused).

usage: python3 tests/test_scale_codes_retain.py      (torch; no GPU)
"""
import contextlib
import dataclasses
import importlib.util
import io
import os
import re
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapter")
sys.path.insert(0, ADAPTER)
import scale_codes as sc  # noqa: E402

FAILS = []
E, N, K, TOPK = 4, 576, 512, 6           # small K keeps the fake layer cheap; N % 128 == 64 as at TP4/EP1


def check(c, m):
    if not c:
        FAILS.append(m)


# ---- fake b12x_next prepared experts (the fields moe_b12x_next and scale_codes touch) ----------------------------
@dataclasses.dataclass(frozen=True, kw_only=True)
class _Val:
    w13_rp: torch.Tensor
    w13_sfb: torch.Tensor
    w2_rp: torch.Tensor
    w2_sfb: torch.Tensor
    n64_repack: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Rep:
    value: object


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Impl:
    w1_blockscale: torch.Tensor
    w2_blockscale: torch.Tensor
    representation: object

    def representation_for(self, mode):
        assert mode == "w4a8_mx"
        return self.representation.value


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Experts:
    plan: object
    _impl: object


class _FM:
    """fm.prepare_weights: the in-place N64 repack as b12x_next does it for the adapter (same storages; the scale bytes
    stay where the checkpoint parameter holds them, viewed as int32 words)."""

    class PackedWeights:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    @staticmethod
    def prepare_weights(plan, weights):
        e = weights.w13.shape[0]
        s13 = weights.w13_block_scales.view(e, -1).view(torch.int32)
        s2 = weights.w2_block_scales.view(e, -1).view(torch.int32)
        val = _Val(w13_rp=weights.w13, w13_sfb=s13, w2_rp=weights.w2, w2_sfb=s2)
        return _Experts(plan=plan, _impl=_Impl(w1_blockscale=s13, w2_blockscale=s2, representation=_Rep(value=val)))


class _Geom:
    """Stands in for _Geometry: records forward()'s experts; capacities 1-8 exact, 12, 24, 48, 96, chunk 96."""
    built = []

    def __init__(self, key, num_experts, hidden, inter, topk, ep_rank, limit, device, ep_size=1):
        self.key, self.E, self.K, self.N, self.topk = key, num_experts, hidden, inter, topk
        self.weight_plan = "plan"
        self.chunk = 96
        self.caps = [1, 2, 3, 4, 5, 6, 7, 8, 12, 24, 48, 96]
        self.coded_caps = frozenset()
        self.calls = []
        self.prepared = []

    def build(self, experts):
        _Geom.built.append(self)

    def prepare_coded(self, coded):
        self.prepared.append(coded)
        self.coded_caps = frozenset(c for c in self.caps if c != 3)      # capacity 3: a compact micro plan

    def cap_for(self, m):
        return next(c for c in self.caps if c >= m)

    def forward(self, impl, x, ids, w, out):
        self.calls.append(("forward", x.shape[0], impl))
        return out

    def prequant_views(self, m):
        return (torch.zeros(m, 1, dtype=torch.uint8), torch.zeros(m, 1, dtype=torch.uint8))

    def forward_prequant(self, impl, m, ids, w, out):
        self.calls.append(("prequant", m, impl))
        return out


def _scales(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(100, 120, (E, 1, 1), generator=g)
    return (base + torch.randint(0, 12, (E, rows, cols), generator=g)).to(torch.uint8)


def make_layer(seed=0):
    ly = torch.nn.Module()
    for name, t in (("w13_weight", torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8)),
                    ("w2_weight", torch.zeros(E, K, N // 2, dtype=torch.uint8)),
                    ("w13_weight_scale_inv", _scales(2 * N, K // 32, seed).view(torch.float8_e8m0fnu)),
                    ("w2_weight_scale_inv", _scales(K, N // 32, seed + 1).view(torch.float8_e8m0fnu))):
        setattr(ly, name, torch.nn.Parameter(t, requires_grad=False))
    ly.num_fused_shared_experts = 0
    ly.top_k, ly.moe_ep_rank, ly.moe_ep_size = TOPK, 0, 1
    method = types.SimpleNamespace(moe_runner_config=types.SimpleNamespace(swiglu_limit=10.0, gemm1_clamp_limit=None,
                                                                           gemm1_alpha=None))
    return ly, method


# ---- module loading per scenario -----------------------------------------------------------------------------------
_KEEP = {k: v for k, v in os.environ.items()}


def load(env):
    """Fresh ab_variant (configured from env, as sitecustomize does) and moe_b12x_next with that environment."""
    for k in [k for k in os.environ if k.startswith(("DSV41_", "B12X_NEXT_"))]:
        del os.environ[k]
    os.environ.update(env)
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]
    mods = {}
    for name in ("ab_variant", "moe_b12x_next"):
        sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(name, os.path.join(ADAPTER, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
        if name == "ab_variant" and os.environ.get("DSV41_AB_VARIANTS", "").strip() not in ("", "0", "1"):
            mod.configure()
    m = mods["moe_b12x_next"]
    m._state["b12x_ready"] = True
    m._B.update(fm=_FM)
    m._RealGeometry, m._Geometry = m._Geometry, _Geom
    m._peak_begin = lambda dev: 0
    m._peak_mb = lambda dev, base: None
    m._ENG.update(TopKOutputChecker=types.SimpleNamespace(format_is_bypassed=lambda t: False),
                  StandardCombineInput=lambda hidden_states: hidden_states,
                  is_allocation_symmetric=lambda: False)
    return mods["ab_variant"], m


def restore_env():
    for k in [k for k in os.environ if k not in _KEEP]:
        del os.environ[k]
    os.environ.update(_KEEP)
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]


def dispatch(m_rows, pre=None):
    x = torch.zeros(m_rows, K, dtype=torch.bfloat16)
    topk = types.SimpleNamespace(topk_ids=torch.zeros(m_rows, TOPK, dtype=torch.int32),
                                 topk_weights=torch.zeros(m_rows, TOPK))
    return types.SimpleNamespace(hidden_states=x, topk_output=topk, hidden_states_pre_quant=pre)


def call(m, layer, m_rows, pre=False):
    st = layer._dsv41_b12x_next
    q = None
    if pre:
        views = st.geom.prequant_views(m_rows)
        q = types.SimpleNamespace(_dsv41_b12x_rows=(st.geom, m_rows, views[0].data_ptr(), views[1].data_ptr()))
        st.geom.prequant_views = lambda mm, v=views: v
    quant_info = types.SimpleNamespace(w13_weight=layer.w13_weight, padded_hidden=None)
    st.geom.calls.clear()
    m._fused_b12x_next(dispatch(m_rows, q), quant_info, None)
    kind, rows, impl = st.geom.calls[-1]
    return kind, ("coded" if impl is st.coded_impl and impl is not None else "native" if impl is st.impl else "?")


def convert(m, seed=0):
    layer, method = make_layer(seed)
    m._convert_layer(method, layer)
    return layer


@contextlib.contextmanager
def stdout():
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = real


# ---- tests --------------------------------------------------------------------------------------------------------
def test_default_off():
    for env in ({"DSV41_MOE_B12X_NEXT": "1"},
                {"DSV41_MOE_B12X_NEXT": "1", "DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": "DSV41_INDEXER_NOSYNC=0",
                 "DSV41_AB_V1": "DSV41_INDEXER_NOSYNC=1"}):
        tag = "A/B armed" if "DSV41_AB_VARIANTS" in env else "alone"
        try:
            ab, m = load(env)
            check(not m.SCALE_CODES and not m.RETAIN and m._AB is None, f"{tag}: gate state {m.SCALE_CODES} {m.RETAIN}")
            check("B12X_NEXT_W4A8_SCALE_CODES" not in os.environ, f"{tag}: process switch set")
            layer = convert(m)
            st = layer._dsv41_b12x_next
            check(st.coded is None and st.coded_impl is None and st.geom.prepared == [], f"{tag}: a coded set was built")
            check(m._RETAIN["layers"] == 0 and m._CODES["layers"] == 0, f"{tag}: a layer was converted")
            check(layer.w13_weight_scale_inv.untyped_storage().nbytes() > 0, f"{tag}: native storage released")
            check(st.impl.representation.value.w13_sfb.dtype == torch.int32, f"{tag}: native representation changed")
            got = [call(m, layer, 6), call(m, layer, 1), call(m, layer, 12, pre=True)]
            if ab.ACTIVE:
                with ab.capturing(1):
                    got += [call(m, layer, 6), call(m, layer, 12)]
            check(all(g[1] == "native" for g in got), f"{tag}: {got}")
        finally:
            restore_env()


RETAIN_ENV = {"DSV41_MOE_B12X_NEXT": "1", "DSV41_AB_VARIANTS": "3", "DSV41_AB_V0": "DSV41_MOE_SCALE_CODES=0",
              "DSV41_AB_V1": "DSV41_MOE_SCALE_CODES=1",
              "DSV41_AB_V2": "DSV41_MOE_SCALE_CODES=1;DSV41_MOE_SCALE_CODES_MIN_ROWS=1",
              "DSV41_PREFILL_SP": "1", "DSV41_PREFILL_SP_FP8_MOE": "1"}


def test_retain_keeps_both():
    try:
        ab, m = load(RETAIN_ENV)
        check(ab.ACTIVE and os.environ.get("DSV41_MOE_SCALE_CODES") == "1", "the union did not switch the gate on")
        check(m.SCALE_CODES and m.RETAIN and m._AB is ab, "retain mode not selected")
        check(m.PREQUANT_WARM, "the production stage-2b profile is not on in this scenario")
        check("B12X_NEXT_W4A8_SCALE_CODES" not in os.environ, "retain set the process-wide switch (plans would be coded)")
        layers = []
        for seed in (0, 7):
            ly = convert(m, seed)                              # PREQUANT_WARM on: no refusal in retain mode
            layers.append(ly)
        for i, ly in enumerate(layers):
            st = ly._dsv41_b12x_next
            nv, cv = st.impl.representation.value, st.coded_impl.representation.value
            p13, p2 = ly.w13_weight_scale_inv, ly.w2_weight_scale_inv
            check(st.coded is not None and st.coded_impl is st.coded._impl and st.impl is st.experts._impl,
                  f"layer {i}: both prepared experts not kept")
            check(nv.w13_sfb.dtype == torch.int32 and nv.w2_sfb.dtype == torch.int32
                  and nv.w13_sfb.untyped_storage().data_ptr() == p13.untyped_storage().data_ptr()
                  and nv.w2_sfb.untyped_storage().data_ptr() == p2.untyped_storage().data_ptr(),
                  f"layer {i}: the native representation no longer names the parameters' storage")
            check(p13.untyped_storage().nbytes() == p13.numel() and p2.untyped_storage().nbytes() == p2.numel(),
                  f"layer {i}: a native storage was resized")
            check(cv.w13_sfb.dtype == torch.uint8 and cv.w2_sfb.dtype == torch.uint8
                  and st.coded_impl.w1_blockscale is cv.w13_sfb and st.coded_impl.w2_blockscale is cv.w2_sfb,
                  f"layer {i}: the coded handles are not the codes")
            for c, n_ in ((cv.w13_sfb, nv.w13_sfb), (cv.w2_sfb, nv.w2_sfb)):
                nat = n_.view(torch.uint8).reshape(E, -1)
                check(torch.equal(sc.unpack(c, nat.shape[1]), nat), f"layer {i}: codes do not round-trip")
            check(cv.w13_rp is nv.w13_rp and cv.w2_rp is nv.w2_rp, f"layer {i}: the weights were copied")
        geoms = {id(ly._dsv41_b12x_next.geom) for ly in layers}
        check(len(geoms) == 1 and len(layers[0]._dsv41_b12x_next.geom.prepared) == 1,
              "prepare_coded not run exactly once per geometry")
        native = sum(t.numel() for ly in layers for t in (ly.w13_weight_scale_inv, ly.w2_weight_scale_inv))
        check(m._RETAIN["layers"] == 2 and m._RETAIN["native"] == native
              and m._RETAIN["coded"] == 2 * E * 32 + native // 2, f"retain stats {m._RETAIN}")
        check(m._CODES["layers"] == 0, "the release path ran")
    finally:
        restore_env()


def test_selection_and_receipts():
    try:
        ab, m = load(RETAIN_ENV)
        layer = convert(m)
        with stdout() as out:
            eager = [call(m, layer, r) for r in (1, 6, 12, 96)]
            for v in range(3):
                ab.set_runtime(v)
                eager.append(call(m, layer, 12))
            ab.set_runtime(0)
            check(all(g == ("forward", "native") for g in eager), f"eager calls: {eager}")
            check(out.getvalue() == "", f"eager calls printed: {out.getvalue()!r}")
            want = {(0, 1): "native", (0, 6): "native", (0, 12): "native",
                    (1, 1): "native", (1, 2): "coded", (1, 6): "coded", (1, 12): "coded", (1, 3): "native",
                    (1, 96): "coded",
                    (2, 1): "coded", (2, 6): "coded", (2, 3): "native"}
            got = {}
            for (v, r), w in want.items():
                with ab.capturing(v):
                    got[(v, r)] = call(m, layer, r)[1]
                    call(m, layer, r)                          # a second layer call of the same shape: no new line
            check(got == want, f"capture selection: {got}")
            with ab.capturing(1):
                pre = call(m, layer, 12, pre=True)
                big = call(m, layer, 200)                      # beyond the chunk: never a graph shape here
            check(pre == ("prequant", "native") and big[1] == "native", f"pre-quantized {pre}, oversized {big}")
            ab.set_runtime(1)
            check(call(m, layer, 12)[1] == "native", "a replay-time (eager) call of the coded variant ran codes")
        text = out.getvalue()
        lines = text.splitlines()
        boot = [l for l in lines if "A/B retain:" in l and " layers, " in l]
        check(len(boot) == 1 and re.fullmatch(
            r"\[moe_b12x_next\] scale codes: A/B retain: 1 layers, native \d+\.\d{3} GiB kept \+ codes \d+\.\d{3} GiB, "
            r"min rows v1=2/v2=1, max expert span \d+", boot[0]), f"boot line: {boot}")
        cap = [l for l in lines if "scale codes: capture variant=" in l]
        keys = [re.match(r"\[moe_b12x_next\] scale codes: capture variant=(\d+) M=(\d+) (coded|native) \(top-k 6, "
                         r"capacity (?:\d+|None)", l) for l in cap]
        check(all(keys) and len(cap) == len({k.groups()[:2] for k in keys}) == len(want) + 1,
              f"capture lines not one per (variant, M): {cap}")
        check(all(k and want.get((int(k.group(1)), int(k.group(2))), "native") == k.group(3) for k in keys),
              "a capture line disagrees with the decision")
        check(text.index(boot[0]) < text.index(cap[0]), "the boot line does not precede the first capture line")
        check(m._RETAIN["decisions"][(1, 6, TOPK)] is True and m._RETAIN["decisions"][(1, 1, TOPK)] is False,
              f"decision record {m._RETAIN['decisions']}")
        # a second layer load (the draft's, after the target captured): the boot line again with the new count
        convert(m, 9)
        with stdout() as out2:
            with ab.capturing(1):
                call(m, layer, 6)
        check(re.search(r"A/B retain: 2 layers, ", out2.getvalue()) is not None, f"no updated boot line: {out2.getvalue()!r}")
        # a malformed floor reaching the adapter refuses (ab_variant refuses it at configure; this is the adapter's own)
        ab._specs[1]["DSV41_MOE_SCALE_CODES_MIN_ROWS"] = "0"
        try:
            with ab.capturing(1):
                call(m, layer, 48)
            check(False, "MIN_ROWS=0 accepted")
        except RuntimeError as exc:
            check("refused" in str(exc), f"MIN_ROWS refusal wording: {exc}")
    finally:
        restore_env()


def test_retain_needs_armed_ab():
    """DSV41_AB_VARIANTS >= 2 selects retain mode at import; an A/B that was never configured refuses the boot."""
    try:
        ab, m = load({"DSV41_MOE_B12X_NEXT": "1", "DSV41_MOE_SCALE_CODES": "1", "DSV41_AB_VARIANTS": "2",
                      "DSV41_AB_ALLOW_AA": "1"})
        check(m.RETAIN, "DSV41_AB_VARIANTS=2 did not select retain mode")
        ab.ACTIVE = False
        try:
            convert(m)
            check(False, "retain mode converted without an armed A/B")
        except RuntimeError as exc:
            check("refused" in str(exc) and "in-boot A/B" in str(exc), f"refusal wording: {exc}")
        check("B12X_NEXT_W4A8_SCALE_CODES" not in os.environ, "the process switch was set")
    finally:
        restore_env()


PROD_ENV = {"DSV41_MOE_B12X_NEXT": "1", "DSV41_MOE_SCALE_CODES": "1", "DSV41_MOE_SCALE_CODES_MIN_ROWS": "2",
            "DSV41_PREFILL_SP": "1", "DSV41_PREFILL_SP_FP8_MOE": "1"}


@contextlib.contextmanager
def stream_capturing():
    """torch.cuda as seen inside a CUDA-graph capture (CPU tensors)."""
    real = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: True
    try:
        yield
    finally:
        torch.cuda.is_current_stream_capturing = real


def test_production_retain():
    """v2.7.5's profile: retain mode without the in-boot A/B."""
    try:
        ab, m = load(PROD_ENV)
        check(m.SCALE_CODES and m.RETAIN and not m.RETAIN_AB and not m.RELEASE and m._AB is None,
              f"production retain not selected: {m.SCALE_CODES} {m.RETAIN} {m.RETAIN_AB} {m.RELEASE}")
        check(not ab.ACTIVE, "the A/B harness armed")
        check("B12X_NEXT_W4A8_SCALE_CODES" not in os.environ, "retain set the process-wide switch")
        check(m.PREQUANT_WARM, "the production stage-2b profile is not on in this scenario")
        layer = convert(m)                                 # PREQUANT_WARM on: no refusal
        st = layer._dsv41_b12x_next
        check(st.coded is not None and st.coded_impl is not None and len(st.geom.prepared) == 1,
              "the coded experts were not built beside the native ones")
        check(layer.w13_weight_scale_inv.untyped_storage().nbytes() == layer.w13_weight_scale_inv.numel(),
              "a native storage was resized")
        with stdout() as out:
            eager = [call(m, layer, r) for r in (1, 2, 6, 12, 96)] + [call(m, layer, 12, pre=True)]
            check(all(g[1] == "native" for g in eager), f"eager calls: {eager}")
            check(out.getvalue() == "", f"eager calls printed: {out.getvalue()!r}")
            want = {1: "native", 2: "coded", 3: "native", 6: "coded", 12: "coded", 96: "coded"}
            with stream_capturing():
                got = {r: call(m, layer, r)[1] for r in want}
                call(m, layer, 6)                          # same shape again: no new line
                pre = call(m, layer, 12, pre=True)
                big = call(m, layer, 200)
            check(got == want, f"capture selection: {got}")
            check(pre == ("prequant", "native") and big[1] == "native", f"pre-quantized {pre}, oversized {big}")
            check(call(m, layer, 6)[1] == "native", "a replay-time (eager) call ran codes")
        lines = out.getvalue().splitlines()
        boot = [l for l in lines if " layers, " in l]
        check(len(boot) == 1 and re.fullmatch(
            r"\[moe_b12x_next\] scale codes: retain: 1 layers, native \d+\.\d{3} GiB kept \+ codes \d+\.\d{3} GiB, "
            r"min rows 2, max expert span \d+", boot[0]), f"boot line: {boot}")
        cap = [l for l in lines if "scale codes: capture variant=" in l]
        check(len(cap) == len(want) + 1 and all(l.startswith("[moe_b12x_next] scale codes: capture variant=0 M=")
                                                for l in cap), f"capture lines: {cap}")
        check(m._RETAIN["decisions"][(0, 2, TOPK)] is True and m._RETAIN["decisions"][(0, 1, TOPK)] is False,
              f"decision record {m._RETAIN['decisions']}")
    finally:
        restore_env()
    for bad in ("0", "x"):
        try:
            ab, m = load(dict(PROD_ENV, DSV41_MOE_SCALE_CODES_MIN_ROWS=bad))
            try:
                convert(m)
                check(False, f"MIN_ROWS={bad} accepted at load")
            except RuntimeError as exc:
                check("refused" in str(exc), f"MIN_ROWS refusal wording: {exc}")
        finally:
            restore_env()
    try:   # the real prepare_coded refuses a production retain boot without the load warm-up
        ab, m = load(PROD_ENV)
        m.WARM = False
        g = object.__new__(m._RealGeometry)
        g.caps, g.chunk, g.K, g.topk, g.E, g.N, g.device = [1, 2], 2, 8, TOPK, E, N, torch.device("cpu")
        g.table = [None] + [g._lookup(r) for r in range(1, m.EXACT_MAX + 1)]
        g.plans = {c: types.SimpleNamespace(state=types.SimpleNamespace(compact_launches=None)) for c in g.caps}
        try:
            g.prepare_coded(types.SimpleNamespace(_impl="CODED"))
            check(False, "production retain without the load warm-up accepted")
        except RuntimeError as exc:
            check("load warm-up" in str(exc), f"warm refusal wording: {exc}")
    finally:
        restore_env()


def test_release_mode_unchanged():
    """DSV41_MOE_SCALE_CODES=release (TEST ONLY) keeps the old codes-only path: process switch set, PREQUANT refused."""
    try:
        ab, m = load({"DSV41_MOE_B12X_NEXT": "1", "DSV41_MOE_SCALE_CODES": "release"})
        check(m.SCALE_CODES and m.RELEASE and not m.RETAIN and os.environ.get("B12X_NEXT_W4A8_SCALE_CODES") == "1",
              "release mode not selected")
        m.PREQUANT_WARM = True
        try:
            convert(m)
            check(False, "release mode accepted the pre-quantized prefill input")
        except RuntimeError as exc:
            check("refused" in str(exc), f"PREQUANT refusal: {exc}")
    finally:
        restore_env()


def test_prepare_coded():
    """The real _Geometry.prepare_coded on a stub geometry: graph capacities (cap_for of 1..EXACT_MAX) minus compact
    micro plans; every kept capacity runs once with the coded experts (compiled before any capture)."""
    try:
        ab, m = load(RETAIN_ENV)
        g = object.__new__(m._RealGeometry)
        g.caps = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 96, 128, 4096]
        g.chunk, g.K, g.topk, g.E, g.N, g.device = 4096, 8, TOPK, E, N, torch.device("cpu")
        g.table = [None] + [g._lookup(r) for r in range(1, m.EXACT_MAX + 1)]
        micro = types.SimpleNamespace(compact_launches=object())
        dyn = types.SimpleNamespace(compact_launches=None)
        g.plans = {c: types.SimpleNamespace(state=micro if c == 3 else dyn) for c in g.caps}
        runs = []
        g._run = lambda c, x, ids, w, out, impl: runs.append((c, x.shape[0], impl))
        coded = types.SimpleNamespace(_impl="CODED")
        sync, torch.cuda.synchronize = torch.cuda.synchronize, (lambda *a: None)
        try:
            with stdout() as out:
                g.prepare_coded(coded)
        finally:
            torch.cuda.synchronize = sync
        want = [c for c in g.caps if c <= 96 and c != 3]
        check(sorted(g.coded_caps) == want, f"coded caps {sorted(g.coded_caps)}")
        check(runs == [(c, c, "CODED") for c in want], f"warm runs {runs}")
        check("native only (compact micro plans) [3]" in out.getvalue(), f"geometry line: {out.getvalue()!r}")
    finally:
        restore_env()


if __name__ == "__main__":
    for fn in (test_default_off, test_retain_keeps_both, test_selection_and_receipts, test_retain_needs_armed_ab,
               test_production_retain, test_release_mode_unchanged, test_prepare_coded):
        n = len(FAILS)
        fn()
        print(f"{fn.__name__}: {'ok' if len(FAILS) == n else 'FAIL'}", flush=True)
    for f in FAILS:
        print("  " + f)
    assert not FAILS, f"{len(FAILS)} check(s) failed"
    print("test_scale_codes_retain: ok")
