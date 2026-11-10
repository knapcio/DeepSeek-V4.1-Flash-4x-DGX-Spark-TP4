"""DSV41_PREFILL_GEMM_TACTIC, CPU: gate / table parsing, the 1-D scale view CUTLASS SM120 needs, the raw-mm wrapper's
dispatch (only agreed table shapes, only b12x / bf16 / 128x4, fail closed on any error), the boot check (bit identity
on three datasets, determinism, one TP collective, fail closed), the install order under mxfp8_b12x (the row split's
blocks reach the wrapper), the ModelRunner.load_model hook (__wrapped__, target only, once), the in-boot A/B registry,
and, where the image's FlashInfer is importable, that its sources match the pins.

The fake GEMMs compute in float64 on dequantized MXFP8 operands and really read the swizzled scales they are handed.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_prefill_gemm_tactic.py   (or pytest)
"""
import importlib
import importlib.util
import inspect
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
import prefill_gemm_tactic as pgt  # noqa: E402

pgt.AGREE[0] = False
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------------------------------------------------- fakes --

def swizzle(lin):
    """Row-major uint8 [rows, kb] -> FlashInfer SfLayout.layout_128x4, 1-D."""
    rows, kb = lin.shape
    mp, kp = (rows + 127) // 128 * 128, (kb + 3) // 4 * 4
    full = torch.full((mp, kp), 127, dtype=torch.uint8)
    full[:rows, :kb] = lin
    return full.view(mp // 128, 4, 32, kp // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def unswizzle(flat, rows, kb):
    mp, kp = (rows + 127) // 128 * 128, (kb + 3) // 4 * 4
    t = flat.reshape(-1)[:mp * kp].view(mp // 128, kp // 4, 32, 4, 4)
    return t.permute(0, 3, 2, 1, 4).reshape(mp, kp)[:rows, :kb]


def fake_quantize(x, is_sf_swizzled_layout=True, alignment=32):
    """MXFP8: per 32-wide block a power-of-two scale (UE8M0 byte), E4M3 values; 1-D swizzled scales."""
    r, k = x.shape
    xb = x.to(torch.float32).view(r, k // 32, 32)
    amax = xb.abs().amax(-1).clamp_min(2.0 ** -100)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (xb / torch.exp2(e)[..., None]).view(r, k).to(FP8)
    return q, swizzle((e + 127).to(torch.uint8))


def dequant(q, sf_flat, rows, k):
    s = unswizzle(sf_flat, rows, k // 32).to(torch.float64) - 127.0
    return q.to(torch.float64) * torch.exp2(s).repeat_interleave(32, dim=1)


def gemm(a, b, a_descale, b_descale, out, out_dtype, bump=0.0):
    m, k = a.shape
    n = b.shape[1]
    res = dequant(a, a_descale, m, k) @ dequant(b.t(), b_descale, n, k).t() + bump
    res = res.to(out_dtype)
    if out is None:
        return res
    out.copy_(res)
    return out


def make_serving(calls=None):
    def is_backend_supported(*a, **k):
        return True

    def mm_mxfp8(a, b, a_descale, b_descale, out=None, out_dtype=BF16, use_8x4_sf_layout=False, backend="auto"):
        if calls is not None:
            calls.append(("serving", a.shape[0], backend))
        return gemm(a, b, a_descale, b_descale, out, out_dtype)

    mm_mxfp8.is_backend_supported = is_backend_supported
    return mm_mxfp8


class FakeRunner:
    """The CUTLASS runner's contract: inputs [a, b, a_sf (1-D), b_sf (1-D), out_dtype, out, workspace], tactic."""

    def __init__(self, bump_m=None, raise_m=None, calls=None, n_tactics=6):
        self.bump_m, self.raise_m, self.calls, self.n = bump_m, raise_m, calls, n_tactics

    def get_valid_tactics(self, inputs, profile):
        return list(range(self.n))

    def __call__(self, inputs, tactic=-1):
        a, b, sa, sb, out_dtype, out, ws = inputs
        assert tactic == pgt.TACTIC and sa.dim() == 1 and sb.dim() == 1 and ws is not None
        m = a.shape[0]
        if self.calls is not None:
            self.calls.append(("tactic", m))
        if self.raise_m == m:
            raise RuntimeError("kernel rejected the problem")
        return gemm(a, b, sa, sb, out, out_dtype, bump=1.0 if self.bump_m == m else 0.0)


def use_runner(runner):
    pgt._RUNNER[0] = (runner, lambda dev: torch.empty(16, dtype=torch.uint8))


def operands(n=64, k=256, m=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g).to(BF16)
    w = torch.randn(n, k, generator=g).to(BF16)
    qx, sx = fake_quantize(x)
    qw, sw = fake_quantize(w)
    return qx, qw, sx, sw


def reset(shapes=None):
    pgt._DECIDED.clear()
    pgt._LOGGED.clear()
    pgt._CONFIG[0] = None
    pgt._ENABLED[0] = None
    pgt._RUNNER[0] = None
    pgt._ORIG_RAW[0] = None
    pgt._QUANTIZE[0] = None
    pgt.STATE.update(armed=False, booted=False, pins=None, boot_s=None, reason="not installed")
    for k in pgt.STATS:
        pgt.STATS[k] = 0
    for name in (pgt.GATE, pgt.SHAPES_ENV):
        os.environ.pop(name, None)
    if shapes is not None:
        os.environ[pgt.SHAPES_ENV] = shapes


# --------------------------------------------------------------------------------------------------------- tests --

def test_gate_and_table_parsing():
    reset()
    for v in (None, "", "0", "off", "false", "no"):
        assert pgt.gate_on(v) is False
    for v in ("1", "on", "true", "cutlass:5"):
        assert pgt.gate_on(v) is True
    for bad in ("2", "cutlass:4", "b12x"):
        try:
            pgt.gate_on(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    t = pgt.parse_shapes(None)
    assert len(t) == 8 and (5120, 2048, 8192) in t and (8192, 1280, 8192) in t
    assert (1792, 5120, 4096) in t and (1152, 5120, 4096) in t and (4096, 1280, 8192) in t
    assert (1792, 5120, 8192) not in t                      # not measured: the row split runs it as 4096 blocks
    assert not any(n == 5120 and k == 640 for n, k, _ in t)  # FC2: inside its A/A noise, not in the default
    assert all(m >= 4096 for _, _, m in t)                  # decode / verify shapes can never match
    assert pgt.parse_shapes(" 64x256@256 , 128x512@128 ") == {(64, 256, 256), (128, 512, 128)}
    for bad in ("64x256", "64x250@256", "a@b", ",", "0x256@1"):
        try:
            pgt.parse_shapes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_flat_scales():
    rows, k = 200, 256
    lin = torch.randint(0, 255, (rows, k // 32), dtype=torch.uint8)
    flat = swizzle(lin)
    assert pgt.flat_scales(flat, rows, k) is flat
    two = flat.view(256, 8)
    assert torch.equal(pgt.flat_scales(two, rows, k), flat)
    assert torch.equal(pgt.flat_scales(two.t(), rows, k), flat)          # the .t() view the linear hands over
    assert pgt.flat_scales(flat[:-4], rows, k) is None                    # wrong byte count
    assert pgt.flat_scales(flat.view(256, 8)[:, :4], rows, k) is None     # not dense
    assert pgt.flat_scales(flat.to(torch.int32), rows, k) is None         # not a UE8M0 byte tensor
    assert pgt.flat_scales(None, rows, k) is None


def test_run_tactic_matches_serving_and_respects_out():
    reset()
    calls = []
    use_runner(FakeRunner(calls=calls))
    qx, qw, sx, sw = operands()
    ref = make_serving()(qx, qw.t(), sx, sw)
    got = pgt.run_tactic(qx, qw.t(), sx, sw, None, BF16)
    assert torch.equal(got, ref) and calls == [("tactic", 256)]
    out = torch.zeros(512, 64, dtype=BF16)
    blk = out[256:]                                                       # the row split's out=out[r0:r1]
    assert pgt.run_tactic(qx, qw.t(), sx, sw, blk, BF16) is blk and torch.equal(out[256:], ref)
    assert pgt.run_tactic(qx, qw.t(), sx, sw, torch.zeros(64, 256, dtype=BF16).t(), BF16) is None
    assert pgt.run_tactic(qx.to(BF16), qw.t(), sx, sw, None, BF16) is None
    assert pgt.run_tactic(qx, qw, sx, sw, None, BF16) is None             # B must be weight.t()
    assert pgt.run_tactic(qx, qw.t(), sx[:-1], sw, None, BF16) is None
    reset()


def test_wrapper_dispatch_and_fail_closed():
    reset(shapes="64x256@256")
    os.environ[pgt.GATE] = "1"
    calls = []
    serving = make_serving(calls)
    use_runner(FakeRunner(calls=calls))
    mm = pgt.make_raw_mm(serving)
    assert mm.__wrapped__ is serving and mm.is_backend_supported("b12x") is True
    assert inspect.unwrap(mm) is serving
    qx, qw, sx, sw = operands()
    ref = serving(qx, qw.t(), sx, sw)
    calls.clear()
    # not decided yet (no boot check): the serving call
    assert torch.equal(mm(qx, qw.t(), sx, sw, out_dtype=BF16, backend="b12x"), ref)
    assert calls == [("serving", 256, "b12x")]
    pgt._DECIDED[(64, 256, 256)] = True
    calls.clear()
    assert torch.equal(mm(qx, qw.t(), sx, sw, out_dtype=BF16, backend="b12x"), ref)
    assert calls == [("tactic", 256)] and pgt.STATS["tactic_calls"] == 1
    for kw in ({"backend": "cutlass"}, {"backend": "b12x", "use_8x4_sf_layout": True},
               {"backend": "b12x", "out_dtype": torch.float16}):
        calls.clear()
        mm(qx, qw.t(), sx, sw, **kw)
        assert calls and calls[0][0] == "serving", (kw, calls)
    # a row count outside the table (decode, a short last chunk): serving
    qx2, _, sx2, _ = operands(m=128)
    calls.clear()
    mm(qx2, qw.t(), sx2, sw, out_dtype=BF16, backend="b12x")
    assert calls == [("serving", 128, "b12x")]
    # gate off (base env read once: re-read after a reset of the cache)
    pgt._ENABLED[0] = None
    os.environ[pgt.GATE] = "0"
    calls.clear()
    mm(qx, qw.t(), sx, sw, out_dtype=BF16, backend="b12x")
    assert calls == [("serving", 256, "b12x")]
    # a per-call error: serving result, that shape stays on serving from now on
    pgt._ENABLED[0] = None
    os.environ[pgt.GATE] = "1"
    use_runner(FakeRunner(raise_m=256, calls=calls))
    calls.clear()
    assert torch.equal(mm(qx, qw.t(), sx, sw, out_dtype=BF16, backend="b12x"), ref)
    assert calls == [("tactic", 256), ("serving", 256, "b12x")] and pgt._DECIDED[(64, 256, 256)] is False
    assert pgt.STATS["errors"] == 1
    calls.clear()
    mm(qx, qw.t(), sx, sw, out_dtype=BF16, backend="b12x")
    assert calls == [("serving", 256, "b12x")]
    reset()


def _arm(serving, shapes):
    reset(shapes=shapes)
    os.environ[pgt.GATE] = "1"
    pgt._ORIG_RAW[0] = serving
    pgt._QUANTIZE[0] = fake_quantize
    pgt.STATE["armed"] = True


def test_boot_check_decides_per_shape():
    _arm(make_serving(), "64x256@256,128x512@128,32x256@512")
    use_runner(FakeRunner())
    got = pgt.boot_check(device=torch.device("cpu"))
    assert got == {(64, 256, 256): True, (128, 512, 128): True, (32, 256, 512): True}, got
    assert pgt.STATE["booted"] and pgt.STATE["reason"] == "boot check 3/3"
    # a tactic that changes the numbers at one M: only that shape stays on serving
    _arm(make_serving(), "64x256@256,128x512@128")
    use_runner(FakeRunner(bump_m=128))
    assert pgt.boot_check(device=torch.device("cpu")) == {(64, 256, 256): True, (128, 512, 128): False}
    # a tactic that raises at one M
    _arm(make_serving(), "64x256@256,128x512@128")
    use_runner(FakeRunner(raise_m=256))
    assert pgt.boot_check(device=torch.device("cpu")) == {(64, 256, 256): False, (128, 512, 128): True}
    # a non-deterministic tactic (second run differs)
    _arm(make_serving(), "64x256@256")
    state = {"n": 0}

    def flaky(a, b, sa, sb, out, out_dtype):
        state["n"] += 1
        return gemm(a, b, sa, sb, out, out_dtype, bump=1.0 if state["n"] == 2 else 0.0)

    assert pgt.boot_check(device=torch.device("cpu"), tactic=flaky) == {(64, 256, 256): False}
    # not armed (pins / backend / install failed): every shape off, the collective still happens
    reset(shapes="64x256@256,128x512@128")
    assert pgt.boot_check(device=torch.device("cpu")) == {(64, 256, 256): False, (128, 512, 128): False}
    # runner cannot be resolved (wrong tactic count): off
    _arm(make_serving(), "64x256@256")
    use_runner(None)
    pgt._RUNNER[0] = None
    orig = pgt._resolve_runner
    pgt._resolve_runner = lambda: (_ for _ in ()).throw(RuntimeError("CUTLASS SM120 MXFP8 offers 4 tactics"))
    try:
        assert pgt.boot_check(device=torch.device("cpu")) == {(64, 256, 256): False}
        assert "runner unavailable" in pgt.STATE["reason"] or pgt.STATE["reason"] == "boot check 0/1"
    finally:
        pgt._resolve_runner = orig
    reset()


def test_operands_are_deterministic_and_cover_the_datasets():
    a1, w1 = pgt.operands(64, 256, 128, "cancel", torch.device("cpu"))
    a2, w2 = pgt.operands(64, 256, 128, "cancel", torch.device("cpu"))
    assert torch.equal(a1, a2) and torch.equal(w1, w2) and a1.dtype == BF16
    assert torch.equal(a1[:, 128:], -a1[:, :128])
    m1, _ = pgt.operands(64, 256, 128, "mixed", torch.device("cpu"))
    g1, _ = pgt.operands(64, 256, 128, "gauss", torch.device("cpu"))
    assert not torch.equal(m1, g1)
    assert m1.float().abs().amax() > 16 * g1.float().abs().amax() / 4           # the row / block scales spread


def test_agreement_is_one_min_collective_on_the_cpu_group():
    reset(shapes="64x256@256,128x512@128")
    pgt.AGREE[0] = True
    seen = []
    other = torch.tensor([1, 0], dtype=torch.int32)        # the other rank refuses the second shape

    def all_reduce(t, op=None, group=None):
        seen.append((op, group, t.device.type))
        t.copy_(torch.minimum(t, other))

    cpu_group = object()
    fake_dist_mod = types.ModuleType("sglang.srt.distributed")
    fake_dist_mod.get_tp_group = lambda: types.SimpleNamespace(cpu_group=cpu_group, device_group=None)
    saved = {k: sys.modules.get(k) for k in ("sglang", "sglang.srt", "sglang.srt.distributed")}
    sys.modules.setdefault("sglang", types.ModuleType("sglang"))
    sys.modules.setdefault("sglang.srt", types.ModuleType("sglang.srt"))
    sys.modules["sglang.srt.distributed"] = fake_dist_mod
    d = torch.distributed
    orig = (d.is_available, d.is_initialized, d.all_reduce, getattr(d, "get_rank", None))
    d.is_available, d.is_initialized, d.all_reduce = (lambda: True), (lambda: True), all_reduce
    d.get_rank = lambda: 0
    try:
        _arm(make_serving(), "64x256@256,128x512@128")
        use_runner(FakeRunner())
        got = pgt.boot_check(device=torch.device("cpu"))
        assert got == {(64, 256, 256): True, (128, 512, 128): False}, got
        assert len(seen) == 1 and seen[0][1] is cpu_group and seen[0][2] == "cpu"
        assert seen[0][0] == torch.distributed.ReduceOp.MIN
    finally:
        d.is_available, d.is_initialized, d.all_reduce = orig[:3]
        if orig[3] is not None:
            d.get_rank = orig[3]
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        pgt.AGREE[0] = False
        reset()


def test_pins_fail_closed(tmp=None):
    import tempfile
    root = tempfile.mkdtemp()
    ok, detail = pgt.check_pins(root)
    assert not ok and "FileNotFoundError" in detail
    for rel in pgt.PINS:
        p = os.path.join(root, "flashinfer", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("another version\n")
    ok, detail = pgt.check_pins(root)
    assert not ok and "!=" in detail


def fake_fp8_utils(calls):
    mod = types.SimpleNamespace()
    mod._raw_flashinfer_mm_mxfp8 = make_serving(calls)
    mod.flashinfer_mxfp8_quantize = fake_quantize
    mod.block_fp8_scale_to_mxfp8_e8m0 = lambda *a: None

    def original(input, weight, weight_scale, input_scale=None, bias=None, output_dtype=None, backend="cutlass",
                 pin_tactic=False):
        """fp8_utils.flashinfer_mxfp8_blockscaled_linear: the raw call is looked up at CALL time (a module global)."""
        x2 = input.reshape(-1, input.shape[-1])
        q, xs = (mod.flashinfer_mxfp8_quantize(x2) if input_scale is None else (x2, input_scale))
        ws_t = weight_scale.t() if weight_scale.ndim == 2 else weight_scale
        out = mod._raw_flashinfer_mm_mxfp8(q, weight.t(), xs, ws_t, out_dtype=output_dtype or BF16,
                                           use_8x4_sf_layout=False, backend=backend)
        return out.view(*input.shape[:-1], weight.shape[0])

    mod.flashinfer_mxfp8_blockscaled_linear = original
    return mod, original


def test_install_fp8_utils_gates_and_pins():
    calls = []
    for env, backend, pins, want in (("0", "b12x", True, False), ("1", "cudnn", True, False),
                                     ("1", "b12x", False, False), ("1", "b12x", True, True)):
        reset()
        os.environ[pgt.GATE] = env
        mod, _ = fake_fp8_utils(calls)
        raw = mod._raw_flashinfer_mm_mxfp8
        orig = pgt.check_pins
        pgt.check_pins = lambda root=None, ok=pins: (ok, "fake pins")
        try:
            got = pgt.install_fp8_utils(mod, backend)
        finally:
            pgt.check_pins = orig
        assert got is want, (env, backend, pins)
        assert (mod._raw_flashinfer_mm_mxfp8 is not raw) is want
        if want:
            assert mod._raw_flashinfer_mm_mxfp8.__wrapped__ is raw and pgt.STATE["armed"]
            assert pgt.install_fp8_utils(mod, backend) is True              # idempotent: no double wrap
            assert mod._raw_flashinfer_mm_mxfp8.__wrapped__ is raw
    reset()


def test_mxfp8_b12x_installs_the_tactic_before_the_row_split():
    """With both knobs on, the row split's 4096-row blocks of an 8192-row GEMM go through the tactic wrapper; the row
    split's own first-call check (split vs unsplit) still sees identical bytes."""
    import prefill_gemm_split as pgs
    reset(shapes="64x5120@4096")
    os.environ[pgt.GATE] = "1"
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT"] = "4096"
    os.environ["DSV41_PREFILL_GEMM_ROW_SPLIT_N"] = "all"
    pgs.AGREE[0] = False
    pgs._DECIDED.clear()
    calls = []
    orig_pins = pgt.check_pins
    pgt.check_pins = lambda root=None: (True, "fake pins")
    try:
        mod, original = fake_fp8_utils(calls)
        import mxfp8_b12x
        importlib.reload(mxfp8_b12x)
        mxfp8_b12x.install(mod)
        assert getattr(mod._raw_flashinfer_mm_mxfp8, "_dsv41_prefill_gemm_tactic", False)
        use_runner(FakeRunner(calls=calls))
        pgt._DECIDED[(64, 5120, 4096)] = True                 # as an agreed boot check would have left it
        linear = mod.flashinfer_mxfp8_blockscaled_linear
        g = torch.Generator().manual_seed(3)
        x = torch.randn(8192, 5120, generator=g).to(BF16)
        w, ws = fake_quantize(torch.randn(64, 5120, generator=g).to(BF16))
        q, xs = fake_quantize(x)
        ref = gemm(q, w.t(), xs, ws, None, BF16)
        calls.clear()
        first = linear(x, w, ws, None, None, BF16, backend="cutlass", pin_tactic=True)   # the row split's check
        assert torch.equal(first, ref)
        assert calls == [("tactic", 4096), ("tactic", 4096), ("serving", 8192, "b12x")], calls
        calls.clear()
        assert torch.equal(linear(x, w, ws, None, None, BF16, backend="cutlass", pin_tactic=True), ref)
        assert calls == [("tactic", 4096), ("tactic", 4096)], calls
        # decode-sized rows: serving, one launch
        calls.clear()
        linear(x[:64], w, ws, None, None, BF16, backend="cutlass", pin_tactic=True)
        assert calls == [("serving", 64, "b12x")], calls
    finally:
        pgt.check_pins = orig_pins
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT", None)
        os.environ.pop("DSV41_PREFILL_GEMM_ROW_SPLIT_N", None)
        pgs._DECIDED.clear()
        reset()


def test_mxfp8_b12x_off_by_default_leaves_the_raw_call():
    reset()
    calls = []
    mod, _ = fake_fp8_utils(calls)
    raw = mod._raw_flashinfer_mm_mxfp8
    import mxfp8_b12x
    importlib.reload(mxfp8_b12x)
    mxfp8_b12x.install(mod)
    assert mod._raw_flashinfer_mm_mxfp8 is raw and not pgt.STATE["armed"]


def test_load_model_hook():
    reset(shapes="64x256@256")
    ran = []

    class ModelRunner:
        def __init__(self, draft):
            self.is_draft_worker = draft

        def load_model(self):
            ran.append("load")
            return "loaded"

    stock = ModelRunner.load_model
    module = types.SimpleNamespace(ModelRunner=ModelRunner)
    assert pgt.install_model_runner(module) is False and ModelRunner.load_model is stock     # gate off: untouched
    os.environ[pgt.GATE] = "1"
    orig = pgt.boot_check
    pgt.boot_check = lambda: ran.append("check") or {}
    try:
        assert pgt.install_model_runner(module) is True
        assert ModelRunner.load_model.__wrapped__ is stock and inspect.unwrap(ModelRunner.load_model) is stock
        assert pgt.install_model_runner(module) is True and ModelRunner.load_model.__wrapped__ is stock
        assert ModelRunner(True).load_model() == "loaded" and ran == ["load"]          # draft: no check
        assert ModelRunner(False).load_model() == "loaded" and ran == ["load", "load", "check"]
        pgt.STATE["booted"] = True
        ModelRunner(False).load_model()
        assert ran == ["load", "load", "check", "load"]                                # once per process
        # a boot check that raises: the model still loads, nothing switches
        pgt.STATE["booted"] = False
        pgt._DECIDED[(64, 256, 256)] = True
        pgt.boot_check = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        assert ModelRunner(False).load_model() == "loaded"
        assert not pgt._DECIDED and pgt.STATE["booted"] and "boom" in pgt.STATE["reason"]
    finally:
        pgt.boot_check = orig
        ModelRunner.load_model = stock
        reset()


def test_ab_variant_registry_and_per_call_gate():
    import ab_variant
    assert pgt.GATE in ab_variant.RUNTIME and pgt.GATE in ab_variant.KNOWN
    assert pgt.GATE not in ab_variant.UNION_BOOL and "prefill_gemm_tactic" in ab_variant.HASHED_SOURCES
    reset()
    fake = types.SimpleNamespace(ACTIVE=True, KNOWN=frozenset([pgt.GATE]), _specs=[{pgt.GATE: "0"}, {pgt.GATE: "1"}],
                                 cur=[0])
    fake.env = lambda name, default=None: fake._specs[fake.cur[0]].get(name, default)
    saved = pgt._ab
    pgt._ab = fake
    try:
        assert pgt.wanted_at_install()                    # the base env is off, the A/B names it: installed
        assert pgt.enabled() is False
        fake.cur[0] = 1
        assert pgt.enabled() is True                      # per call, not cached
        fake._specs[1][pgt.GATE] = "garbage"
        assert pgt.enabled() is False                     # never raises inside a forward
    finally:
        pgt._ab = saved
        reset()


def test_sitecustomize_hooks_model_runner():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter", "sitecustomize.py")).read()
    i = src.index("'sglang.srt.model_executor.model_runner':")
    assert "prefill_gemm_tactic.install_model_runner(module)" in src[i:i + 1200]
    b12x = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter", "mxfp8_b12x.py")).read()
    assert b12x.index("prefill_gemm_tactic.install_fp8_utils") < b12x.index("prefill_gemm_split.make_linear_split")


def test_image_flashinfer_matches_the_pins():
    """In the image only: the FlashInfer sources that define tactic 5 are the measured 0.7.2 ones."""
    if importlib.util.find_spec("flashinfer") is None:
        print("  (skipped: flashinfer not installed)")
        return
    ok, detail = pgt.check_pins()
    assert ok, detail
    assert "sm121a=" in pgt.so_receipt(), pgt.so_receipt()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}", flush=True)
    print("all passed")
