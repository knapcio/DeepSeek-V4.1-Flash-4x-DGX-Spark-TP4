"""Checks for adapter/wo_a_cfg_table.py (DSV41_WO_A_CFG_TABLE) and its hooks in adapter/wo_a_w8.py.

CPU (TRITON_INTERPRET=1, the default here):
1. the table's partial kernel IS the measured variant: its AST equals the woa bench's generator (the anchored
   substitutions of woa_api.variant_source, blockscale=True) applied to wo_a_w8's own `_wo_a_partial_w8` text;
2. the table: M = 6 -> p:64:2:2:b (BN 64 = production's grid and partial layout), M = 96 -> mid:1:32:64:2:3 (production's
   split_k / BM at 96, the _MID_CFG row it was measured against), nothing else;
3. the kernels bitwise under the interpreter: the table's launch vs production's (`_partial_w8_stock`, `_wo_a_mid_stock`) at
   M = 6 (and 2 / 8: same kernel) and M = 96 (interpreter numerics are not the GPU's: this proves the harness logic and
   the launch arguments; the GPU proof is the module's own load / first-call self-check and the woa sweep);
4. the dispatch (fake launches): gate off / qx on / an untabled M / no load check / a mid drift -> production, the first
   eager call runs both and serves the table only on identical bytes (+0.0 vs -0.0 differ) with TP agreement; a capture
   with an unchecked key -> production, nothing recorded; a failed key stays on production;
5. the load self-check: ON for both entries on identical launches, OFF on differing bytes, on an equal negative control,
   on an exception (and the one all-reduce still runs with every entry), on a missing twin; runs once per process;
6. wo_a_w8's wiring: `_partial_w8` / `wo_a_mid` ask the table first, the load hook runs the self-check only when the
   key is on at install, the stock functions are production's launches.
GPU-less, real Triton (TRITON_INTERPRET=0, the image / a Triton wheel with the CUDA backend): 7. the table's kernels lower for
sm_121 at the table's num_warps / num_stages, with the evict-first policy on the 8-bit twin loads only.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_wo_a_cfg_table.py
  CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=0 PYTHONPATH=adapter python3 tests/test_wo_a_cfg_table.py   (PTX only)
"""
import ast
import inspect
import os
import re
import sys
import textwrap
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
for _k in [k for k in os.environ if k.startswith(("DSV41_WO_A_CFG", "DSV41_AB", "DSV41_WO_A_INPUT"))]:
    del os.environ[_k]
os.environ["DSV41_WO_A_W8"] = "1"
os.environ.setdefault("TRITON_INTERPRET", "1")
INTERP = os.environ.get("TRITON_INTERPRET") == "1"

import torch  # noqa: E402

import wo_a_cfg_table as T  # noqa: E402
import wo_a_w8 as W  # noqa: E402


def checkpoint_like(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(2, 1024, 4096, generator=g) * 60).clamp(-448, 448).to(torch.float8_e4m3fn)
    e = torch.randint(-14, -6, (2, 32, 128), generator=g).float()
    return (q.float().view(2, 32, 32, 128, 32) * torch.exp2(e)[:, :, None, :, None]).view(2, 1024, 4096).to(torch.bfloat16)


# ---- 1. the kernel is the measured variant ------------------------------------------------------------------------------
# the woa bench's generator (tier12o-kit/kit/bench/woa/woa_api.py variant_source), its anchors verbatim
_DEF = "def _wo_a_partial_w8("
_EV_PARAM = 'EV: tl.constexpr = ""):'
_N = "n = tile * 64 + tl.arange(0, 64)"
_ACC = "acc = tl.zeros((16, 64), tl.float32)"
_E = "e = tl.load(S + (group * 32 + n[None, :] // 32) * 128 + offsets[:, None] // 32, eviction_policy=EV)"
_W = "w = (w8.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)"
_BLOCK_E = ("nb = tile * (BN // 32) + tl.arange(0, BN // 32)\n"
            "{ind}kb = (split * 512 + i * 128) // 32 + tl.arange(0, 4)\n"
            "{ind}e = tl.load(S + (group * 32 + nb[None, :]) * 128 + kb[:, None], eviction_policy=EV)\n"
            "{ind}sc = tl.exp2(e.to(tl.float32) - 127.0)")
_BLOCK_W = ("w = tl.reshape(tl.reshape(w8.to(tl.float32), (4, 32, BN // 32, 32)) * sc[:, None, :, None],"
            " (128, BN)).to(tl.bfloat16)")


def _sub_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def bench_variant_source(kernel_src):
    t = _sub_once(kernel_src, _DEF, "def _woa_partial_var(")
    t = _sub_once(t, _EV_PARAM, 'EV: tl.constexpr = "", BN: tl.constexpr = 64):')
    t = _sub_once(t, _N, "n = tile * BN + tl.arange(0, BN)")
    t = _sub_once(t, _ACC, "acc = tl.zeros((16, BN), tl.float32)")
    m = re.search(r"^([ \t]*)" + re.escape(_E) + r"$", t, flags=re.M)
    t = _sub_once(t, _E, _BLOCK_E.format(ind=m.group(1)))
    return _sub_once(t, _W, _BLOCK_W)


def _fn_src(fn):
    return textwrap.dedent(inspect.getsource(getattr(fn, "fn", fn)))


def _ast_of(src, name="K"):
    node = ast.parse(src).body[0]
    node.name = name
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
            and isinstance(node.body[0].value.value, str):
        node.body = node.body[1:]                    # docstrings do not change a kernel / launch
    return ast.dump(node, include_attributes=False)


def test_kernel_is_the_measured_variant():
    want = bench_variant_source(_fn_src(W._wo_a_partial_w8))
    got = _fn_src(T._wo_a_partial_w8_bs)
    assert _ast_of(got) == _ast_of(want), "the table's kernel is not the bench's generated p:*:b variant of wo_a_w8's kernel"
    # and the stock launches are production's (the body the image ran before this module existed)
    assert "num_warps=4, num_stages=3" in _fn_src(W._partial_w8_stock) and "_wo_a_partial_w8[(16, 2, 8)]" in _fn_src(W._partial_w8_stock)
    assert "num_stages=3" in _fn_src(W._wo_a_mid_stock) and "next(cfg for lim, cfg in _MID_CFG if m <= lim)" in _fn_src(W._wo_a_mid_stock)


# ---- 2. the table ------------------------------------------------------------------------------------------------------
def test_table():
    assert sorted(T.TABLE) == [6, 96]
    kind, name, cfg = T.TABLE[6]
    assert (kind, name, cfg) == ("partial", "p:64:2:2:b", {"BN": 64, "num_warps": 2, "num_stages": 2})
    assert 1024 // cfg["BN"] == 16                       # production's grid (16, 2, 8)
    kind, name, cfg = T.TABLE[96]
    assert (kind, name) == ("mid", "mid:1:32:64:2:3")
    assert cfg == {"split_k": 1, "BM": 32, "BN": 64, "num_warps": 2, "num_stages": 3}
    prod = T.mid_row(W._MID_CFG, 96)
    assert prod == T.PROD_MID[96] == (1, 32, 64, 4)
    assert (cfg["split_k"], cfg["BM"]) == prod[:2]       # the k association and the row tile are production's
    assert T.PARTIAL_ROWS[0] <= 6 <= T.PARTIAL_ROWS[1] and T.MID_ROWS[0] <= 96 <= T.MID_ROWS[1]
    assert T.PROD_NAME == {6: "p:64:4:3", 96: "mid:1:32:64:4:3"}
    assert not T.wanted_at_install()


# ---- 3. bitwise under the interpreter ----------------------------------------------------------------------------------
def _twin():
    tw = W.make_twin(checkpoint_like())
    assert tw is not None
    return tw


def test_interpreter_bitwise_partial():
    if not INTERP:
        print("SKIP interpreter partial (TRITON_INTERPRET != 1)")
        return
    tw = _twin()
    g = torch.Generator().manual_seed(7)
    for m in (6, 2, 8):
        x = torch.randn(m, 2, 4096, generator=g).to(torch.bfloat16)
        for ev in ("evict_first", ""):
            a = W._partial_w8_stock(x, tw, m)
            b = T.launch_partial(x, tw, m, ev, T.TABLE[6][2])
            assert T._bytewise_equal(a, b), f"M={m} ev={ev!r}: the table's partials differ from production's"
    neg = T.launch_partial(x, T._negative_twin(tw), 8, "evict_first", T.TABLE[6][2])
    assert not T._bytewise_equal(W._partial_w8_stock(x, tw, 8), neg)


def test_interpreter_bitwise_mid():
    if not INTERP:
        print("SKIP interpreter mid (TRITON_INTERPRET != 1)")
        return
    tw = _twin()
    x = torch.randn(96, 2, 4096, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16)
    a = W._wo_a_mid_stock(x, tw)
    b = T.launch_mid(W._wo_a_mid_kernel, x, tw, 96, "evict_first", T.TABLE[96][2])
    assert a.dtype == torch.bfloat16 and a.shape == (96, 2, 1024)
    assert T._bytewise_equal(a, b), "M=96: the table's mid launch differs from production's"
    # a split-k config of the same launcher reduces as wo_a_mid does (the M=24 production row: split_k 2)
    x24 = x[:24].contiguous()
    sk2 = T.launch_mid(W._wo_a_mid_kernel, x24, tw, 24, "", {"split_k": 2, "BM": 32, "BN": 32, "num_warps": 4, "num_stages": 3})
    assert T._bytewise_equal(sk2, W._wo_a_mid_stock(x24, tw))


# ---- 4. the dispatch -------------------------------------------------------------------------------------------------
class Rec:
    def __init__(self):
        self.calls = []


def reset(gate="1", boot=(6, 96), capturing=False, agree=None):
    T._CHECKED.clear()
    T._BOOT.clear()
    for k in T._STATS:
        T._STATS[k] = 0
    T._STATE["boot_ran"] = False
    for m in boot:
        T._BOOT[m] = True
    T._env = lambda name, default=None: gate if name == T.ENV else os.environ.get(name, default)
    T._capturing = lambda: capturing
    T._agree = (lambda flags, device: [agree(f) for f in flags]) if agree is not None else (lambda flags, device: [bool(f) for f in flags])


_REAL = {"launch_partial": T.launch_partial, "launch_mid": T.launch_mid, "_agree": T._agree, "_capturing": T._capturing,
         "_env": T._env, "_negative_twin": T._negative_twin}


def restore():
    for k, v in _REAL.items():
        setattr(T, k, v)


def fakes(differ=False, negzero=False, raise_table=False):
    rec = Rec()

    def stock_out(m):
        out = torch.ones(8, m, 2, 1024)
        if negzero:
            out.view(-1)[0] = -0.0
        return out

    def table_partial(x, twin, m, ev, cfg):
        rec.calls.append(("table", m, ev, cfg["num_warps"]))
        if raise_table:
            raise RuntimeError("compile failed")
        out = torch.ones(8, m, 2, 1024)
        if differ:
            out.view(-1)[5] += 1
        if negzero:
            out.view(-1)[0] = 0.0
        return out

    def table_mid(kernel, o, twin, m, ev, cfg):
        rec.calls.append(("table-mid", m, ev, cfg["num_warps"]))
        return torch.zeros(m, 2, 1024, dtype=torch.bfloat16) + (1 if differ else 0)

    T.launch_partial = table_partial
    T.launch_mid = table_mid

    def stock(m):
        def f():
            rec.calls.append(("stock", m))
            return stock_out(m)
        return f

    def stock_mid(m):
        def f():
            rec.calls.append(("stock-mid", m))
            return torch.zeros(m, 2, 1024, dtype=torch.bfloat16)
        return f
    return rec, stock, stock_mid


def x_of(m):
    return torch.zeros(m, 2, 4096, dtype=torch.bfloat16)


def test_dispatch_first_call_serves_identical():
    try:
        reset()
        rec, stock, _ = fakes()
        out = T.partial(x_of(6), None, 6, False, "evict_first", stock(6))
        assert out is not None and rec.calls == [("stock", 6), ("table", 6, "evict_first", 2)]
        assert list(T._CHECKED.values()) == [True] and T._STATS["same"] == 1
        T.partial(x_of(6), None, 6, False, "evict_first", stock(6))
        assert rec.calls[-1] == ("table", 6, "evict_first", 2) and len(rec.calls) == 3      # no stock call any more
        T.partial(x_of(6), None, 6, False, "", stock(6))                                     # a new eviction policy: a new key
        assert len(T._CHECKED) == 2
    finally:
        restore()


def test_dispatch_mismatch_and_negzero_keep_production():
    try:
        for kw in ({"differ": True}, {"negzero": True}):
            reset()
            rec, stock, _ = fakes(**kw)
            out = T.partial(x_of(6), None, 6, False, "evict_first", stock(6))
            assert T._bytewise_equal(out, stock(6)()), kw            # production's bytes returned
            assert list(T._CHECKED.values()) == [False] and T._STATS["diff"] == 1
            assert T.partial(x_of(6), None, 6, False, "evict_first", stock(6)) is None   # production for good
    finally:
        restore()


def test_dispatch_raise_keeps_production():
    try:
        reset()
        rec, stock, _ = fakes(raise_table=True)
        seen = []
        T._agree = lambda flags, device: seen.append(list(flags)) or [bool(f) for f in flags]
        out = T.partial(x_of(6), None, 6, False, "evict_first", stock(6))
        assert out is not None and seen == [[False]] and list(T._CHECKED.values()) == [False]
    finally:
        restore()


def test_dispatch_capture_and_gates():
    try:
        reset(capturing=True)
        rec, stock, stock_mid = fakes()
        assert T.partial(x_of(6), None, 6, False, "evict_first", stock(6)) is None
        assert T._CHECKED == {} and T._STATS["skipped_capture"] == 1 and rec.calls == []
        reset(gate="0")
        assert T.partial(x_of(6), None, 6, False, "evict_first", stock(6)) is None and T._CHECKED == {}
        reset()
        assert T.partial(x_of(6), None, 6, True, "evict_first", stock(6)) is None             # qx on
        assert T.partial(x_of(5), None, 5, False, "evict_first", stock(5)) is None            # untabled M
        assert T.partial(x_of(96), None, 96, False, "evict_first", stock(96)) is None         # 96 is a mid entry
        assert T.mid(x_of(6), None, False, "evict_first", None, W._MID_CFG, stock_mid(6)) is None
        assert T.mid(x_of(24), None, False, "evict_first", None, W._MID_CFG, stock_mid(24)) is None
        assert rec.calls == [] and T._CHECKED == {}
        reset(boot=())                                                                         # no load self-check
        assert T.partial(x_of(6), None, 6, False, "evict_first", stock(6)) is None and T._STATS["no_boot"] == 1
        reset(boot=(6,))
        T._BOOT[96] = False
        assert T.mid(x_of(96), None, False, "evict_first", None, W._MID_CFG, stock_mid(96)) is None
        reset()
        drifted = ((16, (4, 16, 64, 8)), (32, (2, 32, 32, 4)), (64, (1, 64, 32, 4)), (128, (1, 32, 64, 8)), (192, (1, 64, 32, 4)))
        assert T.mid(x_of(96), None, False, "evict_first", None, drifted, stock_mid(96)) is None and T._STATS["drift"] == 1
        assert rec.calls == []
    finally:
        restore()


def test_dispatch_mid_and_tp_disagreement():
    try:
        reset()
        rec, stock, stock_mid = fakes()
        out = T.mid(x_of(96), None, False, "evict_first", "K", W._MID_CFG, stock_mid(96))
        assert out is not None and rec.calls == [("stock-mid", 96), ("table-mid", 96, "evict_first", 2)]
        assert list(T._CHECKED.values()) == [True]
        reset(agree=lambda f: False)
        rec, stock, stock_mid = fakes()
        out = T.partial(x_of(6), None, 6, False, "evict_first", stock(6))
        assert out is not None and list(T._CHECKED.values()) == [False] and T._STATS["disagree"] == 1
        reset()
        rec, stock, stock_mid = fakes(differ=True)
        out = T.mid(x_of(96), None, False, "evict_first", "K", W._MID_CFG, stock_mid(96))
        assert T._bytewise_equal(out, stock_mid(96)()) and list(T._CHECKED.values()) == [False]
    finally:
        restore()


# ---- 5. the load self-check ------------------------------------------------------------------------------------------
def fake_W(twins=True, mid_cfg=None):
    calls = []

    def ps(x, tw, m, qx=False):
        calls.append(("stock", m))
        return (x.float().sum() * 0 + tw[1].float().view(-1)[0]) * torch.ones(8, m, 2, 1024)

    def ms(x, tw, qx=False):
        calls.append(("stock-mid", x.shape[0]))
        return ((x.float().sum() * 0 + tw[1].float().view(-1)[0]) * torch.ones(x.shape[0], 2, 1024)).to(torch.bfloat16)
    tw = (torch.zeros(2, 1024, 4096, dtype=torch.float8_e4m3fn), torch.full((2, 32, 128), 120, dtype=torch.uint8))
    return types.SimpleNamespace(_TWINS={1: tw} if twins else {}, _MID_CFG=mid_cfg or W._MID_CFG, _ev=lambda: "evict_first",
                                 _partial_w8_stock=ps, _wo_a_mid_stock=ms, _wo_a_mid_kernel="K"), calls


def table_like_stock(differ_m=None, raise_m=None):
    def lp(x, tw, m, ev, cfg):
        if raise_m == m:
            raise RuntimeError("ptxas error")
        out = (x.float().sum() * 0 + tw[1].float().view(-1)[0]) * torch.ones(8, m, 2, 1024)
        return out + (1 if differ_m == m else 0)

    def lm(kernel, x, tw, m, ev, cfg):
        if raise_m == m:
            raise RuntimeError("ptxas error")
        out = ((x.float().sum() * 0 + tw[1].float().view(-1)[0]) * torch.ones(m, 2, 1024)).to(torch.bfloat16)
        return out + (1 if differ_m == m else 0)
    T.launch_partial, T.launch_mid = lp, lm


def test_boot_check():
    try:
        seen = []
        reset(boot=())
        T._agree = lambda flags, device: seen.append(list(flags)) or [bool(f) for f in flags]
        Wf, calls = fake_W()
        table_like_stock()
        assert T.boot_check(Wf) == {6: True, 96: True} and seen == [[True, True]]
        assert ("stock", 6) in calls and ("stock-mid", 96) in calls
        assert T.boot_check(Wf) == {6: True, 96: True} and len(seen) == 1                   # once per process
        for kw, want in (({"differ_m": 6}, {6: False, 96: True}), ({"raise_m": 96}, {6: True, 96: False})):
            seen.clear()
            reset(boot=())
            T._agree = lambda flags, device: seen.append(list(flags)) or [bool(f) for f in flags]
            table_like_stock(**kw)
            assert T.boot_check(fake_W()[0]) == want and len(seen) == 1 and len(seen[0]) == 2, (kw, seen)
        reset(boot=())
        table_like_stock()
        T._negative_twin = lambda tw: tw                       # a compare that cannot see a change proves nothing
        assert T.boot_check(fake_W()[0]) == {6: False, 96: False}
        restore()
        reset(boot=())
        table_like_stock()
        assert T.boot_check(fake_W(twins=False)[0]) == {6: False, 96: False}
        reset(boot=())
        table_like_stock()
        drifted = ((16, (4, 16, 64, 8)), (32, (2, 32, 32, 4)), (64, (1, 64, 32, 4)), (128, (1, 32, 64, 8)), (192, (1, 64, 32, 4)))
        assert T.boot_check(fake_W(mid_cfg=drifted)[0]) == {6: True, 96: False}
        reset(boot=(), agree=lambda f: False)
        table_like_stock()
        assert T.boot_check(fake_W()[0]) == {6: False, 96: False}
    finally:
        restore()


def test_boot_check_real_kernels():
    if not INTERP:
        print("SKIP boot check on the interpreter (TRITON_INTERPRET != 1)")
        return
    try:
        reset(boot=())
        Wr = types.SimpleNamespace(_TWINS={1: _twin()}, _MID_CFG=W._MID_CFG, _ev=lambda: "evict_first",
                                   _partial_w8_stock=W._partial_w8_stock, _wo_a_mid_stock=W._wo_a_mid_stock,
                                   _wo_a_mid_kernel=W._wo_a_mid_kernel)
        assert T.boot_check(Wr) == {6: True, 96: True}
    finally:
        restore()


# ---- 6. wo_a_w8's wiring ---------------------------------------------------------------------------------------------
def test_wo_a_w8_wiring():
    try:
        reset()
        rec, stock, stock_mid = fakes()
        seen = []
        real_p, real_m = W._partial_w8_stock, W._wo_a_mid_stock
        W._partial_w8_stock = lambda x, twin, m, qx=False: seen.append(("stock", m, qx)) or torch.ones(8, m, 2, 1024)
        W._wo_a_mid_stock = lambda o, twin, qx=False: seen.append(("stock-mid", o.shape[0], qx)) or torch.zeros(o.shape[0], 2, 1024, dtype=torch.bfloat16)
        W._partial_w8(x_of(6), None, 6)                          # first call: both, then the table
        W._partial_w8(x_of(6), None, 6)
        assert seen == [("stock", 6, False)] and [c[0] for c in rec.calls] == ["table", "table"]
        W._partial_w8(x_of(4), None, 4)                          # untabled: production only
        W._partial_w8(x_of(6), None, 6, qx=True)
        assert seen[-2:] == [("stock", 4, False), ("stock", 6, True)]
        W.wo_a_mid(x_of(96), None)
        W.wo_a_mid(x_of(40), None)
        assert ("stock-mid", 96, False) in seen and seen[-1] == ("stock-mid", 40, False) and rec.calls[-1][0] == "table-mid"
        reset(gate="0")
        n = len(rec.calls)
        W._partial_w8(x_of(6), None, 6)
        assert len(rec.calls) == n and seen[-1] == ("stock", 6, False)
        W._partial_w8_stock, W._wo_a_mid_stock = real_p, real_m
        # the load hook: the self-check only when the key is on at install
        booted = []
        real_boot = T.boot_check
        T.boot_check = lambda mod: booted.append(mod) or {}
        real_build, real_drop = W.build_twins, W.drop_bf16
        W.build_twins = lambda model: (0, 0)
        W.drop_bf16 = lambda model: None

        class M:
            def load_weights(self, *a, **kw):
                return "loaded"
        W._wrap_load(M)
        assert M().load_weights() == "loaded" and booted == []
        os.environ[T.ENV] = "1"

        class M2:
            def load_weights(self, *a, **kw):
                return "loaded"
        W._wrap_load(M2)
        assert M2().load_weights() == "loaded" and booted == [W]
        T.boot_check, W.build_twins, W.drop_bf16 = real_boot, real_build, real_drop
        assert "p:64:2:2:b" in T.armed_line() and "mid:1:32:64:2:3" in T.armed_line()
    finally:
        os.environ.pop(T.ENV, None)
        restore()


# ---- 7. PTX for sm_121 (real Triton, no GPU) -------------------------------------------------------------------------
def _ptx(kernel, signature, constexprs, warps, stages):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    src = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("cuda", 121, 32), options={"num_warps": warps, "num_stages": stages}).asm["ptx"]


def test_ptx_sm121():
    if INTERP:
        print("SKIP ptx (TRITON_INTERPRET=1; run with TRITON_INTERPRET=0 where Triton has its CUDA backend)")
        return
    try:
        import triton  # noqa: F401
        from triton.backends.compiler import GPUTarget  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print("SKIP ptx (no CUDA triton backend):", exc)
        return
    sig = {"X": "*bf16", "W8": "*fp8e4nv", "S": "*u8", "P": "*fp32", "M": "constexpr", "SX": "constexpr",
           "QX": "constexpr", "EV": "constexpr", "BN": "constexpr"}
    cfg = T.TABLE[6][2]
    ptx = _ptx(T._wo_a_partial_w8_bs, sig, {"M": 6, "SX": 8192, "QX": False, "EV": "evict_first", "BN": cfg["BN"]},
               cfg["num_warps"], cfg["num_stages"])
    lines = ptx.splitlines()
    assert any(".target sm_121" in ln for ln in lines), "not lowered for sm_121"
    hinted = [ln for ln in lines if "ld.global" in ln and "L2::cache_hint" in ln]
    plain = [ln for ln in lines if "ld.global" in ln and "L2::cache_hint" not in ln]
    policy = [ln for ln in lines if "createpolicy.fractional.L2::evict_first" in ln]
    assert policy and hinted and plain, (len(policy), len(hinted), len(plain))
    assert all("evict" not in ln for ln in plain)
    exp2 = sum(1 for ln in lines if "ex2.approx" in ln)
    print(f"partial p:64:2:2:b (warps {cfg['num_warps']}, stages {cfg['num_stages']}): {len(hinted)} hinted loads, "
          f"{len(plain)} plain loads, {len(policy)} policies, {exp2} ex2; triton {triton.__version__}")
    sig_mid = {"X": "*bf16", "W8": "*fp8e4nv", "S": "*u8", "Y": "*bf16", "M": "i32", "SXM": "i32",
               "SPLIT_K": "constexpr", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
               "QX": "constexpr", "EV": "constexpr"}
    c = T.TABLE[96][2]
    ptx = _ptx(W._wo_a_mid_kernel, sig_mid, {"SPLIT_K": c["split_k"], "BM": c["BM"], "BN": c["BN"], "BK": 128, "QX": False,
                                            "EV": "evict_first"}, c["num_warps"], c["num_stages"])
    assert ".target sm_121" in ptx and "createpolicy.fractional.L2::evict_first" in ptx
    print(f"mid mid:1:32:64:2:3 (warps {c['num_warps']}, stages {c['num_stages']}): lowered, hinted")
    print("WOA_TABLE_PTX_OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"test_wo_a_cfg_table: ALL {len(tests)} OK")
