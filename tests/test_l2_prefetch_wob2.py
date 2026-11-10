"""CPU checks for adapter/l2_prefetch.py v6 W2 (torch on CPU; the plan's device tensors are made on the CPU here):

1. WOB2 off: no ("W", n) event is recorded and a plan built from an event sequence that carries one anyway is
   the v4 plan bit for bit (the marker is ignored);
2. WOB2 on, the production attention-core layout (WOA_MB=0 WOB_MB=6): the ("W", 1) window takes the wo_b bytes
   the attention-core window did not take (weight remainder, then scales), continues exactly where that take
   stopped (no byte twice), 16-byte pieces, cut at the budget;
3. WOB2 on, the v2.7 layout (WOA_MB=6 WOB_MB=0): the attention-core window holds wo_a only and ("W", 1) starts at
   wo_b's first byte;
4. a WOB take that already holds every wo_b byte leaves the scales only; one that holds everything plans no W2;
5. two layers: one W2 window per wo_a call, each on its own layer's wo_b;
6. the fork: _before_wo_a numbers the wo_a calls, forks the planned window on the second side stream, is a no-op
   past the learned count, when the forward is not bracketed, and when WOB2 is off (no event either);
7. every wrapper l2_prefetch installs (every install_* with every sub-gate on, incl. the v6 W2 hook inside the wo_a
   recorder) unwraps to the engine's own function (inspect.unwrap: the adapters that hash-pin engine sources, e.g.
   prefill_sp's v4.DeepseekV4Model.forward, see the stock source after the bracket), and a second install changes
   nothing; the sweep compares every callable of the fake modules, so a new wrapper without __wrapped__ fails here,
   not in the image's engine-pin probe.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_l2_prefetch_wob2.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
for k in [k for k in os.environ if k.startswith(("DSV41_L2_PREFETCH", "DSV41_AB"))]:
    del os.environ[k]
os.environ["DSV41_L2_PREFETCH"] = "1"
os.environ["DSV41_L2_PREFETCH_WOA"] = "1"

import torch  # noqa: E402

import l2_prefetch as l2  # noqa: E402

_real_tensor = torch.tensor
l2.torch = types.SimpleNamespace(**{k: getattr(torch, k) for k in ("int64", "cuda")},
                                 tensor=lambda data, dtype=None, device=None: _real_tensor(data, dtype=dtype))
MB = 1 << 20
WOA_EXP, WOA_VAL = 2 * 32 * 128, 2 * 1024 * 4096          # the fp8 twin: exponents (8 KiB) then e4m3 (8 MiB)
WOB, WOB_S = 5120 * 2048, 5120 * 64                        # wo_b weight (10 MiB) and its e8m0 scales


def layer(base, idx0=1, lin0=1, n=1):
    """One decode layer's attention window: AGc, wq_b, the wo_a call (twin exponents + values, then the W2
    mark), wo_b weight + scales, the post-attention AR."""
    qb, exp, val, wob, wobs = base, base + 16 * MB, base + 32 * MB, base + 48 * MB, base + 64 * MB
    ev = [("c", idx0), ("r", qb, 8192 * 1280, 8192), ("r", qb + 12 * MB, 8192 * 40, 8192), ("L", lin0),
          ("w", exp, WOA_EXP), ("w", val, WOA_VAL), ("W", n),
          ("r", wob, WOB, 5120), ("r", wobs, WOB_S, 5120), ("c", idx0 + 1)]
    return ev, dict(qb=qb, exp=exp, val=val, wob=wob, wobs=wobs)


def segs(plan, key):
    return [tuple(v) for v in plan[key][0].view(-1, 2).tolist()]


def setenv(woa_mb, wob_mb, wob2_mb):
    os.environ["DSV41_L2_PREFETCH_WOA_MB"] = str(woa_mb)
    os.environ["DSV41_L2_PREFETCH_WOB_MB"] = str(wob_mb)
    if wob2_mb:
        os.environ["DSV41_L2_PREFETCH_WOB2_MB"] = str(wob2_mb)
    else:
        os.environ.pop("DSV41_L2_PREFETCH_WOB2_MB", None)


def no_overlap(plan):
    allsegs = [(k, s) for k in plan for s in segs(plan, k)]
    for i, (ka, (a, n)) in enumerate(allsegs):
        for kb, (b, m) in allsegs[i + 1:]:
            assert a + n <= b or b + m <= a, (ka, kb)


def test_off_is_v4():
    setenv(0, 6, 0)
    ev, _ = layer(1 << 32)
    plain = [e for e in ev if e[0] != "W"]
    a, b = l2._make_plan(ev), l2._make_plan(plain)
    assert a.keys() == b.keys() and all(torch.equal(a[k][0], b[k][0]) and a[k][1] == b[k][1] for k in a)
    assert not any(k[0] == "W" for k in a)
    # the marker is not even recorded while off
    l2._state.update(sig=("s",), learning={("s",): []}, woa=0)
    l2._before_wo_a()
    assert l2._state["learning"][("s",)] == [] and l2._state["woa"] == 0
    l2._state.update(sig=None, learning={})


def test_production_layout_continues_the_wob_take():
    setenv(0, 6, 6)
    ev, w = layer(1 << 32)
    plan = l2._make_plan(ev)
    att = segs(plan, ("l", 1))
    assert att == [(w["wob"], 6 * MB)], att                       # WOA_MB=0: no wo_a, 6 MB of wo_b
    w2 = segs(plan, ("W", 1))
    assert w2[0] == (w["wob"] + 6 * MB, WOB - 6 * MB), w2        # the weight remainder (4 MiB)
    assert w2[1] == (w["wobs"], WOB_S & ~15), w2                 # then the scales
    total = sum(n for _, n in w2)
    assert total <= 6 * MB and all(a % 16 == 0 and n % 16 == 0 for a, n in w2)
    no_overlap(plan)
    # the AG window before it is the v4 plan (q_b first 6 MB)
    assert segs(plan, ("c", 1))[0] == (w["qb"], 6 * MB)
    # a smaller budget cuts the remainder
    setenv(0, 6, 2)
    plan = l2._make_plan(ev)
    assert segs(plan, ("W", 1)) == [(w["wob"] + 6 * MB, 2 * MB)]


def test_v27_layout_starts_at_wo_b():
    setenv(6, 0, 6)
    ev, w = layer(1 << 32)
    plan = l2._make_plan(ev)
    att = segs(plan, ("l", 1))
    assert att[0] == (w["exp"], WOA_EXP) and att[1] == (w["val"], 6 * MB - WOA_EXP), att   # wo_a only
    assert segs(plan, ("W", 1)) == [(w["wob"], 6 * MB)]
    no_overlap(plan)
    # a WOA budget larger than wo_a reaches into wo_b; W2 continues after that reach
    setenv(9, 0, 6)
    plan = l2._make_plan(ev)
    reach = 9 * MB - WOA_EXP - WOA_VAL
    assert segs(plan, ("l", 1))[-1] == (w["wob"], reach)
    assert segs(plan, ("W", 1))[0] == (w["wob"] + reach, 6 * MB), segs(plan, ("W", 1))
    no_overlap(plan)


def test_everything_held_leaves_scales_or_nothing():
    ev, w = layer(1 << 32)
    setenv(0, 10, 6)                                             # 10 MiB = the whole weight
    plan = l2._make_plan(ev)
    assert segs(plan, ("W", 1)) == [(w["wobs"], WOB_S & ~15)]
    setenv(0, 11, 6)                                             # weight + scales
    plan = l2._make_plan(ev)
    assert ("W", 1) not in plan
    no_overlap(plan)


def test_two_layers():
    setenv(0, 6, 6)
    ev1, w1 = layer(1 << 32, idx0=1, lin0=1, n=1)
    ev2, w2 = layer(1 << 33, idx0=3, lin0=2, n=2)
    plan = l2._make_plan(ev1 + ev2)
    assert segs(plan, ("W", 1))[0][0] == w1["wob"] + 6 * MB
    assert segs(plan, ("W", 2))[0][0] == w2["wob"] + 6 * MB
    no_overlap(plan)
    # a "W" that does not fall in a wo_a window (no wo_a recorded before it) plans nothing
    assert ("W", 1) not in l2._make_plan([("c", 1), ("W", 1), ("r", 1 << 32, 4 * MB, 5120)])


def test_fork():
    setenv(0, 6, 6)
    forks = []
    saved = l2._fork
    l2._fork = lambda item, stream=None, second=False: forks.append((item, second))
    try:
        sig = ("model", 6, "decode")
        item = ("segs", 2)
        l2._state.update(sig=sig, idx=1, woa=0, woa_forks=0,
                         plans={sig: (2, {("W", 1): item}, None, (2, 1, 0, 0, 1))}, learning={sig: []})
        l2._before_wo_a()
        l2._before_wo_a()        # a second wo_a call past the learned count: nothing
        assert forks == [(item, True)] and l2._state["woa"] == 2 and l2._state["woa_forks"] == 1
        assert l2._state["learning"][sig] == [("W", 1), ("W", 2)]   # recorded while learning
        l2._state["sig"] = None
        l2._before_wo_a()
        assert len(forks) == 1
        # a plan from before v6 (4-tuple counts) never forks
        l2._state.update(sig=sig, woa=0, plans={sig: (2, {("W", 1): item}, None, (2, 1, 0, 0))})
        l2._before_wo_a()
        assert len(forks) == 1
    finally:
        l2._fork = saved
        l2._state.update(sig=None, plans={}, learning={}, woa=0, woa_forks=0)


def test_wrapper_order():
    """The wo_a dispatch wrapper records the wo_a bytes and then the W2 mark, outermost call only."""
    calls = []
    saved_rec, saved_before = l2._record_wo_a, l2._before_wo_a
    l2._record_wo_a = lambda o, w: calls.append("record")
    l2._before_wo_a = lambda: calls.append("W")
    try:
        inner = l2._wrap_wo_a(lambda o, w, *a, **k: "out")
        outer = l2._wrap_wo_a(lambda o, w, *a, **k: inner(o, w))
        assert outer(None, None) == "out" and calls == ["record", "W"], calls
    finally:
        l2._record_wo_a, l2._before_wo_a = saved_rec, saved_before



def test_every_wrapper_unwraps_to_engine():
    """Every install_* of l2_prefetch, every install-time sub-gate on: each engine attribute it replaced unwraps to
    the stock function (the sweep compares every callable of the fake modules, so it catches a new wrapper too).
    Nine wrappers: no v5 SHPF in this tree, so DeepseekV2MoE._forward_shared_experts and
    DeepseekV4DecoderLayer._hc_mix_stats must stay the engine's own."""
    import inspect

    def stock(name):
        def fn(*args, **kwargs):
            return (name, args[-1] if args else None)
        fn.__name__ = fn.__qualname__ = name
        return fn

    def cls(name, *methods):
        return type(name, (), {m: stock(f"{name}.{m}") for m in methods})

    v2 = types.SimpleNamespace(DeepseekV2MoE=cls("DeepseekV2MoE", "_forward_shared_experts"),
                               tiny_gemm_bf16=stock("tiny_gemm_bf16"))
    fp8 = types.SimpleNamespace(flashinfer_mxfp8_blockscaled_linear=stock("flashinfer_mxfp8_blockscaled_linear"))
    roce = types.SimpleNamespace(RoceOneshotAllReduce=cls("RoceOneshotAllReduce", "all_reduce", "all_gather"))
    v4 = types.SimpleNamespace(DeepseekV4Model=cls("DeepseekV4Model", "forward"),
                               DeepseekV4DecoderLayer=cls("DeepseekV4DecoderLayer", "_hc_mix_stats"),
                               DeepseekV4ForCausalLM=cls("DeepseekV4ForCausalLM", "forward"),
                               _apply_wo_a_bf16_matmul=stock("_apply_wo_a_bf16_matmul"))
    draft = types.SimpleNamespace(DeepseekV4ForCausalLMDSpark=cls("DeepseekV4ForCausalLMDSpark", "forward"))
    engram = types.SimpleNamespace(EngramEmbedding=cls("EngramEmbedding", "_owned_rows"))
    mods = dict(v2=v2, fp8=fp8, roce=roce, v4=v4, draft=draft, engram=engram)

    def callables():
        out = {}
        for mname, m in mods.items():
            for k, v in vars(m).items():
                if isinstance(v, type):
                    out.update({(mname, k, a): f for a, f in vars(v).items() if callable(f) and not a.startswith("__")})
                elif callable(v):
                    out[(mname, k)] = v
        return out

    def install_all():
        l2.install_fp8_utils(fp8)
        l2.install_router(v2)
        l2.install_roce(roce)
        l2.install_model(v4)
        l2.install_draft(draft)
        l2.install_engram(engram)
        l2._ensure_wo_a([v4])

    before = callables()
    gates = {f"DSV41_L2_PREFETCH_{g}": "1" for g in ("WOA", "LMHEAD", "DRAFT", "ENGRAM", "MOE", "AHEAD")}
    gates.update(DSV41_L2_PREFETCH_WOB2_MB="10", DSV41_L2_PREFETCH_WOA_MB="6", DSV41_L2_PREFETCH_WOB_MB="0",
                 DSV41_L2_PREFETCH_ENGRAM_MB="12")
    saved = {k: os.environ.get(k) for k in gates}
    lib, l2._lib = l2._lib, lambda: None    # install_roce builds the CUDA helper; not needed for wrapping
    try:
        os.environ.update(gates)
        install_all()
        after = callables()
        install_all()                        # idempotent: no wrapper of a wrapper
        assert callables() == after
    finally:
        l2._lib = lib
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert before.keys() == after.keys()
    wrapped = {k for k in before if after[k] is not before[k]}
    expect = {("fp8", "flashinfer_mxfp8_blockscaled_linear"), ("v2", "tiny_gemm_bf16"),
              ("roce", "RoceOneshotAllReduce", "all_reduce"), ("roce", "RoceOneshotAllReduce", "all_gather"),
              ("v4", "DeepseekV4Model", "forward"), ("v4", "DeepseekV4ForCausalLM", "forward"),
              ("v4", "_apply_wo_a_bf16_matmul"), ("draft", "DeepseekV4ForCausalLMDSpark", "forward"),
              ("engram", "EngramEmbedding", "_owned_rows")}
    assert wrapped == expect, sorted(wrapped ^ expect)
    for k in sorted(wrapped):
        w = after[k]
        assert getattr(w, "__module__", None) == "l2_prefetch", k
        assert inspect.unwrap(w) is before[k], f"{'.'.join(k)} does not unwrap to the engine's function"
        assert inspect.getsource(inspect.unwrap(w)) == inspect.getsource(before[k])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
