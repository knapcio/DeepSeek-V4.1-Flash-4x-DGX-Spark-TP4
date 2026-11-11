"""CPU checks for adapter/l2_prefetch.py with v5 SHPF (perf/decode-batch-20261109) merged onto v2.7.3's v6 W2
(perf/shpf-nogather-v273-20261110; adapted from the tier-8 merge test, whose slot order differed):

1. _counts() is the six-tuple (idx, lin, eng, moe, woa, hc): v6's wo_a count keeps v2.7.3's slot [4], the v5 hc count
   is appended at [5]; _before_wo_a reads [4], _before_hc_stats [5]; neither hook reads the other's slot;
2. one layer with both marks and both budgets on: the plan has the W2 window (wo_b continuation) and the SHPF window
   (shared expert), disjoint from each other and from every other window;
3. each budget alone: the other's key is absent and the plan equals the single-lever plan on the same events with the
   other lever's marks stripped;
4. the plan / capture receipt lines carry shpf= and wob2= only when that lever's budget is on, both when both are on,
   in the order " shpf=", " wob2=", then the A/B annotation.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_l2_prefetch_shpf_wob2.py
"""
import logging
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
WOA_EXP, WOA_VAL = 2 * 32 * 128, 2 * 1024 * 4096
WOB, WOB_S = 5120 * 2048, 5120 * 64


def layer(base):
    """AGc, wq_b, wo_a (+ W2 mark), wo_b, the post-attention AR with the c1 SHPF mark before it (fused path), router,
    the shared expert bracket, the post-MoE AR."""
    qb, exp, val, wob, wobs = base, base + 16 * MB, base + 32 * MB, base + 48 * MB, base + 64 * MB
    router, fc1, fc1s, fc2, fc2s = base + 80 * MB, base + 96 * MB, base + 112 * MB, base + 128 * MB, base + 144 * MB
    ev = [("c", 1), ("r", qb, 8192 * 1280, 8192), ("L", 1),
          ("w", exp, WOA_EXP), ("w", val, WOA_VAL), ("W", 1),
          ("r", wob, WOB, 5120), ("r", wobs, WOB_S, 5120),
          ("h", 1), ("c", 2),
          ("r", router, int(3.9 * MB)), ("S",),
          ("r", fc1, int(5.9 * MB), 1152), ("r", fc1s, int(0.18 * MB), 1152),
          ("r", fc2, int(3.3 * MB), 5120), ("r", fc2s, int(0.1 * MB), 5120), ("s",),
          ("c", 3)]
    return ev, dict(wob=wob, wobs=wobs, fc1=fc1, fc2=fc2)


def segs(plan, key):
    return [tuple(v) for v in plan[key][0].view(-1, 2).tolist()]


def setenv(shpf, wob2, woa_mb=6, wob_mb=0):
    os.environ["DSV41_L2_PREFETCH_WOA_MB"] = str(woa_mb)
    os.environ["DSV41_L2_PREFETCH_WOB_MB"] = str(wob_mb)
    for k, v in (("DSV41_L2_PREFETCH_SHPF_MB", shpf), ("DSV41_L2_PREFETCH_WOB2_MB", wob2)):
        if v:
            os.environ[k] = str(v)
        else:
            os.environ.pop(k, None)


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k][0], b[k][0]) and a[k][1] == b[k][1] for k in a)


def no_overlap(plan):
    allsegs = [(k, s) for k in plan for s in segs(plan, k)]
    for i, (ka, (a, n)) in enumerate(allsegs):
        for kb, (b, m) in allsegs[i + 1:]:
            assert a + n <= b or b + m <= a, (ka, kb)


def test_counts_six_tuple():
    l2._state.update(idx=7, lin=6, eng=5, moe=4, woa=3, hc=2)
    assert l2._counts() == (7, 6, 5, 4, 3, 2)
    l2._state.update(idx=0, lin=0, eng=0, moe=0, hc=0, woa=0)


def test_hooks_read_their_own_slot():
    setenv(6, 10)
    forks = []
    saved = l2._fork
    l2._fork = lambda item, stream=None, second=False: forks.append(item)
    try:
        sig = ("model", 6, "decode")
        # learned: 1 hc entry, 0 wo_a calls -> the SHPF window forks, the W2 one never does
        l2._state.update(sig=sig, idx=1, hc=0, hc_forks=0, woa=0, woa_forks=0,
                         plans={sig: (2, {("h", 1): "H", ("W", 1): "W"}, None, (2, 1, 0, 0, 0, 1))},
                         learning={sig: []})
        l2._before_hc_stats()
        l2._before_wo_a()
        assert forks == ["H"] and l2._state["hc_forks"] == 1 and l2._state["woa_forks"] == 0, forks
        # learned: 0 hc entries, 1 wo_a call -> only W2 forks
        forks.clear()
        l2._state.update(hc=0, hc_forks=0, woa=0, woa_forks=0,
                         plans={sig: (2, {("h", 1): "H", ("W", 1): "W"}, None, (2, 1, 0, 0, 1, 0))})
        l2._before_hc_stats()
        l2._before_wo_a()
        assert forks == ["W"] and l2._state["hc_forks"] == 0 and l2._state["woa_forks"] == 1, forks
        assert l2._state["learning"][sig] == [("h", 1), ("W", 1), ("h", 1), ("W", 1)]
    finally:
        l2._fork = saved
        l2._state.update(sig=None, plans={}, learning={}, hc=0, woa=0, hc_forks=0, woa_forks=0)


def test_both_windows_disjoint():
    setenv(6, 10)
    ev, w = layer(1 << 32)
    plan = l2._make_plan(ev)
    assert ("W", 1) in plan and ("h", 1) in plan, sorted(plan)
    assert segs(plan, ("W", 1))[0][0] == w["wob"]                  # v2.7 layout: W2 starts at wo_b
    assert any(a >= w["fc1"] for a, _ in segs(plan, ("h", 1)))
    no_overlap(plan)


def test_each_alone_is_its_own_plan():
    ev, _ = layer(1 << 32)
    setenv(6, 0)
    a = l2._make_plan(ev)
    b = l2._make_plan([e for e in ev if e[0] != "W"])
    assert same(a, b) and not any(k[0] == "W" for k in a)
    setenv(0, 10)
    a = l2._make_plan(ev)
    b = l2._make_plan([e for e in ev if e[0] not in ("h", "S", "s")])
    assert same(a, b) and not any(k[0] == "h" for k in a)
    setenv(0, 0)
    a = l2._make_plan(ev)
    b = l2._make_plan([e for e in ev if e[0] not in ("h", "S", "s", "W")])
    assert same(a, b)


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _receipts(shpf, wob2, capturing):
    setenv(shpf, wob2)
    ev, _ = layer(1 << 33)
    sig = ("model", 6, "decode", 1)
    grab = _Grab()
    l2.logger.addHandler(grab)
    saved_torch, saved_side, saved_join = l2.torch, l2._side_stream, l2._join
    try:
        counts = dict(idx=3, lin=1, eng=0, moe=0, hc=1 if shpf else 0, woa=1 if wob2 else 0)
        l2._state.update(sig=sig, learning={sig: list(ev)}, frozen=set(), logged=set(), plans={},
                         hc_forks=0, woa_forks=0, stats=[0, 0], **counts)
        l2.torch = types.SimpleNamespace(int64=torch.int64, tensor=saved_torch.tensor, cuda=types.SimpleNamespace(
            is_current_stream_capturing=lambda: False))
        l2._side_stream = lambda second=False: None
        l2._join = lambda: None
        l2._end_forward()                                   # eager: learns the plan, logs the plan line
        if capturing:
            l2._state.update(sig=sig, stats=[0, 4], hc_forks=1 if shpf else 0, woa_forks=1 if wob2 else 0, **counts)
            l2.torch.cuda.is_current_stream_capturing = lambda: True
            l2._end_forward()                               # capture: logs the capture line
    finally:
        l2.torch, l2._side_stream, l2._join = saved_torch, saved_side, saved_join
        l2.logger.removeHandler(grab)
        l2._state.update(sig=None, learning={}, plans={}, frozen=set(), logged=set(), hc=0, woa=0)
    key = "DSV41_L2_PREFETCH captured" if capturing else "DSV41_L2_PREFETCH plan"
    return [m for m in grab.lines if key in m]


def test_receipts():
    for shpf, wob2 in ((0, 0), (6, 0), (0, 10), (6, 10)):
        line = _receipts(shpf, wob2, False)[-1]
        assert (" shpf=" in line) == bool(shpf) and (" wob2=" in line) == bool(wob2), line
        assert " variant=" not in line, line          # single-arm receipt (the release carries no A/B annotation)
        if wob2:
            assert " wob2=1/" in line, line
        cap = _receipts(shpf, wob2, True)[-1]
        assert "MISMATCH" not in cap, cap
        assert (f" shpf={1 if shpf else 0}" in cap) == bool(shpf) and (" wob2=1" in cap) == bool(wob2), cap
        assert " variant=" not in cap, cap
        if shpf and wob2:
            assert line.index(" shpf=") < line.index(" wob2="), line
            assert cap.index(" shpf=") < cap.index(" wob2="), cap
        print("   ", shpf, wob2, "|", line.split("MB/forward", 1)[1].strip(), "|", cap.split(")", 1)[1].strip())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
