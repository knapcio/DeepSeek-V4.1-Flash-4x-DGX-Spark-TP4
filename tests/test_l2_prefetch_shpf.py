"""CPU checks for adapter/l2_prefetch.py v5 SHPF (torch on CPU; the device tensors of the plan are made on the CPU here):

1. SHPF off: the v5 markers are never recorded, and a plan built from an event sequence that carries them anyway is
   the v4 plan bit for bit;
2. SHPF on, both production layouts of the FFN-side hyper-connection entry (c1: the attention-side statistics are
   recorded before the post-attention all-reduce, inside the fused all_reduce_mhc_norm path; > 8 rows: after it):
   the ("h", n) window takes the shared expert's FC1 remainder (what the post-attention AR window leaves), its scale,
   then FC2, cut at the budget; never a byte another window already holds; 16-byte pieces;
3. the fork: _before_hc_stats numbers the boundaries, forks the planned window on the second side stream, and is a
   no-op when SHPF is off or the forward is not bracketed (v2.7.3 merge: the learned hc count is the counts' sixth
   slot, [5], after v6's wo_a count at [4]);
4. _subtract on overlapping / unaligned ranges;
5. the v5 wrappers (DeepseekV2MoE._forward_shared_experts, DeepseekV4DecoderLayer._hc_mix_stats) unwrap to the
   engine's own functions, so the adapters that hash-pin those sources through inspect.unwrap (prefill_sp) still see
   the stock source (tests/test_adapter_engine_pins.py failed on the v2.7.2 image without it), and install once;
6. the window policy: DSV41_L2_PREFETCH_SHPF_EL unset / 1 = l2pf_el items (evict_last; each window demotes the
   previous SHPF window's lines, wrapping like MOE), 0 = plain items; the same segments either way.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=adapter python3 tests/test_l2_prefetch_shpf.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
for k in [k for k in os.environ if k.startswith(("DSV41_L2_PREFETCH", "DSV41_AB"))]:
    del os.environ[k]
os.environ["DSV41_L2_PREFETCH"] = "1"

import torch  # noqa: E402

import l2_prefetch as l2  # noqa: E402

_real_tensor = torch.tensor
l2.torch = types.SimpleNamespace(**{k: getattr(torch, k) for k in ("int64", "cuda")},
                                 tensor=lambda data, dtype=None, device=None: _real_tensor(data, dtype=dtype))
MB = 1 << 20


def layer(base, fused=True, idx0=1, n=1):
    """One decode layer's events from the post-attention side on: router 3.9 MB, then the shared expert
    (FC1 weight 5.9 MB + scale 0.18 MB, FC2 weight 3.3 MB + scale 0.1 MB), then the post-MoE AR."""
    router, fc1, fc1s, fc2, fc2s = base, base + 16 * MB, base + 32 * MB, base + 48 * MB, base + 64 * MB
    h = [("h", n)]
    ar = [("c", idx0)]
    body = [("r", router, int(3.9 * MB)), ("S",),
            ("r", fc1, int(5.9 * MB), 1152), ("r", fc1s, int(0.18 * MB), 1152),
            ("r", fc2, int(3.3 * MB), 5120), ("r", fc2s, int(0.1 * MB), 5120), ("s",),
            ("c", idx0 + 1)]
    ev = (h + ar if fused else ar + h) + body
    return ev, dict(router=router, fc1=fc1, fc1s=fc1s, fc2=fc2, fc2s=fc2s)


def segs(plan, key):
    return [tuple(v) for v in plan[key][0].view(-1, 2).tolist()]


def test_off_is_v4():
    os.environ.pop("DSV41_L2_PREFETCH_SHPF_MB", None)
    ev, _ = layer(1 << 32)
    plain = [e for e in ev if e[0] not in ("h", "S", "s")]
    a, b = l2._make_plan(ev), l2._make_plan(plain)
    assert a.keys() == b.keys() and all(torch.equal(a[k][0], b[k][0]) and a[k][1] == b[k][1] for k in a)
    assert not any(k[0] == "h" for k in a)
    # the markers are not even recorded while off
    l2._state.update(sig=("s",), learning={("s",): []})
    l2._shared_mark("S")
    l2._before_hc_stats()
    assert l2._state["learning"][("s",)] == [] and l2._state["hc"] == 0
    l2._state.update(sig=None, learning={})


def test_plan_both_layouts():
    os.environ["DSV41_L2_PREFETCH_SHPF_MB"] = "6"
    try:
        for fused in (True, False):
            ev, w = layer(1 << 32, fused=fused)
            plan = l2._make_plan(ev)
            ar = segs(plan, ("c", 1))
            # the post-attention AR window: the router, then the first 6 - 3.9 MB of FC1 (v4 rule, unchanged)
            assert ar[0] == (w["router"], int(3.9 * MB) & ~15) and ar[1][0] == w["fc1"]
            fc1_taken = ar[1][1]
            sh = segs(plan, ("h", 1))
            assert sh[0] == (w["fc1"] + fc1_taken, int(5.9 * MB) - fc1_taken - ((int(5.9 * MB) - fc1_taken) % 16)), sh
            assert sh[1] == (w["fc1s"], int(0.18 * MB) & ~15)
            assert sh[2][0] == w["fc2"]
            total = sum(n for _, n in sh)
            assert total <= 6 * MB and total > 6 * MB - 16, total
            assert all(a % 16 == 0 and n % 16 == 0 for a, n in sh)
            # never overlaps another window's bytes
            others = [s for k in plan if k[0] != "h" for s in segs(plan, k)]
            for a, n in sh:
                for b, m in others:
                    assert a + n <= b or b + m <= a
            print(f"layout {'fused c1' if fused else '> 8 rows'}: AR window {sum(n for _, n in ar) / MB:.2f} MB, "
                  f"shpf {total / MB:.2f} MB in {len(sh)} pieces")
        # two layers: one window per boundary, each on its own layer's shared expert
        ev1, w1 = layer(1 << 32, idx0=1, n=1)
        ev2, w2 = layer(1 << 33, idx0=3, n=2)
        plan = l2._make_plan(ev1 + ev2)
        assert segs(plan, ("h", 1))[0][0] >= w1["fc1"] and segs(plan, ("h", 2))[0][0] >= w2["fc1"]
        assert segs(plan, ("h", 2))[0][0] < w2["fc1s"]
        # a boundary with no shared-expert call after it plans nothing
        assert ("h", 1) not in l2._make_plan([("h", 1), ("c", 1), ("r", 1 << 32, 4 * MB)])
    finally:
        del os.environ["DSV41_L2_PREFETCH_SHPF_MB"]


def test_fork():
    os.environ["DSV41_L2_PREFETCH_SHPF_MB"] = "6"
    forks = []
    saved = l2._fork
    l2._fork = lambda item, stream=None, second=False: forks.append((item, second))
    try:
        sig = ("model", 6, "decode")
        item = ("segs", 3)
        l2._state.update(sig=sig, idx=0, hc=0, hc_forks=0, plans={sig: (2, {("h", 1): item}, None, (2, 0, 0, 0, 0, 1))},
                         learning={})
        l2._before_hc_stats()
        l2._before_hc_stats()        # a second boundary past the learned count: nothing
        assert forks == [(item, True)] and l2._state["hc"] == 2 and l2._state["hc_forks"] == 1
        l2._state["sig"] = None
        l2._before_hc_stats()
        assert len(forks) == 1
    finally:
        l2._fork = saved
        l2._state.update(sig=None, plans={}, hc=0)
        del os.environ["DSV41_L2_PREFETCH_SHPF_MB"]


def test_window_policy():
    os.environ["DSV41_L2_PREFETCH_SHPF_MB"] = "6"
    try:
        ev1, _ = layer(1 << 32, idx0=1, n=1)
        ev2, _ = layer(1 << 33, idx0=3, n=2)
        os.environ.pop("DSV41_L2_PREFETCH_SHPF_EL", None)
        el = l2._make_plan(ev1 + ev2)
        h1, h2 = el[("h", 1)], el[("h", 2)]
        assert h1[2] == "el" and h2[2] == "el"
        # window 1 demotes window 2's lines (the previous step's last layer, wrapping), window 2 demotes window 1's
        assert h1[3] is h2[0] and h1[4] == h2[1] and h2[3] is h1[0] and h2[4] == h1[1]
        # the other windows keep their own policy (the AR windows stay plain)
        assert all(len(el[k]) == 2 for k in el if k[0] == "c")
        one = l2._make_plan(ev1)[("h", 1)]
        assert one[2] == "el" and one[4] == 0          # a single window demotes nothing
        os.environ["DSV41_L2_PREFETCH_SHPF_EL"] = "0"
        plain = l2._make_plan(ev1 + ev2)
        assert len(plain[("h", 1)]) == 2 and len(plain[("h", 2)]) == 2
        assert el.keys() == plain.keys() and all(torch.equal(el[k][0], plain[k][0]) for k in el)
    finally:
        os.environ.pop("DSV41_L2_PREFETCH_SHPF_EL", None)
        del os.environ["DSV41_L2_PREFETCH_SHPF_MB"]


def test_subtract():
    assert l2._subtract([(0, 100)], [(16, 32)]) == [(0, 16), (48, 48)]
    assert l2._subtract([(0, 100)], [(0, 100)]) == []
    assert l2._subtract([(3, 100)], []) == [(16, 80)]                 # trimmed to 16-byte bounds
    assert l2._subtract([(0, 64), (128, 64)], [(32, 128)]) == [(0, 32), (160, 32)]


def test_wrappers_unwrap_to_engine():
    import inspect

    def stock_fse(self, hidden_states):
        return hidden_states

    def stock_stats(self, x, hc_fn, hc_scale, hc_base, stats_stream=None):
        return x

    v2 = types.SimpleNamespace(DeepseekV2MoE=type("DeepseekV2MoE", (), {"_forward_shared_experts": stock_fse}))
    v4 = types.SimpleNamespace(DeepseekV4DecoderLayer=type("DeepseekV4DecoderLayer", (), {"_hc_mix_stats": stock_stats}))
    l2._install_shared_marks(v2)
    assert l2._install_hc_entry(v4)
    fse, stats = v2.DeepseekV2MoE._forward_shared_experts, v4.DeepseekV4DecoderLayer._hc_mix_stats
    assert fse is not stock_fse and stats is not stock_stats
    assert inspect.unwrap(fse) is stock_fse and inspect.unwrap(stats) is stock_stats
    assert inspect.getsource(inspect.unwrap(fse)) == inspect.getsource(stock_fse)
    # idempotent: a second install keeps the one wrapper
    l2._install_shared_marks(v2)
    assert not l2._install_hc_entry(v4)
    assert v2.DeepseekV2MoE._forward_shared_experts is fse and v4.DeepseekV4DecoderLayer._hc_mix_stats is stats
    # pass-through with SHPF off (no bracketed forward): same results
    assert fse(None, 7) == 7 and stats(None, 5, object(), 1, 2) == 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
