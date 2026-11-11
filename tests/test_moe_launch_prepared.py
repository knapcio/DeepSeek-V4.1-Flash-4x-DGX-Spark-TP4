"""CPU checks of adapter/moe_launch_prepared.py against the REAL b12x_next stock code (no GPU).

Needs torch, triton and nvidia-cutlass-dsl importable (the serving image, or the colima VM image
ds-b3-cutecheck:1); only the device queries and the side effects are stubbed:
  torch.cuda.current_device / _impl.get_num_sm / _current_compute_capability / current_cuda_stream (a new object per
  call, so a stale stream would show), _impl.b12x_compile (a recording "compiled" kernel per cache key),
  _impl._zero_barrier_state and _impl.launch_triton (recorders).

- pins: the stock sources hash to PINNED_SOURCES; a drifted pin refuses to arm.
- equivalence: over production-like contracts (E=384 k=5120 n=576 top-6 W4A8-MX N64 / E=128 top-3 draft geometry,
  rows 1..4096, deterministic on/off, external route plan requested on/off, volatile state on/off, planned tile
  16/32/64/128, a too-small materialized scratch), the stock body and the prepared body (first call: plan build,
  later calls: plan hit, each with freshly allocated per-call tensors) issue the same side effects in the same order
  with the same arguments: pointer wrappers compared by (address, dtype, alignment, space), tensors by identity,
  scalars by value, the stream by identity with the one current at that call. Errors match too.
- invalidation: _DYNAMIC_KERNEL_CACHE.clear(), a B12X_NEXT_* env change, a programmatic tile override, the
  pre-quantized-input context, each produce a new plan equal to stock.
- fallbacks to the stock body: mode off, planning(), CUDA-graph capture, trellis weights.
- plan,direct: eager calls reach the op body without the dispatcher; capture and .default go to the packet.
- timing: stages recorded per (switch seq, variant, row bucket), only the launch body's kernel lookup is proxied,
  one JSON line per seq at the next seq's first record.

usage: PYTHONPATH=adapter:runtime/b12x_next python3 tests/test_moe_launch_prepared.py
"""
import io
import itertools
import json
import os
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "adapter"), os.path.join(ROOT, "runtime", "b12x_next")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    from b12x_next.moe.fused_moe import _impl as I
except Exception as exc:  # noqa: BLE001
    print(f"SKIP test_moe_launch_prepared: needs torch + b12x_next importable ({exc!r})")
    sys.exit(0)

import moe_launch_prepared as P  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("[OK]   " if cond else "[FAIL] ") + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------------------- stubs
EVENTS = []
STREAMS = itertools.count(1000)


class Stream:
    def __init__(self):
        self.id = next(STREAMS)

    def __repr__(self):
        return f"Stream({self.id})"


class FakeCompiled:
    def __init__(self, key):
        self.key = key

    def __call__(self, *args):
        EVENTS.append(("compiled", self.key, args))


def fake_compile(launch, *args, compile_spec=None, dsl_compile_options=None, **kw):
    return FakeCompiled(repr(compile_spec)[-60:] + f"#{len(I._DYNAMIC_KERNEL_CACHE)}")


def fake_zero(bc, be):
    EVENTS.append(("zero", bc, be))


def fake_launch_triton(kernel, grid, *args, **kw):
    EVENTS.append(("triton", kernel, grid, args, tuple(sorted(kw.items()))))


CURRENT = {"stream": None}


def fake_stream():
    CURRENT["stream"] = Stream()
    return CURRENT["stream"]


torch.cuda.current_device = lambda: 0
torch.cuda.is_current_stream_capturing = lambda: CAPTURING[0]
CAPTURING = [False]
I.get_num_sm = lambda device=None: 48
import b12x_next._lib.utils as U  # noqa: E402
U._num_sm_cache[0] = 48                  # get_max_active_clusters(1) -> utils.get_num_sm -> this cache
I._current_compute_capability = lambda: (12, 1)
I.current_cuda_stream = fake_stream
I.b12x_compile = fake_compile
I._zero_barrier_state = fake_zero
I.launch_triton = fake_launch_triton
STOCK_BODY = I._launch_dynamic_flat


def norm(ev):
    """Comparable form of one recorded event."""
    def one(x):
        if isinstance(x, torch.Tensor):
            return ("T", id(x))
        if isinstance(x, Stream):         # the stream current at the call (each call resolves a new one)
            return ("S", x is CURRENT["stream"])
        if type(x).__name__ == "_Pointer":
            return ("P", int(x._pointer), x._dtype, x._assumed_align, str(x._addr_space))
        if isinstance(x, (tuple, list)):
            return tuple(one(y) for y in x)
        return x
    return one(ev)


# ---------------------------------------------------------------------------------------------------------- contracts
U8, I32, F32, BF16 = torch.uint8, torch.int32, torch.float32, torch.bfloat16
STATIC = {}


def static_tensors(E, k, n):
    """Weights / per-geometry tensors: allocated once per geometry (the layer's prepared experts)."""
    key = (E, k, n)
    if key not in STATIC:
        STATIC[key] = dict(
            w13_fp4=torch.empty(64, dtype=U8), w13_sf=torch.empty(64, dtype=U8), down_fp4=torch.empty(64, dtype=U8),
            down_sf=torch.empty(64, dtype=U8), sfb_w13_mx=torch.empty(64, dtype=U8),
            sfb_down_mx=torch.empty(64, dtype=U8), w13_residual=torch.empty(64, dtype=U8),
            down_residual=torch.empty(64, dtype=U8), w13_rp=torch.empty(64, dtype=I32),
            w13_sfb_rp=torch.empty(64, dtype=I32), down_rp=torch.empty(64, dtype=I32),
            down_sfb_rp=torch.empty(64, dtype=I32), w1_alpha=torch.empty(E, dtype=F32),
            w2_alpha=torch.empty(E, dtype=F32))
    return STATIC[key]


def scratch(materialized_bytes):
    """Workspace arena views: shared across calls of one capacity (stable addresses)."""
    key = ("scratch", materialized_bytes)
    if key not in STATIC:
        names_i32 = ("barrier_count", "barrier_epoch", "pair_head", "producers_done_count", "all_work_published",
                     "task_head", "task_tail", "task_ready", "task_expert", "task_m_tile", "task_slice_begin",
                     "task_slice_count", "task_valid_rows", "tile_write_count", "row_counts", "expert_write_rows",
                     "expert_tile_base", "token_map")
        d = {nm: torch.zeros(64, dtype=I32) for nm in names_i32}
        d.update(packed_a_view=torch.empty(256, dtype=U8), packed_a_flat=torch.empty(256, dtype=U8),
                 scale_flat=torch.empty(256, dtype=U8),
                 materialized_intermediate=torch.empty(max(16, materialized_bytes), dtype=U8),
                 input_gs=torch.empty(8, dtype=F32), down_input_scale=torch.empty(8, dtype=F32),
                 token_weights=torch.empty(64, dtype=F32))
        STATIC[key] = d
    return STATIC[key]


def contract(E=384, k=5120, n=576, topk=6, m=34, cap=None, det=True, ext=False, vol=True, tile=16, direct=False,
             n64=True, i32=True, mac=-1, enough=True):
    cap = cap or max(m, 1)
    routed = m * topk
    ptc = max(1, (cap * topk + tile - 1) // tile + E)
    need = I._dynamic_materialized_intermediate_bytes(quant_mode="w4a8_mx", rows_padded=ptc * tile, n=n)
    kw = dict(E=E, m=m, k=k, n=n, num_topk=topk, routed_rows=routed, max_rows=cap, physical_tiles_capacity=ptc,
              task_capacity=ptc * 8, topk_ids_are_i32=i32, fast_math=True, activation="silu", quant_mode="w4a8_mx",
              w4a8_repacked=True, w4a8_n64_repacked=n64, nvfp4_materialize_intermediate=False,
              share_input_across_experts=True, deterministic_output=det, swiglu_limit=10.0, swiglu_alpha=1.0,
              swiglu_beta=0.0, external_route_plan_requested=ext, policy_max_active_clusters=mac,
              volatile_launch_state=vol, planned_tile_m=tile, planned_direct_routing=direct, planned_num_tokens=cap)
    kw.update(static_tensors(E, k, n))
    kw.update(scratch(need if enough else need // 2))
    return kw


def per_call(kw):
    """Fresh activation / routing / output tensors, as the caching allocator would hand out."""
    m, topk, k = kw["m"], kw["num_topk"], kw["k"]
    d = dict(kw)
    d.update(a=torch.empty(m, 8, dtype=BF16), flat_ids=torch.empty(m * topk, dtype=I32 if kw["topk_ids_are_i32"] else torch.int64),
             flat_weights=torch.empty(m * topk, dtype=F32), scatter_output=torch.empty(m, 8, dtype=BF16))
    return d


def run(body, kw):
    EVENTS.clear()
    try:
        body(**kw)
        err = None
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    return [norm(e) for e in EVENTS], err


def same_as_stock(name, kw, calls=3, expect_error=False):
    """stock once, then prepared `calls` times (each with new per-call tensors, and a stock call on the SAME
    tensors right before it). The stock body must succeed unless expect_error (then both must raise the same)."""
    ok = True
    for i in range(calls):
        d = per_call(kw)
        ref, ref_err = run(STOCK_BODY, d)
        got, got_err = run(P.prepared_launch_dynamic_flat, d)
        if (ref_err is not None) != expect_error:
            check(name, False, f"call {i}: stock error {ref_err!r}, expected error={expect_error}")
            return False
        if ref != got or ref_err != got_err:
            ok = False
            detail = f"call {i}: stock err={ref_err} prepared err={got_err}"
            if ref != got:
                for j, (a, b) in enumerate(zip(ref, got)):
                    if a != b:
                        if a[0] == b[0] == "compiled" and len(a[2]) == len(b[2]):
                            pos = [x for x in range(len(a[2])) if a[2][x] != b[2][x]]
                            detail += f"; compiled args differ at {pos}: {a[2][pos[0]]} vs {b[2][pos[0]]}"
                        else:
                            detail += f"; first event diff #{j}: {str(a)[:200]} vs {str(b)[:200]}"
                        break
                else:
                    detail += f"; {len(ref)} vs {len(got)} events"
            check(name, False, detail)
            return False
        if ref_err is None and not any(e[0] == "compiled" for e in got):
            check(name, False, "no compiled launch recorded")
            return False
    return ok


# ---------------------------------------------------------------------------------------------------------- tests
check("pins: stock sources match PINNED_SOURCES", P.check_sources(I) == {}, str(P.check_sources(I)))

saved = dict(P.PINNED_SOURCES)
P.PINNED_SOURCES["_launch_dynamic_flat"] = "0" * 64
try:
    P.install(I, None, environ={"DSV41_MOE_LAUNCH_PREPARED": "plan"})
    check("pins: a drifted pin refuses to arm", False, "installed")
except RuntimeError as exc:
    check("pins: a drifted pin refuses to arm", "drifted" in str(exc), str(exc))
P.PINNED_SOURCES.update(saved)
check("install: both gates off -> nothing", P.install(I, None, environ={}) is False and I._launch_dynamic_flat is STOCK_BODY)

# install in plan mode (no direct stand-in), no timing
P.install(I, None, environ={"DSV41_MOE_LAUNCH_PREPARED": "plan"})
check("install: plan mode replaces the op body", I._launch_dynamic_flat is P.prepared_launch_dynamic_flat
      and not isinstance(getattr(torch.ops.b12x_next, "tp_moe_dynamic_launch"), P.DirectOp))

grid = []
for (E, topk), m, det, ext, vol, tile in itertools.product(((384, 6), (128, 3)), (1, 6, 8, 34, 96, 136, 469, 4096),
                                                          (True, False), (False, True), (True, False), (16, 64)):
    grid.append(dict(E=E, topk=topk, m=m, det=det, ext=ext, vol=vol, tile=tile))
grid += [dict(m=34, tile=32), dict(m=2048, tile=128, n64=False, n=1152), dict(m=34, i32=False),
         dict(m=8, tile=16, mac=48, ext=True), dict(m=6, mac=24), dict(m=6, E=256, k=6144, n=1024, n64=False, tile=32)]
n_ok = sum(same_as_stock(f"equivalence {g}", contract(**g)) for g in grid)
check(f"equivalence: stock == prepared (build + 2 hits) on {len(grid)} contracts", n_ok == len(grid),
      f"{len(grid) - n_ok} failed")
check("equivalence: plans were built and hit", P._STATE["builds"] > 0 and P._STATE["hits"] > P._STATE["builds"],
      str(P._STATE))
n_err = sum(same_as_stock(f"too-small scratch {g}", contract(m=g, enough=False), calls=2, expect_error=True)
            for g in (34, 469))
check("equivalence: a too-small materialized scratch raises the same error", n_err == 2)
check("equivalence: an unsupported planned direct routing raises the same error (production plans are grouped)",
      same_as_stock("direct m=96", contract(m=96, direct=True, tile=16), calls=2, expect_error=True))

# ds41 scale codes: a coded binding (uint8 code buffers in both sfb slots) plans the coded kernel, never reuses a
# native plan of the same scalars, and a mixed or non-N64 coded binding raises the stock error
CODED = {}


def coded(kw, w13=True, down=True):
    d = dict(kw)
    key = (kw["E"], kw["k"], kw["n"])
    if key not in CODED:
        CODED[key] = (torch.empty(64, dtype=U8), torch.empty(64, dtype=U8))
    if w13:
        d["w13_sfb_rp"] = CODED[key][0]
    if down:
        d["down_sfb_rp"] = CODED[key][1]
    return d


cgrid = [dict(m=1), dict(m=6), dict(m=34), dict(m=136, tile=64), dict(m=8, det=False, ext=True)]
n_ok = sum(same_as_stock(f"coded {g}", coded(contract(**g))) for g in cgrid)
check(f"scale codes: stock == prepared on {len(cgrid)} coded contracts", n_ok == len(cgrid),
      f"{len(cgrid) - n_ok} failed")
kc, kn = P._key(coded(contract(m=34))), P._key(contract(m=34))
check("scale codes: coded and native bindings of the same scalars key different plans", kc != kn)
pc, pn = P._PLANS.get(kc), P._PLANS.get(kn)
check("scale codes: the coded plan holds the coded kernel, the native plan the native one",
      pc is not None and pn is not None and pc.kernel_key is not None and pn.kernel_key is not None
      and "w4a8_coded_scales" in pc.kernel_key and "w4a8_coded_scales" not in pn.kernel_key,
      f"{getattr(pc, 'kernel_key', None)} / {getattr(pn, 'kernel_key', None)}")
check("scale codes: w13 coded, down native raises the stock error",
      same_as_stock("coded w13 only", coded(contract(m=34), down=False), calls=2, expect_error=True))
check("scale codes: down coded, w13 native raises the stock error",
      same_as_stock("coded down only", coded(contract(m=34), w13=False), calls=2, expect_error=True))
check("scale codes: a coded non-N64 binding raises the stock error",
      same_as_stock("coded non-N64", coded(contract(m=2048, tile=128, n64=False, n=1152)), calls=2,
                    expect_error=True))

# negative controls: the comparison catches a stale per-call pointer and a skipped barrier reset
real_ptr, stale = P._ptr, {}


def stale_ptr(dtype, t, align=16):
    if dtype is I.cutlass.BFloat16:          # a = this call's activations: keep the first one ever seen
        return stale.setdefault((dtype, align), real_ptr(dtype, t, align))
    return real_ptr(dtype, t, align)


P._ptr = stale_ptr
quiet = io.StringIO()
with contextlib.redirect_stdout(quiet):
    caught_ptr = not same_as_stock("neg: stale pointer", contract(m=34, tile=32), calls=3)
P._ptr = real_ptr
real_zero = I._zero_barrier_state
P._PLANS.clear()
with contextlib.redirect_stdout(quiet):
    same_as_stock("neg warm", contract(m=40), calls=1)
    for p_ in P._PLANS.values():
        p_.zero_barrier = False              # a plan that forgot the per-call reset
    caught_zero = not same_as_stock("neg: no reset", contract(m=40), calls=1)
P._PLANS.clear()
FAILS[:] = [f for f in FAILS if not f.startswith("neg")]
check("negative controls: a stale activation pointer and a skipped barrier reset are both caught",
      caught_ptr and caught_zero, f"ptr={caught_ptr} zero={caught_zero}")

# invalidation
kw = contract(m=34)
same_as_stock("warm", kw, calls=1)
b0 = P._STATE["builds"]
I._DYNAMIC_KERNEL_CACHE.clear()
check("invalidation: kernel cache cleared -> rebuilt, equal to stock",
      same_as_stock("after clear", kw, calls=2) and P._STATE["builds"] > b0)
# the default implementation, consecutive calls (no recheck interval exists any more)
check("env tracking: os.environ is the tracked subclass after install", getattr(type(os.environ), "_dsv41_tracked", False))
kw8 = contract(m=8, tile=16, ext=True)
os.environ["B12X_NEXT_DYNAMIC_EXTERNAL_ROUTE_PLAN"] = "1"
ok_route = same_as_stock("route env 1 (warm)", kw8, calls=2)
os.environ["B12X_NEXT_DYNAMIC_EXTERNAL_ROUTE_PLAN"] = "0"      # the very next call must see the change
EVENTS.clear()
d8 = per_call(kw8)
ref8, _ = run(STOCK_BODY, d8)
got8, _ = run(P.prepared_launch_dynamic_flat, d8)
check("env tracking: route plan 1 -> 0 followed on the next call (stock [zero, compiled], not [triton, compiled])",
      ok_route and ref8 == got8 and [e[0] for e in got8] == ["zero", "compiled"], f"{[e[0] for e in got8]}")
os.environ["B12X_NEXT_DYNAMIC_EXTERNAL_ROUTE_PLAN"] = "1"
d8 = per_call(kw8)
ref8, _ = run(STOCK_BODY, d8)
got8, _ = run(P.prepared_launch_dynamic_flat, d8)
check("env tracking: 0 -> 1 followed on the next call", ref8 == got8 and [e[0] for e in got8] == ["triton", "compiled"],
      f"{[e[0] for e in got8]}")
os.environ.pop("B12X_NEXT_DYNAMIC_EXTERNAL_ROUTE_PLAN")
d8 = per_call(kw8)
ref8, _ = run(STOCK_BODY, d8)
got8, _ = run(P.prepared_launch_dynamic_flat, d8)
check("env tracking: pop() followed on the next call", ref8 == got8)
os.environ.update({"B12X_NEXT_DYNAMIC_MAX_ACTIVE_CLUSTERS": "20"})
check("env tracking: update() of the MAC override followed", same_as_stock("mac env", contract(m=34, mac=48), calls=2))
del os.environ["B12X_NEXT_DYNAMIC_MAX_ACTIVE_CLUSTERS"]
check("env tracking: delete followed", same_as_stock("mac env del", contract(m=34, mac=48), calls=2))
os.environ["B12X_NEXT_DYNAMIC_REUSE_COMPILED"] = "0"            # kernel cache bypassed: no plan is kept
b0, n0 = P._STATE["builds"], len(P._PLANS)
check("env tracking: kernel cache disabled -> decided per call, equal to stock, nothing cached",
      same_as_stock("reuse off", contract(m=34), calls=3) and P._STATE["builds"] >= b0 + 3)
del os.environ["B12X_NEXT_DYNAMIC_REUSE_COMPILED"]
# negative control: with the generation detached (the old periodic-recheck behaviour) the stale plan is caught
os.environ["B12X_NEXT_DYNAMIC_EXTERNAL_ROUTE_PLAN"] = "1"
same_as_stock("neg warm", kw8, calls=1)
os.environ["B12X_NEXT_DYNAMIC_EXTERNAL_ROUTE_PLAN"] = "0"
P._STATE["env_gen"] = P._ENV_GEN[0]          # an invalidation that did not happen (the old 1-in-256 recheck)
d8 = per_call(kw8)
ref8, _ = run(STOCK_BODY, d8)
got8, _ = run(P.prepared_launch_dynamic_flat, d8)
check("env tracking negative control: a missed invalidation reuses the stale plan and the test sees it", ref8 != got8)
del os.environ["B12X_NEXT_DYNAMIC_EXTERNAL_ROUTE_PLAN"]
P._PLANS.clear()
gen = P._ENV_GEN[0]
os.environ["UNRELATED_X"] = "1"
del os.environ["UNRELATED_X"]
check("env tracking: unrelated keys do not drop plans", P._ENV_GEN[0] == gen)
I._DYNAMIC_TILE_MN_OVERRIDE = (64, 128)
check("invalidation: programmatic tile override is part of the key",
      same_as_stock("tile override", contract(m=34, tile=64), calls=2))
I._DYNAMIC_TILE_MN_OVERRIDE = None
if callable(getattr(I, "prequantized_input", None)):
    with I.prequantized_input():
        check("invalidation: pre-quantized input context -> its own plan, equal to stock",
              same_as_stock("prequant", contract(m=4096, tile=64), calls=2))

# fallbacks
before = P._STATE["stock"]
P._STATE["mode"] = "off"
same_as_stock("mode off", contract(m=34), calls=1)
P._STATE["mode"] = "plan"
CAPTURING[0] = True
same_as_stock("capture", contract(m=34), calls=1)
CAPTURING[0] = False
kw = contract(m=34)
kw = dict(kw, w13_sfb_rp=torch.empty(64, dtype=torch.float16))
d = per_call(kw)
EVENTS.clear()
try:
    P.prepared_launch_dynamic_flat(**d)
except Exception:  # noqa: BLE001  (trellis extents do not match a fake: stock raises, which is the point)
    pass
check("fallbacks: mode off, capture and trellis weights run the stock body", P._STATE["stock"] >= before + 3,
      f"{P._STATE['stock'] - before}")
import b12x_next._lib.compile_plan as CP  # noqa: E402
tok = CP._COMPILE_ONLY_LAUNCHES.set(True)
before = P._STATE["stock"]
same_as_stock("compile-only", contract(m=34), calls=1)
CP._COMPILE_ONLY_LAUNCHES.reset(tok)
check("fallbacks: compile-only launches run the stock body", P._STATE["stock"] > before)

# plan,direct stand-in
packet_calls = []


class FakePacket:
    default = "the-default-overload"

    def __call__(self, *a, **k):
        packet_calls.append(a)


body_calls = []
op = P.DirectOp(FakePacket(), lambda *a, **k: body_calls.append(a))
P._STATE["mode"] = "plan,direct"
op(1, 2)
CAPTURING[0] = True
op(3)
CAPTURING[0] = False
P._STATE["mode"] = "plan"
op(4)
check("plan,direct: eager -> body; capture / plan mode -> packet; attributes forwarded",
      body_calls == [(1, 2)] and packet_calls == [(3,), (4,)] and op.default == "the-default-overload")

# timing
P._STATE["mode"] = "plan"
P._ab = type("AB", (), {"ACTIVE": True, "KNOWN": ("DSV41_MOE_LAUNCH_PREPARED",), "_state": {"seq": 0},
                       "current": staticmethod(lambda: AB_V[0]),
                       "env": staticmethod(lambda name, default=None: "plan" if AB_V[0] == 2 else "0")})
AB_V = [0]


class Geo:
    def _run(self, cap, x, ids, w, out, impl):
        I._launch_dynamic(m=x.shape[0])


def fake_launch_dynamic(**kw):          # stands in for the op: argument assembly + dispatch + body
    I._launch_dynamic_flat(**per_call(contract(m=kw["m"])))


I._launch_dynamic = fake_launch_dynamic
P._ORIG["_launch_dynamic"] = fake_launch_dynamic
adapter_mod = type("M", (), {"_Geometry": Geo})
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    P._install_timing(I, adapter_mod)
    g = Geo()
    for seq, v in ((1, 0), (2, 2), (3, 0)):
        P._ab._state["seq"], AB_V[0] = seq, v
        for _ in range(3):
            g._run(64, torch.empty(34, 8), None, None, None, None)
    P._ab._state["seq"] = 4
    g._run(64, torch.empty(6, 8), None, None, None, None)
lines = [json.loads(ln.split(" ", 1)[1]) for ln in buf.getvalue().splitlines() if ln.startswith("DSV41_MOE_LAUNCH_TIMING ")]
by_seq = {d["seq"]: d for d in lines}
s1, s2 = by_seq.get(1, {}).get("stages", {}), by_seq.get(2, {}).get("stages", {})
check("timing: one line per finished seq, variant-tagged", sorted(by_seq) == [1, 2, 3] and "v0" in s1 and "v2" in s2,
      str(sorted(by_seq)))
st1, st2 = s1.get("v0", {}).get("34", {}), s2.get("v2", {}).get("34", {})
check("timing: stock seq has adapter/op/body/getk/compiled stages, 3 calls each",
      all(st1.get(x, {}).get("n") == 3 for x in ("adapter", "op", "body", "getk", "compiled")), json.dumps(st1)[:400])
check("timing: prepared seq has adapter/op/body/compiled and no kernel lookup on hits",
      all(st2.get(x, {}).get("n") == 3 for x in ("adapter", "op", "body", "compiled"))
      and st2.get("getk", {}).get("n", 0) <= 1, json.dumps(st2)[:400])
check("timing: self cost measured", P._TIMING["self_ns"] > 0)
EVENTS.clear()
compiled, _mac = I._get_dynamic_kernel(384, 34, 5120, 576, 6, 34, topk_ids_dtype=torch.int32, fast_math=True,
                                      activation="silu", quant_mode="w4a8_mx", w4a8_repacked=True,
                                      w4a8_n64_repacked=True, deterministic_output=True, swiglu_limit=10.0,
                                      planned_tile_m=16)
check("timing: a kernel lookup outside the launch body (preparation) is not proxied",
      not isinstance(compiled, P.TimedCompiled))

# eager forward timing hook
class Mode:
    def __init__(self, name, graph):
        self.name, self._graph = name, graph

    def is_cuda_graph(self):
        return self._graph


class Runner:
    is_draft_worker = False

    class decode_cuda_graph_runner:
        @staticmethod
        def can_run_graph(fb):
            return True

    def _forward_raw(self, forward_batch, *a, **k):
        return "out"


runner_mod = type("RM", (), {"ModelRunner": Runner})
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    armed = P.install_forward_timing(runner_mod)
    again = P.install_forward_timing(runner_mod)
    P._ab._state["seq"], AB_V[0] = 7, 1
    fb_ext = type("FB", (), {"forward_mode": Mode("EXTEND", False), "input_ids": torch.zeros(34)})()
    fb_dec = type("FB", (), {"forward_mode": Mode("DECODE", True), "input_ids": torch.zeros(6)})()
    r = Runner()
    outs = [r._forward_raw(fb_ext), r._forward_raw(fb_ext), r._forward_raw(fb_dec)]
    P._ab._state["seq"] = 8
    r._forward_raw(fb_ext)
lines = [json.loads(ln.split(" ", 1)[1]) for ln in buf.getvalue().splitlines() if ln.startswith("DSV41_MOE_LAUNCH_TIMING ")]
seq7 = [d for d in lines if d["seq"] == 7]
st = seq7[-1]["stages"].get("v1", {}).get("34", {}) if seq7 else {}
check("forward timing: eager extend forwards recorded (host), graph replays not, results passed through",
      armed and not again and outs == ["out"] * 3 and st.get("fwd_host_t_extend", {}).get("n") == 2
      and not any("decode" in k for b in seq7[-1]["stages"].get("v1", {}).values() for k in b), json.dumps(seq7)[:300])
try:
    P.install_forward_timing(type("RM", (), {}))
    check("forward timing: a drifted ModelRunner refuses", False)
except RuntimeError:
    check("forward timing: a drifted ModelRunner refuses", True)

print("ALL PASS" if not FAILS else f"FAILED: {len(FAILS)}: {FAILS}")
sys.exit(1 if FAILS else 0)
