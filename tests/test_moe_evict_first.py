"""CPU checks of DSV41_MOE_EVICT_FIRST (b12x_next evict-first patch + adapter/moe_b12x_next.py gate). No GPU.

Sources (plain file reads, always):
- the new intrinsic's asm string is the form the tier-10 efb bench validated on GB10 (variant cpa_hint:
  ``cp.async.cg.shared.global.L2::cache_hint [smem], [gmem], 16, policy`` with the policy from
  ``createpolicy.fractional.L2::evict_first.b64 policy, 1.0``), operands renumbered for LLVM inline asm, with the
  shared destination passed as a 64-bit shared-window address (see CPA_HINT_U64);
- phase 1 / phase 2: the hinted copies are exactly the packed expert-weight (B) copies (2 sites each, every one
  behind ``const_expr(self.evict_first_weights)`` with the stock copy in the else branch), the policy is created
  once per thread at kernel entry, and the scale (SFB), activation (A / SFA) copies keep the stock intrinsic;
- scripts/b12x_next-evict-first.patch round-trips against runtime/b12x_next (reverse then forward = the tree) and
  scripts/build_b12x_next.sh applies and checks it (the SOURCE_PATCH pin is tests/test_b12x_next_claim_barrier.py).
Registry: both keys are ab_variant RUNTIME keys (never unioned, not graph-neutral), the row cap is moot while the
gate is off, gate / cap differences are real A/B differences, a variant cannot share another's graph set across
them.
Adapter (torch, no CUDA; fake b12x_next objects): row-cap parsing; per-variant settings; arming prints the boot
receipt, a runtime without the patch refuses to boot; per capture: on only inside capture, gate on, M <= cap,
not refused; one receipt per (variant, M); _Geometry._run launches inside evict_first_weights() exactly then (and
for the load-time warm, ef=True); dropped flag-on kernels are restored into b12x's memo before a capture launch.
b12x_next _impl (needs cutlass + triton importable: the serving / dry image, skipped otherwise): eligibility
truth table; _get_dynamic_kernel with the compile wrapper stubbed: off = stock key (no suffix), on = the same key
+ ("evict_first_weights",) and both phase kernels flagged, an ineligible plan ignores the request (stock key,
counted); the prepared-launch key carries the state (moe_launch_prepared._evict_first).

usage: python3 tests/test_moe_evict_first.py      (torch needed for the adapter part; b12x part needs cutlass)
"""
import contextlib
import dataclasses
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ADAPTER = os.path.join(ROOT, "adapter")
# the repo tree (runtime/b12x_next), else the image's copy on PYTHONPATH (/opt/b12x_next), located without importing it
_REPO_PKG = os.path.join(ROOT, "runtime", "b12x_next", "b12x_next")
if os.path.isdir(_REPO_PKG):
    PKG = _REPO_PKG
    sys.path.insert(0, os.path.dirname(_REPO_PKG))
else:
    import importlib.util
    _spec = importlib.util.find_spec("b12x_next")
    if _spec is None or not _spec.submodule_search_locations:
        sys.exit("test_moe_evict_first: b12x_next not found (runtime/b12x_next or PYTHONPATH)")
    PKG = list(_spec.submodule_search_locations)[0]
PKG_PARENT = os.path.dirname(PKG)
if ADAPTER not in sys.path:
    sys.path.insert(0, ADAPTER)
FAILS = []
SKIPS = []

# diagnostics/fable-ideas-20261109/tier10-kit/kit/bench/efb/evict_first_bench.py (validated on GB10 2026-11-10,
# t10b-run-181420: a prefetched consumer stays 1.004x warm under a 64 MB cpa_hint stream vs 1.31x default)
EFB_CPA_HINT = "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
EFB_POLICY = "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
# the shipped form: the destination as a 64-bit shared-window address (zero-extend, generic round trip = the same
# address), so ptxas cannot fold a uniform shared offset into the LDGSTS over the cache-policy descriptor pair
# (the plain 32-bit operand compiled to a corrupt descriptor on sm_121a: illegal instruction on GB10)
CPA_HINT_U64 = ("{ .reg .u64 ef_a, ef_g; cvt.u64.u32 ef_a, %0; cvta.shared.u64 ef_g, ef_a; cvta.to.shared.u64 ef_a, ef_g; "
                "cp.async.cg.shared.global.L2::cache_hint [ef_a], [%1], 16, %2; }")


def check(name, cond, detail=""):
    print(("[OK]   " if cond else "[FAIL] ") + name + (f" -- {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def read(*parts):
    with open(os.path.join(*parts)) as f:
        return f.read()


def asm_of(src, fn):
    """The inline-asm string of a @dsl_user_op in intrinsics.py (LLVM operand syntax $n -> PTX-style %n)."""
    m = re.search(rf"def {fn}\(.*?\n(?:.*\n)*?\s+\"([^\"]+)\",\n", src)
    return m.group(1).replace("$", "%") if m else None


# ------------------------------------------------------------------------------------------------- sources
intr = read(PKG, "_lib", "intrinsics.py")
check("intrinsic: cp_async4_shared_global_l2hint is the efb-validated cpa_hint form on a 64-bit shared address",
      asm_of(intr, "cp_async4_shared_global_l2hint") == CPA_HINT_U64
      and EFB_CPA_HINT.replace("[%0]", "[ef_a]") in CPA_HINT_U64, asm_of(intr, "cp_async4_shared_global_l2hint"))
check("intrinsic: create_l2_evict_first_policy is the efb-validated policy",
      asm_of(intr, "create_l2_evict_first_policy") == EFB_POLICY, asm_of(intr, "create_l2_evict_first_policy"))
check("intrinsic: the stock 16-byte copy is unchanged",
      asm_of(intr, "cp_async4_shared_global") == "cp.async.cg.shared.global [%0], [%1], 16;")
efb = os.environ.get("EFB_BENCH", os.path.join(ROOT, "..", "..", "..", "diagnostics", "fable-ideas-20261109",
                                                "tier10-kit", "kit", "bench", "efb", "evict_first_bench.py"))
if os.path.exists(efb):
    bench = read(efb)
    check("intrinsic: the efb bench source still carries both forms", EFB_CPA_HINT in bench and EFB_POLICY in bench)
else:
    SKIPS.append("efb bench source not reachable (EFB_BENCH)")

for name, weights, n_sites in (("w4a8_phase1.py", ("weights",), 2), ("w4a8_phase2.py", ("weights", "down_rp"), 2)):
    src = read(PKG, "moe", "_shared", "kernels", name)
    hinted = re.findall(r"if cutlass\.const_expr\(self\.evict_first_weights\):\n(\s+)cp_async4_shared_global_l2hint\(\n"
                        r"\s+([^\n]+),\n\s+get_ptr_as_int64\((\w+), \w+\),\n\s+ef_policy,\n\s+\)\n\s+else:\n"
                        r"\s+cp_async4_shared_global\(\n\s+\2,\n\s+get_ptr_as_int64\(\3, \w+\),\n", src)
    check(f"{name}: {n_sites} hinted sites, each the B copy with the stock copy as its else branch",
          len(hinted) == n_sites and all(h[2] in weights for h in hinted)
          and src.count("cp_async4_shared_global_l2hint(") == n_sites, str(hinted))
    check(f"{name}: one policy per thread at kernel entry",
          src.count("create_l2_evict_first_policy()") == 1
          and re.search(r"def kernel\(.*?ef_policy = None\n\s+if cutlass\.const_expr\(self\.evict_first_weights\):\n"
                        r"\s+ef_policy = create_l2_evict_first_policy\(\)", src, re.S) is not None)
    check(f"{name}: off by default (constructor flag False)", "evict_first_weights: bool = False," in src)
    stock = re.findall(r"cp_async4_shared_global\(\n\s+[^\n]+,\n\s+get_ptr_as_int64\((\w+),", src)
    check(f"{name}: scale / activation copies keep the stock intrinsic",
          {"scales", "packed_a_u32", "intermediate_u32", "down_sfb_rp"} & set(stock)
          and not ({"scales", "packed_a_u32", "intermediate_u32", "down_sfb_rp"} & {h[2] for h in hinted}), str(stock))

impl_src = read(PKG, "moe", "fused_moe", "_impl.py")
check("_impl: the key suffix is appended only when on",
      'if evict_first:\n        # ds41 patch: appended only when on' in impl_src
      and "cache_key = cache_key + (_EVICT_FIRST_KEY,)" in impl_src and '_EVICT_FIRST_KEY = "evict_first_weights"' in impl_src)

patch = read(ROOT, "scripts", "b12x_next-evict-first.patch")
build = read(ROOT, "scripts", "build_b12x_next.sh")
chain = re.findall(r'"\$HERE/b12x_next-([a-z0-9-]+)\.patch"', build)
check("build script applies the patch last (after scale-codes when present) and checks it",
      chain and chain[-1] == "evict-first" and chain.count("evict-first") == 1
      and ("scale-codes" not in chain or chain.index("scale-codes") < chain.index("evict-first"))
      and 'echo "evict-first patch did not apply"' in build, str(chain))
check("patch credits the idea (no code copied)", "Christopher Owen, spark-ds41f #966 (Apache-2.0)" in patch)
files = sorted(set(re.findall(r"^\+\+\+ (b12x_next/\S+)", patch, re.M)))
if shutil.which("patch"):
    tmp = tempfile.mkdtemp()
    try:
        for f in files:
            os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
            shutil.copy(os.path.join(PKG_PARENT, f), os.path.join(tmp, f))
        rev = subprocess.run(["patch", "-p0", "-R", "--quiet"], input=patch, text=True, cwd=tmp, capture_output=True)
        unpatched = {f: read(tmp, f) for f in files}
        fwd = subprocess.run(["patch", "-p0", "--forward", "--quiet"], input=patch, text=True, cwd=tmp,
                             capture_output=True)
        same = all(read(tmp, f) == read(PKG_PARENT, f) for f in files)
        check(f"patch round-trips against the tree ({len(files)} files)",
              rev.returncode == 0 and fwd.returncode == 0 and same
              and all("evict_first_weights" not in t and "shared_global_l2hint" not in t for t in unpatched.values()),
              rev.stderr + fwd.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    SKIPS.append("patch(1) not installed: round trip")

# ------------------------------------------------------------------------------------------------- registry
import ab_variant as ab  # noqa: E402

check("registry: both keys are RUNTIME keys, never unioned, not graph-neutral",
      {"DSV41_MOE_EVICT_FIRST", "DSV41_MOE_EVICT_FIRST_MAX_ROWS"} <= set(ab.RUNTIME)
      and not {"DSV41_MOE_EVICT_FIRST", "DSV41_MOE_EVICT_FIRST_MAX_ROWS"}
      & (set(ab.UNION_BOOL) | set(getattr(ab, "GRAPH_NEUTRAL", ())))
      and "moe_b12x_next" in ab.HASHED_SOURCES)
n = ab._norm
check("registry: the row cap is moot while the gate is off; unset = 128",
      n({"DSV41_MOE_EVICT_FIRST_MAX_ROWS": "64"})["DSV41_MOE_EVICT_FIRST_MAX_ROWS"] is None
      and n({"DSV41_MOE_EVICT_FIRST": "1"})["DSV41_MOE_EVICT_FIRST_MAX_ROWS"] == 128
      and n({"DSV41_MOE_EVICT_FIRST": "1", "DSV41_MOE_EVICT_FIRST_MAX_ROWS": "128"})
      == n({"DSV41_MOE_EVICT_FIRST": "1"})
      and n({"DSV41_MOE_EVICT_FIRST": "1", "DSV41_MOE_EVICT_FIRST_MAX_ROWS": "64"})
      != n({"DSV41_MOE_EVICT_FIRST": "1"}))
try:
    n({"DSV41_MOE_EVICT_FIRST": "1", "DSV41_MOE_EVICT_FIRST_MAX_ROWS": "0"})
    check("registry: a row cap < 1 is refused", False)
except RuntimeError:
    check("registry: a row cap < 1 is refused", True)


def clean_meta_path():
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]


def configured(env):
    import importlib
    mod = importlib.reload(ab)
    mod.configure(env)
    return mod


try:
    configured({"DSV41_AB_VARIANTS": "2", "DSV41_AB_V0": "DSV41_MOE_EVICT_FIRST=0;DSV41_MOE_EVICT_FIRST_MAX_ROWS=64",
                "DSV41_AB_V1": "DSV41_MOE_EVICT_FIRST=0"})
    check("registry: two variants that differ only in a moot row cap are an A/A (refused)", False)
except RuntimeError as exc:
    check("registry: two variants that differ only in a moot row cap are an A/A (refused)", "A/A" in str(exc))
finally:
    clean_meta_path()

# ------------------------------------------------------------------------------------------------- adapter
try:
    import torch
except ImportError:
    torch = None
    SKIPS.append("torch not importable: adapter checks")

if torch is not None:
    env = {"DSV41_AB_VARIANTS": "3", "DSV41_AB_V0": "DSV41_MOE_EVICT_FIRST=0",
           "DSV41_AB_V1": "DSV41_MOE_EVICT_FIRST=1", "DSV41_AB_V2": "DSV41_MOE_EVICT_FIRST=1;DSV41_MOE_EVICT_FIRST_MAX_ROWS=12"}
    abm = configured(dict(env))
    import moe_b12x_next as ad  # noqa: E402
    ad._ab = abm
    try:
        check("adapter: row cap parsing", ad.parse_ef_max_rows("") == 128 and ad.parse_ef_max_rows(None) == 128
              and ad.parse_ef_max_rows(" 96 ") == 96)
        for bad in ("0", "-3", "x"):
            try:
                ad.parse_ef_max_rows(bad)
                check(f"adapter: row cap {bad!r} refused", False)
            except RuntimeError as exc:
                check(f"adapter: row cap {bad!r} refused", "refused" in str(exc))
        check("adapter: per-variant settings", ad.ef_settings() == [(0, False, 128), (1, True, 128), (2, True, 12)],
              str(ad.ef_settings()))
        abm.ACTIVE = False
        check("adapter: without the A/B the process env decides",
              ad.ef_settings({"DSV41_MOE_EVICT_FIRST": "1", "DSV41_MOE_EVICT_FIRST_MAX_ROWS": "48"}) == [(0, True, 48)]
              and ad.ef_settings({}) == [(0, False, 128)])
        abm.ACTIVE = True

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                ad.install_evict_first(types.SimpleNamespace())
                refused = False
            except RuntimeError as exc:
                refused = "refusing to boot" in str(exc)
        check("adapter: a runtime without the patch refuses to boot, receipt says refused",
              refused and "[moe_b12x_next] evict-first expert loads: refused" in out.getvalue() and not ad._EF["armed"],
              out.getvalue())
        check("adapter: nothing armed when no variant enables the gate",
              ad.install_evict_first(None, settings=[(0, False, 128), (1, False, 64)]) is False)

        class FakeImpl:
            def __init__(self):
                self._DYNAMIC_KERNEL_CACHE = {}
                self.active = 0
                self.EVICT_FIRST_STATS = {"compiled": 0, "ineligible": 0}

            @contextlib.contextmanager
            def evict_first_weights(self):
                self.active += 1
                try:
                    yield
                finally:
                    self.active -= 1

        fimpl = FakeImpl()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            armed = ad.install_evict_first(fimpl)
        line = out.getvalue().strip()
        check("adapter: boot receipt", armed and ad._EF["armed"] and ad._EF["max_rows"] == 128 and line ==
              "[moe_b12x_next] evict-first expert loads: armed (max rows v1:128,v2:12, sites "
              "w4a8_phase1.fc1_w13_b,w4a8_phase2.fc2_down_b)", line)

        # a minimal geometry: one capacity-6 plan, one capacity-96 plan, one refused capacity-24 plan
        @dataclasses.dataclass
        class Bound:
            compact_launches: object = None
            plan: object = None

        scratch = types.SimpleNamespace(caps=types.SimpleNamespace(w4a16_fast_math=False),
                                        bind=lambda **kw: Bound())
        state = types.SimpleNamespace(scratch=scratch, w4a16_launches=None, compact_launches=None)
        g = object.__new__(ad._Geometry)
        g.plans = {c: ad._PlanEntry(cap=c, plan=f"P{c}", state=state, specs=()) for c in (6, 12, 24, 96)}
        g.chunk, g.caps, g.ef_caps, g.ef_refused = 96, [6, 12, 24, 96], [6, 12, 96], {24}
        g.ef_caps_coded, g.ef_refused_coded = [6, 96], {12}     # scale-codes retain mode: the coded experts' own sets
        runs = []
        ad._B["fm"] = types.SimpleNamespace(run=lambda binding: runs.append(fimpl.active))
        capturing = [False]
        torch.cuda.is_current_stream_capturing = lambda: capturing[0]

        def launch(cap, m, ef=None):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                g._run(cap, torch.empty(m, 4), None, None, None, None, ef=ef)
            return runs[-1] == 1, out.getvalue().strip()

        on, log = launch(6, 6)
        check("adapter: eager launches never take the hint (no receipt)", on is False and log == "")
        on, log = launch(6, 6, ef=True)
        check("adapter: the load-time warm (ef=True) runs inside evict_first_weights()", on is True and log == "")
        capturing[0] = True
        seen = []
        for variant, cap, m, want in ((0, 6, 6, False), (1, 6, 6, True), (1, 96, 96, True), (2, 6, 6, True),
                                      (2, 96, 96, False), (2, 24, 13, False), (1, 24, 13, False)):
            with abm.capturing(variant):
                got, log = launch(cap, m)
                got2, log2 = launch(cap, m)
            seen.append((variant, m, got, log))
            ok = got == want and got2 == want and log2 == "" and log == (
                f"[moe_b12x_next] evict-first: capture variant={variant} M={m} {'on' if want else 'off'}"
                + (" (refused: this capacity's plan does not run the split-materialized W4A8 phase kernels)"
                   if cap == 24 and variant == 1 else ""))
            check(f"adapter: capture variant={variant} M={m} -> {'on' if want else 'off'}, one receipt", ok,
                  f"{got} {got2} {log!r} {log2!r}")
        capturing[0] = False

        if hasattr(ad, "_CODED_IMPLS"):
            # scale codes (A/B retain mode) + evict-first in one variant: a capture that runs the coded experts takes
            # the coded refusals (capacity 12 refused for coded, admitted for native)
            coded_impl = types.SimpleNamespace(name="coded")
            ad._CODED_IMPLS[id(coded_impl)] = coded_impl
            capturing[0] = True
            out = io.StringIO()
            with abm.capturing(1), contextlib.redirect_stdout(out):
                g._run(12, torch.empty(12, 4), None, None, None, coded_impl)
            coded_on = runs[-1] == 1
            with abm.capturing(2), contextlib.redirect_stdout(out):
                g._run(12, torch.empty(12, 4), None, None, None, object())
            native_on = runs[-1] == 1
            capturing[0] = False
            del ad._CODED_IMPLS[id(coded_impl)]
            check("adapter: coded experts use their own evict-first refusals (coded M=12 off/refused, native M=12 on)",
                  not coded_on and native_on and "variant=1 M=12 off (refused" in out.getvalue()
                  and "variant=2 M=12 on" in out.getvalue(), out.getvalue())

        # restore: a later preparation evicted the flag-on kernel from b12x's memo
        ad._EF["kernels"] = {("k", "evict_first_weights"): "compiled-on"}
        fimpl._DYNAMIC_KERNEL_CACHE.clear()
        capturing[0] = True
        with abm.capturing(1):
            launch(6, 6)
        capturing[0] = False
        check("adapter: a dropped flag-on kernel is restored before a capture launch",
              fimpl._DYNAMIC_KERNEL_CACHE.get(("k", "evict_first_weights")) == "compiled-on" and ad._EF["restored"] == 1)
        rep = ad.geometry_reports()["evict_first"]
        check("adapter: report", rep["armed"] is True and rep["captures_on"] >= 3 and rep["captures_off"] >= 3, str(rep))
    finally:
        clean_meta_path()

# ------------------------------------------------------------------------------------------------- b12x_next _impl
try:
    import cutlass  # noqa: F401
    from b12x_next.moe.fused_moe import _impl as I
except Exception as exc:  # noqa: BLE001
    I = None
    SKIPS.append(f"b12x_next _impl not importable ({type(exc).__name__}): eligibility / key checks")

if I is not None and not callable(getattr(I, "evict_first_weights", None)):
    check("_impl: this b12x_next carries the evict-first patch", False, I.__file__)
    I = None
if I is not None:
    el = I._evict_first_weights_eligible
    base = dict(is_w4a8=True, trellis_bits=0, w4a8_repacked=True, w4a8_n64_repacked=True,
                materialize_intermediate=True, mma_tiler_mn=(16, 128))
    table = [(base, True), (dict(base, mma_tiler_mn=(64, 128)), True), (dict(base, mma_tiler_mn=(128, 128)), True),
             (dict(base, w4a8_n64_repacked=False), False), (dict(base, w4a8_n64_repacked=False, mma_tiler_mn=(64, 128)),
                                                            True),
             (dict(base, mma_tiler_mn=(32, 128)), False), (dict(base, trellis_bits=4), False),
             (dict(base, materialize_intermediate=False), False), (dict(base, is_w4a8=False), False),
             (dict(base, w4a8_repacked=False), False)]
    check("_impl: eligibility = W4A8 split-materialized, non-trellis", all(el(**kw) is want for kw, want in table),
          str([(kw, el(**kw)) for kw, want in table if el(**kw) is not want]))
    check("_impl: the context nests and restores", not I._evict_first_weights_requested())
    with I.evict_first_weights():
        with I.evict_first_weights():
            pass
        inner = I._evict_first_weights_requested()
    check("_impl: the context nests and restores", inner and not I._evict_first_weights_requested())

    seen = {}

    def fake_compile(launch, *args, compile_spec=None, dsl_compile_options=None, **kw):
        seen["launch"] = launch
        return object()

    I.b12x_compile = fake_compile
    I.current_cuda_stream = lambda: 0

    def resolve(n=576, n64=True, tile=16, **extra):
        seen.clear()
        I._DYNAMIC_KERNEL_CACHE.clear()
        I._get_dynamic_kernel(384, 6, 5120, n, 6, 4096, topk_ids_dtype=torch.int32, fast_math=False,
                              mac_override=48, activation="silu", quant_mode="w4a8_mx", w4a8_repacked=True,
                              w4a8_n64_repacked=n64, share_input_across_experts=True, deterministic_output=True,
                              swiglu_limit=10.0, planned_tile_m=tile, planned_num_tokens=6, **extra)
        (key,) = I._DYNAMIC_KERNEL_CACHE
        k = seen["launch"]._kernel
        resolve.kernel = k
        return key, (k.materialized_phase1_kernel.evict_first_weights, k.materialized_phase2_kernel.evict_first_weights)

    off_key, off_flags = resolve()
    with I.evict_first_weights():
        on_key, on_flags = resolve()
    check("_impl: off = stock key and stock phase kernels", off_key[-1] != "evict_first_weights"
          and off_flags == (False, False))
    check("_impl: on = the stock key + ('evict_first_weights',), both phase kernels flagged",
          on_key == off_key + ("evict_first_weights",) and on_flags == (True, True))
    before = dict(I.EVICT_FIRST_STATS)
    with I.evict_first_weights():
        n_key, n_flags = resolve(n=1152, n64=False, tile=32)   # EP2 M32: not split-materialized
    ref_key, _ = resolve(n=1152, n64=False, tile=32)
    check("_impl: an ineligible plan ignores the request (stock key, counted)",
          n_key == ref_key and n_flags == (False, False)
          and I.EVICT_FIRST_STATS["ineligible"] == before["ineligible"] + 1)
    import inspect
    if "w4a8_coded_scales" in inspect.signature(I._get_dynamic_kernel).parameters:
        # scale codes (b12x_next-scale-codes.patch, applied before evict-first): the two key suffixes are independent
        c_key, c_flags = resolve(w4a8_coded_scales=True)
        with I.evict_first_weights():
            ce_key, ce_flags = resolve(w4a8_coded_scales=True)
        k1, k2 = resolve.kernel.materialized_phase1_kernel, resolve.kernel.materialized_phase2_kernel
        check("_impl: coded x evict-first: independent key suffixes, coded phase kernels flagged",
              "w4a8_coded_scales" in c_key and c_key != off_key and c_flags == (False, False)
              and ce_key == c_key + ("evict_first_weights",) and ce_flags == (True, True)
              and k1.coded_scales and k2.coded_scales, f"{c_key[-12:]} {ce_key[-12:]}")
    import moe_launch_prepared as P
    with I.evict_first_weights():
        on_state = P._evict_first(I)
    check("prepared key: carries the evict-first state (False without the patch)",
          on_state is True and P._evict_first(I) is False and P._evict_first(types.SimpleNamespace()) is False)

for s in SKIPS:
    print(f"[SKIP] {s}")
if FAILS:
    print(f"FAILED: {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS" + (f" ({len(SKIPS)} skipped)" if SKIPS else ""))
