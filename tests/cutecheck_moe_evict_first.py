"""GPU-less sm_121a compile check of the evict-first expert-weight copies (DSV41_MOE_EVICT_FIRST).

Runs in the colima image ds-b3-cutecheck:1 (no GPU; CUTE_DSL_ARCH=sm_121a, PTX and CUBIN kept), one job per
process (the DSL reads its dump dir once):

  python3 tests/cutecheck_moe_evict_first.py B12X_NEXT_ROOT OUT_DIR JOB | list
  JOB = phase{1,2}_t{16,64}[_coded]_{off,on} | dynamic_{ext,int}[_coded]_{off,on}
  (_coded: the ds41 scale-codes variant of the same kernels, coded_scales / w4a8_coded_scales; needs a tree with
  scripts/b12x_next-scale-codes.patch)

phase*: the split-materialized W4A8 phase kernel the production decode plans launch (compact N64 with the N
tail: TP4 slice N=576, K=5120 hidden, deterministic output, top-k 6) through cute.compile with fake pointers,
with ``evict_first_weights`` off or on (a B12X_NEXT_ROOT without the evict-first patch only has ``off``).
dynamic*: the whole production decode program (front-end + phase 1 + phase 2 in one module) resolved by
fused_moe._impl._get_dynamic_kernel exactly as a launch resolves it (E 384, the plan's tile M16, grouped routing,
ext = Triton route planner (the <= 8 row plan), int = internal planner (b12x's heuristic plans)), with the
compile wrapper replaced by cute.compile[OptLevel(2)] (b12x's own wrapper keys its disk cache on a device),
inside ``_impl.evict_first_weights()`` for ``on``; also records the cache key tail and the phase kernels' flags.
Prints
one JSON line: PTX counts of the plain / hinted ``cp.async.cg.shared.global`` forms and of ``createpolicy``, the
ptxas resource line, and the SASS LDGSTS forms (cuobjdump -sass). A failure after the CUBIN exists (module load
needs a driver) still counts as compiled. The caller diffs the off PTX against a baseline tree's.
Descriptor gate (tests/sass_desc_check.py, exit 1 on failure): in every function of the listing each desc[URd] access
reads the default descriptor or (LDGSTS only) the evict_first policy on every path (value flow over the uniform
registers: no odd pair, no use before write, no overwrite), and a flag-on job holds exactly HINTED policy LDGSTS per
phase kernel (PTX: HINTED hinted cp.async per phase kernel), a flag-off job none.
"""
import collections
import glob
import json
import os
import re
import subprocess
import sys

ROOT, OUT = sys.argv[1], sys.argv[2]
JOB = sys.argv[3] if len(sys.argv) > 3 else "list"
K, N, TOPK = 5120, 576, 6
JOBS = ([f"phase{p}_t{t}{c}_{f}" for p in (1, 2) for t in (16, 64) for c in ("", "_coded") for f in ("off", "on")]
        + [f"dynamic_{r}{c}_{f}" for r in ("ext", "int") for c in ("", "_coded") for f in ("off", "on")])
if JOB == "list":
    print(" ".join(JOBS))
    sys.exit(0)
if JOB not in JOBS:
    sys.exit(f"unknown job {JOB}; one of {JOBS}")
flag = JOB.endswith("_on")
coded = "_coded_" in JOB
if JOB.startswith("phase"):
    phase, tile = int(JOB[5]), int(JOB.split("_")[1][1:])
else:
    phase, tile = 0, 16
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = OUT
sys.path.insert(0, ROOT)

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cutlass_dsl import Int32  # noqa: E402

import cutlass.base_dsl.runtime.cuda as _rc  # noqa: E402

_orig_attr = _rc.get_device_attribute


def _attr(attribute, device_id=0):
    # no GPU in the VM: launches with min_blocks_per_mp > 1 ask for this one attribute (GB10: 100 KiB / SM)
    if attribute == _rc.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
        return 102400
    return _orig_attr(attribute, device_id)


_rc.get_device_attribute = _attr
from b12x_next._lib.utils import make_ptr  # noqa: E402
from b12x_next.moe._shared.kernels import w4a8_phase1 as p1  # noqa: E402
from b12x_next.moe._shared.kernels import w4a8_phase2 as p2  # noqa: E402


def ptr(dtype, align=16):
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


def lay(n=1 << 20):
    return cute.make_layout((n,), stride=(1,))


U, I, F = cutlass.Uint32, cutlass.Int32, cutlass.Float32
P1_TYPES = [(U, 16), (U, 16), (U, 16), (U, 16), (U, 16), (I, 4), (F, 4), (I, 4), (I, 4), (I, 4), (F, 4), (F, 4),
            (U, 16), (cutlass.Float16, 16)]
P2_TYPES = [(U, 16), (U, 16), (U, 16), (cutlass.BFloat16, 16), (I, 4), (F, 4), (I, 4), (I, 4), (I, 4), (F, 4),
            (F, 4), (U, 16)]
KW = dict({"evict_first_weights": True} if flag else {}, **({"coded_scales": True} if coded else {}))
HINTED = 8      # hinted packed-weight copies per phase kernel, both tiles, plain and coded (the dynamic program: 8 + 8)


class Phase1Wrap:
    def __init__(self):
        self.k = p1.W4A8MaterializedPhase1Kernel(fast_math=False, swiglu_limit=10.0, source_tile_m=tile,
                                                 deterministic_output=True, num_topk=TOPK, n64_repacked=True,
                                                 n64_tail=True, **KW)

    @cute.jit
    def __call__(self, a0: cute.Pointer, a1: cute.Pointer, a2: cute.Pointer, a3: cute.Pointer, a4: cute.Pointer,
                 a5: cute.Pointer, a6: cute.Pointer, a7: cute.Pointer, a8: cute.Pointer, a9: cute.Pointer,
                 a10: cute.Pointer, a11: cute.Pointer, a12: cute.Pointer, a13: cute.Pointer, stream: cuda.CUstream):
        T = [cute.make_tensor(x, lay()) for x in (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13)]
        # input K128 tiles, intermediate (N128) tiles incl. the N64 tail, packed w13 tiles, max active clusters
        self.k(*T, Int32(K // 128), Int32((N + 127) // 128), Int32(2 * ((N + 127) // 128)), Int32(48), stream)


class Phase2Wrap:
    def __init__(self):
        self.k = p2.W4A8MaterializedPhase2Kernel(source_tile_m=tile, deterministic_output=True, n64_repacked=True,
                                                 n64_tail=True, **KW)

    @cute.jit
    def __call__(self, a0: cute.Pointer, a1: cute.Pointer, a2: cute.Pointer, a3: cute.Pointer, a4: cute.Pointer,
                 a5: cute.Pointer, a6: cute.Pointer, a7: cute.Pointer, a8: cute.Pointer, a9: cute.Pointer,
                 a10: cute.Pointer, a11: cute.Pointer, stream: cuda.CUstream):
        T = [cute.make_tensor(x, lay()) for x in (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)]
        T[3] = cute.make_tensor(a3, cute.make_layout((4096, K), stride=(K, 1)))   # scatter_output [rows, K]
        self.k(*T, Int32((N + 127) // 128), Int32(K // 256), Int32(48), Int32(64), stream)


def dynamic_job():
    """_get_dynamic_kernel of the production decode plan, compile wrapper -> cute.compile[OptLevel(2)]."""
    import contextlib
    import torch
    from cutlass.base_dsl.compiler import OptLevel
    from b12x_next.moe.fused_moe import _impl
    seen = {}

    def fake_compile(launch, *args, compile_spec=None, dsl_compile_options=None, **kw):
        seen["key"] = compile_spec.key if hasattr(compile_spec, "key") else None
        seen["launch"] = launch
        cc = cute.compile[dsl_compile_options] if dsl_compile_options is not None else cute.compile
        return cc(launch, *args)

    _impl.b12x_compile = fake_compile
    _impl.current_cuda_stream = lambda: cuda.CUstream(0)
    ctx = getattr(_impl, "evict_first_weights", None)
    if flag and ctx is None:
        raise SystemExit("this b12x_next has no evict_first_weights()")
    with (ctx() if flag else contextlib.nullcontext()):
        try:
            _impl._get_dynamic_kernel(
                384, 6, K, N, TOPK, 4096, topk_ids_dtype=torch.int32, fast_math=False, mac_override=48,
                activation="silu", quant_mode="w4a8_mx", w4a8_repacked=True, w4a8_n64_repacked=True,
                direct_routing=False, external_route_plan=JOB.startswith("dynamic_ext"),
                share_input_across_experts=True, deterministic_output=True, swiglu_limit=10.0,
                planned_tile_m=16, planned_num_tokens=6, **({"w4a8_coded_scales": True} if coded else {}))
        finally:
            kern = getattr(seen.get("launch"), "_kernel", None)
            row["kernel_split_materialized"] = getattr(kern, "w4a8_split_materialized", None)
            row["phase1_evict_first"] = getattr(getattr(kern, "materialized_phase1_kernel", None),
                                                "evict_first_weights", None)
            row["phase2_evict_first"] = getattr(getattr(kern, "materialized_phase2_kernel", None),
                                                "evict_first_weights", None)
            row["phase_coded"] = [getattr(getattr(kern, a, None), "coded_scales", None)
                                  for a in ("materialized_phase1_kernel", "materialized_phase2_kernel")]
            keys = [k for k in _impl._DYNAMIC_KERNEL_CACHE]
            row["cache_key_tail"] = list(map(str, keys[-1][-2:])) if keys else None
            row["evict_first_stats"] = dict(getattr(_impl, "EVICT_FIRST_STATS", {}))


row = {"job": JOB, "root": ROOT}
try:
    if phase == 0:
        dynamic_job()
    elif phase == 1:
        cute.compile(Phase1Wrap(), *[ptr(dt, al) for dt, al in P1_TYPES], cuda.CUstream(0))
    else:
        cute.compile(Phase2Wrap(), *[ptr(dt, al) for dt, al in P2_TYPES], cuda.CUstream(0))
    row["status"] = "compiled+loaded"
except TypeError as exc:
    row["status"] = "FAILED"
    row["error"] = repr(exc)[:400]
except Exception as exc:  # noqa: BLE001
    row["status"] = "compiled (load failed: no GPU)"
    row["error"] = repr(exc)[:300]
ptxs = sorted(glob.glob(os.path.join(OUT, "**", "*.ptx"), recursive=True))
cubins = sorted(glob.glob(os.path.join(OUT, "**", "*.cubin"), recursive=True))
row["ptx"], row["cubin"] = len(ptxs), len(cubins)
if not cubins:
    row["status"] = "FAILED"
if ptxs:
    text = open(ptxs[-1]).read()
    row["ptx_file"] = os.path.relpath(ptxs[-1], OUT)
    row["ptx_cp_async_plain"] = len(re.findall(r"cp\.async\.cg\.shared\.global \[", text))
    row["ptx_cp_async_l2hint"] = len(re.findall(r"cp\.async\.cg\.shared\.global\.L2::cache_hint \[", text))
    row["ptx_createpolicy_evict_first"] = len(re.findall(r"createpolicy\.fractional\.L2::evict_first\.b64", text))
    row["ptx_ld_global_u8"] = len(re.findall(r"ld\.global(?:\.[a-z0-9:_]+)*\.u8", text))
if cubins:
    try:
        sass = subprocess.run(["cuobjdump", "-sass", cubins[-1]], capture_output=True, text=True, timeout=300).stdout
        forms = collections.Counter(m.group(1) for m in re.finditer(r"\b(LDGSTS(?:\.[A-Z0-9_]+)*)\b", sass))
        row["sass_ldgsts_forms"] = dict(sorted(forms.items()))
        # Every sm_12x global access names a 64-bit memory descriptor desc[URx]. The default one is loaded from the
        # constant bank (c[0x0][0x358]); an L2::cache_hint access uses the createpolicy value as its descriptor
        # (evict_first fraction 1.0 = 0x12f0000000000000: UR(hi) = 0x12f << 20).
        # Registers are per function: classify inside each "Function :" section.
        by_desc = collections.Counter()
        by_function = {}
        examples = {}
        for section in re.split(r"\n\s*Function : ", sass)[1:]:
            name = section.split("\n", 1)[0].strip()
            short = ("phase1" if "Phase1Kernel" in name else "phase2" if "Phase2Kernel" in name
                     else "frontend" if "MoEDynamicKernel" in name else name[:40])
            default = set(re.findall(r"(?:LDCU|ULDC)\.64 (UR\d+), c\[0x0\]\[0x358\]", section))
            counts = collections.Counter()
            for ln in section.splitlines():
                m = re.search(r"LDGSTS\S* .*desc\[(UR\d+)\]", ln)
                if m:
                    kind = "default" if m.group(1) in default else "policy"
                    counts[kind] += 1
                    examples.setdefault(kind, re.sub(r"\s+", " ", ln.split(";")[0].strip()))
            by_desc.update(counts)
            by_function[short] = dict(counts)
        row["sass_ldgsts_by_desc"] = dict(by_desc)
        row["sass_ldgsts_by_function"] = by_function
        row["sass_ldgsts_examples"] = examples
        row["sass_policy_hi_0x12f"] = bool(re.search(r"UMOV UR\d+, 0x120 ;", sass) and re.search(r"UMOV UR\d+, 0xf0 ;", sass))
        res = subprocess.run(["cuobjdump", "-res-usage", cubins[-1]], capture_output=True, text=True, timeout=120).stdout
        row["res_usage"] = [ln.strip() for ln in res.splitlines() if "REG" in ln][:2]
        with open(os.path.join(OUT, "sass.txt"), "w") as f:
            f.write(sass)
        # the descriptor gate (tests/sass_desc_check.py): every global access's desc[] pair is the default
        # descriptor or the evict_first policy on every path, and the flag-on job has exactly its hinted copies
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import sass_desc_check  # noqa: E402
        expect = {} if not flag else ({"phase1": HINTED, "phase2": HINTED} if phase == 0 else {f"phase{phase}": HINTED})
        ok, lines = sass_desc_check.gate(sass, expect)
        want_ptx = 0 if not flag else HINTED * (2 if phase == 0 else 1)
        if row.get("ptx_cp_async_l2hint") != want_ptx:
            ok = False
            lines.append(f"    PTX: {row.get('ptx_cp_async_l2hint')} hinted cp.async, required {want_ptx}")
        row["sass_gate"] = "PASS" if ok else "FAIL"
        row["sass_gate_expect"] = expect
        row["sass_gate_lines"] = lines[:40]
    except Exception as exc:  # noqa: BLE001
        row["sass_error"] = repr(exc)[:200]
if row["status"] != "FAILED" and row.get("sass_gate") != "PASS":
    row["status"] = "FAILED"            # no listing, a listing error or a descriptor-gate failure fails the job
print(json.dumps(row), flush=True)
sys.exit(0 if row["status"] != "FAILED" else 1)
