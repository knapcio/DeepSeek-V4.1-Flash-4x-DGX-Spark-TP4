"""Prepared eager launch of b12x_next's dynamic MoE kernel, and host-side launcher timing. Default OFF;
the production profile sets DSV41_MOE_LAUNCH_PREPARED=plan (fleet A/B/A 2026-10-26, with DSV41_INDEXER_NOSYNC=1:
cold prefill 4k +16 %, TTFT -4 %, T=0 outputs byte-identical). ``plan,direct`` and the timing hooks stay test only.

Why. In an eager (non-graph) forward every routed-MoE layer goes through the custom op
``b12x_next::tp_moe_dynamic_launch``. The 2026-10-23 c1 traces (34-row prefill, all four ranks) show 1.45-1.77 ms
of host time per call: ~0.98 ms of Python before the first launch (the barrier fill), ~0.52 ms more before the
compiled kernel's launch, and only four launches. 40 calls are ~60 ms of a host-paced ~200 ms prefill. The decode
step is a CUDA graph and never runs this Python.

``DSV41_MOE_LAUNCH_PREPARED`` (off | plan | plan,direct):

* ``plan``: ``_impl._launch_dynamic_flat`` (the custom op's body) is replaced by a version that runs the stock
  host decisions (quant-mode / routing / materialization / MAC / tile policy and the ``_get_dynamic_kernel`` lookup)
  ONCE per launch contract and keeps the result as a plan. The contract key is every scalar argument of the call,
  the materialized-scratch size, the device, the coded-scale format of both sfb slots (ds41 scale codes), the
  pre-quantized-input state, the evict-first state (b12x_next
  ``evict_first_weights()``, DSV41_MOE_EVICT_FIRST: its kernels have their own cache key, so a plan built with the
  flag off must never serve a call with it on, e.g. the load-time warm of the flag-on kernels) and the two
  programmatic overrides (``_DYNAMIC_TILE_MN_OVERRIDE`` / ``_DYNAMIC_SWAP_AB_OVERRIDE``). The ``B12X_NEXT_*`` environment is tracked
  exactly: at install ``os.environ`` becomes a subclass of its own class that counts every write / delete of a
  ``B12X_NEXT_*`` key (b12x reads the environment only through ``os.environ``), and every eager launch compares that
  generation and the identity of ``os.environ`` before using a plan, so a change drops every plan on the very next
  launch. Every call still does, in the stock order and from the stock module
  attributes: the barrier fill (volatile state, no external route plan), the Triton route planner (external route
  plan), then the compiled kernel with pointers to THIS call's tensors and THIS call's current CUDA stream. Only
  the cute pointer wrappers are memoized, keyed by (address, dtype, alignment): a wrapper holds nothing else.
  A plan is used only while ``_DYNAMIC_KERNEL_CACHE`` still maps its kernel key to the same compiled object
  (``clear_kernel_cache`` or a recompile invalidates it). The stock body runs instead under CUDA-graph capture,
  compile planning / compile-only launches / program observers, trellis weights, or a disabled kernel cache.
  Same kernels, same arguments, same order: bit-identical by construction (tests/test_moe_launch_prepared.py runs
  the stock and prepared bodies against recording stubs of every side effect over the production contracts).
* ``plan,direct``: also calls the op's Python body directly in eager mode instead of going through the PyTorch
  dispatcher (``torch.ops.b12x_next.tp_moe_dynamic_launch`` is wrapped; capture / compile keep the op). This skips
  the dispatcher's version-counter bumps of the 21 mutated scratch arguments (nothing in the adapter reads them;
  prepared weights are not mutated). Not measured on the fleet yet.

``DSV41_MOE_LAUNCH_TIMING=1``: perf_counter_ns around the adapter call (``_Geometry._run``: scratch bind + fused
run), the op (``_launch_dynamic``: argument assembly + dispatch + body), the body (``_launch_dynamic_flat``), the
stock kernel lookup (``_get_dynamic_kernel``) and the compiled call, eager calls only (never under capture), per
in-boot A/B variant and row bucket. Every ``DSV41_MOE_LAUNCH_TIMING_EVERY`` (512) op calls a process logs one line
``DSV41_MOE_LAUNCH_TIMING {json}`` (n, mean / p50 / p90 / min us per stage, the per-record self cost measured at
install). Derived: dispatch = op - body; decide = body - compiled call. The same gate also times every eager
forward (``ModelRunner._forward_raw``, hooked from sitecustomize): host issue time and the CUDA-event GPU span, read
without a sync, so a warmed unprofiled prefill shows whether it is host-paced (see install_forward_timing).

Install: from moe_b12x_next._b12x() after its pin / drift checks, only when one of the two gates is set (import
gated before). Refuses to arm when the stock sources differ from the pinned sha256 (the plan mirrors them).
Under the in-boot A/B (adapter/ab_variant.py) DSV41_MOE_LAUNCH_PREPARED is a union gate: the hooks are installed
when any variant enables it and the mode is read per eager launch through ab_variant.env().
"""
import hashlib
import inspect
import json
import logging
import os
import sys
import time

import torch

logger = logging.getLogger(__name__)
_TAG = "[moe_launch_prepared]"
_OFF = ("", "0", "off", "false", "no")

# sha256 of inspect.getsource() of the stock functions this module mirrors or wraps (runtime/b12x_next at
# a7d7d29b + the five ds41 patches; scripts/build_b12x_next.sh). A drifted runtime refuses to arm.
PINNED_SOURCES = {
    "_launch_dynamic_flat": "a9c375d26cd5bc1f505388391fcf4028d77c4cfe01697a580a5c4a8e907514fb",
    "_tp_moe_dynamic_launch_op": "bb840645a86092558eafb1b5381bc1429b184cf1445568583fb009f73fec5ba7",
    "_launch_dynamic": "10f3393562a76692a7861e4eea041647a0a33559c13e756ff19ea612fed620b3",
}

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:
    _ab = None

# every scalar argument of _launch_dynamic_flat, in signature order (the plan key)
SCALARS = ("E", "m", "k", "n", "num_topk", "routed_rows", "max_rows", "physical_tiles_capacity", "task_capacity",
           "topk_ids_are_i32", "fast_math", "activation", "quant_mode", "w4a8_repacked", "w4a8_n64_repacked",
           "nvfp4_materialize_intermediate", "share_input_across_experts", "deterministic_output", "swiglu_limit",
           "swiglu_alpha", "swiglu_beta", "external_route_plan_requested", "policy_max_active_clusters",
           "volatile_launch_state", "planned_tile_m", "planned_direct_routing", "planned_num_tokens")
PTR_CACHE_MAX = 8192
PLAN_CACHE_MAX = 4096

_I = None                  # b12x_next.moe.fused_moe._impl
_ORIG = {}                 # stock callables saved at install (before any timing wrapper)
_PLANS = {}
_PTRS = {}
_STATE = {"mode": "off", "installed": False, "calls": 0, "env_gen": None, "env_obj": None, "hits": 0, "builds": 0,
          "stock": 0,
          "timing": False, "direct": False}


def source_sha(fn):
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def parse_mode(value):
    v = str(value or "0").strip().lower().replace(" ", "")
    if v in _OFF:
        return "off"
    return "plan,direct" if "direct" in v else "plan"


def mode():
    if _ab is not None and getattr(_ab, "ACTIVE", False) and "DSV41_MOE_LAUNCH_PREPARED" in getattr(_ab, "KNOWN", ()):
        return parse_mode(_ab.env("DSV41_MOE_LAUNCH_PREPARED", "0"))
    return _STATE["mode"]


def _variant():
    return _ab.current() if _ab is not None and getattr(_ab, "ACTIVE", False) else 0


_ENV_GEN = [0]
_ENV_PREFIX = "B12X_NEXT_"


def track_environ():
    """Make os.environ count writes / deletes of B12X_NEXT_* keys (update / pop / clear / setdefault all go through
    __setitem__ / __delitem__). Idempotent; False when os.environ cannot be retyped (the caller refuses to arm)."""
    env = os.environ
    if getattr(type(env), "_dsv41_tracked", False):
        return True
    base = type(env)

    class TrackedEnviron(base):
        _dsv41_tracked = True

        def __setitem__(self, key, value):
            base.__setitem__(self, key, value)
            if str(key).startswith(_ENV_PREFIX):
                _ENV_GEN[0] += 1

        def __delitem__(self, key):
            base.__delitem__(self, key)
            if str(key).startswith(_ENV_PREFIX):
                _ENV_GEN[0] += 1

    try:
        env.__class__ = TrackedEnviron
    except TypeError:
        return False
    return True


def _capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001  (no CUDA: CPU tests)
        return False


def _plain_context(I):
    """No compile planning, compile-only launches, program observers or retained-program scopes."""
    cp = sys.modules.get("b12x_next._lib.compile_plan")
    if I.planning():
        return False
    if cp is None:
        return True
    for name, idle in (("_COMPILE_ONLY_LAUNCHES", (False, None)), ("_OBSERVERS", ((), None)),
                       ("_RETAINED_PROGRAMS", (None,))):
        var = getattr(cp, name, None)
        if var is not None and var.get() not in idle:
            return False
    return True


# ------------------------------------------------------------------------------------------------------------------
# the plan: the stock decision code of _launch_dynamic_flat, run once per contract (mirrors the pinned source)
# ------------------------------------------------------------------------------------------------------------------
class Plan:
    __slots__ = ("compiled", "mac", "kernel_key", "zero_barrier", "external_route_plan", "selected_tile_m",
                 "block_e", "routed_rows", "E", "scatter_rows", "tile_rows", "task_capacity",
                 "physical_tiles_capacity", "m", "max_rows", "is_w4a8", "ids_dtype", "ids_align", "packed_dtype",
                 "sf_dtype")


def _decide(kw):
    I = _I
    quant_mode = kw["quant_mode"]
    E, m, k, n = kw["E"], kw["m"], kw["k"], kw["n"]
    num_topk, routed_rows, max_rows = kw["num_topk"], kw["routed_rows"], kw["max_rows"]
    physical_tiles_capacity, task_capacity = kw["physical_tiles_capacity"], kw["task_capacity"]
    topk_ids_are_i32, fast_math, activation = kw["topk_ids_are_i32"], kw["fast_math"], kw["activation"]
    w4a8_repacked, w4a8_n64_repacked = kw["w4a8_repacked"], kw["w4a8_n64_repacked"]
    nvfp4_materialize_intermediate = kw["nvfp4_materialize_intermediate"]
    share_input_across_experts, deterministic_output = kw["share_input_across_experts"], kw["deterministic_output"]
    swiglu_limit, swiglu_alpha, swiglu_beta = kw["swiglu_limit"], kw["swiglu_alpha"], kw["swiglu_beta"]
    external_route_plan_requested = kw["external_route_plan_requested"]
    policy_max_active_clusters, volatile_launch_state = kw["policy_max_active_clusters"], kw["volatile_launch_state"]
    planned_tile_m, planned_direct_routing = kw["planned_tile_m"], kw["planned_direct_routing"]
    planned_num_tokens = kw["planned_num_tokens"]
    materialized_intermediate = kw["materialized_intermediate"]

    # ---- mirrors _launch_dynamic_flat (pinned) from the quant-mode normalization to _get_dynamic_kernel ----
    quant_mode = I._normalize_quant_mode(quant_mode)
    scatter_rows = routed_rows if deterministic_output else m
    # ds41 scale codes: a coded binding carries uint8 code buffers in both sfb slots (mirrors the stock guards)
    w4a8_coded_scales = I._is_w4a8_quant_mode(quant_mode) and kw["w13_sfb_rp"].dtype == torch.uint8
    if w4a8_coded_scales != (I._is_w4a8_quant_mode(quant_mode) and kw["down_sfb_rp"].dtype == torch.uint8):
        raise ValueError("ds41 scale codes: w13 and down scale slots disagree on the coded format")
    if w4a8_coded_scales and not w4a8_n64_repacked:
        raise ValueError("ds41 scale codes: coded scales reached a non-N64 W4A8 launch")
    trellis_bits = 0                      # trellis weights never reach a plan (_eligible)
    trellis_intermediate_hadamard = False
    if w4a8_repacked and int(n) % 128 != 0 and not w4a8_n64_repacked:
        n = I._dynamic_kernel_intermediate_size(n, quant_mode)
    decode_regime = bool(
        w4a8_repacked
        and I._w4a8_dynamic_decode_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=planned_num_tokens * num_topk,
            num_experts=E,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
    )
    direct_routing = bool(
        (quant_mode != "w4a8_mx" or w4a8_repacked)
        and I._dynamic_direct_routing_selected(
            route_mode="direct" if planned_direct_routing else "grouped",
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=E,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
    )
    mac_backend = "dynamic_w4a8_decode" if decode_regime else "dynamic"
    effective_mac = I._get_impl_mac(mac_backend, routed_rows=routed_rows)
    multicta_enabled = I._dynamic_multicta_enabled()
    selected_tile_m = I._select_dynamic_tile_mn(
        m * num_topk,
        n,
        quant_mode,
        num_experts=E,
        activation=activation,
        planned_tile_m=planned_tile_m,
    )[0]
    materialize_intermediate = I._w4a8_dynamic_materialized_enabled(
        quant_mode=quant_mode,
        activation=activation,
        num_tokens=planned_num_tokens,
        routed_rows=planned_num_tokens * num_topk,
        num_experts=E,
        k=k,
        n=n,
        w4a8_repacked=w4a8_repacked,
        share_input_across_experts=share_input_across_experts,
        deterministic_output=deterministic_output,
        planned_tile_m=planned_tile_m,
    ) or (quant_mode == "nvfp4" and nvfp4_materialize_intermediate)
    external_route_plan_supported = I._dynamic_external_route_plan_supported(
        quant_mode=quant_mode,
        activation=activation,
        routed_rows=routed_rows,
        planned_tile_m=selected_tile_m,
        dynamic_route_mode=("direct" if direct_routing else "grouped"),
        deterministic_output=deterministic_output,
        w4a8_n64_repacked=w4a8_n64_repacked,
    )
    external_route_plan = bool(
        external_route_plan_supported
        and I._env_flag(
            I._DYNAMIC_EXTERNAL_ROUTE_PLAN_ENV,
            default=external_route_plan_requested,
        )
    )
    materialize_intermediate = materialize_intermediate or w4a8_n64_repacked
    if materialize_intermediate:
        required_intermediate_bytes = I._dynamic_materialized_intermediate_bytes(
            quant_mode=quant_mode,
            rows_padded=physical_tiles_capacity * selected_tile_m,
            n=n,
        )
        available_intermediate_bytes = (
            materialized_intermediate.numel() * materialized_intermediate.element_size()
        )
        if available_intermediate_bytes < required_intermediate_bytes:
            raise ValueError(
                f"dynamic {quant_mode} materialized scratch exceeds preplanned "
                f"capacity: need {required_intermediate_bytes} bytes, have "
                f"{available_intermediate_bytes}; physical_tiles_capacity="
                f"{physical_tiles_capacity}, selected_tile_m={selected_tile_m}, "
                f"n={n}, routed_rows={routed_rows}, m={m}, num_topk={num_topk}"
            )
    if not multicta_enabled:
        effective_mac = 1
    elif w4a8_n64_repacked and selected_tile_m == 16:
        effective_mac = min(effective_mac, I.get_num_sm(torch.device("cuda")))
    elif w4a8_repacked and selected_tile_m <= 32:
        effective_mac = min(
            effective_mac * 2,
            I.get_num_sm(torch.device("cuda")) * 2,
        )
        if (
            I._current_compute_capability() == (12, 1)
            and E == 256
            and k == 6144
            and n == 1024
            and 24 <= routed_rows <= 48
        ):
            effective_mac = min(effective_mac, 24)
    if direct_routing:
        direct_task_count = routed_rows * max(1, (n + 127) // 128)
        if materialize_intermediate and m == 1:
            direct_task_count = max(
                direct_task_count,
                routed_rows * max(1, (k + 255) // 256),
            )
        effective_mac = min(effective_mac, direct_task_count)
    if (
        (external_route_plan or w4a8_n64_repacked
         or (w4a8_repacked and n % 128 == 0 and selected_tile_m <= 32))
        and policy_max_active_clusters > 0
        and I._first_env(
            f"B12X_NEXT_{mac_backend.upper()}_MAX_ACTIVE_CLUSTERS",
            "B12X_NEXT_DYNAMIC_MAX_ACTIVE_CLUSTERS",
            "B12X_NEXT_LEVEL10_MAX_ACTIVE_CLUSTERS",
        )
        is None
    ):
        effective_mac = min(
            effective_mac,
            policy_max_active_clusters,
        )
    compiled, mac = _ORIG["_get_dynamic_kernel"](
        E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        topk_ids_dtype=torch.int32 if topk_ids_are_i32 else torch.int64,
        fast_math=fast_math,
        mac_override=effective_mac,
        activation=activation,
        quant_mode=quant_mode,
        w4a8_repacked=w4a8_repacked,
        w4a8_n64_repacked=w4a8_n64_repacked,
        w4a8_coded_scales=w4a8_coded_scales,
        nvfp4_materialize_intermediate=nvfp4_materialize_intermediate,
        direct_routing=direct_routing,
        external_route_plan=external_route_plan,
        share_input_across_experts=share_input_across_experts,
        deterministic_output=deterministic_output,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        trellis_bits=trellis_bits,
        trellis_intermediate_hadamard=trellis_intermediate_hadamard,
        planned_tile_m=planned_tile_m,
        planned_num_tokens=planned_num_tokens,
    )
    # ---- end of the mirrored decisions; the rest is what the stock tail derives from them ----
    p = Plan()
    p.compiled, p.mac = compiled, mac
    p.kernel_key = next((key for key, value in I._DYNAMIC_KERNEL_CACHE.items() if value is compiled), None)
    p.zero_barrier = bool(volatile_launch_state and not external_route_plan)
    p.external_route_plan = external_route_plan
    p.selected_tile_m = selected_tile_m
    p.block_e = 1 << (int(E) - 1).bit_length()
    p.routed_rows, p.E, p.scatter_rows = routed_rows, E, scatter_rows
    p.tile_rows = physical_tiles_capacity * I._select_dynamic_tile_mn(
        m * num_topk,
        n,
        quant_mode,
        num_experts=E,
        activation=activation,
        planned_tile_m=planned_tile_m,
    )[0]
    p.task_capacity, p.physical_tiles_capacity, p.m, p.max_rows = (task_capacity, physical_tiles_capacity, m,
                                                                   max_rows)
    cutlass = I.cutlass
    p.is_w4a8 = I._is_w4a8_quant_mode(quant_mode)
    p.ids_dtype = cutlass.Int32 if topk_ids_are_i32 else cutlass.Int64
    p.ids_align = 4 if topk_ids_are_i32 else 8
    is_w6a8 = quant_mode in I._W6A8_QUANT_MODES
    p.packed_dtype = cutlass.Float8E4M3FN if is_w6a8 else cutlass.Float4E2M1FN
    p.sf_dtype = cutlass.Float8E8M0FNU if is_w6a8 else cutlass.Float8E4M3FN
    return p


def _ptr(dtype, t, align=16):
    key = (t.data_ptr(), dtype, align)
    p = _PTRS.get(key)
    if p is None:
        if len(_PTRS) >= PTR_CACHE_MAX:
            _PTRS.clear()
        p = _PTRS[key] = _I.make_ptr(dtype, key[0], _I.cute.AddressSpace.gmem, assumed_align=align)
    return p


def _launch(p, kw):
    """The stock tail of _launch_dynamic_flat for a plan: same side effects, same order, this call's tensors."""
    I = _I
    if p.zero_barrier:
        I._zero_barrier_state(kw["barrier_count"], kw["barrier_epoch"])
    if p.external_route_plan:
        I.launch_triton(
            I._dynamic_route_plan_kernel,
            (1,),
            kw["flat_ids"],
            kw["row_counts"],
            kw["expert_tile_base"],
            kw["barrier_count"],
            kw["barrier_epoch"],
            p.routed_rows,
            NUM_EXPERTS=p.E,
            TILE_M=p.selected_tile_m,
            BLOCK_E=p.block_e,
            BLOCK_ROUTES=I._DYNAMIC_EXTERNAL_ROUTE_PLAN_MAX_ROWS,
            num_warps=8,
        )
    cutlass = I.cutlass
    sf = p.sf_dtype
    sfb_w13_mx = kw["sfb_w13_mx"]
    args = [
        _ptr(cutlass.BFloat16, kw["a"]),
        _ptr(p.ids_dtype, kw["flat_ids"], p.ids_align),
        _ptr(cutlass.Float32, kw["flat_weights"], 4),
        _ptr(p.packed_dtype, kw["packed_a_view"]),
        _ptr(sf, kw["scale_flat"]),
        _ptr(cutlass.Uint8, kw["packed_a_flat"]),
        _ptr(cutlass.Uint8, kw["scale_flat"]),
        _ptr(cutlass.Uint32, kw["materialized_intermediate"]),
        kw["barrier_count"],
        kw["barrier_epoch"],
        kw["pair_head"],
        kw["producers_done_count"],
        kw["all_work_published"],
        kw["task_head"],
        kw["task_tail"],
        _ptr(cutlass.Int32, kw["task_ready"], 4),
        _ptr(cutlass.Int32, kw["task_expert"], 4),
        _ptr(cutlass.Int32, kw["task_m_tile"], 4),
        _ptr(cutlass.Int32, kw["task_slice_begin"], 4),
        _ptr(cutlass.Int32, kw["task_slice_count"], 4),
        _ptr(cutlass.Int32, kw["task_valid_rows"], 4),
        _ptr(cutlass.Int32, kw["tile_write_count"], 4),
        kw["w13_fp4"],
        _ptr(sf, kw["w13_sf"]),
    ]
    if not p.is_w4a8:
        args.append(_ptr(sf, sfb_w13_mx))
    args += [kw["down_fp4"], _ptr(sf, kw["down_sf"])]
    if p.is_w4a8:
        args += [
            _ptr(cutlass.Uint8, sfb_w13_mx),
            _ptr(cutlass.Uint8, kw["sfb_down_mx"]),
            _ptr(cutlass.Uint8, kw["w13_residual"]),
            _ptr(cutlass.Uint8, kw["down_residual"]),
            _ptr(cutlass.Uint32, kw["w13_rp"]),
            _ptr(cutlass.Uint32, kw["w13_sfb_rp"]),
            _ptr(cutlass.Uint32, kw["down_rp"]),
            _ptr(cutlass.Uint32, kw["down_sfb_rp"]),
        ]
    args += [
        kw["row_counts"],
        kw["expert_write_rows"],
        kw["expert_tile_base"],
        kw["input_gs"],
        kw["w1_alpha"],
        kw["w2_alpha"],
        kw["down_input_scale"],
        _ptr(cutlass.BFloat16, kw["scatter_output"]),
        _ptr(cutlass.Int32, kw["token_map"], 4),
        _ptr(cutlass.Float32, kw["token_weights"], 4),
        p.m,
        p.max_rows,
        p.scatter_rows,
        p.tile_rows,
        p.task_capacity,
        p.physical_tiles_capacity,
        p.mac,
        I.current_cuda_stream(),
    ]
    if _STATE["timing"]:
        t0 = time.perf_counter_ns()
        p.compiled(*args)
        _record("compiled", p.m, time.perf_counter_ns() - t0)
    else:
        p.compiled(*args)


def _eligible(kw):
    return (kw["w13_sfb_rp"].dtype is not torch.float16 and not _capturing() and _plain_context(_I))


def _evict_first(I):
    """b12x_next evict_first_weights() state (False on a runtime without the evict-first patch)."""
    fn = getattr(I, "_evict_first_weights_requested", None)
    return bool(fn()) if fn is not None else False


def _key(kw):
    I = _I
    mi = kw["materialized_intermediate"]
    try:
        dev = torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        dev = -1
    return tuple(kw[s] for s in SCALARS) + (
        mi.numel() * mi.element_size(), dev,
        kw["w13_sfb_rp"].dtype == torch.uint8, kw["down_sfb_rp"].dtype == torch.uint8,    # ds41 scale codes
        I._prequantized_input_requested(), _evict_first(I),                                # ds41 evict-first
        I._DYNAMIC_TILE_MN_OVERRIDE, I._DYNAMIC_SWAP_AB_OVERRIDE)


def prepared_launch_dynamic_flat(**kw):
    """Drop-in for _impl._launch_dynamic_flat (same keyword-only signature)."""
    if mode() == "off" or not _eligible(kw):
        _STATE["stock"] += 1
        return _ORIG["_launch_dynamic_flat"](**kw)
    st = _STATE
    st["calls"] += 1
    if st["env_gen"] != _ENV_GEN[0] or st["env_obj"] is not os.environ:    # exact, on every call
        _PLANS.clear()
        st["env_gen"], st["env_obj"] = _ENV_GEN[0], os.environ
    key = _key(kw)
    p = _PLANS.get(key)
    if p is not None and (p.kernel_key is None or _I._DYNAMIC_KERNEL_CACHE.get(p.kernel_key) is not p.compiled):
        p = None
    if p is None:
        p = _decide(kw)
        if p.kernel_key is None:          # kernel cache disabled or bypassed: no plan, stock tail semantics
            _PLANS.pop(key, None)
        else:
            if len(_PLANS) >= PLAN_CACHE_MAX:
                _PLANS.clear()
            _PLANS[key] = p
        st["builds"] += 1
    else:
        st["hits"] += 1
    _launch(p, kw)


# ------------------------------------------------------------------------------------------------------------------
# plan,direct: the op's Python body without the dispatcher, eager only
# ------------------------------------------------------------------------------------------------------------------
class DirectOp:
    """Stands in for torch.ops.b12x_next.tp_moe_dynamic_launch (an OpOverloadPacket): eager calls in
    plan,direct mode go to the op's Python body; anything else (capture, compile, attribute access such as
    .default) goes to the packet."""

    def __init__(self, packet, body):
        self._packet, self._body = packet, body

    def __call__(self, *args, **kwargs):
        if mode() == "plan,direct" and not _capturing() and not _compiling():
            return self._body(*args, **kwargs)
        return self._packet(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._packet, name)


def _compiling():
    try:
        return torch.compiler.is_compiling()
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------------------------
# timing
# ------------------------------------------------------------------------------------------------------------------
# Records are keyed (A/B switch sequence, variant, stage, row bucket). The in-boot A/B advances its switch sequence
# on every rank in the same scheduler iteration, so seq 0 is the boot (load warm-ups, graph-capture warm-ups:
# excluded by the summarizer) and every later seq is one measured block of one variant. A process prints the
# current seq's cumulative stats every DSV41_MOE_LAUNCH_TIMING_EVERY op calls and the final stats of a seq at the
# first record of the next one (the switch's poke request runs a prefill).
_STATS = {}
_RESERVOIR = 256
_TIMING = {"every": 512, "op_calls": 0, "self_ns": 0.0, "rank": None, "emits": 0, "seq": None, "in_body": 0}


def bucket(m):
    m = int(m)
    return m if m <= 128 else 1 << (m - 1).bit_length()


def _seq():
    return int(_ab._state.get("seq", 0)) if _ab is not None and getattr(_ab, "ACTIVE", False) else 0


def _record(stage, m, dt):
    if stage == "_calibration":            # the install-time self-cost loop: its own key, no seq bookkeeping
        seq = -1
    else:
        seq = _seq()
        if seq != _TIMING["seq"]:
            old = _TIMING["seq"]
            if old is not None and any(k[0] == old for k in _STATS):
                emit()
                for key in [k for k in _STATS if k[0] == old]:
                    del _STATS[key]
            _TIMING["seq"] = seq
    key = (seq, _variant(), stage, bucket(m))
    s = _STATS.get(key)
    if s is None:
        s = _STATS[key] = [0, 0, None, 0, []]
    s[0] += 1
    s[1] += dt
    s[2] = dt if s[2] is None or dt < s[2] else s[2]
    s[3] = max(s[3], dt)
    r = s[4]
    if len(r) < _RESERVOIR:
        r.append(dt)
    else:
        r[s[0] % _RESERVOIR] = dt


def summary(seq=None):
    """{"v<variant>": {bucket: {stage: {n, mean_us, p50_us, p90_us, min_us, max_us}}}} of one seq (default current)."""
    seq = _TIMING["seq"] if seq is None else seq
    out = {}
    for (q_seq, v, stage, b), (n, tot, mn, mx, r) in sorted(_STATS.items()):
        if q_seq != seq or stage == "_calibration":
            continue
        q = sorted(r)
        out.setdefault(f"v{v}", {}).setdefault(str(b), {})[stage] = {
            "n": n, "mean_us": round(tot / n / 1e3, 2), "p50_us": round(q[len(q) // 2] / 1e3, 2),
            "p90_us": round(q[min(len(q) - 1, int(len(q) * 0.9))] / 1e3, 2), "min_us": round(mn / 1e3, 2),
            "max_us": round(mx / 1e3, 2)}
    return out


def _rank():
    if _TIMING["rank"] is None:
        try:
            import torch.distributed as dist
            _TIMING["rank"] = dist.get_rank() if dist.is_initialized() else -1
        except Exception:  # noqa: BLE001
            _TIMING["rank"] = -1
    return _TIMING["rank"]


def emit():
    doc = {"rank": _rank(), "pid": os.getpid(), "seq": _TIMING["seq"], "self_ns_per_record": round(_TIMING["self_ns"], 1),
           "prepared": {k: _STATE[k] for k in ("hits", "builds", "stock")}, "plans": len(_PLANS),
           "stages": summary()}
    print("DSV41_MOE_LAUNCH_TIMING " + json.dumps(doc, sort_keys=True, separators=(",", ":")), flush=True)
    _TIMING["emits"] += 1
    return doc


def _timed(stage, fn, m_of):
    def wrapper(*args, **kwargs):
        if _capturing():
            return fn(*args, **kwargs)
        if stage == "body":
            _TIMING["in_body"] += 1
        t0 = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter_ns() - t0
            if stage == "body":
                _TIMING["in_body"] -= 1
            _record(stage, m_of(args, kwargs), dt)
            if stage == "op":
                _TIMING["op_calls"] += 1
                if _TIMING["op_calls"] % _TIMING["every"] == 0:
                    emit()
    wrapper.__wrapped__ = fn
    wrapper._dsv41_timed = stage
    return wrapper


class TimedCompiled:
    """Transparent proxy of a compiled kernel the stock body got from _get_dynamic_kernel: times the call."""

    def __init__(self, inner, m):
        self._inner, self._m = inner, m

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        try:
            return self._inner(*args, **kwargs)
        finally:
            _record("compiled", self._m, time.perf_counter_ns() - t0)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _calibrate(n=2000):
    t0 = time.perf_counter_ns()
    for _ in range(n):
        t = time.perf_counter_ns()
        _record("_calibration", 1, time.perf_counter_ns() - t)
    per = (time.perf_counter_ns() - t0) / n
    for key in [k for k in _STATS if k[2] == "_calibration"]:
        del _STATS[key]
    return per


def _install_timing(I, adapter_module):
    every = os.environ.get("DSV41_MOE_LAUNCH_TIMING_EVERY", "").strip()
    _TIMING["every"] = int(every) if every else 512
    _STATE["timing"] = True

    def get_kernel(*args, **kwargs):
        compiled, mac = _ORIG["_get_dynamic_kernel"](*args, **kwargs)
        if _TIMING["in_body"] and not _capturing():     # only the stock launch body's lookup (not preparation)
            compiled = TimedCompiled(compiled, args[1] if len(args) > 1 else kwargs.get("m", 0))
        return compiled, mac

    I._get_dynamic_kernel = _timed("getk", get_kernel, lambda a, k: a[1] if len(a) > 1 else k.get("m", 0))
    I._launch_dynamic_flat = _timed("body", I._launch_dynamic_flat, lambda a, k: k["m"])
    I._launch_dynamic = _timed("op", I._launch_dynamic, lambda a, k: k["m"])
    geo = getattr(adapter_module, "_Geometry", None) if adapter_module is not None else None
    if geo is not None and hasattr(geo, "_run"):
        geo._run = _timed("adapter", geo._run, lambda a, k: a[2].shape[0] if len(a) > 2 else k["x"].shape[0])
    _TIMING["self_ns"] = _calibrate()


# ------------------------------------------------------------------------------------------------------------------
# forward timing (DSV41_MOE_LAUNCH_TIMING): is an eager prefill host-paced or GPU-paced, without a profiler?
# ------------------------------------------------------------------------------------------------------------------
# ModelRunner._forward_raw of every EAGER forward (no decode graph replay; extend / draft extend / idle) records
#   fwd_host_<kind>  perf_counter of the call (host issue time, plus every host sync inside it)
#   fwd_gpu_<kind>   CUDA-event span from the call's start to its end on the current stream, read without a sync
#                    (events are queried on later forwards; a pending pair is never waited for)
# per seq / variant / row bucket; <kind> = "t" target or "d" draft + the forward mode name. host >= gpu means the GPU
# waited for the host (host-paced); host << gpu means the launch queue ran ahead (GPU-paced).
_PENDING = []
_PENDING_MAX = 64


def _drain_events():
    keep = []
    for start, end, stage, rows, seq_variant in _PENDING:
        try:
            done = end.query()
        except Exception:  # noqa: BLE001
            continue
        if done:
            ms = start.elapsed_time(end)
            _record_at(seq_variant, stage, rows, int(ms * 1e6))
        else:
            keep.append((start, end, stage, rows, seq_variant))
    _PENDING[:] = keep[-_PENDING_MAX:]


def _record_at(seq_variant, stage, m, dt):
    """Record into an explicit (seq, variant): a GPU span read after the A/B switched still counts for its own."""
    seq, v = seq_variant
    if not any(k[0] == seq for k in _STATS) and seq != _TIMING["seq"]:
        return                                   # that seq was already emitted and dropped
    key = (seq, v, stage, bucket(m))
    s = _STATS.get(key)
    if s is None:
        s = _STATS[key] = [0, 0, None, 0, []]
    s[0] += 1
    s[1] += dt
    s[2] = dt if s[2] is None or dt < s[2] else s[2]
    s[3] = max(s[3], dt)
    if len(s[4]) < _RESERVOIR:
        s[4].append(dt)
    else:
        s[4][s[0] % _RESERVOIR] = dt


def install_forward_timing(module):
    """Hook for sglang.srt.model_executor.model_runner (sitecustomize, gated on DSV41_MOE_LAUNCH_TIMING)."""
    cls = getattr(module, "ModelRunner", None)
    if cls is None or not hasattr(cls, "_forward_raw"):
        raise RuntimeError(f"{_TAG} ModelRunner._forward_raw not found (engine drift); unset DSV41_MOE_LAUNCH_TIMING")
    if getattr(cls._forward_raw, "_dsv41_timed", None):
        return False
    orig = cls._forward_raw

    def _forward_raw(self, forward_batch, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        graph = False
        try:
            graph = bool(mode.is_cuda_graph() and self.decode_cuda_graph_runner
                         and self.decode_cuda_graph_runner.can_run_graph(forward_batch))
        except Exception:  # noqa: BLE001
            graph = False
        if graph or _capturing():
            return orig(self, forward_batch, *args, **kwargs)
        kind = ("d_" if getattr(self, "is_draft_worker", False) else "t_") + str(getattr(mode, "name", mode)).lower()
        ids = getattr(forward_batch, "input_ids", None)
        rows = int(ids.shape[0]) if ids is not None and hasattr(ids, "shape") else 0
        _drain_events()
        start = end = None
        if torch.cuda.is_available():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        sv = (_seq(), _variant())
        t0 = time.perf_counter_ns()
        try:
            return orig(self, forward_batch, *args, **kwargs)
        finally:
            dt = time.perf_counter_ns() - t0
            _record("fwd_host_" + kind, rows, dt)
            if end is not None:
                end.record()
                _PENDING.append((start, end, "fwd_gpu_" + kind, rows, sv))

    _forward_raw._dsv41_timed = "forward"
    _forward_raw.__wrapped__ = orig
    cls._forward_raw = _forward_raw
    print(f"{_TAG} forward timing armed on ModelRunner._forward_raw (eager forwards only)", flush=True)
    return True


# ------------------------------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------------------------------
def check_sources(I):
    """{name: actual sha} of the pinned stock sources that differ (empty = ok)."""
    fns = {"_launch_dynamic_flat": I._launch_dynamic_flat, "_launch_dynamic": I._launch_dynamic,
           "_tp_moe_dynamic_launch_op": getattr(I._tp_moe_dynamic_launch_op, "_init_fn", None)}
    bad = {}
    for name, fn in fns.items():
        fn = getattr(fn, "__wrapped__", fn)
        got = source_sha(fn) if fn is not None else "missing"
        if got != PINNED_SOURCES[name]:
            bad[name] = got
    return bad


def _direct_wanted(environ):
    key = "DSV41_MOE_LAUNCH_PREPARED"
    if _ab is not None and getattr(_ab, "ACTIVE", False) and key in getattr(_ab, "KNOWN", ()):
        return any(parse_mode(_ab.env_for(i, key, "0")) == "plan,direct" for i in range(_ab.N))
    return parse_mode(environ.get(key, "0")) == "plan,direct"


def install(I, adapter_module=None, environ=None):
    """Arm per DSV41_MOE_LAUNCH_PREPARED / DSV41_MOE_LAUNCH_TIMING (both off: nothing). Idempotent."""
    global _I
    environ = os.environ if environ is None else environ
    want_prep = parse_mode(environ.get("DSV41_MOE_LAUNCH_PREPARED", "0")) != "off"
    want_timing = str(environ.get("DSV41_MOE_LAUNCH_TIMING", "0")).strip().lower() not in _OFF
    if not (want_prep or want_timing) or _STATE["installed"]:
        return False
    if I is None:
        raise RuntimeError(f"{_TAG} b12x_next fused_moe._impl is not importable; refusing to arm")
    bad = check_sources(I)
    if bad:
        raise RuntimeError(f"{_TAG} b12x_next stock sources drifted from the pinned sha256 {bad}; refusing to arm "
                           f"(unset DSV41_MOE_LAUNCH_PREPARED / DSV41_MOE_LAUNCH_TIMING)")
    _I = I
    _ORIG.update(_launch_dynamic_flat=I._launch_dynamic_flat, _get_dynamic_kernel=I._get_dynamic_kernel,
                 _launch_dynamic=I._launch_dynamic)
    if want_prep:
        _STATE["mode"] = parse_mode(environ.get("DSV41_MOE_LAUNCH_PREPARED", "0"))
        if not track_environ():
            raise RuntimeError(f"{_TAG} os.environ cannot be tracked for B12X_NEXT_* changes; refusing to arm")
        I._launch_dynamic_flat = prepared_launch_dynamic_flat
        if _direct_wanted(environ):     # the dispatcher stand-in only when some variant runs plan,direct
            body = getattr(I._tp_moe_dynamic_launch_op, "_init_fn", None)
            if body is None:
                raise RuntimeError(f"{_TAG} custom op has no _init_fn (torch drift); refusing to arm")
            ns = torch.ops.b12x_next
            packet = getattr(ns, "tp_moe_dynamic_launch")
            if not isinstance(packet, DirectOp):
                setattr(ns, "tp_moe_dynamic_launch", DirectOp(packet, body))
            _STATE["direct"] = True
    if want_timing:
        _install_timing(I, adapter_module)
    _STATE["installed"] = True
    ab = _ab is not None and getattr(_ab, "ACTIVE", False)
    print(f"{_TAG} armed: prepared={'per-variant (in-boot A/B)' if ab and want_prep else _STATE['mode']} "
          f"direct-op={'installed' if _STATE['direct'] else 'no'} "
          f"timing={'on (self %.0f ns/record, every %d op calls)' % (_TIMING['self_ns'], _TIMING['every']) if want_timing else 'off'} "
          f"sources pinned ok", flush=True)
    return True
