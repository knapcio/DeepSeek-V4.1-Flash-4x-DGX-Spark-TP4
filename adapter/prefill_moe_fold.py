"""Prefill MoE: the shared-expert add folded into b12x's top-k sum, bit-identical or not at all (R1).

DSV41_PREFILL_MOE_FOLD=1 (default 0 = off). In production every routed-MoE layer of a prefill chunk ends with two
row-proportional DRAM passes over [M, 5120] bf16 (trace ced-t0, M = 8192, one full-row layer):

  W4A16TopKSumKernel   routed[i] = bf16(sum_k route[i, k])           503 MB in + 84 MB out   2.29 ms
  CUDAFunctor_add      shared[i] = bf16(float(shared[i]) + float(routed[i]))  168 in + 84 out  1.08 ms

(``DeepseekV2MoE.forward_normal`` -> ``maybe_fuse_routed_scale_and_shared_add`` -> ``shared.add_(routed, alpha=1.0)``:
the routed scale is already folded into the top-k weights, ``should_fuse_routed_scaling_factor_in_topk``). Here the
top-k sum reads ``shared[i]`` and writes the sum into ``shared`` in place:

  shared[i] = bf16(float(shared[i]) + float(bf16(sum_k route[i, k])))

which is the same value bit for bit: the routed sum is accumulated in the same order and rounded to bf16 exactly
as the stock kernel stores it (the same instruction), both bf16 -> fp32 widenings are exact, and the binary32 add
is ``add.rn.f32`` without .ftz, i.e. torch's opmath add of a bf16 ``add_`` at alpha 1 (no fast math; fp32 addition
commutes), followed by the same round-to-nearest-even bf16 conversion torch's CUDA kernel uses (``__float2bfloat16``
= ``cvt.rn.bf16.f32``). -168 MB of DRAM traffic and one launch per layer, +84 MB read inside the top-k sum.
The kernel side is the runtime patch scripts/b12x_next-topk-sum-accumulate.patch (b12x_next's
``topk_sum_accumulate()`` context; a launch inside it that would not end in the dynamic deterministic route-buffer
top-k sum raises before anything runs).

Where it applies (all decided from shapes, static config and the per-call gate, so identically on every TP rank):
a ``DeepseekV2MoE.forward_normal`` call with >= DSV41_PREFILL_MOE_FOLD_MIN_ROWS (2048) rows, outside CUDA-graph
capture / torch.compile, on a layer whose routed experts run on moe_b12x_next with
DSV41_MOE_B12X_NEXT_DETERMINISTIC=1, shared experts computed BEFORE the routed experts (runner ``inplace``, so
``defer_shared`` is False), not replicated (``_shared_expert_tp1``), no SBO fusion, routed scale folded into
top-k. Decode / verify / draft never get here (graphs, <= 2048 rows). Prefill SP stage 2b (pre-quantized input) and
its per-size self-check, the GEMM tactic and CED are unaffected: the fold sits after the routed experts' last GEMM
and before the post-experts reduction.

Checked, not assumed: the first eligible call of every b12x plan capacity runs the routed experts the stock way
(into their own buffer) AND the fold (into a clone of the shared output), compares ``clone.add_(routed,
alpha=1.0)`` with the fold bit for bit, and the TP ranks agree (MIN over the TP group) before any fold is used;
that call returns the stock result. A difference, an exception, a drifted engine source (sha256 pins below) or a
broken return chain keeps the stock add: fail closed. At runtime a fold whose result does not come back to the add
site as the same storage raises instead of adding twice.

Read per call through adapter/ab_variant.py when the in-boot A/B is armed (a RUNTIME key), else once. Installed from
sitecustomize.py on sglang.srt.models.deepseek_v2 (forward_normal, _forward_shared_experts, the module's
maybe_fuse_routed_scale_and_shared_add); every wrapper sets __wrapped__. moe_b12x_next calls claim() / run().
"""
import hashlib
import inspect
import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)

GATE = "DSV41_PREFILL_MOE_FOLD"
MIN_ROWS_ENV = "DSV41_PREFILL_MOE_FOLD_MIN_ROWS"
TAG = "DSV41_PREFILL_MOE_FOLD"
_OFF = ("", "0", "off", "false", "no")
RECEIPT_EVERY = 4096        # `<TAG> calls rank <r>: ...` at the first fold and every 4096th call after it
RECEIPT_EVERY_ENV = "DSV41_PREFILL_MOE_FOLD_RECEIPT_EVERY"   # test windows only (per-chunk receipts); read at install

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:  # CPU tests without the adapter dir
    _ab = None

# sha256[:16] of inspect.getsource(inspect.unwrap(obj)) of the engine sources the fold's correctness rests on (sglang
# image of v2.7.3, nightly-dev-20261027-5884e4fe + the ds41 patch train). forward_normal: shared experts before the
# routed experts when the runner is in place, then maybe_fuse...(experts, routed, shared, scale), nothing between;
# the maybe_fuse body: shared.add_(routed, alpha=1.0) for the fused-scale MXFP4 methods; the routed return chain
# (FusedMoE.forward / forward_impl, MoeRunner.run, the MXFP4 CUTLASS method's apply, StandardDispatcher.combine):
# the fused function's output tensor comes back as the same storage. Same scheme as prefill_sp._src_hash.
PINS = {
    "v2.DeepseekV2MoE.forward_normal": "9da361af16dafbec",
    "v2.DeepseekV2MoE._forward_shared_experts": "e90ba60e4d5c3d10",
    "trt.maybe_fuse_routed_scale_and_shared_add": "f43487d273bf0560",
    "fmoe.FusedMoE.forward": "978c59903bdb0333",
    "fmoe.FusedMoE.forward_impl": "2eac8aedf987c339",
    "runner.MoeRunner.run": "9048525d78790cc3",
    "std.StandardDispatcher.combine": "71ea935d67603092",
}
# checked at the first eligible call instead of at install: the deepseek_v2 hook must not import the MXFP4 CUTLASS
# module early (its own sitecustomize hook installs moe_b12x_next when it is first imported)
LAZY_PINS = {
    "cut.Mxfp4FlashinferCutlassMoEMethod.apply": "d4e868319c346691",
}
_MODS = {
    "v2": "sglang.srt.models.deepseek_v2",
    "trt": "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe",
    "fmoe": "sglang.srt.layers.moe.fused_moe_triton.layer",
    "runner": "sglang.srt.layers.moe.moe_runner.runner",
    "cut": "sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe",
    "std": "sglang.srt.layers.moe.token_dispatcher.standard",
}

_VERDICT = {}            # b12x plan capacity -> bool, agreed over TP by the first eligible call of that capacity
STATE = {"installed": False, "reason": "not installed", "chain_ok": True, "lazy_pins": None}
STATS = {"calls": 0, "folds": 0, "checks": 0, "stock": 0, "errors": 0}
AGREE = [True]           # CPU tests switch the TP agreement off
_LOGGED = set()


# ---------------------------------------------------------------------------------------------------------- config --

def gate_on(raw):
    v = str(raw if raw is not None else "0").strip().lower()
    if v in _OFF:
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    raise ValueError(f"{GATE}={raw!r}: expected 0 / 1")


def _ab_active():
    """The in-boot A/B is armed and one of its variants names the gate (else: base env, read once)."""
    if _ab is None or not getattr(_ab, "ACTIVE", False) or GATE not in getattr(_ab, "KNOWN", ()):
        return False
    return any(GATE in spec for spec in getattr(_ab, "_specs", ()) or ())


_ENABLED = [None]


def enabled():
    """Per call under an armed in-boot A/B that names the gate, else read once from the base env. A value that does
    not parse is OFF, never an exception inside a forward."""
    if _ab_active():
        try:
            return gate_on(_ab.env(GATE, "0"))
        except ValueError:
            return False
    if _ENABLED[0] is None:
        try:
            _ENABLED[0] = gate_on(os.environ.get(GATE, "0"))
        except ValueError:
            _ENABLED[0] = False
    return _ENABLED[0]


def wanted_at_install():
    """Install the hooks when the base env enables the gate or the in-boot A/B can switch it per call."""
    if _ab_active():
        return True
    try:
        return gate_on(os.environ.get(GATE, "0"))
    except ValueError:
        return False


def min_rows():
    try:
        return max(1, int(os.environ.get(MIN_ROWS_ENV, "2048") or 2048))
    except ValueError:
        return 2048


def receipt_every():
    """The counter-line cadence: DSV41_PREFILL_MOE_FOLD_RECEIPT_EVERY (a window sets 20 = one line per 8192-row chunk per
    rank, attributable per in-boot A/B variant), else 4096. Logging only; a bad value keeps the default."""
    try:
        return max(1, int(os.environ.get(RECEIPT_EVERY_ENV, "") or 4096))
    except ValueError:
        return 4096


# ------------------------------------------------------------------------------------------------------------ pins --

def src_hash(obj):
    return hashlib.sha256(inspect.getsource(inspect.unwrap(obj)).encode()).hexdigest()[:16]


def _resolve(name):
    import importlib
    mod, _, path = name.partition(".")
    obj = importlib.import_module(_MODS[mod])
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def check_pins(pins=None):
    """(ok, {name: (expected, found)} of the drifted ones)."""
    bad = {}
    for name, want in (pins or PINS).items():
        try:
            got = src_hash(_resolve(name))
        except Exception as exc:  # noqa: BLE001
            got = f"unresolved ({exc!r})"[:120]
        if got != want:
            bad[name] = (want, got)
    return not bad, bad


# ------------------------------------------------------------------------------------------------------- the frame --

class _Frame:
    __slots__ = ("moe", "rows", "shared", "claimed", "folded", "check_out", "done")

    def __init__(self, moe, rows):
        self.moe = moe
        self.rows = rows
        self.shared = None          # the shared experts' output, recorded before the routed experts run
        self.claimed = False        # the routed experts started (no shared output is recorded after this)
        self.folded = None          # the shared output the routed sum was accumulated into
        self.check_out = None       # the self-check call's stock routed output (return-chain check)
        self.done = False           # the add site saw the folded tensor


class _Tls(threading.local):
    def __init__(self):
        self.stack = []


_tls = _Tls()


def _top():
    return _tls.stack[-1] if _tls.stack else None


def _capturing():
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:  # noqa: BLE001
        pass
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def static_reason(moe, v2=None, b12x_layer=None):
    """'' when this DeepseekV2MoE's shared add can fold (shape-free part; cached on the module), else why not."""
    cached = getattr(moe, "_dsv41_moe_fold_static", None)
    if cached is not None:
        return cached
    why = ""
    try:
        experts = moe.experts
        cfg = getattr(experts, "moe_runner_config", None)
        if getattr(moe, "_shared_expert_tp1", False):
            why = "replicated (TP1) shared experts"
        elif getattr(moe, "_fuse_shared_experts_inside_sbo", False):
            why = "SBO shared-expert fusion"
        elif getattr(moe, "num_fused_shared_experts", 0) != 0 or not hasattr(moe, "shared_experts"):
            why = "no separate shared experts"
        elif getattr(moe, "_enable_a2a_moe", False):
            why = "a2a MoE"
        elif cfg is None or not getattr(cfg, "inplace", False):
            why = "runner not in place (shared experts deferred after the routed experts)"
        elif not getattr(experts, "should_fuse_routed_scaling_factor_in_topk", False):
            why = "routed scale not folded into top-k (add alpha != 1)"
        elif getattr(experts, "reduce_results", False):
            why = "routed experts reduce their own results"
        elif getattr(experts, "_dsv41_b12x_next", None) is None:
            why = "routed experts not on moe_b12x_next"
        else:
            import moe_b12x_next as mbn
            if not mbn.DETERMINISTIC:
                why = "DSV41_MOE_B12X_NEXT_DETERMINISTIC is off (atomic reduction, no top-k sum)"
            elif v2 is not None and not getattr(v2, "_is_cuda", False):
                why = "not CUDA"
            else:
                qm = getattr(experts, "quant_method", None)
                cut = _cutlass_method()
                if cut is None or not isinstance(qm, cut):
                    why = f"quant method {type(qm).__name__} is not the MXFP4 CUTLASS method"
    except Exception as exc:  # noqa: BLE001
        why = f"inspection failed ({exc!r})"[:200]
    try:
        moe._dsv41_moe_fold_static = why
    except Exception:  # noqa: BLE001
        pass
    return why


def _cutlass_method():
    """The MXFP4 CUTLASS MoE method class if its module is loaded (never imports it), after its lazy pin check."""
    import sys
    mod = sys.modules.get(_MODS["cut"])
    if mod is None:
        return None
    if STATE["lazy_pins"] is None:
        ok, bad = check_pins(LAZY_PINS)
        STATE["lazy_pins"] = ok
        if not ok:
            print(f"{TAG}: engine source drifted from the pins: {bad}; the fold stays OFF", flush=True)
    return getattr(mod, "Mxfp4FlashinferCutlassMoEMethod", None) if STATE["lazy_pins"] else None


def _log_once(key, msg):
    if key not in _LOGGED and len(_LOGGED) < 32:
        _LOGGED.add(key)
        print(msg, flush=True)


def _rank():
    try:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    except Exception:  # noqa: BLE001
        return -1


def _receipt():
    n = STATS["calls"]
    if STATS["folds"] == 1 or (n and n % RECEIPT_EVERY == 0):
        print(f"{TAG} calls rank {_rank()}: fold {STATS['folds']} check {STATS['checks']} stock {STATS['stock']} "
              f"errors {STATS['errors']} verdicts {dict(sorted(_VERDICT.items()))}", flush=True)


# ---------------------------------------------------------------------------------------------------- engine hooks --

def make_forward_normal(orig, v2=None):
    def forward_normal(self, hidden_states, *args, **kwargs):
        rows = hidden_states.shape[0] if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 2 else 0
        if (rows < min_rows() or kwargs.get("skip_shared_experts", False) or (len(args) >= 4 and args[3])
                or not STATE["installed"] or not STATE["chain_ok"] or not enabled() or _capturing()):
            return orig(self, hidden_states, *args, **kwargs)
        why = static_reason(self, v2)
        if why:
            _log_once(("static", why), f"{TAG}: not on this layer: {why} (stock add)")
            return orig(self, hidden_states, *args, **kwargs)
        frame = _Frame(self, rows)
        _tls.stack.append(frame)
        try:
            out = orig(self, hidden_states, *args, **kwargs)
        finally:
            _tls.stack.pop()
        if frame.folded is not None and not frame.done:
            raise RuntimeError(f"{TAG}: the routed sum was folded into the shared output but the add site never saw "
                               f"it; refusing to continue")
        return out

    forward_normal.__wrapped__ = orig
    forward_normal._dsv41_moe_fold = True
    return forward_normal


def make_forward_shared(orig):
    def _forward_shared_experts(self, hidden_states, *args, **kwargs):
        out = orig(self, hidden_states, *args, **kwargs)
        f = _top()
        if (f is not None and f.moe is self and f.shared is None and not f.claimed
                and isinstance(out, torch.Tensor) and isinstance(hidden_states, torch.Tensor)
                and out.dim() == 2 and out.shape[0] == f.rows and out.shape[0] == hidden_states.shape[0]):
            f.shared = out
        return out

    _forward_shared_experts.__wrapped__ = orig
    _forward_shared_experts._dsv41_moe_fold = True
    return _forward_shared_experts


def make_maybe_fuse(orig):
    def maybe_fuse_routed_scale_and_shared_add(experts, routed, shared, routed_scaling_factor):
        f = _top()
        if f is not None and f.folded is not None and experts is f.moe.experts:
            same = (shared is f.shared and isinstance(routed, torch.Tensor)
                    and routed.data_ptr() == f.folded.data_ptr() and tuple(routed.shape) == tuple(f.folded.shape)
                    and routed.dtype == f.folded.dtype)
            if not same:
                raise RuntimeError(f"{TAG}: the folded routed sum did not come back to the add site as the shared "
                                   f"output's storage; refusing to add it twice")
            f.done = True
            return shared           # what shared.add_(routed, alpha=1.0) returns: shared itself, now shared + routed
        if f is not None and f.check_out is not None and experts is f.moe.experts:
            # the self-check call: the routed output must come back as the same storage, else no fold anywhere
            if not (isinstance(routed, torch.Tensor) and routed.data_ptr() == f.check_out.data_ptr()
                    and tuple(routed.shape) == tuple(f.check_out.shape)):
                STATE["chain_ok"] = False
                print(f"{TAG}: the routed output does not come back to the add site as its own storage "
                      f"({type(routed).__name__}); the fold stays OFF in this process", flush=True)
            f.check_out = None
        return orig(experts, routed, shared, routed_scaling_factor)

    maybe_fuse_routed_scale_and_shared_add.__wrapped__ = orig
    maybe_fuse_routed_scale_and_shared_add._dsv41_moe_fold = True
    return maybe_fuse_routed_scale_and_shared_add


# --------------------------------------------------------------------------------------- the routed-experts side --

def claim(b12x_layer_state, m, k, device):
    """From moe_b12x_next._fused_b12x_next: the open frame this routed call belongs to, or None (stock path). Marks
    the frame claimed either way (a shared output recorded later is never used). Only rank-invariant facts decide
    whether a frame is returned (the frame, the layer, the row count), because the first call of a capacity joins a
    TP agreement; the tensor checks of the shared output (``frame_ok``) only veto the fold locally."""
    f = _top()
    if f is None or f.claimed:
        return None
    f.claimed = True
    if getattr(f.moe.experts, "_dsv41_b12x_next", None) is not b12x_layer_state or m != f.rows:
        return None
    return f


def frame_ok(f, m, k, device):
    """None: the shared output cannot take the fold on ANY rank (absent, other shape / dtype / layout: structural,
    rank-invariant) -> the stock path, no check. False: this rank's buffer is not 16-byte aligned (the kernel's
    pointer contract; allocator-dependent, so a local veto that still joins a pending check). True: foldable."""
    s = f.shared
    if (s is None or tuple(s.shape) != (m, k) or s.dtype != torch.bfloat16 or not s.is_contiguous()
            or s.device != device):
        if s is not None:
            _log_once(("shared", tuple(s.shape), str(s.dtype), s.is_contiguous()),
                      f"{TAG}: shared output {tuple(s.shape)} {s.dtype} contiguous {s.is_contiguous()} does not match "
                      f"the routed call ({m} x {k}); stock add")
        return None
    return s.data_ptr() % 16 == 0


class _NotFoldable(Exception):
    pass


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


def _agree(local):
    if not AGREE[0] or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(local)
    from sglang.srt.distributed import get_tp_group
    g = get_tp_group()
    if g.world_size == 1:
        return bool(local)
    flag = torch.tensor([1 if local else 0], dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=g.device_group)
    return bool(flag.item())


def run(frame, cap, launch, alloc_out, ok_shared):
    """The routed experts of a claimed frame. ``launch(out, accumulate, count)`` runs them writing ``out`` (stock) or
    adding into it (fold); ``alloc_out()`` allocates the stock output; ``ok_shared`` = frame_ok(). Returns the tensor
    the fused function hands back: the shared output (folded) or the stock routed output."""
    STATS["calls"] += 1
    shared = frame.shared
    if ok_shared is None:                            # structural: the same on every rank, no check, no fold
        STATS["stock"] += 1
        out = alloc_out()
        launch(out, False, True)
        return out
    verdict = _VERDICT.get(cap)
    if verdict is None:
        STATS["checks"] += 1
        out = alloc_out()
        launch(out, False, True)                     # this call's result: stock
        local, detail = False, "shared output not 16-byte aligned on this rank"
        try:
            if not ok_shared:
                raise _NotFoldable()
            ref = shared.clone()
            ref.add_(out, alpha=1.0)                 # what maybe_fuse_routed_scale_and_shared_add will do with it
            fused = shared.clone()
            launch(fused, True, False)
            local = _bits_equal(ref, fused)
            detail = "bit-identical" if local else \
                f"{int((ref.view(torch.int16) != fused.view(torch.int16)).sum())} elements differ"
            del ref, fused
        except _NotFoldable:
            pass
        except Exception as exc:  # noqa: BLE001
            detail = f"raised {exc!r}"[:300]
        verdict = _agree(local)
        _VERDICT[cap] = verdict
        print(f"{TAG} capacity {cap} (M={frame.rows}): fold vs stock add {detail} (rank {_rank()} local {local}) -> "
              f"{'ON' if verdict else 'OFF (stock add at this capacity)'}", flush=True)
        frame.check_out = out
        return out
    if not verdict or not ok_shared:
        STATS["stock"] += 1
        out = alloc_out()
        launch(out, False, True)
        return out
    try:
        launch(shared, True, True)
    except Exception as exc:  # noqa: BLE001 - the accumulate launch refuses before writing anything
        STATS["errors"] += 1
        _log_once(("err", repr(exc)[:80]), f"{TAG}: fold launch raised {exc!r}; stock add for this call")
        out = alloc_out()
        launch(out, False, True)
        return out
    STATS["folds"] += 1
    frame.folded = shared
    _receipt()
    return shared


# --------------------------------------------------------------------------------------------------------- install --

def install_v2(module):
    """sglang.srt.models.deepseek_v2 (from sitecustomize.py). Not installed (stock add everywhere) on any pin drift."""
    if not wanted_at_install():
        return False
    if STATE["installed"]:
        return True
    MoE = getattr(module, "DeepseekV2MoE", None)
    if MoE is None or not hasattr(module, "maybe_fuse_routed_scale_and_shared_add"):
        STATE["reason"] = "DeepseekV2MoE / maybe_fuse_routed_scale_and_shared_add not found"
        print(f"{TAG} not installed: {STATE['reason']}", flush=True)
        return False
    ok, bad = check_pins()
    if not ok:
        STATE["reason"] = f"engine sources drifted {bad}"
        print(f"{TAG} not installed (stock add): engine sources drifted from the pins: {bad}", flush=True)
        return False
    if getattr(MoE.forward_normal, "_dsv41_moe_fold", False):
        return True
    global RECEIPT_EVERY
    RECEIPT_EVERY = receipt_every()
    MoE.forward_normal = make_forward_normal(MoE.forward_normal, module)
    MoE._forward_shared_experts = make_forward_shared(MoE._forward_shared_experts)
    module.maybe_fuse_routed_scale_and_shared_add = make_maybe_fuse(module.maybe_fuse_routed_scale_and_shared_add)
    STATE["installed"] = True
    STATE["reason"] = "installed"
    print(f"{TAG} installed: prefill MoE calls >= {min_rows()} rows fold the shared-expert add into b12x_next's "
          f"top-k sum (checked bit for bit per plan capacity, agreed over TP; gate "
          f"{'per call (in-boot A/B)' if _ab_active() else os.environ.get(GATE, '0')})", flush=True)
    return True
