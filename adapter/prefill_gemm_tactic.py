"""Prefill dense MXFP8 GEMMs on FlashInfer's CUTLASS SM120 tactic 5 at the measured shapes, bit-identical or not at all.

DSV41_PREFILL_GEMM_TACTIC=1 (default 0 = off). Production runs every block-fp8-as-MXFP8 dense linear on FlashInfer's
b12x kernel with the tactic pinned (autotune(False, skip_ops={"mxfp8_gemm"}): the 64x128 default plan at prefill M).
The tier-7 pg3 bench (2026-11-10, one GB10, production image rg272-v272-1993630, 9 interleaved CUDA-graph rounds,
three integer-built datasets) found that FlashInfer's CUTLASS SM120 block-scaled MXFP8 GEMM, tactic 5 of
`mm_mxfp8(backend="cutlass")` = CtaShape128x128x128B with swap_ab (KernelScheduleAuto, PersistentScheduler,
StageCountAutoCarveout; getConfigs() order 128x32 / 128x64 / 128x128, each without / with swap_ab), returns
the SAME BYTES as the serving b12x call at every prefill shape and is faster:

  per-rank (N x K) @ M       layer        b12x us   cutlass:5 us   speedup
  5120 x 2048 @ 8192         wo_b          1344.6        1049.4      1.281
  8192 x 1280 @ 8192         q_b           1277.2        1145.0      1.115
  1792 x 5120 @ 4096         wqkv_a         609.1         458.4      1.329   (the row-split blocks of an 8192 chunk)
  1152 x 5120 @ 4096         shared FC1     388.7         284.3      1.367   (idem)
  4096 x 1280 @ 8192         indexer q_b    648.6         589.2      1.101
  5120 x 2048 @ 4096         wo_b           612.4         522.5      1.172
  8192 x 1280 @ 4096         q_b            629.4         545.5      1.154
  4096 x 1280 @ 4096         indexer q_b    318.0         282.3      1.127

Shared FC2 (5120 x 640) is not in the default table: its 8192-row A/A was 7.95 % (1.096 is inside the noise) and at
4096 rows tactic 5 gains 1.038 only. Decode / verify / short-extend shapes never match the table (exact M only).

  DSV41_PREFILL_GEMM_TACTIC_SHAPES   comma list of NxK@M (per-rank output width x K @ rows); default = the table above.
                                     Only exact (N, K, M) matches switch; anything else keeps the serving call.

Exactness is checked, not assumed, at BOOT: after the target model is loaded (ModelRunner.load_model, before the KV
pool and the CUDA graphs), every rank runs, for every table shape, the serving call and tactic 5 on three synthetic
operand sets built and quantized exactly as production quantizes activations (gauss, mixed power-of-two row / K-block
scales, cancel = mirrored K halves that expose any K-order change), compares the bf16 outputs bit for bit (and tactic 5
twice: deterministic), and the TP ranks agree (MIN over the TP group, one collective for the whole table) so a shape
runs tactic 5 on every rank or on none. A difference, an exception, an unpinned FlashInfer source (the tactic index
means 128x128 swap_ab only in the pinned 0.7.2 sources) or a missing boot check keeps the serving call: fail closed.
At runtime a per-call exception drops that shape to the serving call (bit-identical either way).

Read per call through adapter/ab_variant.py when the in-boot A/B is armed (a RUNTIME key), else once. Installed by
adapter/mxfp8_b12x.py on fp8_utils._raw_flashinfer_mm_mxfp8 (BEFORE prefill_gemm_split captures it, so the row-split
blocks are covered) and by sitecustomize.py on ModelRunner.load_model (the boot check). Both wrappers set __wrapped__.
"""
import functools
import hashlib
import importlib.util
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

GATE = "DSV41_PREFILL_GEMM_TACTIC"
SHAPES_ENV = "DSV41_PREFILL_GEMM_TACTIC_SHAPES"
TAG = "DSV41_PREFILL_GEMM_TACTIC"
BACKEND = "b12x"            # the serving backend the table was measured against (DSV41_MXFP8_BACKEND default)
TACTIC = 5                  # CUTLASS SM120 MXFP8: CtaShape128x128x128B, swap_ab=True
TACTIC_NUM = 6              # 3 tiles x swap_ab {false, true}
LABEL = "cutlass:5 (CUTLASS SM120 MXFP8 CtaShape128x128x128B swap_ab)"
_OFF = ("", "0", "off", "false", "no")

# (N, K, M) per rank, measured bitwise + faster (tier-7 pg3, 2026-11-10)
DEFAULT_SHAPES = "5120x2048@8192,8192x1280@8192,1792x5120@4096,1152x5120@4096,4096x1280@8192," \
                 "5120x2048@4096,8192x1280@4096,4096x1280@4096"

# sha256 of the FlashInfer 0.7.2 sources that fix what tactic 5 is (config order, tile, schedule) and how it is called
PINS = {
    "data/include/flashinfer/gemm/mxfp8_gemm_cutlass_template_sm120.h":
        "edbf9b35bf445a27e5c9d0d291e88db2d313244f89aadd26c0c69d8f527dcd5e",
    "data/include/flashinfer/gemm/mxfp8_gemm_template_sm120.h":
        "db9db08005bf4ef26e5152abd823d9fab931d77bef729a62360552f8cc619cf8",
    "data/csrc/mxfp8_gemm_cutlass_sm120.cu":
        "896676c7f8f8238958f3c918c611963a0630729a664f82511ed509e57bb6c962",
}
# receipt only (which prebuilt kernel library the process can load)
SO_RECEIPT = ("flashinfer_jit_cache/providers/sm121a/jit_cache/mxfp8_gemm_cutlass_sm120/mxfp8_gemm_cutlass_sm120.so",
              "flashinfer_jit_cache/providers/sm120f/jit_cache/mxfp8_gemm_cutlass_sm120/mxfp8_gemm_cutlass_sm120.so")

DATASETS = ("gauss", "mixed", "cancel")

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:  # CPU tests without the adapter dir
    _ab = None

_DECIDED = {}            # (n, k, m) -> True after the agreed boot check; absent / False = serving call
_LOGGED = set()
STATE = {"armed": False, "booted": False, "pins": None, "boot_s": None, "reason": "not installed"}
STATS = {"tactic_calls": 0, "serving_calls": 0, "errors": 0}
AGREE = [True]           # CPU tests switch the TP agreement off
_RUNNER = [None]         # (runner, workspace getter) once resolved
_CONFIG = [None]


# ---------------------------------------------------------------------------------------------------------- config --

def _ab_active():
    """The in-boot A/B is armed and one of its variants names the gate (else: base env, read once)."""
    if _ab is None or not getattr(_ab, "ACTIVE", False) or GATE not in getattr(_ab, "KNOWN", ()):
        return False
    return any(GATE in spec for spec in getattr(_ab, "_specs", ()) or ())


def gate_on(raw):
    v = str(raw if raw is not None else "0").strip().lower()
    if v in _OFF:
        return False
    if v in ("1", "on", "true", "yes", "cutlass:5", "cutlass5"):
        return True
    raise ValueError(f"{GATE}={raw!r}: expected 0 / 1 (1 = {LABEL})")


def parse_shapes(raw):
    """'NxK@M,...' -> frozenset{(n, k, m)}; empty / None = the default table."""
    txt = str(DEFAULT_SHAPES if raw is None or not str(raw).strip() else raw).strip().lower().replace(" ", "")
    out = set()
    for item in txt.split(","):
        if not item:
            continue
        try:
            nk, m = item.split("@")
            n, k = nk.split("x")
            key = (int(n), int(k), int(m))
        except ValueError:
            raise ValueError(f"{SHAPES_ENV}: bad entry {item!r} (expected NxK@M)") from None
        if min(key) <= 0 or key[1] % 32 or key[0] % 32:
            raise ValueError(f"{SHAPES_ENV}: {item!r}: N and K must be positive multiples of 32")
        out.add(key)
    if not out:
        raise ValueError(f"{SHAPES_ENV}: empty table")
    return frozenset(out)


def table():
    """The shape table (process-wide, read once from the base env: the boot check covers exactly these)."""
    if _CONFIG[0] is None:
        _CONFIG[0] = parse_shapes(os.environ.get(SHAPES_ENV))
    return _CONFIG[0]


_ENABLED = [None]


def enabled():
    """Per call under an armed in-boot A/B that names the gate, else read once from the base env. A value that does
    not parse (only possible through an A/B variant: the base env is parsed at install) is OFF, never an exception
    inside a forward."""
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
    """Install the wrappers when the base env enables the gate or the in-boot A/B can switch it per call."""
    if _ab_active():
        return True
    return gate_on(os.environ.get(GATE, "0"))


# ------------------------------------------------------------------------------------------------------------ pins --

def _site_root():
    spec = importlib.util.find_spec("flashinfer")
    return os.path.dirname(list(spec.submodule_search_locations)[0])


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_pins(root=None):
    """(ok, detail): every pinned FlashInfer source present with the pinned sha256."""
    try:
        root = root or _site_root()
    except Exception as exc:  # noqa: BLE001
        return False, f"flashinfer not found ({exc!r})"
    bad = []
    for rel, want in PINS.items():
        try:
            got = _sha(os.path.join(root, "flashinfer", rel))
        except OSError as exc:
            bad.append(f"{rel}: {exc.__class__.__name__}")
            continue
        if got != want:
            bad.append(f"{rel}: {got[:16]} != {want[:16]}")
    return (not bad), ("pinned FlashInfer 0.7.2 sources" if not bad else "; ".join(bad))


def so_receipt(root=None):
    out = []
    try:
        root = root or _site_root()
    except Exception:  # noqa: BLE001
        return "unknown"
    for rel in SO_RECEIPT:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            try:
                out.append(f"{rel.split('/')[2]}={_sha(p)[:16]}")
            except OSError:
                out.append(f"{rel.split('/')[2]}=unreadable")
    return ",".join(out) or "no prebuilt .so (JIT)"


# ---------------------------------------------------------------------------------------------------- the kernel --

def _resolve_runner():
    """FlashInfer's CUTLASS SM120 MXFP8 runner (the object mm_mxfp8(backend='cutlass') runs) + its workspace."""
    if _RUNNER[0] is None:
        from flashinfer.gemm import gemm_base as gb
        runner = gb.get_cutlass_mxfp8_gemm_module(12).cutlass_mxfp8_gemm_runner()
        n = len(runner.get_valid_tactics(None, None))
        if n != TACTIC_NUM:
            raise RuntimeError(f"CUTLASS SM120 MXFP8 offers {n} tactics, pinned {TACTIC_NUM}")

        def workspace(device):
            return gb._get_cache_buf("mm_mxfp8_workspace", gb.DEFAULT_WORKSPACE_SIZE, device)

        _RUNNER[0] = (runner, workspace)
    return _RUNNER[0]


def ceil_to(x, a):
    return (x + a - 1) // a * a


def flat_scales(sf, rows, k):
    """The 128x4-swizzled UE8M0 scales as the 1-D buffer CUTLASS SM120 requires, or None.

    1-D contiguous: as is. 2-D: the same bytes in storage order when the tensor is dense (contiguous, or the transposed
    view of a contiguous tensor, which is what flashinfer_mxfp8_blockscaled_linear hands over for a 2-D weight scale).
    The byte count must be ceil128(rows) * ceil4(k / 32)."""
    if sf is None or sf.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8)):
        return None
    if sf.dim() == 1:
        flat = sf if sf.is_contiguous() else None
    elif sf.dim() == 2:
        if sf.is_contiguous():
            flat = sf.reshape(-1)
        elif sf.t().is_contiguous():
            flat = sf.t().reshape(-1)
        else:
            flat = None
    else:
        flat = None
    if flat is None or flat.numel() != ceil_to(rows, 128) * ceil_to(k // 32, 4):
        return None
    return flat


def run_tactic(a, b, a_descale, b_descale, out=None, out_dtype=torch.bfloat16):
    """mm_mxfp8(a, b, ...) on tactic 5 (b = weight.t(), [K, N]); None when the operands do not fit the kernel."""
    if a.dim() != 2 or b.dim() != 2 or a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return None
    m, k = a.shape
    n = b.shape[1]
    if b.shape[0] != k or not a.is_contiguous() or not b.t().is_contiguous():
        return None
    sa = flat_scales(a_descale, m, k)
    sb = flat_scales(b_descale, n, k)
    if sa is None or sb is None:
        return None
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    elif tuple(out.shape) != (m, n) or out.dtype != out_dtype or not out.is_contiguous():
        return None
    runner, workspace = _resolve_runner()
    runner(inputs=[a, b, sa, sb, out_dtype, out, workspace(a.device)], tactic=TACTIC)
    return out


def _capturing(t):
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


def _rank0():
    try:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def make_raw_mm(original):
    """Wrap fp8_utils._raw_flashinfer_mm_mxfp8 (= flashinfer.mm_mxfp8): an agreed table shape with the gate on runs
    tactic 5, everything else the original call. functools.wraps copies the attributes fp8_utils reads from it
    (is_backend_supported, ...) and sets __wrapped__."""

    @functools.wraps(original)
    def mm(a, b, a_descale, b_descale, out=None, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend="auto"):
        if (_DECIDED and backend == BACKEND and not use_8x4_sf_layout and out_dtype == torch.bfloat16
                and a.dim() == 2 and b.dim() == 2):
            key = (int(b.shape[1]), int(a.shape[1]), int(a.shape[0]))
            if _DECIDED.get(key) and enabled() and not _capturing(a):
                try:
                    res = run_tactic(a, b, a_descale, b_descale, out, out_dtype)
                except Exception as exc:  # noqa: BLE001 - never break a forward on the opt-in path
                    _DECIDED[key] = False
                    STATS["errors"] += 1
                    logger.warning("%s N=%s K=%s M=%s raised %r; serving call for this shape from now on",
                                   TAG, key[0], key[1], key[2], exc)
                    res = None
                if res is not None:
                    STATS["tactic_calls"] += 1
                    if key not in _LOGGED and _rank0():
                        _LOGGED.add(key)
                        print(f"{TAG} first live call N={key[0]} K={key[1]} M={key[2]} -> {LABEL}", flush=True)
                    return res
                STATS["serving_calls"] += 1
        return original(a, b, a_descale, b_descale, out=out, out_dtype=out_dtype,
                        use_8x4_sf_layout=use_8x4_sf_layout, backend=backend)

    mm._dsv41_prefill_gemm_tactic = True
    return mm


_ORIG_RAW = [None]
_QUANTIZE = [None]


def install_fp8_utils(module, backend):
    """From adapter/mxfp8_b12x.install, before the row split captures the raw call. Returns True when wrapped."""
    if not wanted_at_install():
        return False
    if backend != BACKEND:
        STATE["reason"] = f"DSV41_MXFP8_BACKEND={backend!r}, the table was measured against {BACKEND!r}"
        logger.warning("%s not installed: %s", TAG, STATE["reason"])
        return False
    raw = getattr(module, "_raw_flashinfer_mm_mxfp8", None)
    quantize = getattr(module, "flashinfer_mxfp8_quantize", None)
    if raw is None or quantize is None:
        STATE["reason"] = "fp8_utils has no raw mm_mxfp8 / quantize"
        logger.warning("%s not installed: %s", TAG, STATE["reason"])
        return False
    if getattr(raw, "_dsv41_prefill_gemm_tactic", False):
        return True
    ok, detail = check_pins()
    STATE["pins"] = detail
    if not ok:
        STATE["reason"] = f"unpinned FlashInfer ({detail})"
        print(f"{TAG}: NOT armed, {STATE['reason']}; every GEMM stays on {BACKEND}", flush=True)
        return False
    table()
    _ORIG_RAW[0] = raw
    _QUANTIZE[0] = quantize
    module._raw_flashinfer_mm_mxfp8 = make_raw_mm(raw)
    STATE["armed"] = True
    STATE["reason"] = "armed, boot check pending"
    print(f"{TAG} armed: {LABEL} at {fmt_shapes(table())} after the boot check; {detail}; prebuilt {so_receipt()}",
          flush=True)
    return True


def fmt_shapes(shapes):
    return ",".join(f"{n}x{k}@{m}" for n, k, m in sorted(shapes, key=lambda s: (-s[2], -s[0] * s[1])))


# ------------------------------------------------------------------------------------------------- boot check --

def operands(n, k, m, ds, device, seed=20261110):
    """bf16 A [m, k], W [n, k] for one dataset, deterministic (the same bytes on every rank)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed + 7919 * DATASETS.index(ds) + n + 3 * k + 5 * m)
    x = torch.randn(m, k, device=device, generator=g, dtype=torch.float32)
    w = torch.randn(n, k, device=device, generator=g, dtype=torch.float32)
    if ds == "mixed":     # every UE8M0 exponent path: power-of-two scales per row and per 32-wide K block
        x = x * torch.exp2((torch.arange(m, device=device) % 13).to(torch.float32) - 6.0)[:, None]
        x = x * torch.exp2(((torch.arange(k, device=device) // 32) % 7).to(torch.float32) - 3.0)[None, :]
        w = w * torch.exp2((torch.arange(n, device=device) % 5).to(torch.float32) - 2.0)[:, None]
    elif ds == "cancel":  # mirrored K halves: any change of the K summation order changes the bytes
        h = k // 2
        x[:, h:2 * h] = -x[:, :h]
        w[:, h:2 * h] = w[:, :h] * (1 + 1e-2 * torch.randn(n, h, device=device, generator=g))
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


def serving_call(original, q, w_t, sx, sw):
    """The production call: b12x with the tactic pinned (fp8_utils' pin_tactic=True path)."""
    try:
        from flashinfer.autotuner import autotune
        ctx = autotune(False, skip_ops={"mxfp8_gemm"})
    except Exception:  # noqa: BLE001 - CPU tests
        import contextlib
        ctx = contextlib.nullcontext()
    with ctx:
        return original(q, w_t, sx, sw, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend=BACKEND)


def check_shape(n, k, m, original, quantize, device, tactic=None):
    """(ok, note): tactic 5 == the serving call, byte for byte, on every dataset, twice."""
    tactic = tactic or run_tactic
    for ds in DATASETS:
        x, w = operands(n, k, m, ds, device)
        qx, sx = quantize(x, is_sf_swizzled_layout=True, alignment=32)
        qw, sw = quantize(w, is_sf_swizzled_layout=True, alignment=32)
        del x, w
        ref = serving_call(original, qx, qw.t(), sx, sw)
        if not bool(torch.isfinite(ref.float()).all()):
            return False, f"{ds}: serving output not finite"
        mine = tactic(qx, qw.t(), sx, sw, None, torch.bfloat16)
        if mine is None:
            return False, f"{ds}: operands rejected by the tactic path"
        if not torch.equal(mine, ref):
            nd = int((mine.view(torch.int16) != ref.view(torch.int16)).sum()) if mine.shape == ref.shape else -1
            return False, f"{ds}: {nd} of {ref.numel()} outputs differ"
        mine2 = tactic(qx, qw.t(), sx, sw, mine.fill_(0), torch.bfloat16)
        if mine2 is None or not torch.equal(mine2, ref):
            return False, f"{ds}: tactic not deterministic"
        del ref, mine, mine2, qx, qw, sx, sw
    return True, "bit-identical on " + "/".join(DATASETS)


def _agree(flags):
    """MIN over the TP group (CPU / gloo group when SGLang has one): a shape is on for every rank or for none."""
    if not AGREE[0] or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return list(flags)
    from sglang.srt.distributed import get_tp_group
    group = get_tp_group()
    cpu = getattr(group, "cpu_group", None)
    if cpu is not None:
        t = torch.tensor(flags, dtype=torch.int32)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=cpu)
    else:
        t = torch.tensor(flags, dtype=torch.int32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
    return [int(v) for v in t.tolist()]


def boot_check(device=None, tactic=None):
    """Decide every table shape once. Every rank that installed the hook calls this exactly once and always reaches
    the one collective (a local failure only zeroes its flags), so the ranks cannot desynchronise."""
    t0 = time.time()
    shapes = sorted(table(), key=lambda s: (-s[2], -s[0] * s[1]))
    local, notes = [], []
    armed = STATE["armed"] and _ORIG_RAW[0] is not None and _QUANTIZE[0] is not None
    if armed:
        try:
            _resolve_runner()
        except Exception as exc:  # noqa: BLE001
            armed = False
            STATE["reason"] = f"runner unavailable ({exc!r})"
    if device is None and armed:
        device = torch.device("cuda", torch.cuda.current_device())
    for n, k, m in shapes:
        if not armed:
            local.append(0)
            notes.append(STATE["reason"])
            continue
        try:
            with torch.no_grad():
                ok, note = check_shape(n, k, m, _ORIG_RAW[0], _QUANTIZE[0], device, tactic)
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"raised {exc!r}"[:300]
        local.append(1 if ok else 0)
        notes.append(note)
    if armed and device is not None and getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()   # the check's transients go back before the KV pool is sized
    agreed = _agree(local)
    _DECIDED.clear()
    for (n, k, m), ok in zip(shapes, agreed):
        _DECIDED[(n, k, m)] = bool(ok)
    STATE["booted"] = True
    STATE["boot_s"] = round(time.time() - t0, 2)
    n_on = sum(1 for v in _DECIDED.values() if v)
    if _rank0():
        for (n, k, m), ok_l, ok, note in zip(shapes, local, agreed, notes):
            print(f"{TAG} N={n} K={k} M={m}: {note} (local {bool(ok_l)}) -> "
                  f"{LABEL if ok else BACKEND + ' (serving call)'}", flush=True)
        print(f"{TAG} boot check: {n_on}/{len(shapes)} shapes on {LABEL.split(' ')[0]}, agreed over TP, "
              f"{STATE['boot_s']} s; gate {'ON' if _gate_any() else 'off'} at start", flush=True)
    STATE["reason"] = f"boot check {n_on}/{len(shapes)}"
    return dict(_DECIDED)


def _gate_any():
    try:
        return enabled() or _ab_active()
    except ValueError:
        return False


def install_model_runner(module):
    """Hook ModelRunner.load_model: the boot check after the TARGET model is loaded (once per process)."""
    if not wanted_at_install():
        return False
    cls = getattr(module, "ModelRunner", None)
    original = getattr(cls, "load_model", None) if cls is not None else None
    if original is None:
        print(f"{TAG}: ModelRunner.load_model not found; no boot check, every GEMM stays on {BACKEND}", flush=True)
        return False
    if getattr(original, "_dsv41_prefill_gemm_tactic", False):
        return True

    @functools.wraps(original)
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not STATE["booted"] and not getattr(self, "is_draft_worker", False):
            try:
                boot_check()
            except Exception as exc:  # noqa: BLE001 - fail closed: no shape switches, the engine boots as stock
                _DECIDED.clear()
                STATE["booted"] = True
                STATE["reason"] = f"boot check raised {exc!r}"[:300]
                print(f"{TAG}: boot check raised {exc!r}; every GEMM stays on {BACKEND}", flush=True)
        return result

    load_model._dsv41_prefill_gemm_tactic = True
    cls.load_model = load_model
    return True
