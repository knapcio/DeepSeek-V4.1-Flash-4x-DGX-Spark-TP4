"""Row-block split of the K >= 5120 MXFP8 dense GEMMs at prefill (wqkv_a, shared-expert FC1), bit-identical or not at all.

DSV41_PREFILL_GEMM_ROW_SPLIT=<rows> (default 0 = off). At an 8192-row prefill chunk the MXFP8 activation of a K = 5120
GEMM is 8192 x 5120 = 42 MB, larger than GB10's 25.2 MB L2, and the persistent b12x dense kernel re-reads it from DRAM
for every N tile: the v2.7 trace (diagnostics/fable-ideas-20261108/prefill.md 1.3) has wqkv_a (5120 -> 1792) and the
shared FC1 (5120 -> 1152) at 51 TFLOP/s against 132 at 4096 rows. With the knob a GEMM of M >= MIN_M rows runs as
ceil(M / rows) launches of the SAME b12x kernel on row blocks of `rows` rows (A per launch 21 MB at 4096, L2-resident),
each writing its rows of one output tensor (mm_mxfp8 out=, no concatenation copy). The activation is quantized once,
as stock; a pre-quantized input (DSV41_PREFILL_SP_FP8's gathered Mxfp8SwizzledInput) is sliced: E4M3 rows are a row
view, and the 128x4 swizzled scales are row-tile-major (prefill_sp.swizzle_128x4), so rows [r0, r1) with r0 % 128 == 0
are the contiguous bytes [r0 * Gp, (r0 + ceil128(r1 - r0)) * Gp) with Gp = ceil4(K / 32).

  DSV41_PREFILL_GEMM_ROW_SPLIT_MIN_K   default 5120: only GEMMs with K >= this (wo_b K=2048, q_b K=1280 keep L2)
  DSV41_PREFILL_GEMM_ROW_SPLIT_MIN_M   default 2 x rows: only M >= this (decode, DSpark verify, short extends: stock);
                                       and only M a multiple of rows (production chunks are 8192; an 8192 + x bypass
                                       batch or a replay of another size keeps the stock call and never re-checks)
  DSV41_PREFILL_GEMM_ROW_SPLIT_N       comma list of N (per-rank output width), default 1792,1152 = wqkv_a and the
                                       shared FC1, the two measured shapes; "all" = every N

Rank invariance of the check's all_reduce rests on (N, K, M) being the same on every rank for every eligible call:
true for the replicated wqkv_a, the TP-even shared FC1 and SP's full-row gathers; a layer whose per-rank N differs must
never be eligible (keep _N to replicated / even shapes).

Each row of a GEMM output depends only on that row of A and on B: the kernel's K loop per output element does not
change with M. That is the claim; it is checked, not assumed: the first eager call of every (N, K, M, pre-quantized,
dtype) key runs the stock GEMM AND the split on the same input, compares the outputs bit for bit, and the TP ranks
agree (MIN) so a key is split on every rank or on none. A difference, an exception or a capture keeps the stock call.
The per-M tactic of b12x (pin_tactic: heuristic by M) may differ between 4096 and 8192 rows; the check is what makes
that safe. Decode is untouched: CUDA-graph capture never splits, and graph shapes are far below MIN_M.

Read per call through adapter/ab_variant.py when the in-boot A/B is armed (a RUNTIME key), else once at install.
"""
import contextlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

TILE = 128
_OFF = ("", "0", "off", "false", "no")

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:  # CPU tests without the adapter dir
    _ab = None

_DECIDED = {}        # key -> bool (agreed over TP)
_LOGGED = set()
STATS = {"split_calls": 0, "stock_calls": 0, "checks": 0, "refused": 0}
AGREE = [True]       # CPU tests switch the TP agreement off


def _ab_active():
    """The in-boot A/B is armed and one of its variants names the gate (else: base env, read once)."""
    name = "DSV41_PREFILL_GEMM_ROW_SPLIT"
    if _ab is None or not getattr(_ab, "ACTIVE", False) or name not in getattr(_ab, "KNOWN", ()):
        return False
    return any(name in spec for spec in getattr(_ab, "_specs", ()) or ())


def _raw_env(name, default):
    if _ab_active():
        return _ab.env(name, default)
    return os.environ.get(name, default)


DEFAULT_N = "1792,1152"


def parse(rows_raw, min_k_raw="5120", min_m_raw="", n_raw=DEFAULT_N):
    """(rows, min_k, min_m, n_set or None) from the env strings; rows 0 = off. rows must be a positive multiple of 128."""
    raw = str(rows_raw or "0").strip().lower()
    if raw in _OFF:
        return 0, 0, 0, None
    rows = int(raw)
    if rows <= 0:
        return 0, 0, 0, None
    if rows % TILE:
        raise ValueError(f"DSV41_PREFILL_GEMM_ROW_SPLIT={rows}: must be a multiple of {TILE}")
    min_k = int(str(min_k_raw or "5120").strip())
    min_m = int(str(min_m_raw).strip()) if str(min_m_raw or "").strip() else 2 * rows
    if min_m <= rows:
        raise ValueError(f"DSV41_PREFILL_GEMM_ROW_SPLIT_MIN_M={min_m} must exceed the block ({rows})")
    n_txt = str(DEFAULT_N if n_raw is None else n_raw).strip().lower()
    ns = None if n_txt in ("all", "*", "") else {int(x) for x in n_txt.split(",") if x.strip()}
    return rows, min_k, min_m, ns


_CONFIG = [None]


def _read_config():
    return parse(_raw_env("DSV41_PREFILL_GEMM_ROW_SPLIT", "0"),
                 _raw_env("DSV41_PREFILL_GEMM_ROW_SPLIT_MIN_K", "5120"),
                 _raw_env("DSV41_PREFILL_GEMM_ROW_SPLIT_MIN_M", ""),
                 _raw_env("DSV41_PREFILL_GEMM_ROW_SPLIT_N", DEFAULT_N))


def config():
    """Per call under an armed in-boot A/B that names the gate, else parsed once (make_linear_split resets it)."""
    if _ab_active():
        return _read_config()
    if _CONFIG[0] is None:
        _CONFIG[0] = _read_config()
    return _CONFIG[0]


def plan_blocks(m, rows):
    """[(r0, r1)] covering [0, m) in blocks of `rows` (the last one may be shorter); every r0 is a multiple of 128."""
    if rows <= 0 or rows % TILE:
        raise ValueError(rows)
    return [(r0, min(m, r0 + rows)) for r0 in range(0, m, rows)]


def eligible(m, n, k, cfg):
    rows, min_k, min_m, ns = cfg
    return bool(rows) and m >= min_m and m % rows == 0 and k >= min_k and (ns is None or n in ns)


def ceil_to(x, a):
    return (x + a - 1) // a * a


def scale_rows(scales, m, k, r0, r1):
    """The 128x4-swizzled UE8M0 scales of rows [r0, r1) of an [m, k] MXFP8 tensor, as a view of `scales`.

    `scales` is the swizzled tensor of all m rows: 1-D (M_pad * Gp) or 2-D (M_pad, Gp) with M_pad = ceil128(m),
    Gp = ceil4(k / 32). Row tiles are outermost in that layout, so a row block that starts on a tile boundary is a
    contiguous byte range. Returns None when the tensor does not have that shape (caller keeps the stock GEMM)."""
    if r0 % TILE or not (0 <= r0 < r1 <= m) or k % 32:
        return None
    gp = ceil_to(k // 32, 4)
    m_pad = ceil_to(m, TILE)
    if scales.numel() != m_pad * gp or not scales.is_contiguous():
        return None
    rows_pad = ceil_to(r1 - r0, TILE)
    flat = scales.reshape(-1)[r0 * gp:(r0 + rows_pad) * gp]
    if scales.dim() == 1:
        return flat
    if scales.dim() == 2 and tuple(scales.shape) == (m_pad, gp):
        return flat.view(rows_pad, gp)
    return None


def split_mm(raw_mm, q, x_scale, w_t, ws_t, out_dtype, backend, pin_tactic, rows, autotune=None):
    """The row-block GEMM: one output, one raw mm_mxfp8 launch per block. Returns None if a block cannot be sliced."""
    m, k = q.shape
    n = w_t.shape[1]
    if not q.is_contiguous():
        return None
    blocks = plan_blocks(m, rows)
    slices = []
    for r0, r1 in blocks:
        s = scale_rows(x_scale, m, k, r0, r1)
        if s is None:
            return None
        slices.append(s)
    out = torch.empty((m, n), dtype=out_dtype, device=q.device)
    ctx = autotune(False, skip_ops={"mxfp8_gemm"}) if (pin_tactic and autotune is not None) else contextlib.nullcontext()
    with ctx:
        for (r0, r1), s in zip(blocks, slices):
            raw_mm(q[r0:r1], w_t, s, ws_t, out=out[r0:r1], out_dtype=out_dtype,
                   use_8x4_sf_layout=False, backend=backend)
    return out


def _agree(local):
    if not AGREE[0] or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return bool(local)
    from sglang.srt.distributed import get_tp_group
    group = get_tp_group()
    t = torch.tensor([1 if local else 0], dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
    return bool(t.item())


def _rank0():
    try:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def _capturing(t):
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


def make_linear_split(fp8_utils):
    """(original, input, weight, weight_scale, input_scale, bias, output_dtype, backend, pin_tactic) -> tensor or None.

    None = not split (the caller runs the stock call). Mirrors fp8_utils.flashinfer_mxfp8_blockscaled_linear for the
    b12x backend: quantize once (swizzled, alignment 32), weight.t(), weight_scale.t(), autotune(False) when pinned."""
    _CONFIG[0] = None
    raw_mm = getattr(fp8_utils, "_raw_flashinfer_mm_mxfp8", None)
    quantize = getattr(fp8_utils, "flashinfer_mxfp8_quantize", None)
    if raw_mm is None or quantize is None:
        logger.warning("DSV41_PREFILL_GEMM_ROW_SPLIT: fp8_utils has no raw mm_mxfp8 / quantize; split inactive")
        return None
    try:
        from flashinfer.autotuner import autotune
    except Exception:  # noqa: BLE001
        autotune = None

    def split(original, input, weight, weight_scale, input_scale, bias, output_dtype, backend, pin_tactic):
        cfg = config()
        if not cfg[0] or backend == "trtllm":
            return None
        k = int(input.shape[-1])
        n = int(weight.shape[0])
        m = input.numel() // k if k else 0
        if not eligible(m, n, k, cfg) or _capturing(input) or bias is not None:
            return None
        if output_dtype is None:
            output_dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
        key = (n, k, m, input_scale is not None, input.dtype, output_dtype, cfg[0])
        decided = _DECIDED.get(key)
        if decided is False:
            STATS["stock_calls"] += 1
            return None

        def run():
            input_2d = input.reshape(-1, k)
            if input_scale is None:
                q, xs = quantize(input_2d, is_sf_swizzled_layout=True, alignment=32)
            else:
                q, xs = input_2d, input_scale
            ws_t = weight_scale.t() if weight_scale.ndim == 2 else weight_scale
            return split_mm(raw_mm, q, xs, weight.t(), ws_t, output_dtype, backend, pin_tactic, cfg[0], autotune)

        if decided is None:
            STATS["checks"] += 1
            try:
                mine = run()
                ref = original(input, weight, weight_scale, input_scale, bias, output_dtype,
                               backend=backend, pin_tactic=pin_tactic)
                local = mine is not None and mine.shape == ref.reshape(m, n).shape and \
                    bool(torch.equal(mine, ref.reshape(m, n)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("DSV41_PREFILL_GEMM_ROW_SPLIT check N=%s K=%s M=%s raised %r", n, k, m, exc)
                mine, ref, local = None, None, False
            ok = _agree(local)
            _DECIDED[key] = ok
            if key not in _LOGGED and _rank0():
                _LOGGED.add(key)
                print(f"DSV41_PREFILL_GEMM_ROW_SPLIT N={n} K={k} M={m} prequant={input_scale is not None} rows={cfg[0]}: "
                      f"{'bit-identical on every rank -> SPLIT' if ok else 'differs or failed -> stock'} (local {local})",
                      flush=True)
            if not ok:
                STATS["refused"] += 1
            if ref is None:
                return None
            return ref  # the checked call returns the stock result either way
        try:
            out = run()
        except Exception as exc:  # noqa: BLE001 - never break a forward on the opt-in path
            _DECIDED[key] = False
            logger.warning("DSV41_PREFILL_GEMM_ROW_SPLIT N=%s K=%s M=%s raised %r; stock from now on", n, k, m, exc)
            return None
        if out is None:
            _DECIDED[key] = False
            return None
        STATS["split_calls"] += 1
        return out.view(*input.shape[:-1], n)

    return split


def wanted_at_install():
    """Install the wrapper when the base env enables it or the in-boot A/B can switch it per call."""
    if _ab_active():
        return True
    return parse(os.environ.get("DSV41_PREFILL_GEMM_ROW_SPLIT", "0"))[0] > 0
