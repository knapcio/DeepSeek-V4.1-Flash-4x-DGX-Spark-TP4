"""DSV41_DENSE_TILE_TABLE=1 (off by default): the narrow decode dense MXFP8 GEMMs on the b12x tile a measured sweep found
faster than the runner's default plan, bit for bit, self-checked at the first eager call of every (shape, M) on every
rank with TP-wide agreement, fail closed.

Measured 2026-11-10 (tier10-kit window B2, bench/d1: one GB10, production stopped, every tile x tile_k x stage-cap
config of the b12x SM120 dense MXFP8 kernel against the production plan on the same bytes, two activations per cell,
whole-output byte compare; all 24 configs were bitwise on every cell: the tile changes which rows / columns a CTA
owns, never an element's K order or MMA sequence). The runner's default plan for an N x 5120 weight at decode row
counts is the M-independent (64, 64) tile ("maximizes CTAs"); at M <= 16 the (16, 64) tile, at 17..32 the (32, 64) tile,
with the same CTA count, cut the per-call time by 2x (warm L2, us per call):

    qa_kv     512 x 5120  (replicated_split window, 40 calls / step)  M6: 17.42 -> 11.07   M24: 17.44 -> 12.85 (32x64)
    FC1      1152 x 5120  (shared expert gate+up, 43 calls / step)    M6: 29.22 -> 13.10   M24: 23.46 -> 13.17 (32x64)
    draft    1536 x 5120  (DSpark draft wkv, 3 calls / step)          M6: 37.86 -> 15.71   M24: 30.32 -> 15.17 (32x64)
    (M96: the default (64, 64) is the right tile; FC2 5120 x 640 and idx_q_b 4096 x 1280 already run (16|32, 128): no
    entry; cold-L2 times are unchanged by the tile: the gain is the latency-bound warm regime)

Per step at M = 6 that is 40 x 6.4 + 43 x 16.1 + 3 x 22 = ~1.0 ms warm-isolated; the engine's gain is smaller (weights
are L2-warm only where the prefetch plan put them): the in-boot A/B decides (tier10-kit window S, arm X_d1, promote at
<= -0.4 ms on the paired c1 step).

Mechanics (adapter/mxfp8_b12x.py calls `call()` first for every b12x-routed linear): the table keys (N, K) -> M buckets
with the tile and the default tile the sweep measured against. A call whose (N, K, M) has an entry and whose runner
default plan for that (M, N, K) is still the measured one goes to the SAME b12x runner (gemm_base._b12x_gemm_mxfp8_runner,
the object mm_mxfp8 builds) with an explicit tactic (tile, (1, 1), swap_ab False, prefetch False, "sm120", None) in
place of the autotuner's fallback (production pins tactics: autotune(False, skip_ops={"mxfp8_gemm"}) -> the default
plan); the activation is quantized by the same SGLang op, the scales / bias / output dtype handled as
flashinfer_mxfp8_blockscaled_linear does. The first eager call of every key runs both (the production path and the
tabled tile) on the same inputs and compares the output bytes (uint8 views: +0.0 / -0.0 differ); the verdict is
TP-agreed (MIN: every rank serves the same kernel for a key, or none does); a mismatch, a plan drift (the image's
default plan is not the measured one) or a disagreement keeps the production path for that key for good. Inside a
graph capture an unchecked key takes the production path and records nothing (SGLang's warm-up forward before each
capture checks it). Prefill rows (M > 32) and every other shape are untouched. Nothing is cached across processes;
nothing is written to the environment; the kernel compile cache is the runner's own (its key carries the tile).
Under DSV41_AB_VARIANTS the gate is read per call through ab_variant.env (a per-variant key; the install is the union).
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

ENV = "DSV41_DENSE_TILE_TABLE"
MAX_M = 32
# (N, K) per TP4 rank -> ((m_max, tile, measured default tile), ...) in ascending m_max
TABLE = {
    (512, 5120): ((16, (16, 64), (64, 64)), (32, (32, 64), (64, 64))),
    (1152, 5120): ((16, (16, 64), (64, 64)), (32, (32, 64), (64, 64))),
    (1536, 5120): ((16, (16, 64), (64, 64)), (32, (32, 64), (64, 64))),
}
_CHECKED = {}      # key -> True (the tabled tile serves it) / False (production path kept)
_STATS = {"checked": 0, "same": 0, "diff": 0, "drift": 0, "disagree": 0, "skipped_capture": 0}
_RUNNERS = {}      # (device index, out dtype) -> the b12x runner object
_QUANT = None      # SGLang's flashinfer_mxfp8_quantize op (set at install)

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): per-variant value; plain os.environ when off
    from ab_variant import env as _env
except ImportError:
    _env = os.environ.get


def _truthy(v):
    return str(v or "0").strip().lower() in ("1", "on", "true", "yes")


def wanted_at_install():
    """Install-time union gate (the A/B harness unions the variants); the per-call value is read through ab_variant."""
    return _truthy(os.environ.get(ENV))


def entry(n, k, m):
    """(tile, measured default tile) for (N, K, M), or None."""
    rows = TABLE.get((int(n), int(k)))
    if not rows or m > MAX_M:
        return None
    for m_max, tile, prod_tile in rows:
        if m <= m_max:
            return tile, prod_tile
    return None


def tactic(tile):
    return (tuple(tile), (1, 1), False, False, "sm120", None)


def _default_tile(m, n, k, device):
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import _select_default_dense_gemm_plan
    from flashinfer.utils import get_device_sm_count
    plan = _select_default_dense_gemm_plan(m, n, k, get_device_sm_count(device), is_mxfp8=True, expected_m=m)
    return tuple(plan.mma_tiler_mn)


def _runner(device, out_dtype):
    key = (int(device.index or 0), out_dtype)
    r = _RUNNERS.get(key)
    if r is None:
        from flashinfer.gemm import gemm_base
        major, minor = torch.cuda.get_device_capability(device)
        r = gemm_base._b12x_gemm_mxfp8_runner(major, minor, True, out_dtype)
        _RUNNERS[key] = r
    return r


def _workspace(device):
    from flashinfer.gemm import gemm_base
    return gemm_base._get_cache_buf("mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, device)


def run(args, tile):
    """flashinfer_mxfp8_blockscaled_linear's b12x body with an explicit tactic (the bench's api_shape.mm_tile)."""
    input, weight, weight_scale, input_scale, bias, output_dtype = args
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    if input_scale is None:
        q_input, x_sf = _QUANT(input_2d, is_sf_swizzled_layout=True, alignment=32)
    else:
        q_input, x_sf = input_2d, input_scale
    if output_dtype is None:
        output_dtype = input_2d.dtype if input_2d.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
    w_sf = weight_scale.t() if weight_scale.ndim == 2 else weight_scale
    out_dtype = output_dtype if output_dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    out = torch.empty((q_input.shape[0], weight.shape[0]), device=q_input.device, dtype=out_dtype)
    runner = _runner(q_input.device, out_dtype)
    runner(inputs=[q_input, weight.t(), x_sf, w_sf, out_dtype, out, _workspace(q_input.device)], tactic=tactic(tile))
    if bias is not None:
        out = out + bias
    return out.to(dtype=output_dtype).view(*output_shape)


def _bytewise_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _agree(flag, device):
    from mxfp8_b12x import _agree as agree
    return agree(flag, device)


def call(original, args, want, pin_tactic):
    """The dispatch: None = not this lever's call (the caller takes its own route)."""
    if want != "b12x" or not _truthy(_env(ENV, "0")):
        return None
    input, weight, _ws, _is, _bias, output_dtype = args
    m = int(input.view(-1, input.shape[-1]).shape[0])
    ent = entry(int(weight.shape[0]), int(weight.shape[1]), m)
    if ent is None:
        return None
    tile, prod_tile = ent
    key = (int(weight.shape[0]), int(weight.shape[1]), m, str(input.dtype), str(output_dtype), int(input.device.index or 0))
    verdict = _CHECKED.get(key)
    if verdict is None and torch.cuda.is_current_stream_capturing():
        _STATS["skipped_capture"] += 1
        return None                     # never an unchecked kernel inside a graph; not recorded
    if verdict is False:
        return None
    if verdict is True:
        return run(args, tile)
    stock = original(*args, backend=want, pin_tactic=pin_tactic)
    try:
        got = _default_tile(m, key[0], key[1], input.device)
    except Exception as exc:  # noqa: BLE001 - the plan reader is part of the measurement's precondition
        got = ("unreadable", repr(exc)[:80])
    drift = tuple(got) != tuple(prod_tile)
    same = False
    cand = None
    if not drift:
        cand = run(args, tile)
        same = _bytewise_equal(stock, cand)
    agreed = _agree(same and not drift, input.device)
    _CHECKED[key] = agreed
    _STATS["checked"] += 1
    if drift:
        _STATS["drift"] += 1
    else:
        _STATS["same" if same else "diff"] += 1
    if same and not drift and not agreed:
        _STATS["disagree"] += 1
    logger.warning("DSV41 dense tile table N=%d K=%d M=%d tile=%s: %s", key[0], key[1], m, "x".join(map(str, tile)),
                   "bit-identical to the default plan %s, serves this key" % "x".join(map(str, prod_tile)) if agreed else
                   ("default plan %s is not the measured %s (plan drift); production path kept" % (got, prod_tile) if drift else
                    ("MISMATCH against the default plan; production path kept" if not same else
                     "bit-identical here but another rank differs; production path kept on every rank")))
    return cand if agreed else stock


def stats():
    return dict(_STATS, keys=len(_CHECKED), on=sum(1 for v in _CHECKED.values() if v))


def make(module, want):
    """Bind SGLang's quant op; returns the dispatch for adapter/mxfp8_b12x.py's linear (None when nothing to arm)."""
    global _QUANT
    _QUANT = getattr(module, "flashinfer_mxfp8_quantize", None)
    if _QUANT is None:
        logger.warning("DSV41_DENSE_TILE_TABLE NOT armed: fp8_utils has no flashinfer_mxfp8_quantize")
        return None
    logger.warning("DSV41_DENSE_TILE_TABLE armed: %d shapes (N x K %s) at M <= %d on the measured b12x tiles, self-checked "
                   "bit for bit against the default plan per (shape, M), TP-agreed; backend %r",
                   len(TABLE), ", ".join(f"{n}x{k}" for n, k in sorted(TABLE)), MAX_M, want)
    return call
