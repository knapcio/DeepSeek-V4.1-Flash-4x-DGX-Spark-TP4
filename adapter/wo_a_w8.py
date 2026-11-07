"""wo_a from its fp8 checkpoint bytes instead of the bf16 copy, same Triton tiling. Gated on DSV41_WO_A_W8.

The checkpoint ships wo_a as e4m3 with 32x32 ue8m0 block scales. The engine dequantizes it to bf16
at load (its fp8 absorb path needs 128x128 blocks) and the verify/draft small-batch kernel
`_wo_a_partial` then streams 16.8 MB per layer and rank. This adapter keeps an fp8 twin of every
wo_a (e4m3 values plus one exponent per 32x32 block), built after load and kept only when it
reconstructs the bf16 weight exactly, and runs a copy of `_wo_a_partial` that loads the e4m3 tile
and its exponent and rebuilds the same bf16 tile in registers before the same `tl.dot`.

Half the weight bytes for the same operand values. Measured on GB10 (TP4 shape, 43 layers, M=6):
3.43 -> 2.20 ms; the MXFP8-epilogue path (what production runs) is bitwise identical to the stock
kernel; the plain bf16 path differs only in the MMA's fp32 accumulation order (1 bf16 ulp on
0.016 % of outputs). Only 2 <= M <= 8 rows take the new kernel (verify at bs=1, the draft block);
every other shape and any layer whose twin is not exact stays on the stock path.

DSV41_WO_A_W8_MID=1 (with DSV41_WO_A_W8=1): verify/draft calls with 9..192 rows (c2..c32) also read
the twin, in a second Triton kernel tuned on GB10 with DRAM-resident weights (bf16 bmm 84-134 us per
layer at 12-96 rows, the twin 49-74 us); prefill keeps the stock kernel. Split-K partials are summed
in a fixed order, so the result is deterministic.
DSV41_WO_A_W8_DROP=1: after the twins exist and a dequantized copy was checked bit-identical, the
bf16 storage of every twinned wo_a is released (16.8 MB per layer: 722 MB per rank incl. the draft,
which the head node returns to the KV pool); every row count is then served from the twin, prefill
through a per-call dequantized bf16 buffer, so its numerics are unchanged.
DSV41_WO_A_INPUT_FP8=1 (adapter/wo_a_input_fp8.py): the routes this adapter serves without calling the
stock dispatch (DROP's 2-8 rows, MID's 9-192) round the loaded `o` tile in registers as DeepSeek's
reference rounds o (QX); every call that reaches the stock dispatch was already rounded by that
adapter's wrapper, so the kernels it reaches (`small`, `small_mx`) run with QX off.
"""
import os

import torch
import triton
import triton.language as tl

from wo_a_input_fp8 import ENABLED as QX
from wo_a_input_fp8 import qdq_tile

ENABLED = os.environ.get("DSV41_WO_A_W8", "0").strip() not in ("0", "", "off", "false")
_TWINS = {}          # bf16 weight data_ptr -> (e4m3 [2,1024,4096], exponent uint8 [2,32,128])


@triton.jit
def _wo_a_partial_w8(X, W8, S, P, M: tl.constexpr, SX: tl.constexpr, QX: tl.constexpr = False):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        if QX:  # 128 columns = four whole 32-blocks of the group's 4096
            x = qdq_tile(x, 16, 128)
        w8 = tl.load(W8 + (group * 1024 + n[None, :]) * 4096 + offsets[:, None])
        e = tl.load(S + (group * 32 + n[None, :] // 32) * 128 + offsets[:, None] // 32)
        w = (w8.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * 2 + group) * 1024 + n[None, :],
        acc,
        m[:, None] < M,
    )


def make_twin(w: torch.Tensor):
    """bf16 [2, 1024, 4096] -> (e4m3, exponent) or None if the twin does not reproduce w exactly."""
    g, r, d = w.shape
    wf = w.float().view(g, r // 32, 32, d // 32, 32)
    amax = wf.abs().amax(dim=(2, 4)).clamp_min(2.0 ** -126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.exp2(e)[:, :, None, :, None]
    q = (wf / scale).to(torch.float8_e4m3fn)
    back = (q.float() * scale).to(torch.bfloat16).view(g, r, d)
    if not torch.equal(back, w):
        return None
    return q.view(g, r, d).contiguous(), (e + 127).to(torch.uint8).contiguous()


def _partial_w8(x, twin, m, qx=False):
    w8, s = twin
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial_w8[(16, 2, 8)](x, w8, s, partial, m, x.stride(0), QX=qx, num_warps=4, num_stages=3)
    return partial


MID = os.environ.get("DSV41_WO_A_W8_MID", "0").strip() not in ("0", "", "off", "false")
# tuned on GB10 with 43 distinct layers (DRAM-bound, not L2): rows -> (split_k, BM, BN, warps)
_MID_CFG = ((16, (4, 16, 64, 8)), (32, (2, 32, 32, 4)), (64, (1, 64, 32, 4)), (128, (1, 32, 64, 4)),
            (192, (1, 64, 32, 4)))


@triton.jit
def _wo_a_mid_kernel(X, W8, S, Y, M, SXM, SPLIT_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, QX: tl.constexpr = False):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    pid_mk = tl.program_id(2)
    pid_m = pid_mk // SPLIT_K
    pid_k = pid_mk % SPLIT_K
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    KC: tl.constexpr = 4096 // SPLIT_K
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(pid_k * KC, (pid_k + 1) * KC, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * SXM + g * 4096 + k[None, :], m[:, None] < M, 0.0)
        if QX:  # k0 is a multiple of BK = 128: whole 32-blocks
            x = qdq_tile(x, BM, BK)
        w8 = tl.load(W8 + g * 1024 * 4096 + n[None, :] * 4096 + k[:, None])
        e = tl.load(S + g * 32 * 128 + (n[None, :] // 32) * 128 + k[:, None] // 32)
        w = (w8.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x, w)
    if SPLIT_K == 1:
        tl.store(Y + m[:, None] * 2048 + g * 1024 + n[None, :], acc.to(tl.bfloat16), m[:, None] < M)
    else:
        # per-split partials, summed in a fixed order afterwards (atomics made the order random)
        tl.store(Y + pid_k * M * 2048 + m[:, None] * 2048 + g * 1024 + n[None, :], acc, m[:, None] < M)


def wo_a_mid(o, twin, qx=False):
    """o [M, 2, 4096] (stride (>=8192, 4096, 1)) bf16, 9 <= M <= 192 -> bf16 [M, 2, 1024].
    qx: round o as DeepSeek's reference does (adapter/wo_a_input_fp8.py) on the loaded tiles."""
    w8, s = twin
    m = o.shape[0]
    sk, bm, bn, warps = next(cfg for lim, cfg in _MID_CFG if m <= lim)
    grid = (1024 // bn, 2, triton.cdiv(m, bm) * sk)
    if sk == 1:
        y = torch.empty((m, 2, 1024), dtype=torch.bfloat16, device=o.device)
        _wo_a_mid_kernel[grid](o, w8, s, y, m, o.stride(0), 1, bm, bn, 128, QX=qx, num_warps=warps, num_stages=3)
        return y
    y = torch.empty((sk, m, 2, 1024), dtype=torch.float32, device=o.device)
    _wo_a_mid_kernel[grid](o, w8, s, y, m, o.stride(0), sk, bm, bn, 128, QX=qx, num_warps=warps, num_stages=3)
    return y.sum(dim=0).to(torch.bfloat16)


DROP = os.environ.get("DSV41_WO_A_W8_DROP", "0").strip() not in ("0", "", "off", "false")
_REFS = {}
_DROPPED = set()          # tag data_ptrs of wo_a weights whose bf16 storage was released


@triton.jit
def _dequant_kernel(W8, S, OUT):
    g = tl.program_id(0)
    rb = tl.program_id(1)
    cb = tl.program_id(2)
    r = rb * 32 + tl.arange(0, 32)
    c = cb * 32 + tl.arange(0, 32)
    off = g * 1024 * 4096 + r[:, None] * 4096 + c[None, :]
    e = tl.load(S + g * 32 * 128 + rb * 128 + cb)
    w = tl.load(W8 + off).to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)
    tl.store(OUT + off, w.to(tl.bfloat16))


def dequant_into_scratch(twin, device):
    # a fresh buffer per call: the caching allocator orders reuse per stream, so target and draft
    # work on different streams never share it (one shared scratch raced)
    buf = torch.empty((2, 1024, 4096), dtype=torch.bfloat16, device=device)
    _dequant_kernel[(2, 32, 128)](twin[0], twin[1], buf)
    return buf


def drop_bf16(model):
    """After the twins exist: release the bf16 storage of every exactly-twinned wo_a. The weight
    becomes a zero-stride view of a 1-element tag (unique data_ptr, correct shape), and every call
    is served from the twin by the wrapper in install_mid."""
    freed = mismatched = 0
    for mod in model.modules():
        wo_a = getattr(mod, "wo_a", None)
        w = getattr(wo_a, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.numel() != 2 * 1024 * 4096:
            continue
        old_ptr = w.data.view(2, 1024, 4096).data_ptr()
        twin = _TWINS.get(old_ptr)
        if twin is None:
            continue
        if not torch.equal(dequant_into_scratch(twin, w.device), w.data.view(2, 1024, 4096)):
            mismatched += 1
            continue                   # the stock path must see the identical bf16 weight
        del _TWINS[old_ptr]            # the freed address may be reused by another tensor
        tag = torch.zeros(1, dtype=torch.bfloat16, device=w.device)
        w.data = tag.as_strided(tuple(w.shape), (0,) * w.dim())
        _TWINS[tag.data_ptr()] = twin
        _DROPPED.add(tag.data_ptr())
        freed += 1
    torch.cuda.empty_cache()
    # Fall-through site: a twin whose dequantized copy is not bit-identical keeps its bf16 storage
    # (DSV41_WO_A_W8_DROP skipped for it). Kept; the line starts with WARNING when any is kept.
    print(f"[wo_a_w8] {'WARNING DSV41_WO_A_W8_DROP: ' if mismatched else ''}released the bf16 copy of {freed} wo_a weights ({freed * 16.8:.0f} MB); "
          f"{mismatched} kept (dequantized copy not bit-identical)", flush=True)


def install_mid(dsv4_module):
    """deepseek_v4._apply_wo_a_bf16_matmul: 9..192 rows (verify/draft at c2..c32) from the fp8 twin."""
    if not (ENABLED and (MID or DROP)) or getattr(dsv4_module, "_dsv41_wo_a_mid", False):
        return
    orig = dsv4_module._apply_wo_a_bf16_matmul
    if QX and not getattr(orig, "_dsv41_wo_a_input_fp8", False):
        # the routes below that fall through to `orig` must find o rounded there
        raise RuntimeError("DSV41_WO_A_INPUT_FP8: install adapter/wo_a_input_fp8.py before wo_a_w8's mid/drop wrapper")
    dsv4_module._dsv41_wo_a_mid = True

    swizzled = getattr(dsv4_module, "Mxfp8SwizzledInput", None)

    def _dropped_path(o, wo_a, twin, m, a, kw):
        std = (o.dtype == torch.bfloat16 and o.shape[1:] == (2, 4096) and o.stride(2) == 1
               and o.stride(1) == 4096 and o.stride(0) >= 8192)
        if std and 2 <= m <= 8:
            p = _partial_w8(o, twin, m, qx=QX)
            if kw.get("fuse_mxfp8_quant", False) and swizzled is not None:
                return swizzled(*_REFS["quantize_partial"](p))
            result = torch.empty((m, 2, 1024), dtype=o.dtype, device=o.device)
            _REFS["reduce_k"][(triton.cdiv(m * 2048, 256),)](p, result, m * 2048, num_warps=4)
            return result
        if std and 9 <= m <= 192 and kw.get("is_target_verify", False) and not kw.get("is_prefill", False):
            return wo_a_mid(o, twin, qx=QX)
        return orig(o, dequant_into_scratch(twin, o.device), *a, **kw)   # bit-identical bf16 weight

    def _apply_wo_a_bf16_matmul(o, wo_a, *a, **kw):
        m = o.shape[0]
        if _DROPPED and wo_a.data_ptr() in _DROPPED:
            return _dropped_path(o, wo_a, _TWINS[wo_a.data_ptr()], m, a, kw)
        if not MID:
            return orig(o, wo_a, *a, **kw)
        # verify/draft graphs only: prefill keeps the stock kernel (identical prefill numerics)
        if 9 <= m <= 192 and kw.get("is_target_verify", False) and not kw.get("is_prefill", False):
            twin = _TWINS.get(wo_a.data_ptr())
            if (twin is not None and o.dtype == torch.bfloat16 and o.shape[1:] == (2, 4096)
                    and o.stride(2) == 1 and o.stride(1) == 4096 and o.stride(0) >= 8192):
                return wo_a_mid(o, twin, qx=QX)
        return orig(o, wo_a, *a, **kw)

    dsv4_module._apply_wo_a_bf16_matmul = _apply_wo_a_bf16_matmul
    print("[wo_a_w8] mid path armed: 9..192 rows from the fp8 twin", flush=True)


def _patch_kernels(dsv4_module, kernel_module):
    for name in ("wo_a_bf16_small_batch", "wo_a_bf16_small_batch_mxfp8", "_wo_a_reduce", "_quantize_partial"):
        if not hasattr(kernel_module, name):
            raise RuntimeError(f"DSV41_WO_A_W8: {kernel_module.__name__}.{name} is gone; engine drifted")
    orig_small = kernel_module.wo_a_bf16_small_batch
    orig_mx = kernel_module.wo_a_bf16_small_batch_mxfp8
    reduce_k = kernel_module._wo_a_reduce
    quantize_partial = kernel_module._quantize_partial
    _REFS["reduce_k"], _REFS["quantize_partial"] = reduce_k, quantize_partial

    def small(x, weight):
        twin = _TWINS.get(weight.data_ptr())
        if twin is None:
            return orig_small(x, weight)
        m = x.shape[0]
        result = torch.empty((m, 2, 1024), dtype=x.dtype, device=x.device)
        reduce_k[(triton.cdiv(m * 2048, 256),)](_partial_w8(x, twin, m), result, m * 2048, num_warps=4)
        return result

    def small_mx(x, weight):
        twin = _TWINS.get(weight.data_ptr())
        if twin is None:
            return orig_mx(x, weight)
        return quantize_partial(_partial_w8(x, twin, x.shape[0]))

    kernel_module.wo_a_bf16_small_batch = small
    kernel_module.wo_a_bf16_small_batch_mxfp8 = small_mx
    # deepseek_v4 imported both names into its own namespace
    if getattr(dsv4_module, "wo_a_bf16_small_batch", None) is not orig_small or \
            getattr(dsv4_module, "wo_a_bf16_small_batch_mxfp8", None) is not orig_mx:
        raise RuntimeError("DSV41_WO_A_W8: deepseek_v4 no longer imports the wo_a small-batch kernels by name")
    dsv4_module.wo_a_bf16_small_batch = small
    dsv4_module.wo_a_bf16_small_batch_mxfp8 = small_mx


def build_twins(model: torch.nn.Module) -> tuple[int, int]:
    made = kept = 0
    for mod in model.modules():
        w = getattr(getattr(mod, "wo_a", None), "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.numel() != 2 * 1024 * 4096:
            continue
        w3 = w.data.view(2, 1024, 4096)
        twin = make_twin(w3)
        if twin is None:
            kept += 1
            continue
        _TWINS[w3.data_ptr()] = twin
        made += 1
    # Override site: when no wo_a weight has an exact fp8 twin, a set DSV41_WO_A_W8 took no effect
    # (every weight stayed on bf16). It now stops the boot.
    if made == 0 and kept > 0:
        raise RuntimeError(f"DSV41_WO_A_W8={os.environ.get('DSV41_WO_A_W8', '')!r}: none of {kept} wo_a weights has "
                           f"an exact fp8 twin, so every call stays on bf16; refusing to boot")
    # Fall-through site: a wo_a weight whose fp8 twin is not bit-exact stays on the bf16 path. Kept
    # (per-weight eligibility); the line starts with WARNING when any weight is kept, at load time.
    print(f"[wo_a_w8] {'WARNING DSV41_WO_A_W8: ' if kept else ''}fp8 twins for {made} wo_a weights, "
          f"{kept} left on bf16", flush=True)
    return made, kept


def _wrap_load(cls):
    if getattr(cls, "_dsv41_wo_a_w8", False):
        return
    cls._dsv41_wo_a_w8 = True
    orig = cls.load_weights

    def load_weights(self, *a, **kw):
        out = orig(self, *a, **kw)
        build_twins(self)
        if DROP:
            drop_bf16(self)
        return out

    cls.load_weights = load_weights


def install_model(dsv4_module):
    """sglang.srt.models.deepseek_v4 (target model; also owns the wo_a dispatch the draft uses)."""
    if not ENABLED:
        return
    import importlib
    _patch_kernels(dsv4_module, importlib.import_module("sglang.kernels.ops.attention.dsv4.wo_a_bf16"))
    install_mid(dsv4_module)
    _wrap_load(dsv4_module.DeepseekV4ForCausalLM)


def install_dspark(dspark_module):
    """sglang.srt.models.deepseek_v4_dspark: twins for the draft's three wo_a as well."""
    if ENABLED:
        _wrap_load(dspark_module.DeepseekV4ForCausalLMDSpark)
