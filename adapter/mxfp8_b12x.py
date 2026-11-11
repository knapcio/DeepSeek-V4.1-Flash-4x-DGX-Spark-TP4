"""Route SGLang's block-fp8-as-MXFP8 dense linears to a FlashInfer backend that
suits GB10 (SM121).

SGLang's --fp8-gemm-backend only knows the cutlass / cute-dsl / trtllm MXFP8
kernels. The CUTLASS SM120 kernel uses a 128x32x128 tile: at M=6 (DSpark verify
at batch 1) it pads M to 128 and streams 32 weight columns per tile, which the
2026-09-10 profile measured at 50-75 GB/s on every projection (52 ms of a 118 ms
decode step, the same as the Triton kernel it replaced). FlashInfer's b12x
warp-level MMA kernel has 16|32 x 64|128 tiles for small M and is what
mm_mxfp8's own 'auto' heuristic prefers on SM120/121; 'cudnn' is the other
SM12x option. DSV41_MXFP8_BACKEND selects it (default b12x; '', 'cutlass',
'off', or '0' restores SGLang's choice). A shape the chosen backend rejects falls back to
cutlass for the rest of the process, logged once.
"""
import hashlib
import importlib.util
import logging
import os

import torch

from mxfp8_mode import mxfp8_backend_disabled

logger = logging.getLogger(__name__)

_REJECTED = set()

# FlashInfer PR #7676 (issue #7675): the b12x dense GEMM's TMA stage refill needs a proxy fence after producer_acquire.
# The image applies it at build time (runtime/flashinfer_fi7676/apply_fi7676.py); this only reports which source the
# process will compile. sha256 of flashinfer 0.6.18's file without / with the fence.
_FI7676_ABSENT = 'eee668e5c143f1c31690ad8a7b44f65050fee3232df1e1c484a3789c2ef0d960'
_FI7676_PRESENT = 'd583991c917f8cbd609f412a82651445b3f9fa1e4cac9adf24e7176e3c741d05'
# v2.7: flashinfer 0.7.2 (d84335d8) and 0.7.3 (775207de) ship the same file as main at the PR's base (blob 4451824a);
# with the fence it is the PR head's blob e8b94b00 exactly. Every pristine -> patched pair the receipt knows:
_FI7676_PAIRS = {
    _FI7676_ABSENT: _FI7676_PRESENT,
    '6739702e27afad21767b71024678b35c6892e3f9c74aaff375fb3f6734399f86':
        '4924ff095448f59b204b9f5ba8627ec9769eddd23777b34a1212bfc754b0e8b6',
}
_FI7676_LINE = 'cute.arch.fence_proxy("async.shared", space="cta")'
_FI7676_REL = os.path.join('gemm', 'kernels', 'dense_blockscaled_gemm_sm120_b12x.py')


def fi7676_receipt(path=None):
    """One log line naming the b12x dense GEMM source this process compiles and whether it has the #7676 fence."""
    try:
        if path is None:
            spec = importlib.util.find_spec('flashinfer')
            path = os.path.join(list(spec.submodule_search_locations)[0], _FI7676_REL)
        with open(path, 'rb') as f:
            data = f.read()
    except Exception as exc:  # a receipt never breaks loading
        return f'DSV41 b12x dense GEMM fence (FlashInfer #7676): unknown ({exc})'
    digest = hashlib.sha256(data).hexdigest()
    if digest in _FI7676_PAIRS.values():
        state = 'present'
    elif digest in _FI7676_PAIRS:
        state = 'ABSENT'
    else:
        state = 'unpinned (fence line %s)' % ('found' if _FI7676_LINE.encode() in data.split(
            b'mainloop_pipeline.producer_acquire(mainloop_producer_state)', 1)[-1][:200] else 'not found')
    return f'DSV41 b12x dense GEMM fence (FlashInfer #7676): {state} sha256={digest[:16]}'


def _agree(flag, device):
    """TP-wide MIN of a rank-local verdict (adapter/replicated_split.py's pattern): every rank serves the same kernel for
    a key, or none does. Rank-local only when no TP group exists (unit tests, single process). Used by
    adapter/dense_tile_table.py's per-(shape, M) self-check."""
    try:
        from sglang.srt.distributed import get_tp_group
        group = get_tp_group()
        if group is None or group.world_size <= 1:
            return bool(flag)
        t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
        return bool(t.item())
    except Exception as exc:  # noqa: BLE001 - no group: the verdict stays rank-local and says so
        logger.warning('DSV41 MXFP8: no TP agreement (%r): rank-local verdict', exc)
        return bool(flag)


def install(module):
    want = os.environ.get('DSV41_MXFP8_BACKEND', 'b12x').strip()
    if mxfp8_backend_disabled(want):
        return
    original = module.flashinfer_mxfp8_blockscaled_linear
    # DSV41_PREFILL_GEMM_TACTIC (adapter/prefill_gemm_tactic.py, off by default): measured prefill shapes on CUTLASS
    # tactic 5 after a boot bit-identity check. Wraps the raw mm_mxfp8 FIRST, so the row split below captures it.
    try:
        import prefill_gemm_tactic
        prefill_gemm_tactic.install_fp8_utils(module, want)
    except Exception as exc:  # an opt-in knob must not stop the stock route
        logger.warning('DSV41_PREFILL_GEMM_TACTIC not installed: %r', exc)
    # DSV41_PREFILL_GEMM_ROW_SPLIT (adapter/prefill_gemm_split.py, off by default): K >= 5120 GEMMs of >= 8192 rows as
    # row blocks of the same kernel, bit-identity checked per shape. None = not installed, the path below is unchanged.
    # DSV41_DENSE_TILE_TABLE (adapter/dense_tile_table.py, off by default): the narrow decode GEMMs on the measured b12x tile,
    # self-checked bit for bit per (shape, M), TP-agreed. None = not installed, the path below is unchanged.
    tile_table = None
    try:
        import dense_tile_table
        if dense_tile_table.wanted_at_install():
            tile_table = dense_tile_table.make(module, want)
    except Exception as exc:  # an opt-in knob must not stop the stock route
        logger.warning('DSV41_DENSE_TILE_TABLE not installed: %r', exc)
        tile_table = None
    row_split = None
    try:
        import prefill_gemm_split
        if prefill_gemm_split.wanted_at_install():
            row_split = prefill_gemm_split.make_linear_split(module)
            if row_split is not None:
                logger.warning('DSV41_PREFILL_GEMM_ROW_SPLIT armed: %s (rows, min K, min M, N filter)',
                               prefill_gemm_split.config())
    except Exception as exc:  # an opt-in knob must not stop the stock route
        logger.warning('DSV41_PREFILL_GEMM_ROW_SPLIT not installed: %r', exc)
        row_split = None

    def linear(input, weight, weight_scale, input_scale=None, bias=None,
               output_dtype=None, backend='cutlass', pin_tactic=False):
        key = (int(weight.shape[0]), int(weight.shape[1]))
        if backend == 'cutlass' and key not in _REJECTED:
            if row_split is not None:
                out = row_split(original, input, weight, weight_scale, input_scale, bias,
                                output_dtype, want, pin_tactic)
                if out is not None:
                    return out
            try:
                if tile_table is not None:
                    out = tile_table(original, (input, weight, weight_scale, input_scale, bias, output_dtype), want, pin_tactic)
                    if out is not None:
                        return out
                return original(input, weight, weight_scale, input_scale, bias,
                                output_dtype, backend=want, pin_tactic=pin_tactic)
            except Exception as exc:  # shape/backend rejection: keep serving
                _REJECTED.add(key)
                logger.warning('DSV41 MXFP8 backend %r rejected N=%s K=%s (%s); '
                               'cutlass for this shape', want, key[0], key[1], exc)
        return original(input, weight, weight_scale, input_scale, bias,
                        output_dtype, backend=backend, pin_tactic=pin_tactic)

    module.flashinfer_mxfp8_blockscaled_linear = linear
    logger.warning(fi7676_receipt())

    # TP=3 pads heads 64->96 and o_groups 8->12, so rank 2's wq_b / wo_b shards are
    # entirely padding: zero weights with zero block scales. A zero scale is not a
    # power of two, so block_fp8_scale_to_mxfp8_e8m0 raised and those 86 layers stayed
    # on the Triton kernel (22 ms/step on rank 2, the slowest rank). The fp8 values in
    # such blocks are zero, so any scale gives the same product; use 1.0 (e8m0 code 127).
    original_encode = module.block_fp8_scale_to_mxfp8_e8m0

    def encode(weight_scale, weight_shape, weight_block_size):
        scale = weight_scale.detach()
        if scale.numel() and bool((scale <= 0).any()):
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return original_encode(scale, weight_shape, weight_block_size)

    module.block_fp8_scale_to_mxfp8_e8m0 = encode
    logger.warning('DSV41 MXFP8 dense linears routed to FlashInfer backend %r '
                   '(zero block scales of padded shards encoded as 1.0)', want)


def _repair_padded_scales(layer, block_size):
    """Set every block scale that is not a positive power of two to 1.0 when its
    weight block is entirely zero (a TP-padded shard). Returns a diagnostic string
    for anything that was, or could not be, repaired; None when nothing was wrong."""
    ws = getattr(layer, 'weight_scale_inv', None)
    w = getattr(layer, 'weight', None)
    if ws is None or w is None or ws.data.ndim != 2:
        return None
    s = ws.data
    if s.dtype != torch.float32:
        return f'scale dtype {s.dtype} (not float32), left alone'
    f = s.contiguous()
    bits = f.view(torch.int32)
    bad = ~(((bits & 0x7FFFFF) == 0) & (f > 0))
    if not bool(bad.any()):
        return None
    n, k = w.shape
    bn, bk = block_size
    sn, sk = f.shape
    if n % bn or k % bk or sn != n // bn or sk != k // bk:
        return (f'{int(bad.sum())} bad scales but shape {tuple(w.shape)} / {tuple(f.shape)} '
                f'does not tile by {block_size}, left alone')
    zero = ((w.data.view(torch.uint8) & 0x7F) == 0).view(n // bn, bn, k // bk, bk).all(3).all(1)
    fix = bad & zero
    vals = f[bad][:4].tolist()
    if bool(fix.any()):
        f[fix] = 1.0
        s.copy_(f)
    return (f'{int(bad.sum())} invalid scales (e.g. {vals}), {int(fix.sum())} on all-zero '
            f'weight blocks set to 1.0, {int((bad & ~zero).sum())} on non-zero blocks left alone')


def install_fp8(module):
    """Hook Fp8LinearMethod._prepare_block_fp8_as_mxfp8 so padded shards pass the e8m0
    check whatever their (never written) scale memory holds, and log what was found."""
    if mxfp8_backend_disabled(os.environ.get('DSV41_MXFP8_BACKEND', 'b12x')):
        return
    cls = getattr(module, 'Fp8LinearMethod', None)
    if cls is None or not hasattr(cls, '_prepare_block_fp8_as_mxfp8'):
        logger.warning('DSV41: Fp8LinearMethod._prepare_block_fp8_as_mxfp8 not found; no scale repair')
        return
    original = cls._prepare_block_fp8_as_mxfp8

    def prepare(self, layer):
        try:
            note = _repair_padded_scales(layer, self.weight_block_size)
        except Exception as exc:  # never let diagnostics break loading
            note = f'repair failed: {exc}'
        if note:
            logger.warning('DSV41 block scales for %s: %s', getattr(layer, 'prefix', layer.__class__.__name__), note)
        return original(self, layer)

    cls._prepare_block_fp8_as_mxfp8 = prepare
    logger.warning('DSV41 padded-shard block-scale repair installed')
