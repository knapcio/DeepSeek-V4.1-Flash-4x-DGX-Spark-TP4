"""wo_a input rounded to FP8 as DeepSeek's reference does. Gated on DSV41_WO_A_INPUT_FP8 (default off).

DeepSeek's reference inference code for V4.1-Flash (Hugging Face deepseek-ai/DeepSeek-V4.1-Flash,
inference/model.py, revision 4dd75a8b44 of 2026-10-12; the same file ships with
deepseek-ai/DeepSeek-V4.1-Flash-1017) quantizes the attention output `o` to FP8 and dequantizes it
before the bf16 `wo_a` einsum, in `Attention.forward` and in the DSpark draft's
`DSparkAttention.forward`:

    o = o.view(bsz, seqlen, n_local_groups, -1)
    act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)     # 32, "ue8m0", e8m0, inplace

`act_quant(..., inplace=True)` (inference/kernel.py) works on blocks of 32 consecutive values along
the last dim (each group's 4096 features): amax floored at 1e-4, scale 2^ceil(log2(amax * (1/448)))
taken from the fp32 bits, x / scale clamped to +-448, rounded to e4m3 (nearest, ties to even) and
multiplied back to bf16. The SGLang engine and adapter/wo_a_w8.py feed bf16 `o` unrounded (sglang#45185
adds the same step upstream, opt-in).

What this does, with the flag on:
  * `deepseek_v4._apply_wo_a_bf16_matmul` (the bf16 wo_a dispatch of the target's MQALayer and, through
    its by-name import, of the DSpark draft's attention) is wrapped: `o` is rounded in place by one
    small Triton kernel, then the stock dispatch runs (GEMV, small-batch, bmm, einsum).
  * adapter/wo_a_w8.py's own routes (2-8 rows after DSV41_WO_A_W8_DROP, 9-192 verify/draft rows
    under DSV41_WO_A_W8_MID) round the `o` tile in registers right after loading it (`qdq_tile`),
    so those routes need no extra launch. Every other route goes through the wrapper.
The rounded value is exactly representable in bf16 (an e4m3 value times a power of two between
2^-22 and 2^126), so the in-register and the standalone rounding give the same bf16 operand.

TP: the feature dim of `o` is split over the ranks by whole groups (8 groups of 8 heads x 512; 2 groups
per rank at TP4) and 32 divides 512, so no 32-block straddles a head, a group or a rank boundary:
each rank rounding its own slice is the reference's rounding. Elementwise per block, no reduction
across blocks or rows: deterministic and identical on every rank.

Not covered: the fp8 wo_a GEMM path (DeepGEMM fp8_einsum, taken only for 128x128-block checkpoints;
both V4.1-Flash checkpoints use 32x32 blocks, so wo_a is dequantized to bf16 at load and takes the
bf16 dispatch) and the draft's fp32 einsum under SGLANG_DSPARK_FAST_KERNEL=0 (warned at install).
"""
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:          # the CPU tests run without Triton; the image always has it
    triton = None
    tl = None

ENABLED = os.environ.get("DSV41_WO_A_INPUT_FP8", "0").strip().lower() not in ("0", "", "off", "false")
BLOCK = 32
_WARNED = []
_FP8_MAX = 448.0
_AMAX_FLOOR = 1e-4


# ------------------------------------------------------------------------------------------
# torch reference (CPU tests, and the CPU fallback of round_o_)
# ------------------------------------------------------------------------------------------
def _scale_exponent(amax: torch.Tensor) -> torch.Tensor:
    """int32 e with scale = 2^e = 2^ceil(log2(max(amax, 1e-4) * fp32(1/448))), from the fp32 bits."""
    t = amax.float().clamp_min(_AMAX_FLOOR) * torch.tensor(1.0 / _FP8_MAX, dtype=torch.float32)
    bits = t.view(torch.int32)
    return ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def reference_qdq(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Out-of-place torch version of the reference act_quant(x, block, "ue8m0", ..., inplace=True)."""
    if x.shape[-1] % block:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of {block}")
    f = x.float().reshape(-1, block)
    e = _scale_exponent(f.abs().amax(dim=1))
    s = ((e + 127) << 23).view(torch.float32)[:, None]
    v = (f / s).clamp(-_FP8_MAX, _FP8_MAX)
    return (v.to(torch.float8_e4m3fn).float() * s).to(x.dtype).reshape(x.shape)


# ------------------------------------------------------------------------------------------
# Triton
# ------------------------------------------------------------------------------------------
if triton is not None:
    # The e4m3 rounding: the hardware conversion (cvt.rn.satfinite, round to nearest even, as the
    # reference's tilelang cast) by default; DSV41_WO_A_INPUT_FP8_EMULATE=1 does the same rounding in
    # fp32 arithmetic (for checking it against the hardware, and under TRITON_INTERPRET, whose fp8
    # conversion does not round ties to even).
    _E4M3_NATIVE = tl.constexpr(os.environ.get("DSV41_WO_A_INPUT_FP8_EMULATE", "0").strip() in ("0", ""))

    @triton.jit
    def _e4m3_round(v):
        """v fp32, |v| <= 448 -> the nearest e4m3 value (ties to even, signed zero kept), fp32."""
        if _E4M3_NATIVE:
            q = v.to(tl.float8e4nv).to(tl.float32)
        else:
            a = tl.abs(v)
            ea = tl.maximum(((a.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, -6)  # subnormals below 2^-6
            ulp = ((ea - 3 + 127) << 23).to(tl.float32, bitcast=True)                 # 3 mantissa bits
            r = a * ((127 - (ea - 3)) << 23).to(tl.float32, bitcast=True)              # a / ulp, exact, < 16
            r = (r + 12582912.0) - 12582912.0                                          # 1.5 * 2^23: ties to even
            q = r * ulp
            q = (q.to(tl.int32, bitcast=True) | (v.to(tl.int32, bitcast=True) & -2147483648)).to(
                tl.float32, bitcast=True)
        return q

    @triton.jit
    def _qdq_rows32(x):
        """x fp32 [N, 32] -> fp32 [N, 32]: each row rounded as act_quant rounds one block.
        x * 2^-e instead of x / 2^e: both are the exact quotient (a power of two), and the
        multiply cannot pick up an approximate division."""
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        t = amax * (1.0 / 448.0)
        bits = t.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)   # -22 <= e <= 120
        s = ((e + 127) << 23).to(tl.float32, bitcast=True)                         # 2^e
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)                       # 2^-e
        v = tl.minimum(tl.maximum(x * inv[:, None], -448.0), 448.0)
        return _e4m3_round(v) * s[:, None]

    @triton.jit
    def qdq_tile(x, R: tl.constexpr, C: tl.constexpr):
        """A loaded bf16 tile [R, C] of o (C a multiple of 32, columns 32-aligned within a group)
        -> the same tile rounded as the reference rounds o, bf16. Rows padded with 0 stay 0."""
        y = _qdq_rows32(tl.reshape(x.to(tl.float32), (R * (C // 32), 32)))
        return tl.reshape(y, (R, C)).to(tl.bfloat16)

    @triton.jit
    def _qdq_inplace_kernel(X, S0, S1, NB: tl.constexpr):
        row, group, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        ptr = X + row.to(tl.int64) * S0 + group.to(tl.int64) * S1 + chunk * (NB * 32) + tl.arange(0, NB * 32)
        x = tl.load(ptr)
        y = qdq_tile(tl.reshape(x, (NB, 32)), NB, 32)
        tl.store(ptr, tl.reshape(y, (NB * 32,)))
else:
    qdq_tile = None


def round_o_(o: torch.Tensor) -> torch.Tensor:
    """o [T, G, D] bf16 (last dim contiguous, D a multiple of 32), rounded in place along D in
    blocks of 32, as the reference's act_quant(o, 32, "ue8m0", inplace=True). Returns o."""
    if o.dim() != 3 or o.shape[-1] % BLOCK or o.dtype != torch.bfloat16 or (o.numel() and o.stride(-1) != 1):
        raise ValueError(f"DSV41_WO_A_INPUT_FP8: unexpected o {tuple(o.shape)} {o.dtype} {o.stride()}")
    if o.numel() == 0:
        return o
    # Fall-through site (GB10 / sm_121 relevant): a CUDA tensor without Triton is rounded by the torch
    # reference instead of the Triton kernel. Same values; kept, one WARNING line per process.
    if not o.is_cuda or triton is None:
        if o.is_cuda and not _WARNED:
            _WARNED.append(True)
            print("[wo_a_input_fp8] WARNING DSV41_WO_A_INPUT_FP8: Triton unavailable; kernel chosen: torch "
                  "reference_qdq on the GPU (logged once)", flush=True)
        o.copy_(reference_qdq(o))
        return o
    t, g, d = o.shape
    nb = 32 if d % 1024 == 0 else 1
    _qdq_inplace_kernel[(t, g, d // (nb * 32))](o, o.stride(0), o.stride(1), NB=nb, num_warps=4)
    return o


# ------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------
def wrap(fn):
    def _apply_wo_a_bf16_matmul(o, wo_a, *args, **kwargs):
        return fn(round_o_(o), wo_a, *args, **kwargs)

    _apply_wo_a_bf16_matmul._dsv41_wo_a_input_fp8 = True
    return _apply_wo_a_bf16_matmul


def install_model(dsv4_module):
    """sglang.srt.models.deepseek_v4, BEFORE adapter/wo_a_w8.py (its mid/drop wrapper must call this one
    as its `orig`, and round in registers on its own routes)."""
    if not ENABLED or getattr(dsv4_module, "_dsv41_wo_a_input_fp8", False):
        return
    fn = getattr(dsv4_module, "_apply_wo_a_bf16_matmul", None)
    if fn is None:
        raise RuntimeError("DSV41_WO_A_INPUT_FP8: deepseek_v4._apply_wo_a_bf16_matmul is gone; engine drifted")
    if getattr(dsv4_module, "_dsv41_wo_a_mid", False):
        raise RuntimeError("DSV41_WO_A_INPUT_FP8: must be installed before adapter/wo_a_w8.py's mid/drop wrapper")
    dsv4_module._dsv41_wo_a_input_fp8 = True
    dsv4_module._apply_wo_a_bf16_matmul = wrap(fn)
    print("[wo_a_input_fp8] wo_a input rounded to e4m3 (32-blocks, power-of-two scales) as the "
          "2026-10-12 reference does", flush=True)


def install_dspark(dspark_module):
    """sglang.srt.models.deepseek_v4_dspark: the draft must call the wrapped dispatch."""
    if not ENABLED:
        return
    import sys
    v4 = sys.modules.get("sglang.srt.models.deepseek_v4")
    if v4 is None or getattr(dspark_module, "_apply_wo_a_bf16_matmul", None) is not v4._apply_wo_a_bf16_matmul:
        raise RuntimeError("DSV41_WO_A_INPUT_FP8: the draft does not call deepseek_v4's wrapped wo_a dispatch")
    envs = getattr(dspark_module, "envs", None)
    fast = getattr(envs, "SGLANG_DSPARK_FAST_KERNEL", None)
    # Override site: SGLANG_DSPARK_FAST_KERNEL=0 sends the draft through an fp32 einsum wo_a that is not
    # rounded, so DSV41_WO_A_INPUT_FP8 did not take effect on the draft (only a log line). Both are user
    # settings in conflict, so this now stops the boot.
    if fast is not None and not fast.get():
        raise RuntimeError(f"DSV41_WO_A_INPUT_FP8={os.environ.get('DSV41_WO_A_INPUT_FP8', '')!r} conflicts with "
                           f"SGLANG_DSPARK_FAST_KERNEL=0: the draft's fp32 einsum wo_a would not be rounded; "
                           f"refusing to boot")
