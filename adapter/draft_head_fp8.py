"""The DSpark draft's LM head from an fp8 copy of the (shared) target LM head. Default OFF.

DSV41_DRAFT_HEAD_FP8=1. The draft borrows the target's vocab-parallel lm_head (bf16, 32320 x 5120
per rank at TP4) and reads it every step for its 5 block rows. Only the DRAFT gets the fp8 copy
(e4m3 + one power-of-two scale per 32x32 block): the target's logits, and therefore every accepted
token, are untouched. Offline the draft's acceptance moved 3.0882 -> 3.0875 tokens/step.

DSV41_DRAFT_HEAD_FP4=1 (default OFF; needs DSV41_DRAFT_HEAD_FP8=1): a second, MXFP4 copy of the same head (OCP MX: e2m1
values, two per byte with the even k in the low nibble, one e8m0 scale per 1 x 32 run along K; 82.7 + 5.2 MB per rank
instead of the fp8 twin's 165.5 + 0.2 MB) is read for draft calls of at most DSV41_DRAFT_HEAD_FP4_MAX_M rows (default
5 = one request's DSPARK block); larger batches keep the fp8 twin. Draft only: verification decides every token, so
the output is unchanged and only the acceptance can move (offline, diagnostics/drafthead-fp4-20261112: 1017 drafter,
owner thinking prose 3.1214 -> 3.1118 tokens/step = -0.31 %, fenced code -0.28 %, synthetic prose -0.41 %, code 0.00 %).
Quantizer: e = floor(log2(amax)) - 2 per run, values round to nearest e2m1 (ties to even), saturating at 6.
Kernel (M <= 8 rows): a CUDA-core GEMV that streams the codes once (_head_fp4_gemv_kernel): each thread holds BN / (rows
per CTA layer) rows x one 32-k run (16 bytes = one 128-bit load) per K tile, decodes every nibble with three integer ops
straight to the fp32 bits of e2m1 x 2^-126, one multiply by 2^(e-1) gives the exact dequantized weight, and every
activation row runs one fp32 FMA chain per (row, run) in k order; the next tile's codes and scales are loaded before the
current one is decoded (STAGES >= 2). No tensor cores, no shared memory or barrier in the K loop. 9..16 rows and more keep
the tl.dot kernel (_head_fp4_kernel). DSV41_DRAFT_HEAD_FP4_TILE=BN,BK,WARPS,STAGES overrides the M <= 8 launch (BK = the
K tile = 32 x the run lanes; STAGES 1 = no prefetch; scripts/bench_draft_head_fp4.py prints the fastest). The default
launch 32,512,8,1 is the bench's fastest M = 5 launch on a GB10 (2026-11-12: 399 us against the fp8 twin's 733 us, paired
gain +340 us [+331, +351]), the launch the in-boot A/B on v2.7.6 ran. Idea: pico-dgx-ds4 docs/stack4.md lever 1
(Picom-code; flag from blaesus/dsgx10).
Every key is read from the process env (the copy is built at attach, the gate and the row cap per call, so per capture in
CUDA graphs); the in-boot A/B registration of the test branch is not part of this release.
"""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("DSV41_DRAFT_HEAD_FP8", "0").strip() not in ("0", "", "off", "false")
# Draft head rows = DSPARK_BLOCK_SIZE (5) x decode bs: 5..80 on the bs <= 16 graph list. Row tiles cover
# the whole batch so the fp8 head is streamed once per call (16-row tiles re-read it per tile, which
# made M >= 40 slower than the bf16 head). Measured on Spark_04 (real head shard 32320 x 5120, us per
# call, CUDA graph; diagnostics/dsv41-moe-plan-table): M=5 721 (bf16 1374); M=40 16-row 2087 -> 794
# (bf16 1627); M=60 2824 -> 797 (bf16 1559); M=80 16-row 3593 -> 1138 (bf16 1566, today's fallback).
# Every tiling gives bit-identical fp32 logits per row (same K order, one tl.dot chain per row).
# Rows above 64 (bs 14 and 16) stay on the bf16 head as before: moving them to the fp8 copy changes the
# draft proposals there, and a fleet A/B showed no gain at c16 (the gain is at c4-c12, where the retiled
# kernel is bit-identical to the old one). DSV41_DRAFT_HEAD_FP8_MAX_M raises it (<= 128).
MAX_M = min(128, int(os.environ.get("DSV41_DRAFT_HEAD_FP8_MAX_M", "64")))
_TILES = ((16, 16, 256, 4, 3), (64, 64, 128, 4, 3), (128, 128, 64, 8, 3))   # max M, BM, BK, warps, stages
_TWIN = {}          # lm_head weight data_ptr -> (e4m3, exponent)
_TWIN4 = {}         # lm_head weight data_ptr -> (packed e2m1 uint8 [N, K/2], e8m0 uint8 [N, K/32])
_OFF = ("", "0", "off", "false", "no")
FP4_DEFAULT_MAX_M = 5
FP4_DEFAULT_TILE = (32, 512, 8, 1)      # BN, BK, warps, stages of the M <= FP4_GEMV_MAX_M GEMV (stages >= 2: prefetch)
FP4_GEMV_MAX_M = 8                      # rows the GEMV takes (one fp32 accumulator set per row)
_FP4_DOT16 = (16, 32, 256, 4, 3)        # BM, BN, BK, warps, stages: the tl.dot launch for 9..16 rows (the old M <= 16 one)
_FP4_TILES = ((64, 64, 128, 4, 3), (128, 128, 64, 8, 3))   # max M, BM, BK, warps, stages (BN 32) above 16 rows
# E2M1 magnitudes by code 0..7; MIDS are the rounding boundaries between neighbours
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_stats = {"fp4": 0, "fp8": 0, "stock": 0}

def env(name, default=""):
    return os.environ.get(name, default)


def fp4_installed() -> bool:
    """Build gate of the MXFP4 copy: the process env."""
    return os.environ.get("DSV41_DRAFT_HEAD_FP4", "0").strip().lower() not in _OFF


def parse_max_m(value) -> int:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return FP4_DEFAULT_MAX_M
    try:
        v = int(raw)
    except ValueError:
        raise RuntimeError(f"DSV41_DRAFT_HEAD_FP4_MAX_M={value!r}: expected a row count") from None
    if not 1 <= v <= 128:
        raise RuntimeError(f"DSV41_DRAFT_HEAD_FP4_MAX_M={value!r}: must be 1..128")
    return v


def parse_tile(value) -> tuple:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return FP4_DEFAULT_TILE
    try:
        bn, bk, warps, stages = (int(x) for x in raw.split(","))
    except ValueError:
        raise RuntimeError(f"DSV41_DRAFT_HEAD_FP4_TILE={value!r}: expected BN,BK,WARPS,STAGES") from None
    if bn not in (16, 32, 64, 128) or bk not in (64, 128, 256, 512) or warps not in (1, 2, 4, 8) or not 1 <= stages <= 6:
        raise RuntimeError(f"DSV41_DRAFT_HEAD_FP4_TILE={value!r}: BN 16/32/64/128, BK 64..512, warps 1..8, stages 1..6")
    return bn, bk, warps, stages


def fp4_rows() -> int:
    """Row cap of the MXFP4 copy for the call being captured / replayed: 0 while the gate is off for it."""
    if str(env("DSV41_DRAFT_HEAD_FP4", "0") or "0").strip().lower() in _OFF:
        return 0
    return parse_max_m(env("DSV41_DRAFT_HEAD_FP4_MAX_M", ""))


@triton.jit
def _head_fp8_kernel(X, W8, S, Y, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = pid * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < M, 0.0)
        w8 = tl.load(W8 + n[None, :] * K + k[:, None], nmask[None, :], 0.0)
        e = tl.load(S + (n[None, :] // 32) * (K // 32) + k[:, None] // 32, nmask[None, :], 127)
        w = (w8.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x, w)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & nmask[None, :])


def make_twin(w: torch.Tensor):
    """bf16 [N, K] -> (e4m3 [N, K], exponent uint8 [ceil(N/32), K/32]); lossy, draft-only."""
    n, k = w.shape
    pad = (-n) % 32
    wf = torch.nn.functional.pad(w.float(), (0, 0, 0, pad)).view((n + pad) // 32, 32, k // 32, 32)
    amax = wf.abs().amax(dim=(1, 3)).clamp_min(2.0 ** -126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (wf / torch.exp2(e)[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(n + pad, k)[:n].contiguous(), (e + 127).to(torch.uint8).contiguous()


def head_fp8(x: torch.Tensor, twin) -> torch.Tensor:
    """x [M, K] bf16 (M <= MAX_M) -> fp32 logits [M, N]."""
    w8, s = twin
    m, k = x.shape
    n = w8.shape[0]
    y = torch.empty((m, n), dtype=torch.float32, device=x.device)
    # BN 32; M <= 16 keeps the M=5 tuning (BK 256, 4 warps, 3 stages = 230 GB/s of fp8)
    _, bm, bk, warps, stages = next(t for t in _TILES if m <= t[0])
    _head_fp8_kernel[(triton.cdiv(n, 32), triton.cdiv(m, bm))](x.contiguous(), w8, s, y, m, n, k, bm, 32, bk,
                                                              num_warps=warps, num_stages=stages)
    return y


@triton.jit
def _e2m1_x2(c):
    """4-bit e2m1 code (int32, sign in bit 3) -> 2 x its value as int32: 0 1 2 3 4 6 8 12 for codes 0..7."""
    m = c & 7
    sh = tl.maximum((m >> 1) - 1, 0)
    v = tl.where(m < 2, m, (2 + (m & 1)) << sh)
    return tl.where(c >= 8, -v, v)


@triton.jit
def _head_fp4_kernel(X, W4, S, Y, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = pid * BN + tl.arange(0, BN)
    nmask = n < N
    mmask = m[:, None] < M
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kh = k0 // 2 + tl.arange(0, BK // 2)                       # byte index: k = 2 kh (low), 2 kh + 1 (high)
        xe = tl.load(X + m[:, None] * K + 2 * kh[None, :], mmask, 0.0)
        xo = tl.load(X + m[:, None] * K + 2 * kh[None, :] + 1, mmask, 0.0)
        p = tl.load(W4 + n[None, :] * (K // 2) + kh[:, None], nmask[None, :], 0).to(tl.int32)
        e = tl.load(S + n[None, :] * (K // 32) + kh[:, None] // 16, nmask[None, :], 128)
        sc = tl.exp2(e.to(tl.float32) - 128.0)                     # 2^(e - 127) / 2: the codes decode to 2 x value
        wl = (_e2m1_x2(p & 15).to(tl.float32) * sc).to(tl.bfloat16)
        wh = (_e2m1_x2(p >> 4).to(tl.float32) * sc).to(tl.bfloat16)
        acc += tl.dot(xe, wl)
        acc += tl.dot(xo, wh)
    tl.store(Y + m[:, None] * N + n[None, :], acc, mmask & nmask[None, :])


# ---------------------------------------------------------------- the M <= 8 GEMV (bandwidth-bound: the codes stream once)
# Layout (Triton's coalescing of the [BN, RT, 16]-byte code tile): 16 bytes per thread (one 128-bit load = one 32-k run of
# one row), RT run lanes x (32 / RT) row lanes per warp, warps along N; a thread owns BN * RT / (32 * warps) rows. The
# activation pairs are loaded with the SAME [BN, RT, 16] shape (broadcast along BN), so they land in the codes' layout
# (no shared-memory conversion); the scale bytes are loaded as [BN, RT] with the lane -> run map permuted (r * 3 mod RT:
# no lane contiguity), which gives them the reduction's layout. Nothing in the K loop touches shared memory.
# Decode: a code byte b; (b << 28) >> 6 (arithmetic) & 0x81C00000 moves the low nibble's sign to bit 31 and its three
# magnitude bits to bits 24..22, the fp32 bits of e2m1 x 2^-126 (codes 0 / 1 = 0 / 0.5 as exact subnormals; nothing here
# flushes denormals); (b << 24) >> 6 the high nibble. x 2^(e-1) = 2^126 x 2^(e-127) is the exact dequantized weight
# (dequant_fp4's value) for e8m0 bytes <= 128; above 128 (|w| >= 16, never in a real head) the rest, 2^(e-128), scales the
# run's partial sum (exact: a power of two). Each (row, run, activation row) is one fp32 FMA chain in k order.


@triton.jit
def _fp4_halves(t, BN: tl.constexpr, RT: tl.constexpr, H: tl.constexpr):
    """[BN, RT, 2H] -> its even and odd positions along the last dim ([BN, RT, H] each; in-thread, no data movement)."""
    return tl.split(tl.reshape(t, (BN, RT, H, 2)))


@triton.jit
def _fp4_split16(t, BN: tl.constexpr, RT: tl.constexpr):
    """[BN, RT, 16] -> its 16 [BN, RT] slices, in order."""
    a, b = _fp4_halves(t, BN, RT, 8)
    a0, a1 = _fp4_halves(a, BN, RT, 4)
    b0, b1 = _fp4_halves(b, BN, RT, 4)
    a00, a01 = _fp4_halves(a0, BN, RT, 2)
    a10, a11 = _fp4_halves(a1, BN, RT, 2)
    b00, b01 = _fp4_halves(b0, BN, RT, 2)
    b10, b11 = _fp4_halves(b1, BN, RT, 2)
    c0, c8 = tl.split(a00)
    c4, c12 = tl.split(a01)
    c2, c10 = tl.split(a10)
    c6, c14 = tl.split(a11)
    c1, c9 = tl.split(b00)
    c5, c13 = tl.split(b01)
    c3, c11 = tl.split(b10)
    c7, c15 = tl.split(b11)
    return c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15


@triton.jit
def _fp4_row_dot(X, mr, K: tl.constexpr, kp, BN: tl.constexpr, RT: tl.constexpr, acc, pf,
                 l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
                 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    """Activation row mr against the tile: per (n, run) one fp32 FMA chain over its 32 k in order, then acc += part x pf."""
    xi = tl.load(X + mr * (K // 2) + kp)            # int32 pairs: bf16 x[2i] in the low half, x[2i + 1] in the high half
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = _fp4_split16(
        (xi << 16).to(tl.float32, bitcast=True), BN, RT)
    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 = _fp4_split16(
        (xi & -65536).to(tl.float32, bitcast=True), BN, RT)
    s = l0 * e0
    s = tl.fma(h0, o0, s)
    s = tl.fma(l1, e1, s)
    s = tl.fma(h1, o1, s)
    s = tl.fma(l2, e2, s)
    s = tl.fma(h2, o2, s)
    s = tl.fma(l3, e3, s)
    s = tl.fma(h3, o3, s)
    s = tl.fma(l4, e4, s)
    s = tl.fma(h4, o4, s)
    s = tl.fma(l5, e5, s)
    s = tl.fma(h5, o5, s)
    s = tl.fma(l6, e6, s)
    s = tl.fma(h6, o6, s)
    s = tl.fma(l7, e7, s)
    s = tl.fma(h7, o7, s)
    s = tl.fma(l8, e8, s)
    s = tl.fma(h8, o8, s)
    s = tl.fma(l9, e9, s)
    s = tl.fma(h9, o9, s)
    s = tl.fma(l10, e10, s)
    s = tl.fma(h10, o10, s)
    s = tl.fma(l11, e11, s)
    s = tl.fma(h11, o11, s)
    s = tl.fma(l12, e12, s)
    s = tl.fma(h12, o12, s)
    s = tl.fma(l13, e13, s)
    s = tl.fma(h13, o13, s)
    s = tl.fma(l14, e14, s)
    s = tl.fma(h14, o14, s)
    s = tl.fma(l15, e15, s)
    s = tl.fma(h15, o15, s)
    return tl.fma(s, pf, acc)


@triton.jit
def _fp4_tile_rows(X, K: tl.constexpr, MR: tl.constexpr, BN: tl.constexpr, RT: tl.constexpr, pi, ei, kb,
                   acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7):
    """One K tile: decode the [BN, RT, 16] code bytes pi with their e8m0 bytes ei, run every activation row over it."""
    f = ((tl.minimum(ei, 128) + 126) << 23).to(tl.float32, bitcast=True)[:, :, None]   # 2^(e-1)
    pf = ((tl.maximum(ei, 128) - 1) << 23).to(tl.float32, bitcast=True)               # 2^(e-128) above 128, else 1
    q = pi.to(tl.int32)
    wl = (((q << 28) >> 6) & -2118123520).to(tl.float32, bitcast=True) * f     # even k (low nibble), exact
    wh = (((q << 24) >> 6) & -2118123520).to(tl.float32, bitcast=True) * f     # odd k (high nibble)
    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15 = _fp4_split16(wl, BN, RT)
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15 = _fp4_split16(wh, BN, RT)
    acc0 = _fp4_row_dot(X, 0, K, kb, BN, RT, acc0, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14,
                        l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    if MR > 1:
        acc1 = _fp4_row_dot(X, 1, K, kb, BN, RT, acc1, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                            l14, l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    if MR > 2:
        acc2 = _fp4_row_dot(X, 2, K, kb, BN, RT, acc2, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                            l14, l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    if MR > 3:
        acc3 = _fp4_row_dot(X, 3, K, kb, BN, RT, acc3, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                            l14, l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    if MR > 4:
        acc4 = _fp4_row_dot(X, 4, K, kb, BN, RT, acc4, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                            l14, l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    if MR > 5:
        acc5 = _fp4_row_dot(X, 5, K, kb, BN, RT, acc5, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                            l14, l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    if MR > 6:
        acc6 = _fp4_row_dot(X, 6, K, kb, BN, RT, acc6, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                            l14, l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    if MR > 7:
        acc7 = _fp4_row_dot(X, 7, K, kb, BN, RT, acc7, pf, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                            l14, l15, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)
    return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7


@triton.jit
def _head_fp4_gemv_kernel(X, W4, S, Y, N, K: tl.constexpr, MR: tl.constexpr, BN: tl.constexpr, RT: tl.constexpr,
                          PF: tl.constexpr):
    """X: [MR, K / 2] int32 (bf16 pairs), W4: [N, K / 2] codes, S: [N, K / 32] e8m0, Y: [MR, N] fp32. Grid: cdiv(N, BN)."""
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    rs = (tl.arange(0, RT) * 3) % RT                  # lane -> run within the tile (see the layout note above)
    b = tl.arange(0, 16)
    acc0 = tl.zeros((BN, RT), tl.float32)
    acc1 = tl.zeros((BN, RT), tl.float32)
    acc2 = tl.zeros((BN, RT), tl.float32)
    acc3 = tl.zeros((BN, RT), tl.float32)
    acc4 = tl.zeros((BN, RT), tl.float32)
    acc5 = tl.zeros((BN, RT), tl.float32)
    acc6 = tl.zeros((BN, RT), tl.float32)
    acc7 = tl.zeros((BN, RT), tl.float32)
    m3 = tl.broadcast_to(nmask[:, None, None], (BN, RT, 16))
    wrow = W4 + n[:, None, None] * (K // 2)
    srow = S + n[:, None] * (K // 32)
    kb = tl.broadcast_to(rs[None, :, None] * 16 + b[None, None, :], (BN, RT, 16))   # byte = activation-pair index
    if PF:
        pi = tl.load(wrow + kb, m3, 0)
        ei = tl.load(srow + rs[None, :], nmask[:, None], 0).to(tl.int32)
    for k0 in range(0, K, 32 * RT):
        run = k0 // 32 + rs
        kb = tl.broadcast_to(run[None, :, None] * 16 + b[None, None, :], (BN, RT, 16))
        if PF:
            # the next tile's codes and scales are in flight while this one is decoded (masked off after the last)
            more = k0 + 32 * RT < K
            run_n = run + RT
            kb_n = tl.broadcast_to(run_n[None, :, None] * 16 + b[None, None, :], (BN, RT, 16))
            p_n = tl.load(wrow + kb_n, m3 & more, 0)
            e_n = tl.load(srow + run_n[None, :], nmask[:, None] & more, 0).to(tl.int32)
            acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _fp4_tile_rows(
                X, K, MR, BN, RT, pi, ei, kb, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7)
            pi = p_n
            ei = e_n
        else:
            p_c = tl.load(wrow + kb, m3, 0)
            e_c = tl.load(srow + run[None, :], nmask[:, None], 0).to(tl.int32)
            acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _fp4_tile_rows(
                X, K, MR, BN, RT, p_c, e_c, kb, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7)
    tl.store(Y + n, tl.sum(acc0, axis=1), nmask)
    if MR > 1:
        tl.store(Y + N + n, tl.sum(acc1, axis=1), nmask)
    if MR > 2:
        tl.store(Y + 2 * N + n, tl.sum(acc2, axis=1), nmask)
    if MR > 3:
        tl.store(Y + 3 * N + n, tl.sum(acc3, axis=1), nmask)
    if MR > 4:
        tl.store(Y + 4 * N + n, tl.sum(acc4, axis=1), nmask)
    if MR > 5:
        tl.store(Y + 5 * N + n, tl.sum(acc5, axis=1), nmask)
    if MR > 6:
        tl.store(Y + 6 * N + n, tl.sum(acc6, axis=1), nmask)
    if MR > 7:
        tl.store(Y + 7 * N + n, tl.sum(acc7, axis=1), nmask)


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """|w| / scale (fp32, >= 0) -> e2m1 magnitude code 0..7: nearest value, ties to the even code, saturating at 6."""
    mids = torch.tensor([(a + b) / 2 for a, b in zip(E2M1[:-1], E2M1[1:])], device=v.device, dtype=v.dtype)
    lo = torch.bucketize(v, mids, right=False)
    hi = torch.bucketize(v, mids, right=True)
    return torch.where((lo != hi) & (hi % 2 == 0), hi, lo)


def make_fp4_twin(w: torch.Tensor, rows: int = 2048):
    """bf16 [N, K] (K % 32 == 0) -> (packed uint8 [N, K/2], e8m0 uint8 [N, K/32]); lossy, draft-only. Row chunks keep the
    fp32 temporaries at ~rows x K x 4 bytes x a few."""
    n, k = w.shape
    if k % 32:
        raise RuntimeError(f"DSV41_DRAFT_HEAD_FP4: K={k} is not a multiple of 32")
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    scale = torch.empty((n, k // 32), dtype=torch.uint8, device=w.device)
    for r0 in range(0, n, rows):
        wf = w[r0:r0 + rows].float().view(-1, k // 32, 32)
        amax = wf.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
        e = (torch.floor(torch.log2(amax)) - 2).clamp(-127, 127)
        code = e2m1_codes(wf.abs() / torch.exp2(e)) + 8 * (wf < 0).to(torch.int64)
        code = code.view(-1, k)
        packed[r0:r0 + rows] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        scale[r0:r0 + rows] = (e.view(-1, k // 32) + 127).to(torch.uint8)
    return packed, scale


def dequant_fp4(twin) -> torch.Tensor:
    """The MXFP4 copy back to bf16 [N, K] (exact: every e2m1 x 2^e value is a bf16)."""
    packed, scale = twin
    n = packed.shape[0]
    code = torch.stack((packed & 15, packed >> 4), -1).view(n, -1).long()
    mag = torch.tensor(E2M1, device=packed.device)[code & 7]
    val = torch.where(code >= 8, -mag, mag)
    k = val.shape[1]
    s = torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=1)[:, :k]
    return (val * s).to(torch.bfloat16)


def head_fp4(x: torch.Tensor, twin, tile=None) -> torch.Tensor:
    """x [M, K] bf16 -> fp32 logits [M, N] from the MXFP4 copy (M <= 8: the GEMV with `tile` or the env launch)."""
    w4, s = twin
    m, k = x.shape
    n = w4.shape[0]
    y = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m <= FP4_GEMV_MAX_M:
        bn, bk, warps, stages = tile or parse_tile(env("DSV41_DRAFT_HEAD_FP4_TILE", ""))
        if k % bk == 0:
            xi = x.contiguous()
            if xi.data_ptr() % 4:                       # the int32 view of the bf16 pairs needs a 4-byte aligned start
                xi = xi.clone()
            _head_fp4_gemv_kernel[(triton.cdiv(n, bn),)](xi.view(torch.int32), w4, s, y, n, k, m, bn, bk // 32,
                                                         stages >= 2, num_warps=warps, num_stages=1)
            return y
    if m <= 16:
        bm, bn, bk, warps, stages = _FP4_DOT16
    else:
        _, bm, bk, warps, stages = next(t for t in _FP4_TILES if m <= t[0])
        bn = 32
    _head_fp4_kernel[(triton.cdiv(n, bn), triton.cdiv(m, bm))](x.contiguous(), w4, s, y, m, n, k, bm, bn, bk,
                                                              num_warps=warps, num_stages=stages)
    return y


def stats():
    """Python-side call counts: graph captures plus eager calls (a replayed graph does not count)."""
    return dict(_stats)


_receipts = set()


def _receipt(rows, path):
    """One line per (rows, head) captured into a CUDA graph while the MXFP4 copy is built: the boot receipt."""
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 - no CUDA (CPU tests)
        capturing = False
    if not capturing or not fp4_installed():
        return
    key = (rows, path)
    if key in _receipts:
        return
    _receipts.add(key)
    print(f"[draft_head_fp8] capture rows={rows} head={path}", flush=True)


def install(dspark_module):
    """sglang.srt.models.deepseek_v4_dspark: fp8 twin built when the target's lm_head is attached,
    used by _logits_from_x_post_hc for up to MAX_M rows; everything else is the stock path."""
    if not ENABLED:
        return
    cls = dspark_module.DeepseekV4ForCausalLMDSpark
    for name in ("attach_shared_modules", "_logits_from_x_post_hc"):
        if not hasattr(cls, name):
            raise RuntimeError(f"DSV41_DRAFT_HEAD_FP8: DeepseekV4ForCausalLMDSpark.{name} is gone; engine drifted")
    if getattr(cls, "_dsv41_draft_head_fp8", False):
        return
    cls._dsv41_draft_head_fp8 = True
    orig_attach = cls.attach_shared_modules
    orig_logits = cls._logits_from_x_post_hc
    gather = getattr(dspark_module, "gather_and_crop_vocab")

    def attach_shared_modules(self, *a, **kw):
        out = orig_attach(self, *a, **kw)
        w = getattr(self.lm_head, "weight", None)
        if w is not None and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] % 32 == 0:
            _TWIN[w.data_ptr()] = make_twin(w.data)
            if fp4_installed():
                _TWIN4[w.data_ptr()] = make_fp4_twin(w.data)
            torch.cuda.empty_cache()          # the fp32 temporaries must not shrink the KV pool
            print(f"[draft_head_fp8] fp8 twin of the draft LM head {tuple(w.shape)}", flush=True)
            if w.data_ptr() in _TWIN4:
                p4, s4 = _TWIN4[w.data_ptr()]
                print(f"[draft_head_fp8] mxfp4 copy of the draft LM head {tuple(w.shape)}: "
                      f"{(p4.numel() + s4.numel()) / 2**20:.1f} MiB, default max rows {FP4_DEFAULT_MAX_M}", flush=True)
        return out

    def _logits_from_x_post_hc(self, x_post_hc):
        w = getattr(self.lm_head, "weight", None)
        twin = None if w is None else _TWIN.get(w.data_ptr())
        rows = x_post_hc.shape[0]
        if twin is None or rows > MAX_M or rows == 0:
            _stats["stock"] += 1
            return orig_logits(self, x_post_hc)
        x = self.stages[-1].norm(x_post_hc)
        twin4 = _TWIN4.get(w.data_ptr()) if _TWIN4 else None
        if twin4 is not None and rows <= fp4_rows():
            _stats["fp4"] += 1
            _receipt(rows, "mxfp4")
            local = head_fp4(x.to(torch.bfloat16), twin4).to(w.dtype)
        else:
            _stats["fp8"] += 1
            _receipt(rows, "fp8")
            local = head_fp8(x.to(torch.bfloat16), twin).to(w.dtype)
        if self._opt_markov_w2_tp_shard:
            return local
        return gather(local, self.lm_head)

    cls.attach_shared_modules = attach_shared_modules
    cls._logits_from_x_post_hc = _logits_from_x_post_hc
    print("[draft_head_fp8] armed" + (" (mxfp4 copy built; DSV41_DRAFT_HEAD_FP4 read per call)" if fp4_installed()
                                       else ""), flush=True)
