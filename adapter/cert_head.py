"""Certified target LM head for DSpark verify: a 1-byte screen of the head, exact logits only where the argmax can be.
Default OFF.

DSV41_CERT_HEAD=1. The target-verify forward reads the bf16 head shard (32320 x 5120 = 331 MB per rank at TP4) for
bs x 6 rows, only for the in-graph greedy accept (DsparkVerifyEpilogue: argmax of the gathered logits). Instead:

  xnorm   per row: ||x_g|| per 128-column group, ||x||, a non-finite flag; resets the row threshold T to -inf
  screen  s^ = x . D^T from an MXINT8 copy D of the shard (int8 + a power-of-two scale per 32 weights of a row,
          165 MB), the bound B = sum_g R_g ||x_g|| + 2 C_ACC N_v ||x|| (R_g = ||W_vg - D_vg||, N_v = max(||W_v||,
          ||D_v||)); writes hi = rn_bf16(s^ + B) and atomically max-reduces lo = rn_bf16(s^ - B) per row into T
  refine  one CTA per 16 head rows: if the tile holds a token with hi >= T in ANY row (or the step is in full mode,
          or a row is non-finite) it computes the tile's logits for every row with the exact kernel (same MMA, same K
          order as the stock cuBLAS kernel, bf16 output: bit-identical where qualified), else it writes -inf
The output replaces torch.matmul(x, W.T) in LogitsProcessor._compute_lm_head: same shape, same dtype, so the vocab
all-gather, the fp32 logits buffer and the in-graph accept run unchanged. No host sync, graph-capturable, and the
collective sequence is the stock one on every rank whatever each rank's candidates are.

Why the argmax is exact. With l_v the stock logit, lo_v <= l_v <= hi_v (the bound; C_ACC models tensor-core
accumulation, validated on the GPU; the same constants as overlay/cert_math.py in knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4). On a rank, T = max_u lo_u <= max_u l_u, so the
local maximiser and every token tied with it have hi >= T and are computed exactly; every -inf token u has
l_u <= hi_u < T <= (local max) <= (global max). The global argmax of the masked logits is therefore the stock one,
ties included (the tied tokens carry identical values at identical positions).

Full mode. A single flag buffer, set by the host before the replay (only when it changes), selects masked (cert)
or full logits. Full = the exact kernel on every tile (xnorm / screen exit at once): the exact stock logits, needed
by every consumer other than a greedy argmax: sampled rows (block verify / rejection sampling), logprobs, grammar
masks, penalties / logit bias / custom processors. The flag is armed only when every row is greedy and none of
those is present (step_armed; per-batch scheduler state, identical on every TP rank). The default (never set) is full.

min_new_tokens (sparkDash sends min_tokens = max_tokens). SGLang then adds -inf to each request's stop ids after the
head (eager accept path). Stop ids of the batch (the penalizer's own set: stop_token_ids | eos ids | tokenizer
additional stop ids | eos_token_id, at most 64) are written to a slot buffer; the screen gives them lo = -inf (they
never raise T) and hi = +inf (their tiles are always computed exactly). Then the argmax is exact whether the mask is
active or not: masked, T <= the best non-stop logit; unmasked, a stop-id maximiser is computed exactly and every
non-stop token u that is -inf has l_u < T <= best non-stop logit <= the maximum.

Qualification (fail closed). A verify row count M uses the certified head only when it is listed in
DSV41_CERT_HEAD_M (e.g. "6,12,18,24"): tests/gpu_test_cert_ds.py compares the exact kernel with torch.matmul on every
logit of every rank's shard and prints the list of M that passed everywhere. Every other M runs the stock matmul.

Modes
  DSV41_CERT_HEAD=0|1|check   check: the stock matmul is returned; the certified logits are computed too (both
                              flag modes) and compared on device: local argmax, value at it, every computed tile
                              (and every logit in full mode). Counters are logged every DSV41_CERT_HEAD_LOG_EVERY
                              verify steps (one host sync then).
  DSV41_CERT_HEAD_M=6,12      qualified verify row counts (from gpu_test_cert_ds.py); empty = stock everywhere
  DSV41_CERT_HEAD_MASK_MAX_M=30   X3b: masked mode only for verify steps of at most this many rows; a greedy step
                              above it runs full mode (every tile exact, the stock logits). Unset = no cap (the
                              production behaviour). The graphs are the same; only the host-set flag differs.
  DSV41_CERT_HEAD_GROUP=128   residual-norm group width
  DSV41_CERT_HEAD_TILE=64,128,4,3        screen BN,BK,warps,stages (BM = next pow2 >= M, >= 16)
  DSV41_CERT_HEAD_EXACT=16,128,4,3       exact kernel BV,BK,warps,stages (BMX = next pow2 >= M, >= 16)
  DSV41_CERT_HEAD_LOG_EVERY=2000
In-boot A/B (adapter/ab_variant.py): DSV41_CERT_HEAD is a union gate (tables are built when any variant enables
it) and is read per capture through ab_variant.env(), so each variant's verify graphs contain the certified head or
the stock matmul. DSV41_CERT_HEAD_MASK_MAX_M is read per verify step (host side, the replayed variant).
DSV41_CERT_HEAD_M is read per capture too (qualified_m(), TEST ONLY): a variant may qualify a subset of the process
list (the others take the stock matmul); the process value still gates the table build and the rank handshake.

X3b (DSV41_CERT_HEAD_MASK_MAX_M). The certified head's masked mode is slower than stock from M 36 (K4c, Tier-1
2026-11-08: 1935 vs 1636 us at M 48), its full mode faster (1528 us). Above the cap, step_armed returns (False, ()):
the step takes full mode, the exact stock logits (bit-identical where qualified), whose greedy argmax is the stock
one; sampled / logprob / grammar steps were never armed. The row count is the verify forward's (verify_ids_2d: bs x
stride); SGLang captures every speculative decode bs 1..8, so it equals the replayed graph's M wherever the head is
qualified (M <= 48). Nothing the draft confidence or the verify policy reads changes (they read the draft's state).
All DSV41_CERT_HEAD* values must be identical on every rank (a config hash is MAX-all-reduced at build; any
difference or a failed build disables the feature on every rank).
"""
from __future__ import annotations

import hashlib
import os
import sys

import torch

_OFF = ("", "0", "off", "false", "no")
GROUP = int(os.environ.get("DSV41_CERT_HEAD_GROUP", "128"))
LOG_EVERY = int(os.environ.get("DSV41_CERT_HEAD_LOG_EVERY", "2000") or 0)
_st = [int(v) for v in os.environ.get("DSV41_CERT_HEAD_TILE", "64,128,4,3").split(",")]
SBN, SBK, SWARPS, SSTAGES = _st
_ex = [int(v) for v in os.environ.get("DSV41_CERT_HEAD_EXACT", "16,128,4,3").split(",")]
XBV, XBK, XWARPS, XSTAGES = _ex
MAX_M = 128
NSTOP = 64        # stop-id slots (min_new_tokens masks); a batch with more distinct stop ids takes full mode
CONFIG_KEYS = ("DSV41_CERT_HEAD", "DSV41_CERT_HEAD_M", "DSV41_CERT_HEAD_GROUP", "DSV41_CERT_HEAD_TILE",
               "DSV41_CERT_HEAD_EXACT")
# hashed only when set, so the hash (and the handshake) of a config without it is unchanged
OPTIONAL_CONFIG_KEYS = ("DSV41_CERT_HEAD_MASK_MAX_M",)

# certificate constants (identical to cert_math.py of the GLM-5.3 TP4 recipe)
INFL_Q = 1.0 + 2.0 ** -18
INFL_B = 1.0 + 2.0 ** -20
INFL_X = 1.0 + 2.0 ** -16
SLOP_REL = 2.0 ** -22
SLOP_ABS = 2.0 ** -126
X_ABS = 2.0 ** -59

try:  # DSV41_AB_VARIANTS in-boot A/B (test only)
    import ab_variant as _ab
except ImportError:
    _ab = None


def env(name, default=""):
    if _ab is not None and getattr(_ab, "ACTIVE", False) and name in getattr(_ab, "KNOWN", ()):
        return _ab.env(name, default)
    return os.environ.get(name, default)


def mode() -> str:
    """'0' | '1' | 'check', read per call (per capture under the A/B harness)."""
    v = str(env("DSV41_CERT_HEAD", "0") or "0").strip().lower()
    if v in _OFF:
        return "0"
    return "check" if v == "check" else "1"


def enabled_anywhere() -> bool:
    """Install/build gate: the process env (the A/B harness unions the variants into it)."""
    return os.environ.get("DSV41_CERT_HEAD", "0").strip().lower() not in _OFF


def parse_m(spec: str) -> frozenset:
    out = set()
    for part in filter(None, (p.strip() for p in (spec or "").split(","))):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        if not 1 <= lo <= hi <= MAX_M:
            raise ValueError(f"DSV41_CERT_HEAD_M: bad entry {part!r} (1..{MAX_M})")
        out.update(range(lo, hi + 1))
    return frozenset(out)


QUALIFIED_M = parse_m(os.environ.get("DSV41_CERT_HEAD_M", ""))
_QM: dict = {}
_CAP: dict = {}


def qualified_m() -> frozenset:
    """The qualified verify row counts for the variant being captured / replayed: QUALIFIED_M itself unless the
    in-boot A/B harness is armed; then that variant's DSV41_CERT_HEAD_M intersected with QUALIFIED_M (a variant can
    only drop row counts, never qualify one the process did not). Parsed once per distinct spec."""
    if _ab is None or not getattr(_ab, "ACTIVE", False):
        return QUALIFIED_M
    spec = env("DSV41_CERT_HEAD_M", "")
    got = _QM.get(spec)
    if got is None:
        got = _QM[spec] = parse_m(spec) & QUALIFIED_M
    return got


def parse_mask_cap(spec):
    """DSV41_CERT_HEAD_MASK_MAX_M -> the row cap of the masked mode; '' = None (no cap, the production code path).
    Values above MAX_M are MAX_M (no larger M is ever qualified)."""
    raw = str(spec or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"DSV41_CERT_HEAD_MASK_MAX_M={spec!r}: expected an integer row count") from None
    if v < 0:
        raise ValueError(f"DSV41_CERT_HEAD_MASK_MAX_M={spec!r}: must be >= 0")
    return min(v, MAX_M)


def mask_max_m():
    """The masked-mode row cap of the variant being replayed, None = no cap (per verify step, host side; parsed once
    per spec). A malformed value takes no cap (the production behaviour), logged once; ab_variant refuses it at boot."""
    spec = env("DSV41_CERT_HEAD_MASK_MAX_M", "") or ""
    if spec in _CAP:
        return _CAP[spec]
    try:
        got = parse_mask_cap(spec)
    except ValueError as e:
        _log(f"{e}; no cap")
        got = None
    _CAP[spec] = got
    return got


def mask_capped(rows) -> bool:
    """X3b: True when this verify step must take full mode because of DSV41_CERT_HEAD_MASK_MAX_M."""
    if rows is None:
        return False
    cap = mask_max_m()
    return cap is not None and rows > cap


def c_acc(K: int) -> float:
    """ASSUMED tensor-core accumulation bound per unit of ||W_v|| ||x|| (Fasi et al. 2021 model; GPU-validated)."""
    return (K / 16.0) * 17.0 * 2.0 ** -23


def config_hash(environ=None) -> int:
    environ = os.environ if environ is None else environ
    keys = CONFIG_KEYS + tuple(k for k in OPTIONAL_CONFIG_KEYS if environ.get(k, "") != "")
    s = "|".join(f"{k}={environ.get(k, '')}" for k in keys)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:7], "little")


def _log(msg: str) -> None:
    sys.stderr.write(f"[cert_head] {msg}\n")
    sys.stderr.flush()


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# ---------------------------------------------------------------------------------------------------
# tables (pure torch: CPU tests build them too)
# ---------------------------------------------------------------------------------------------------
def _round_up_f32(x64: torch.Tensor) -> torch.Tensor:
    """float64 (>= 0 or inf) -> smallest fp32 >= x."""
    f = x64.float()
    bump = f.double() < x64
    return torch.where(bump, (f.view(torch.int32) + 1).view(torch.float32), f)


def _round_up_f16(x32: torch.Tensor) -> torch.Tensor:
    """fp32 (>= 0 or inf) -> smallest fp16 >= x (inf above the fp16 range)."""
    h = x32.half()
    bump = h.float() < x32
    return torch.where(bump, (h.view(torch.int16) + 1).view(torch.float16), h)


def make_mxint8(w: torch.Tensor, chunk_rows: int = 2048):
    """bf16 [V, K] -> (int8 [V, K], exponent byte uint8 [V, K/32]); scale 2**(e - 127) >= amax / 127."""
    V, K = w.shape
    q = torch.empty((V, K), dtype=torch.int8, device=w.device)
    e = torch.empty((V, K // 32), dtype=torch.uint8, device=w.device)
    for r0 in range(0, V, chunk_rows):
        r1 = min(V, r0 + chunk_rows)
        wf = w[r0:r1].float().view(r1 - r0, K // 32, 32)
        amax = wf.abs().amax(-1)
        ex = torch.where(amax > 0, torch.ceil(torch.log2(amax.clamp_min(1e-38) / 127.0)), torch.zeros_like(amax))
        eb = (ex + 127.0).clamp(1.0, 254.0)
        qq = torch.round(wf / torch.exp2(eb - 127.0)[..., None]).clamp(-127, 127)
        q[r0:r1] = qq.view(r1 - r0, K).to(torch.int8)
        e[r0:r1] = eb.to(torch.uint8)
    return q, e


def deq_rows(q: torch.Tensor, e: torch.Tensor, r0: int, r1: int):
    """Rows r0:r1 of the screening copy as the screen kernel builds them: (bf16 values as fp32, raw fp32 products)."""
    K = q.shape[1]
    qf = q[r0:r1].float()
    eb = e[r0:r1].to(torch.int32)
    sc = (eb << 23).view(torch.float32).repeat_interleave(32, dim=1)
    raw = qf * sc[:, :K]
    return raw.to(torch.bfloat16).float(), raw


class State:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@torch.inference_mode()
def build_tables(W: torch.Tensor, group: int = GROUP) -> State:
    """State for one rank's head shard W [V, K] bf16."""
    V, K = W.shape
    if W.dtype != torch.bfloat16 or K % group or group % 32 or group & (group - 1) or K % SBK or K % XBK:
        raise ValueError(f"head shard {tuple(W.shape)} {W.dtype} does not fit GROUP={group} / BK={SBK},{XBK}")
    q, e = make_mxint8(W)
    NG = K // group
    rt = torch.empty((NG, V), dtype=torch.float16, device=W.device)
    nv = torch.empty((V,), dtype=torch.float32, device=W.device)
    n_unsafe = 0
    for r0 in range(0, V, 1024):
        r1 = min(V, r0 + 1024)
        d_bf, d_raw = deq_rows(q, e, r0, r1)
        unsafe = (d_bf != d_raw).any(dim=1)       # the kernel's bf16 cast of a dequantised weight must be exact
        w64 = W[r0:r1].double()
        res = (w64 - d_bf.double()).view(r1 - r0, NG, group)
        rg = _round_up_f32(res.pow(2).sum(-1).sqrt() * (1.0 + 2.0 ** -40))
        rg[unsafe] = float("inf")
        rt[:, r0:r1] = _round_up_f16(rg).t()
        nrm = torch.maximum(w64.norm(dim=1), d_bf.double().norm(dim=1)) * (1.0 + 2.0 ** -40)
        nv[r0:r1] = _round_up_f32(nrm)
        n_unsafe += int(unsafe.sum())
    if not bool(torch.isfinite(W).all()):
        raise ValueError("head shard has non-finite weights")
    d0, _ = deq_rows(q, e, 0, min(V, 4096))
    w0 = W[:d0.shape[0]].float()
    eps = float(((w0 - d0).norm(dim=1) / w0.norm(dim=1).clamp_min(1e-30)).mean())
    return State(W=W, q=q, e=e, rt=rt, nv=nv, V=V, K=K, group=group, n_unsafe=n_unsafe, eps=eps, vstart=0)


# ---------------------------------------------------------------------------------------------------
# torch reference of the device pipeline (CPU tests; tests/gpu_test_cert_ds.py compares the kernels with it)
# ---------------------------------------------------------------------------------------------------
def _bf16_down(x32: torch.Tensor) -> torch.Tensor:
    return x32.to(torch.bfloat16).float()


def reference_bounds(st: State, x: torch.Tensor, screen_acc=None, stop_ids=()):
    """-> (hi [M,V] fp32 holding bf16 values, T [M], bad [M]) exactly as xnorm + screen define them.
    screen_acc: optional fp32 [M, V] screen accumulation (defaults to fp32 x @ D^T)."""
    M, K = x.shape
    G, NG = st.group, K // st.group
    xf = x.float()
    xg = xf.view(M, NG, G).pow(2).sum(-1).sqrt() * INFL_X + X_ABS
    xn = xg.pow(2).sum(-1).sqrt() * INFL_X
    bad = ~torch.isfinite(xn)
    if screen_acc is None:
        d_bf, _ = deq_rows(st.q, st.e, 0, st.V)
        screen_acc = xf @ d_bf.t()
    qb = xg @ st.rt.float()
    B = (qb * INFL_Q + 2.0 * c_acc(K) * (xn[:, None] * st.nv[None, :])) * INFL_B
    B = torch.where(torch.isnan(B), torch.full_like(B, float("inf")), B)
    slop = (screen_acc.abs() + B) * SLOP_REL + SLOP_ABS
    lo = _bf16_down(screen_acc - B - slop)
    hi = _bf16_down(screen_acc + B + slop)
    nonfin = ~torch.isfinite(screen_acc)
    lo = torch.where(nonfin | torch.isnan(lo), torch.full_like(lo, float("-inf")), lo)
    hi = torch.where(nonfin | torch.isnan(hi), torch.full_like(hi, float("inf")), hi)
    loc = [i - st.vstart for i in stop_ids if 0 <= i - st.vstart < st.V]
    if loc:
        lo[:, loc] = float("-inf")
        hi[:, loc] = float("inf")
    T = lo.max(dim=1).values
    return hi, T, bad


def reference_needed_tiles(hi, T, bad, full: bool, bv: int = XBV) -> torch.Tensor:
    """bool [ceil(V / bv)]: tiles the refine kernel computes."""
    M, V = hi.shape
    nt = (V + bv - 1) // bv
    if full or bool(bad.any()):
        return torch.ones(nt, dtype=torch.bool)
    cand = (hi >= T[:, None]).any(dim=0)
    pad = nt * bv - V
    if pad:
        cand = torch.cat([cand, torch.zeros(pad, dtype=torch.bool)])
    return cand.view(nt, bv).any(dim=1)


def reference_masked(exact_logits_bf16: torch.Tensor, needed: torch.Tensor, bv: int = XBV) -> torch.Tensor:
    """The refine output given the exact logits [M, V] (bf16): exact on needed tiles, -inf elsewhere."""
    M, V = exact_logits_bf16.shape
    col = needed.repeat_interleave(bv)[:V]
    out = exact_logits_bf16.clone()
    out[:, ~col] = float("-inf")
    return out


# ---------------------------------------------------------------------------------------------------
# Triton kernels (built on first use on the GPU)
# ---------------------------------------------------------------------------------------------------
_K: dict = {}
_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    sqrt_rn = getattr(tl, "sqrt_rn", None)
    RN = sqrt_rn is not None

    @triton.jit
    def _xnorm_kernel(X, sx, XG, XN, BAD, THR, FLAG, K: tl.constexpr, G: tl.constexpr, NGP: tl.constexpr,
                      INFL: tl.constexpr, XABS: tl.constexpr, RN: tl.constexpr):
        m = tl.program_id(0)
        if tl.load(FLAG) == 0:      # full mode: nothing to screen
            return
        NG: tl.constexpr = K // G
        gi = tl.arange(0, NGP)      # NG (40 at K=5120, G=128) padded to a power of two
        gok = gi < NG
        x = tl.load(X + m * sx + gi[:, None] * G + tl.arange(0, G)[None, :], mask=gok[:, None],
                    other=0.0).to(tl.float32)
        s = tl.sum(x * x, axis=1)
        if RN:
            xg = tl.sqrt_rn(s) * INFL + XABS
        else:
            xg = tl.sqrt(s) * INFL + XABS
        xg = tl.where(gok, xg, 0.0)
        tl.store(XG + m * NG + gi, xg, mask=gok)
        t = tl.sum(xg * xg, axis=0)
        if RN:
            xn = tl.sqrt_rn(t) * INFL
        else:
            xn = tl.sqrt(t) * INFL
        tl.store(XN + m, xn)
        bad = (xn != xn) | (xn == float("inf"))
        tl.store(BAD + m, bad.to(tl.int32))
        tl.store(THR + m, float("-inf"))

    @triton.jit
    def _screen_kernel(X, sx, Q, E, RT, NV, XG, XN, HI, THR, FLAG, STOP, vstart, M, V,
                       K: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       CACC2: tl.constexpr, INFL_Q: tl.constexpr, INFL_B: tl.constexpr, SLOP_REL: tl.constexpr,
                       SLOP_ABS: tl.constexpr, NSTOP: tl.constexpr, IP: tl.constexpr):
        pid = tl.program_id(0)
        if tl.load(FLAG) == 0:
            return
        m = tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        mok = m < M
        nok = n < V
        n64 = n.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * sx + k[None, :], mask=mok[:, None], other=0.0)
            w = tl.load(Q + n64[None, :] * K + k[:, None], mask=nok[None, :], other=0)
            eb = tl.load(E + n64[None, :] * (K // 32) + k[:, None] // 32, mask=nok[None, :], other=127)
            sc = (eb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            wd = (w.to(tl.float32) * sc).to(tl.bfloat16)
            if IP:   # Triton interpreter (CPU smoke test): its bf16 dot is wrong, fp32 operands hold the same values
                acc = tl.dot(x.to(tl.float32), wd.to(tl.float32), acc)
            else:
                acc = tl.dot(x, wd, acc)
        NG: tl.constexpr = K // G
        qb = tl.zeros((BM, BN), dtype=tl.float32)
        for g in range(0, NG):
            r = tl.load(RT + g * V + n, mask=nok, other=0.0).to(tl.float32)
            xg = tl.load(XG + m * NG + g, mask=mok, other=0.0)
            qb += xg[:, None] * r[None, :]
        nv = tl.load(NV + n, mask=nok, other=0.0)
        xn = tl.load(XN + m, mask=mok, other=0.0)
        B = (qb * INFL_Q + CACC2 * (xn[:, None] * nv[None, :])) * INFL_B
        B = tl.where(B != B, float("inf"), B)
        slop = (tl.abs(acc) + B) * SLOP_REL + SLOP_ABS
        lo = (acc - B - slop).to(tl.bfloat16).to(tl.float32)
        hi = (acc + B + slop).to(tl.bfloat16).to(tl.float32)
        nonfin = (acc != acc) | (tl.abs(acc) == float("inf"))
        lo = tl.where(nonfin | (lo != lo), float("-inf"), lo)
        hi = tl.where(nonfin | (hi != hi), float("inf"), hi)
        # stop ids (min_new_tokens may mask them after the head): never raise T, always computed exactly
        gid = vstart + n
        isstop = tl.zeros((BN,), dtype=tl.int32)
        for j in tl.static_range(NSTOP):
            isstop = isstop | (gid == tl.load(STOP + j)).to(tl.int32)
        lo = tl.where(isstop[None, :] != 0, float("-inf"), lo)
        hi = tl.where(isstop[None, :] != 0, float("inf"), hi)
        ok = mok[:, None] & nok[None, :]
        tl.store(HI + m[:, None] * V + n[None, :], hi.to(tl.bfloat16), mask=ok)
        lof = tl.where(ok, lo, float("-inf"))
        tl.atomic_max(THR + m, tl.max(lof, axis=1), mask=mok)

    @triton.jit
    def _refine_kernel(X, sx, W, HI, THR, BAD, FLAG, OUT, so, NEED, M, V,
                       K: tl.constexpr, BV: tl.constexpr, BMX: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr):
        pid = tl.program_id(0)
        j = pid * BV + tl.arange(0, BV)
        ok = j < V
        mm = tl.arange(0, BMX)
        mok = mm < M
        full = tl.load(FLAG) == 0
        # in full mode xnorm / screen did not run: BAD / HI / THR hold stale values, and `full` decides alone
        bad = tl.max(tl.load(BAD + mm, mask=mok, other=0), axis=0)
        h = tl.load(HI + mm[:, None] * V + j[None, :], mask=mok[:, None] & ok[None, :],
                    other=float("-inf")).to(tl.float32)
        T = tl.load(THR + mm, mask=mok, other=float("inf"))
        hit = tl.max(tl.max((h >= T[:, None]).to(tl.int32), axis=1), axis=0)
        need = full | (hit > 0) | (bad > 0)
        if need:
            j64 = j.to(tl.int64)
            acc = tl.zeros((BV, BMX), dtype=tl.float32)
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                w = tl.load(W + j64[:, None] * K + k[None, :], mask=ok[:, None], other=0.0)
                xt = tl.load(X + mm[None, :] * sx + k[:, None], mask=mok[None, :], other=0.0)
                if IP:
                    acc = tl.dot(w.to(tl.float32), xt.to(tl.float32), acc)
                else:
                    acc = tl.dot(w, xt, acc)
            tl.store(OUT + mm[None, :] * so + j[:, None], acc.to(tl.bfloat16), mask=ok[:, None] & mok[None, :])
            tl.atomic_add(NEED, 1)
        else:
            ninf = tl.full((BV, BMX), float("-inf"), tl.float32).to(tl.bfloat16)
            tl.store(OUT + mm[None, :] * so + j[:, None], ninf, mask=ok[:, None] & mok[None, :])

    _K.update(triton=triton, xnorm=_xnorm_kernel, screen=_screen_kernel, refine=_refine_kernel, RN=RN)
    return _K


class Buffers:
    """Per-device persistent buffers, allocated outside any graph capture."""

    def __init__(self, device, max_m: int):
        self.flag = torch.zeros((1,), dtype=torch.int32, device=device)      # 0 full (default), 1 certified
        self.flag_host = 0
        self.stop = torch.full((NSTOP,), -1, dtype=torch.int32, device=device)  # global stop ids, -1 padded
        self.stop_host = ()
        self.need = torch.zeros((1,), dtype=torch.int32, device=device)      # tiles computed (stats, wraps)
        self.counters = torch.zeros((8,), dtype=torch.int64, device=device)  # check mode


def certified_logits(st: State, bufs: Buffers, x: torch.Tensor, debug: dict | None = None) -> torch.Tensor:
    """x [M, K] bf16 -> bf16 [M, V]: masked (flag 1) or full exact logits (flag 0). No host sync.
    debug (tests only): filled with the internal hi / thr / bad / xg / xn tensors."""
    k = _kernels()
    triton = k["triton"]
    x = x.contiguous()
    M, K = x.shape
    V = st.V
    dev = x.device
    NG = K // st.group
    xg = torch.empty((M, NG), dtype=torch.float32, device=dev)
    xn = torch.empty((M,), dtype=torch.float32, device=dev)
    bad = torch.empty((M,), dtype=torch.int32, device=dev)
    thr = torch.empty((M,), dtype=torch.float32, device=dev)
    hi = torch.empty((M, V), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, V), dtype=torch.bfloat16, device=dev)
    BM = max(16, next_pow2(M))
    k["xnorm"][(M,)](x, x.stride(0), xg, xn, bad, thr, bufs.flag, K=K, G=st.group, NGP=next_pow2(NG), INFL=INFL_X,
                     XABS=X_ABS, RN=k["RN"])
    k["screen"][(triton.cdiv(V, SBN),)](
        x, x.stride(0), st.q, st.e, st.rt, st.nv, xg, xn, hi, thr, bufs.flag, bufs.stop, st.vstart, M, V,
        K=K, G=st.group, BM=BM, BN=SBN, BK=SBK, CACC2=2.0 * c_acc(K), INFL_Q=INFL_Q, INFL_B=INFL_B,
        SLOP_REL=SLOP_REL, SLOP_ABS=SLOP_ABS, NSTOP=NSTOP, IP=_INTERP, num_warps=SWARPS if BM <= 32 else max(SWARPS, 8), num_stages=SSTAGES)
    k["refine"][(triton.cdiv(V, XBV),)](
        x, x.stride(0), st.W, hi, thr, bad, bufs.flag, out, out.stride(0), bufs.need, M, V,
        K=K, BV=XBV, BMX=max(16, next_pow2(M)), BK=XBK, IP=_INTERP, num_warps=XWARPS, num_stages=XSTAGES)
    if debug is not None:
        debug.update(hi=hi, thr=thr, bad=bad, xg=xg, xn=xn)
    return out


def _check(bufs: Buffers, stock: torch.Tensor, cert: torch.Tensor) -> None:
    """Device-side comparison (graph-capturable). counters: [steps, cert steps, argmax mismatches, value mismatches,
    computed-logit mismatches, full-mode logit mismatches, rows, -]."""
    a = stock.float()
    b = cert.float()
    full = bufs.flag.to(torch.int64)[0] == 0
    ia, ib = a.argmax(dim=-1), b.argmax(dim=-1)
    va, vb = a.gather(-1, ia[:, None]), b.gather(-1, ib[:, None])
    arg_bad = (ia != ib).sum()
    val_bad = (va != vb).sum()
    computed = b != float("-inf")
    diff = (a != b) & computed
    comp_bad = diff.sum()
    full_bad = torch.where(full, (a != b).sum(), torch.zeros_like(comp_bad))
    one = torch.ones((), dtype=torch.int64, device=a.device)
    upd = torch.stack([one, (~full).to(torch.int64), arg_bad, val_bad, comp_bad, full_bad,
                       torch.full_like(one, a.shape[0]), torch.zeros_like(one)])
    bufs.counters.add_(upd)


# ---------------------------------------------------------------------------------------------------
# engine integration
# ---------------------------------------------------------------------------------------------------
_S = {"st": None, "bufs": None, "lp": None, "why": "not built", "calls": 0, "verify_calls": 0,
      "logged_modes": set(), "capture_warned": False}


def _rank() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return 0


def tp_agree(ok: bool):
    """MAX-all-reduce (failed, hash, -hash) over the TP CPU group; every rank calls it exactly once."""
    try:
        from sglang.srt.distributed import get_tp_group
        grp = get_tp_group()
        if grp.world_size == 1:
            return ok, "tp=1"
        h = config_hash()
        t = torch.tensor([0 if ok else 1, h, -h], dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=grp.cpu_group)
        failed, hmax, hneg = (int(v) for v in t.tolist())
        if failed:
            return False, "a TP rank failed to build its tables"
        if hmax != -hneg:
            return False, "DSV41_CERT_HEAD* settings differ between TP ranks"
        return ok, "agreed"
    except Exception as e:  # noqa: BLE001
        return False, f"agreement failed: {e!r}"


def _build_for_model(model):
    lm_head = getattr(model, "lm_head", None)
    lp = getattr(model, "logits_processor", None)
    W = getattr(lm_head, "weight", None)
    if lp is None or W is None:
        return "no lm_head / logits_processor"
    if os.environ.get("DSV41_DRAFT_CAPTURE", "0").strip().lower() not in _OFF:
        return "DSV41_DRAFT_CAPTURE is on (the capture reads the target logits)"
    if W.dtype != torch.bfloat16 or W.dim() != 2 or not W.is_contiguous() or not W.is_cuda:
        return f"lm_head.weight is not a contiguous bf16 CUDA matrix ({W.dtype}, {tuple(W.shape)})"
    qm = type(getattr(lm_head, "quant_method", None)).__name__
    from sglang.srt.layers import logits_processor as lpm
    if lpm.should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None)):
        return f"lm_head uses a quant method ({qm})"
    if hasattr(lm_head, "set_lora"):
        return "LoRA lm_head"
    if getattr(lp, "use_fp32_lm_head", False) or getattr(lp, "rl_on_policy_target", None) is not None:
        return "fp32 / rl-on-policy lm_head path"
    if getattr(lp, "logit_scale", None) is not None or getattr(lp, "final_logit_softcapping", None):
        return "logit scale / softcapping"
    if not QUALIFIED_M:
        return "DSV41_CERT_HEAD_M is empty: stock everywhere"
    try:
        st = build_tables(W.data)
    except ValueError as e:
        return str(e)
    st.lm_head = lm_head
    si = getattr(lm_head, "shard_indices", None)
    st.vstart = int(getattr(si, "org_vocab_start_index", 0)) if si is not None else 0
    return st


def install_model(dsv4_module):
    """sglang.srt.models.deepseek_v4: tables built after the TARGET's weights loaded (before the KV pool is sized)."""
    if not enabled_anywhere():
        return
    cls = dsv4_module.DeepseekV4ForCausalLM
    if getattr(cls, "_dsv41_cert_head", False):
        return
    cls._dsv41_cert_head = True
    orig = cls.load_weights

    def load_weights(self, *a, **kw):
        out = orig(self, *a, **kw)
        if type(self) is not cls or _S["lp"] is not None:
            return out
        try:
            st = _build_for_model(self)
        except Exception as e:  # noqa: BLE001  never take the boot down; stock stays
            st = f"build failed: {e!r}"
        ok, why = tp_agree(isinstance(st, State))
        if not ok:
            reason = st if isinstance(st, str) else f"disabled on every rank: {why}"
            _S["why"] = reason
            _log(f"disabled: {reason}")
            torch.cuda.empty_cache()
            return out
        _S["st"], _S["lp"] = st, self.logits_processor
        _S["bufs"] = Buffers(st.W.device, MAX_M)
        _S["bufs"].need_per_full = (st.V + XBV - 1) // XBV
        torch.cuda.empty_cache()
        mb = (st.q.numel() + st.e.numel() + st.rt.numel() * 2 + st.nv.numel() * 4) / 1e6
        _log(f"built on rank {_rank()}: MXINT8 copy of the {st.V} x {st.K} head shard ({mb:.1f} MB read per step vs "
             f"{st.V * st.K * 2 / 1e6:.1f} MB bf16), eps {st.eps * 100:.3f} %, unsafe rows {st.n_unsafe}, "
             f"qualified M {sorted(QUALIFIED_M)} ({why})")
        return out

    cls.load_weights = load_weights
    _log(f"armed (mode {mode()}, qualified M {sorted(QUALIFIED_M) or 'none'})")


def install_logits(lp_module):
    """sglang.srt.layers.logits_processor: the target's verify head goes through certified_logits."""
    if not enabled_anywhere():
        return
    cls = lp_module.LogitsProcessor
    if getattr(cls, "_dsv41_cert_head", False):
        return
    cls._dsv41_cert_head = True
    orig_get, orig_head = cls._get_logits, cls._compute_lm_head
    ctx = {"on": False}

    def _get_logits(self, hidden_states, lm_head, logits_metadata, *a, **kw):
        st = _S["st"]
        fm = getattr(logits_metadata, "forward_mode", None)
        if (st is None or self is not _S["lp"] or lm_head is not st.lm_head or fm is None
                or not fm.is_target_verify()):
            return orig_get(self, hidden_states, lm_head, logits_metadata, *a, **kw)
        ctx["on"] = True
        try:
            return orig_get(self, hidden_states, lm_head, logits_metadata, *a, **kw)
        finally:
            ctx["on"] = False

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        st = _S["st"]
        md = mode()
        M = int(hidden_states.shape[0])
        if (not ctx["on"] or embedding_bias is not None or md == "0" or M not in qualified_m()
                or hidden_states.dim() != 2 or lm_head.weight is not st.lm_head.weight):
            return orig_head(self, hidden_states, lm_head, embedding_bias)
        key = (md, M)
        if key not in _S["logged_modes"] and _rank() == 0:
            _S["logged_modes"].add(key)
            _log(f"verify head M={M}: {'certified' if md == '1' else 'check (stock returned)'}"
                 f"{' [capture]' if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() else ''}")
        x = hidden_states.to(torch.bfloat16)
        cert = certified_logits(st, _S["bufs"], x)
        if md == "1":
            return cert
        stock = orig_head(self, hidden_states, lm_head, embedding_bias)
        _check(_S["bufs"], stock, cert)
        return stock

    cls._get_logits = _get_logits
    cls._compute_lm_head = _compute_lm_head


def set_flag(armed: bool) -> None:
    """Host side, before the verify forward is launched: stream-ordered, written only when it changes."""
    bufs = _S["bufs"]
    if bufs is None:
        return
    v = 1 if armed else 0
    if bufs.flag_host != v:
        bufs.flag.fill_(v)
        bufs.flag_host = v


def _req_stop_ids(req):
    """The stop ids BatchedMinNewTokensPenalizer._prepare masks for this request (cached on the request)."""
    got = getattr(req, "_dsv41_cert_stop", None)
    if got is None:
        tok = getattr(req, "tokenizer", None)
        ids = set(req.sampling_params.stop_token_ids or set()) | set(getattr(req, "eos_token_ids", None) or set())
        if tok is not None:
            ids |= set(getattr(tok, "additional_stop_token_ids", None) or set())
            if getattr(tok, "eos_token_id", None) is not None:
                ids.add(tok.eos_token_id)
        got = frozenset(int(i) for i in ids if i is not None)
        try:
            req._dsv41_cert_stop = got
        except Exception:  # noqa: BLE001
            pass
    return got


def verify_rows(executor, batch, verify_ids_2d=None):
    """Rows of the target-verify forward (bs x stride), from host metadata only (no sync); None if unknown."""
    shape = getattr(verify_ids_2d, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0]) * int(shape[1])
    w, reqs = getattr(executor, "verify_num_draft_tokens", None), getattr(batch, "reqs", None)
    if w is None or reqs is None:
        return None
    return len(reqs) * int(w)


def step_armed(executor, batch, rows=None):
    """-> (armed, stop ids). Certified (masked) logits are safe when every consumer of the verify logits is the
    greedy argmax: all rows greedy, no logprobs, no grammar, no custom logit processor, no logit bias, and no
    penalizer other than min_new_tokens (its -inf stop ids are handled by the stop-id rule of the screen).
    rows (X3b): the verify forward's row count; above DSV41_CERT_HEAD_MASK_MAX_M the step is not armed (full mode,
    the stock logits) because the masked mode is the slower one there. None = no cap check.
    Everything read here is per-batch scheduler state, identical on every TP rank."""
    del executor
    if mask_capped(rows):
        return False, ()
    if getattr(batch, "return_logprob", False) or getattr(batch, "has_grammar", False):
        return False, ()
    si = getattr(batch, "sampling_info", None)
    if si is None:
        return True, ()
    if (not getattr(si, "is_all_greedy", False) or getattr(si, "has_custom_logit_processor", False)
            or getattr(si, "logit_bias", None) is not None or getattr(si, "grammar_mask", None) is not None):
        return False, ()
    stops = set()
    pen = getattr(si, "penalizer_orchestrator", None)
    if pen is not None and getattr(pen, "is_required", False):
        for p in pen.penalizers.values():
            if p.is_prepared() and type(p).__name__ != "BatchedMinNewTokensPenalizer":
                return False, ()
        for req in batch.reqs:
            stops |= _req_stop_ids(req)
        if len(stops) > NSTOP:
            return False, ()
    return True, tuple(sorted(stops))


def set_stop(ids) -> None:
    """Host side, before the forward: the stop-id slots, rewritten only when the set changes (rare)."""
    bufs = _S["bufs"]
    if bufs is None or bufs.stop_host == ids:
        return
    t = torch.full((NSTOP,), -1, dtype=torch.int32)
    if ids:
        t[:len(ids)] = torch.tensor(ids, dtype=torch.int32)
    bufs.stop.copy_(t.to(bufs.stop.device), non_blocking=False)
    bufs.stop_host = ids


def _log_stats():
    bufs = _S["bufs"]
    if bufs is None:
        return
    c = [int(v) for v in bufs.counters.tolist()]
    need = int(bufs.need.item())
    msg = (f"rank {_rank()} verify steps {_S['verify_calls']} (armed {_S.get('armed_calls', 0)}, mask-capped "
           f"{_S.get('capped_calls', 0)}), tiles computed "
           f"{need} (all-tile steps = {bufs.need_per_full or '?'} each)")
    if c[0]:
        msg += (f"; check: steps {c[0]} (certified {c[1]}), argmax mismatches {c[2]}, value mismatches {c[3]}, "
                f"computed-logit mismatches {c[4]}, full-mode logit mismatches {c[5]}, rows {c[6]}")
    if _rank() == 0 or any(c[2:6]):
        _log(msg)


def install_verify(verify_module):
    """sglang.srt.speculative.dspark_components.dspark_verify: set the flag before every verify forward."""
    if not enabled_anywhere():
        return
    ex = verify_module.TargetVerifyExecutor
    if getattr(ex, "_dsv41_cert_head", False):
        return
    ex._dsv41_cert_head = True
    orig_nc, orig_c, orig_idle = ex.run_non_compact, ex.run_compact, ex.run_idle_participation

    def run_non_compact(self, *, batch, **kw):
        rows = verify_rows(self, batch, kw.get("verify_ids_2d"))
        armed, stops = step_armed(self, batch, rows)
        if armed and stops:
            set_stop(stops)       # a superset left from earlier steps is harmless (those ids are simply computed)
        set_flag(armed)
        _S["verify_calls"] += 1
        _S["armed_calls"] = _S.get("armed_calls", 0) + int(armed)
        if mask_capped(rows):
            _S["capped_calls"] = _S.get("capped_calls", 0) + 1
        if LOG_EVERY and _S["verify_calls"] % LOG_EVERY == 0:
            _log_stats()
        return orig_nc(self, batch=batch, **kw)

    def run_compact(self, *a, **kw):
        set_flag(False)
        return orig_c(self, *a, **kw)

    def run_idle_participation(self, *a, **kw):
        set_flag(False)
        return orig_idle(self, *a, **kw)

    ex.run_non_compact = run_non_compact
    ex.run_compact = run_compact
    ex.run_idle_participation = run_idle_participation
