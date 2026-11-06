"""Block verification for sampled DSpark rows with ~8 launches instead of ~57, BIT-IDENTICAL to
adapter/block_verify.block_accept_probs. Default OFF.

DSV41_BLOCK_VERIFY_FOLD=1 (needs DSV41_BLOCK_VERIFY=1): adapter/block_verify.py's execute calls fold() instead of
block_accept_probs() for CUDA inputs. DSV41_BLOCK_VERIFY_FOLD=check runs both on the same random draws, returns the
torch result (production behaviour unchanged) and counts on device every row whose tau or correction token differs
(the contract is 0); the counters are logged every DSV41_BLOCK_VERIFY_FOLD_LOG_EVERY calls (one host sync then, check
mode only). Every mode logs a host-side receipt of its calls per path (no sync) at the same interval, and a terminal
receipt (+ the check counters) at each busy -> idle transition of the scheduler (flush / install_scheduler).

Exactness contract: the output (tau, correction token) equals block_accept_probs' bit for bit, for the same inputs
and the same random draws. How:
  * every elementwise / per-request value is computed in Triton with torch's FP32 operations in torch's association
    (IEEE division via div_rn, no FMA contraction: enable_fp_fusion=False), so it is bit-identical;
  * the three vocabulary-wide reductions are NOT reimplemented: the kernels write the exact tensors production
    reduces (same values, shape, dtype, contiguity), and the reductions are production's own torch ops on them:
      S    = terms.sum(-1)                         terms [bs, g-1, V] = clamp_min(w_i * p_{i+1} - q_{i+1}, 0)
      mass = res.sum(-1, keepdim=True)             res   [bs, V]      = clamp_min(w_tau * p_tau - q_tau, 0)
      corr = keys.argmax(dim=-1)                   keys  [bs, V]      = where(mass > 0, res / mass, p_tau) / gumbel
    so their results are those of production by construction (torch's reductions are deterministic for a given
    input tensor, shape and device).
Random stream: the caller draws eta = torch.rand(bs, g) and then torch.empty(bs, V).exponential_() exactly as
block_accept_probs does (same calls, shapes, dtypes, order), so the Philox offsets and the values are unchanged.
What is saved: the w recursion loop, the gathers, the masked draft copy, the q_pad cat and ~45 small torch launches
(launches: 3 Triton + 2 sums + 1 argmax + the 2 RNG draws both paths make).
"""
import os

import torch

MODE = os.environ.get("DSV41_BLOCK_VERIFY_FOLD", "0").strip().lower()
ENABLED = MODE not in ("0", "", "off", "false", "no")
CHECK = MODE == "check"
LOG_EVERY = int(os.environ.get("DSV41_BLOCK_VERIFY_FOLD_LOG_EVERY", "2000") or 0)
CHUNK = int(os.environ.get("DSV41_BLOCK_VERIFY_FOLD_CHUNK", "4096"))


def _pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# ------------------------------------------------------------------------------------------------ kernels --
try:
    import triton
    import triton.language as tl

    _NAN_ALL = tl.constexpr(tl.PropagateNan.ALL)

    @triton.jit
    def _pick(v, gi, i):
        return tl.sum(tl.where(gi == i, v, 0.0), axis=0)

    @triton.jit
    def _scalars(b, TP, DP, X, sxb, sxg, LIVE, V, G: tl.constexpr, GP: tl.constexpr, HAS_LIVE: tl.constexpr,
                 LIVE_SUB: tl.constexpr):
        """Per request: alive [GP] (int1) and w [GP] fp32 (w_1..w_g, 0 beyond g), in torch's operation order:
        ratio = p_tok / clamp_min(q_tok, 1e-30); ratio = where(alive, ratio, 0); w_i = minimum(w_{i-1} * ratio_i, 1)."""
        gi = tl.arange(0, GP)
        gok = gi < G
        x = tl.load(X + b * sxb + gi * sxg, mask=gok, other=0).to(tl.int64)
        rowp = (b * (G + 1) + gi).to(tl.int64) * V
        rowq = (b * G + gi).to(tl.int64) * V
        p_tok = tl.load(TP + rowp + x, mask=gok, other=0.0)
        q_tok = tl.load(DP + rowq + x, mask=gok, other=0.0)
        ratio = tl.math.div_rn(p_tok, tl.maximum(q_tok, 1e-30, propagate_nan=_NAN_ALL))
        if HAS_LIVE:
            live = tl.load(LIVE + b).to(tl.int64) - LIVE_SUB
            alive = gok & (gi < live)
        else:
            alive = gok
        ratio = tl.where(alive, ratio, 0.0)
        cur = tl.full((), 1.0, tl.float32)
        w = tl.zeros((GP,), tl.float32)
        for i in tl.static_range(G):
            cur = tl.minimum(cur * _pick(ratio, gi, i), 1.0, propagate_nan=_NAN_ALL)
            w = tl.where(gi == i, cur, w)
        return alive, w

    @triton.jit
    def _terms_kernel(TP, DP, X, sxb, sxg, LIVE, TERMS, V,
                      G: tl.constexpr, GP: tl.constexpr, HAS_LIVE: tl.constexpr, LIVE_SUB: tl.constexpr,
                      CH: tl.constexpr):
        """TERMS[b, i, :] = clamp_min(w[:, i] * target_probs[b, i+1] - draft_probs'[b, i+1], 0), i < g-1, where
        draft_probs' = draft_probs * alive (block_accept_probs, the S operand)."""
        b = tl.program_id(0)
        c = tl.program_id(1)
        gi = tl.arange(0, GP)
        alive, w = _scalars(b, TP, DP, X, sxb, sxg, LIVE, V, G, GP, HAS_LIVE, LIVE_SUB)
        offs = c * CH + tl.arange(0, CH)
        m = offs < V
        for i in tl.static_range(G - 1):
            w_i = _pick(w, gi, i)
            a_n = tl.sum(tl.where((gi == i + 1) & alive, 1.0, 0.0), axis=0)      # alive[i+1] as 1.0 / 0.0
            t = tl.load(TP + (b * (G + 1) + i + 1).to(tl.int64) * V + offs, mask=m, other=0.0)
            q = tl.load(DP + (b * G + i + 1).to(tl.int64) * V + offs, mask=m, other=0.0)
            if HAS_LIVE:
                q = q * a_n
            x = tl.maximum(w_i * t - q, 0.0, propagate_nan=_NAN_ALL)
            tl.store(TERMS + (b * (G - 1) + i).to(tl.int64) * V + offs, x, mask=m)

    @triton.jit
    def _decide(b, TP, DP, X, sxb, sxg, LIVE, ETA, S, V, G: tl.constexpr, GP: tl.constexpr,
                HAS_LIVE: tl.constexpr, LIVE_SUB: tl.constexpr):
        """-> (tau, wt, alive[tau] as 1.0 / 0.0, h [GP]) from the production-reduced S."""
        gi = tl.arange(0, GP)
        alive, w = _scalars(b, TP, DP, X, sxb, sxg, LIVE, V, G, GP, HAS_LIVE, LIVE_SUB)
        s = tl.load(S + b * (G - 1) + gi, mask=gi < G - 1, other=0.0)
        den = tl.maximum(s + (1.0 - w), 1e-30, propagate_nan=_NAN_ALL)
        h = tl.math.div_rn(s, den)
        h = tl.where(gi == G - 1, w, h)
        eta = tl.load(ETA + b * G + gi, mask=gi < G, other=2.0)
        ok = (eta <= h) & alive
        tau = tl.max(tl.where(ok, gi + 1, 0), axis=0)
        wt = tl.where(tau == 0, 1.0, _pick(w, gi, tau - 1))
        if HAS_LIVE:
            a_t = tl.sum(tl.where((gi == tau) & alive, 1.0, 0.0), axis=0)
        else:
            a_t = 1.0
        return tau, wt, a_t, h

    @triton.jit
    def _residual_kernel(TP, DP, X, sxb, sxg, LIVE, ETA, S, RES, TAU, HDBG, V,
                         G: tl.constexpr, GP: tl.constexpr, HAS_LIVE: tl.constexpr, LIVE_SUB: tl.constexpr,
                         CH: tl.constexpr):
        """RES[b, :] = clamp_min(w_tau * target_probs[b, tau] - q_pad[b, tau], 0); TAU[b] = tau."""
        b = tl.program_id(0)
        c = tl.program_id(1)
        tau, wt, a_t, h = _decide(b, TP, DP, X, sxb, sxg, LIVE, ETA, S, V, G, GP, HAS_LIVE, LIVE_SUB)
        offs = c * CH + tl.arange(0, CH)
        m = offs < V
        p = tl.load(TP + (b * (G + 1) + tau).to(tl.int64) * V + offs, mask=m, other=0.0)
        q = tl.load(DP + (b * G + tau).to(tl.int64) * V + offs, mask=m & (tau < G), other=0.0)
        if HAS_LIVE:
            q = q * a_t
        r = tl.maximum(wt * p - q, 0.0, propagate_nan=_NAN_ALL)
        tl.store(RES + b.to(tl.int64) * V + offs, r, mask=m)
        if c == 0:
            tl.store(TAU + b, tau.to(tl.int64))
            gi = tl.arange(0, GP)
            tl.store(HDBG + b * GP + gi, h)

    @triton.jit
    def _keys_kernel(TP, RES, MASS, TAU, GUM, KEYS, V, G: tl.constexpr, CH: tl.constexpr):
        """KEYS[b, :] = where(mass > 0, res / clamp_min(mass, 1e-30), p_tau) / gumbel (two roundings, as torch)."""
        b = tl.program_id(0)
        c = tl.program_id(1)
        offs = c * CH + tl.arange(0, CH)
        m = offs < V
        tau = tl.load(TAU + b)
        mass = tl.load(MASS + b)
        r = tl.load(RES + b.to(tl.int64) * V + offs, mask=m, other=0.0)
        p = tl.load(TP + (b * (G + 1) + tau).to(tl.int64) * V + offs, mask=m, other=0.0)
        rn = tl.where(mass > 0, tl.math.div_rn(r, tl.maximum(mass, 1e-30, propagate_nan=_NAN_ALL)), p)
        gum = tl.load(GUM + b.to(tl.int64) * V + offs, mask=m, other=1.0)
        tl.store(KEYS + b.to(tl.int64) * V + offs, tl.math.div_rn(rn, gum), mask=m)
except Exception as _triton_exc:                     # pragma: no cover - no triton on the host
    # Override site: without Triton (or when the fold kernels fail to define on GB10 sm_121) the
    # fold kernels are absent and fold() raises only at the first sampled verify. A user-set
    # DSV41_BLOCK_VERIFY_FOLD now raises at import, so the engine start fails instead.
    if ENABLED:
        raise RuntimeError(
            f"DSV41_BLOCK_VERIFY_FOLD={MODE!r} is set but the Triton fold kernels are unavailable "
            f"({_triton_exc!r})") from _triton_exc
    triton = None


def _launch_opts():
    # torch rounds w * p and the subtraction separately: no FMA contraction in the kernels
    return {"enable_fp_fusion": False}


def fold(X, target_probs, draft_probs, eta, gumbel, live=None, live_sub=0, debug=None):
    """X [bs, g] int (any strides), target_probs [bs, g+1, V] fp32, draft_probs [bs, g, V] fp32 (contiguous),
    eta [bs, g] fp32, gumbel [bs, V] fp32, live [bs] (optional) = drafts verified per request (<= g);
    live_sub = 1 takes the verify lengths (anchor included) as they are, without a subtraction launch.
    Returns (tau int64 [bs], correction token int64 [bs]); bit-identical to block_accept_probs.
    debug (tests): filled with S, h, mass, res, keys."""
    if triton is None:
        raise RuntimeError("DSV41_BLOCK_VERIFY_FOLD needs triton")
    bs, g = X.shape
    V = target_probs.shape[-1]
    assert target_probs.shape == (bs, g + 1, V) and draft_probs.shape == (bs, g, V), (target_probs.shape,
                                                                                        draft_probs.shape)
    assert target_probs.dtype == torch.float32 and draft_probs.dtype == torch.float32
    assert target_probs.is_contiguous() and draft_probs.is_contiguous()
    eta = eta.contiguous()
    gumbel = gumbel.contiguous()
    assert gumbel.dtype == torch.float32 and gumbel.shape == (bs, V) and eta.shape == (bs, g)
    dev = X.device
    ch = min(CHUNK, _pow2(V))
    nch = (V + ch - 1) // ch
    gp = max(2, _pow2(g + 1))
    has_live = live is not None
    tau = torch.empty((bs,), dtype=torch.int64, device=dev)
    lv = live if has_live else tau
    common = dict(G=g, GP=gp, HAS_LIVE=has_live, LIVE_SUB=int(live_sub))
    opts = _launch_opts()
    if g > 1:
        terms = torch.empty((bs, g - 1, V), dtype=torch.float32, device=dev)
        _terms_kernel[(bs, nch)](target_probs, draft_probs, X, X.stride(0), X.stride(1), lv, terms, V,
                                 CH=ch, **common, **opts)
        s = terms.sum(-1)                               # production's reduction, production's operand
    else:
        terms = None
        s = torch.zeros((bs, 1), dtype=torch.float32, device=dev)
    s = s.contiguous()
    res = torch.empty((bs, V), dtype=torch.float32, device=dev)
    hdbg = torch.empty((bs, gp), dtype=torch.float32, device=dev)
    _residual_kernel[(bs, nch)](target_probs, draft_probs, X, X.stride(0), X.stride(1), lv, eta, s, res, tau,
                                hdbg, V, CH=ch, **common, **opts)
    mass = res.sum(-1, keepdim=True)                    # production's reduction
    keys = torch.empty((bs, V), dtype=torch.float32, device=dev)
    _keys_kernel[(bs, nch)](target_probs, res, mass, tau, gumbel, keys, V, G=g, CH=ch, **opts)
    corr = keys.argmax(dim=-1)                          # production's argmax
    if debug is not None:
        debug.update(s=None if terms is None else s, h=hdbg[:, :g], mass=mass.squeeze(-1), res=res, keys=keys,
                     tau=tau)
    return tau, corr


# --------------------------------------------------------------------------------------- receipts / check --
_C = {"counters": None, "check_calls": 0}
_R = {"torch": 0, "fold": 0, "check": 0, "rows": 0, "calls": 0}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def receipt(path, rows):
    """Host-side count of sampled block-verify calls per path (torch / fold / check); no device sync. Logged every
    LOG_EVERY calls: the window's evidence that the arm it measured ran the path it claims."""
    _R[path] += 1
    _R["rows"] += int(rows)
    _R["calls"] += 1
    if LOG_EVERY and _R["calls"] % LOG_EVERY == 0:
        print(f"[block_verify_fold] receipt rank={_rank()} calls={_R['calls']} torch={_R['torch']} "
              f"fold={_R['fold']} check={_R['check']} rows={_R['rows']}", flush=True)


def check_counts(fold_out, ref_out):
    """Device-side: [calls, rows, tau mismatches, token mismatches] (no host sync until the log line)."""
    t_f, c_f = fold_out
    t_r, c_r = ref_out
    c = _C["counters"]
    if c is None or c.device != t_f.device:
        c = _C["counters"] = torch.zeros(4, dtype=torch.int64, device=t_f.device)
    one = torch.ones((), dtype=torch.int64, device=t_f.device)
    c.add_(torch.stack([one, torch.full_like(one, t_f.numel()), (t_f != t_r).sum(), (c_f != c_r).sum()]))
    _C["check_calls"] += 1
    if LOG_EVERY and _C["check_calls"] % LOG_EVERY == 0:
        v = c.tolist()
        print(f"[block_verify_fold] check: rank={_rank()} calls {v[0]} rows {v[1]} tau mismatches {v[2]} "
              f"token mismatches {v[3]}", flush=True)


# Terminal receipts: the periodic lines above leave up to LOG_EVERY - 1 calls unreported. On the scheduler's busy ->
# idle transition (every TP rank reaches it after the same step, once the work in flight has drained) each rank logs
# its totals, and in check mode its device counters, once per transition that saw a sampled call. While idle the hook
# costs one dict comparison per scheduler loop; the periodic cadence is unchanged.
_LOGGED = {"calls": 0}


def flush(at="idle"):
    if _R["calls"] == _LOGGED["calls"]:
        return
    _LOGGED["calls"] = _R["calls"]
    try:
        print(f"[block_verify_fold] receipt rank={_rank()} calls={_R['calls']} torch={_R['torch']} "
              f"fold={_R['fold']} check={_R['check']} rows={_R['rows']} at={at}", flush=True)
        c = _C["counters"]
        if c is not None:
            v = c.tolist()
            print(f"[block_verify_fold] check: rank={_rank()} calls {v[0]} rows {v[1]} tau mismatches {v[2]} "
                  f"token mismatches {v[3]} at={at}", flush=True)
    except Exception as exc:  # noqa: BLE001 - a receipt never stops the scheduler; a missing one fails the gate
        print(f"[block_verify_fold] terminal receipt failed: {exc!r}", flush=True)


def install_scheduler(module):
    """On import of sglang.srt.managers.scheduler (sitecustomize, DSV41_BLOCK_VERIFY_FOLD set): flush() in on_idle."""
    cls = module.Scheduler
    if getattr(cls, "_dsv41_fold_receipt", False):
        return
    name = next((n for n in ("on_idle", "self_check_during_idle") if hasattr(cls, n)), None)
    if name is None:
        print("[block_verify_fold] Scheduler has no on_idle hook: no terminal receipts", flush=True)
        return
    original = getattr(cls, name)

    def idle_hook(self, *args, **kwargs):
        flush("idle")
        return original(self, *args, **kwargs)

    idle_hook.__wrapped__ = original
    setattr(cls, name, idle_hook)
    cls._dsv41_fold_receipt = True
    print(f"[block_verify_fold] terminal receipts on Scheduler.{name}", flush=True)
