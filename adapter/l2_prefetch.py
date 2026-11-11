"""Prefetch the next dense weights into L2 while DRAM is idle in a decode step (decode only).

Every decode step spends ~3.6 ms (c1, rank 1-3 profile 2026-09-24) inside ~89 one-shot
all-reduces and ~49 all-gathers during which no other kernel runs: the RoCE kernel is a 1-8 CTA
grid polling flags, so the SMs and DRAM are idle. The kernel that follows each window is a dense
MXFP8 GEMM (post-MoE AR -> next layer's wqkv_a; all-gather -> wq_b) or the router + shared expert
(post-attention AR), all weight-bound at M<=16. This adapter issues
``cp.async.bulk.prefetch.L2`` for the first DSV41_L2_PREFETCH_MB of those weights right before
the collective is launched, so the GEMM's first K tiles come from L2. The prefetch kernel runs on
a side stream forked right before the collective (issuing 8 MB of bulk prefetches takes ~20 us
because the TMA queue drains at DRAM speed, so it must not sit on the main stream) and is joined
at the next collective or at the end of the forward, long after it finished. Weights are static and a
prefetch is only a cache hint: outputs are bit-identical by construction.

Which weights follow which window is learned, not hard-coded: during the eager warm-up
forwards that SGLang runs before capturing each CUDA graph, the collectives are numbered per
forward and every MXFP8 dense linear records itself in the event sequence. The capture pass
(the same Python sequence) then forks the prefetch right before the collective, so it becomes a
graph branch. A forward whose signature was never learned prefetches nothing; weights are static,
so a stale plan can only waste bandwidth, never change a result.

Gate ``DSV41_L2_PREFETCH=1`` (off by default). ``DSV41_L2_PREFETCH_MB`` budget per window
(default 6), ``DSV41_L2_PREFETCH_AG=0`` skips all-gather windows.

Production: DSV41_L2_PREFETCH=1 DSV41_L2_PREFETCH_WOA=1 (fleet A/B 2026-09-25: WOA -0.4 ms/step,
greedy output byte-identical). Every other sub-gate below (AHEAD, DRAFT, ENGRAM, LMHEAD, SKIP_N,
WOB_MB, MOE) was measured on the fleet the same day, came out flat or slower, and is NOT
recommended; they stay in the code, off by default (0 / unset), for in-boot A/B runs only.

v3 sub-gates (each needs DSV41_L2_PREFETCH=1; all off = the v2 plan, bit for bit):
  DSV41_L2_PREFETCH_WOA=1     wo_a (the fp8 twin of adapter/wo_a_w8.py, or the bf16 weight) is
                              recorded, and the MXFP8 linear right before it (wq_b, or the
                              indexer's last linear in compressed layers) becomes a window: after
                              that linear, a side branch prefetches DSV41_L2_PREFETCH_WOA_MB (6)
                              of wo_a while rope / paging / sparse MLA run (~22-40 us, latency
                              bound). This branch is joined at the end of the forward.
  DSV41_L2_PREFETCH_AHEAD=1   a window whose own weights are smaller than its budget continues
                              into the next window's weights (post-MoE AR: wqkv_a slice 2.6 MB,
                              then the first MB of wq_b); the next window resumes where it stopped.
  DSV41_L2_PREFETCH_DRAFT=1   the DSpark draft forward is bracketed too (its own plans).
  DSV41_L2_PREFETCH_ENGRAM=1  the Engram row join (host-callback wait, 0.2-0.7 ms idle per step)
                              becomes a window of DSV41_L2_PREFETCH_ENGRAM_MB (12) that always
                              looks ahead into the engram.wkv slice read after the row all-reduce.
                              Joined at the end of the forward.
  DSV41_L2_PREFETCH_LMHEAD=1  the target LM head weight is recorded after the last collective.
v4 knobs (read per call through ab_variant.env(); all unset = v3):
  DSV41_L2_PREFETCH_SKIP_N=512  MXFP8 weights with N <= this many output rows leave the plan (the
                              wqkv_a slice, a latency-bound GEMM); a window left empty reaches into
                              the next window (post-MoE AR -> wq_b).
  DSV41_L2_PREFETCH_WOB_MB=4  the attention-core window takes WOA_MB of wo_a, then this much of wo_b.
  DSV41_L2_PREFETCH_MOE=1     (measured NO-GO, off) after the router GEMM, a side branch prefetches
                              DSV41_L2_PREFETCH_MOE_MB (10) of the next layer's dense weights with an
                              L2::evict_last policy while the MoE streams, demoting the previous
                              window's lines first; later windows start after those bytes. Against
                              the real b12x MoE it slows the MoE by ~12.6 us per 10 MB and the lines
                              do not outlive the next layer's dense stream: -13 us per layer.
  (v3's TILED experiment, scales + a K prefix of every row, lost on GB10 and was removed.)
v5 knob (read per plan through ab_variant.env(); unset / 0 = v4 bit for bit: no extra event is even recorded):
  DSV41_L2_PREFETCH_SHPF_MB=6  shared-expert window. The FFN-side hyper-connection entry (the attention-side
                              mix statistics, hc_attn_fn: TF32x3 stats + Sinkhorn on the stats stream, then the
                              post / combine / norm, ~25 us of mostly idle DRAM on every rank, right before the
                              router) forks a prefetch, on the second side stream, of the shared expert's gate_up
                              (FC1) remainder and down_proj (FC2): the bytes of the next shared-expert call that
                              no other window of the plan already covers (the post-attention all-reduce window
                              takes the router and the first ~2 MB of FC1; a v6 W2 window, when on, only wo_b
                              bytes). Fork point: right before DeepseekV4DecoderLayer._hc_mix_stats(hc_attn_fn) is
                              recorded (at c1 inside the fused all_reduce_mhc_norm path, before the post-attention
                              AR; at >8 rows after it). The consumer is the next GEMM chain (router, then FC1 / FC2
                              on the alt stream), so no dense stream sits between the prefetch and its use (the
                              X2e failure was 13-18 MB of streaming in between). Joined at the end of the forward.
                              Cache hints only: outputs are bit-identical by construction. Receipt: the plan line's
                              " shpf=<windows>/<MB> MB" and the capture line's " shpf=<forks>".
  DSV41_L2_PREFETCH_SHPF_EL=1  (default 1; only read while SHPF_MB > 0) the SHPF windows go through l2pf_el, the
                              v4 MOE kernel: the lines are prefetched with an L2::evict_last policy, so FC2's bytes
                              outlive the router + FC1 reads in front of them, and each window first demotes the
                              previous SHPF window's lines (consumed one layer earlier) back to evict_normal, so at
                              most one window (<= SHPF_MB) of evict_last lines is ever resident. 0 = the plain bulk
                              prefetch of v5 (perf/decode-batch-20261109). Same receipts either way.
v6 knob (read per plan through ab_variant.env(); unset / 0 = v4 bit for bit: no extra event is even recorded;
needs DSV41_L2_PREFETCH_WOA=1, the production gate, because the wo_a call is the fork point):
  DSV41_L2_PREFETCH_WOB2_MB=6 second wo_b window ("W2"), forked at the wo_a call itself, on the second side
                              stream, right before the wo_a kernel launches: it takes the wo_b bytes (weight,
                              then scales) that the attention-core window after wq_b did not take (WOA_MB of
                              wo_a, then WOB_MB of wo_b; this one continues where that take stopped, so no
                              byte is prefetched twice), cut at the budget. The window is the wo_a kernel's own
                              run: with wo_a L2-warm (WOA_MB > 0) DRAM is ~60 % idle for ~29 us there, and the
                              only traffic between the prefetch and its consumer is wo_a's L2-served reads,
                              so the lines do not have to survive a DRAM stream (the X2e failure mode). Joined
                              at the end of the forward. Cache hints only: outputs bit-identical by
                              construction. Pairs with DSV41_WO_A_W8_EVICT_FIRST=1 (adapter/wo_a_w8.py), which
                              marks wo_a's own single-use stream evict-first so the W2 lines outlive it.
                              Receipt: the plan line's " wob2=<windows>/<MB> MB" and the capture line's
                              " wob2=<forks>". Stage 1 of diagnostics/megakernel-20261109/DESIGN.md.

Wiring (all gated by the env flag in sitecustomize.py):
  sglang.srt.layers.quantization.fp8_utils  -> l2_prefetch.install_fp8_utils(module)  (after mxfp8_b12x)
  b12x.comm.roce(_ring).roce_oneshot        -> l2_prefetch.install_roce(module)
  sglang.srt.models.deepseek_v4             -> l2_prefetch.install_model(module)   (after wo_a_w8)
  sglang.srt.models.deepseek_v2             -> l2_prefetch.install_router(module)
  sglang.srt.models.deepseek_v4_dspark      -> l2_prefetch.install_draft(module)   (_DRAFT)
  sglang.srt.layers.engram                  -> l2_prefetch.install_engram(module)  (_ENGRAM, after engram_prefetch)
Microbench and go/no-go: sparks/diagnostics/dsv41-l2-pdl/RESULTS.md, sparks/diagnostics/dsv41-l2-v3/RESULTS.md.
"""
import ctypes as C
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): per-variant gates; plain os.environ when off
    from ab_variant import env as _env, sig_tag as _ab_tag
except ImportError:
    _env, _ab_tag = os.environ.get, tuple

logger = logging.getLogger(__name__)

_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
// segs: [n][2] int64 {address, bytes}; one bulk prefetch per <=chunk piece.
__global__ void l2pf(const long long* __restrict__ segs, int n, long long chunk) {
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = threadIdx.x * chunk; off < b; off += (long long)blockDim.x * chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a + off), "r"((unsigned)sz) : "memory");
    }
  }
}
extern "C" int dsv41_l2pf(const void* segs, int n, long long chunk, uintptr_t stream) {
  l2pf<<<n < 8 ? n : 8, 64, 0, (cudaStream_t)stream>>>((const long long*)segs, n, chunk);
  return (int)cudaGetLastError();
}
// v4 MOE: the same bulk prefetch with an L2::evict_last cache policy, so lines prefetched during
// the MoE stream survive it; first, the previous MoE window's lines go back to evict_normal
// (applypriority), so at most one window's worth of evict_last lines is ever resident.
__global__ void l2pf_el(const long long* __restrict__ segs, int n, const long long* __restrict__ old, int n_old,
                        long long chunk) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long long)gridDim.x * blockDim.x;
  for (int s = 0; s < n_old; ++s)
    for (long long off = tid * 128; off < old[2 * s + 1]; off += nth * 128)
      asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :: "l"(old[2 * s] + off) : "memory");
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = threadIdx.x * chunk; off < b; off += (long long)blockDim.x * chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                     :: "l"(a + off), "r"((unsigned)sz), "l"(pol) : "memory");
    }
  }
}
extern "C" int dsv41_l2pf_el(const void* segs, int n, const void* old, int n_old, long long chunk, uintptr_t stream) {
  l2pf_el<<<8, 64, 0, (cudaStream_t)stream>>>((const long long*)segs, n, (const long long*)old, n_old, chunk);
  return (int)cudaGetLastError();
}
"""

_state = {"lib": None, "learning": {}, "plans": {}, "sig": None, "idx": 0, "lin": 0, "eng": 0, "moe": 0,
          "woa": 0, "woa_forks": 0, "hc": 0, "hc_forks": 0,
          "side": {}, "side2": {}, "pending": None, "pending2": None, "stats": [0, 0],
          "installed": set(), "logged": set(), "frozen": set(), "keepalive": [], "lm_head": {},
          "woa_depth": 0}
_CHUNK = 16384
# RoCEnante packages whose collectives fork the prefetch: the stock one, the switchless-ring
# one (DSV41_ROCE_RING=1, runtime/b12x/b12x/comm/roce_ring) and the two-shot copy
# (DSV41_ROCE_TWOSHOT=1, runtime/b12x/b12x/comm/roce_twoshot); the overlay picks one.
_ROCE_MODULES = ("b12x.comm.roce.roce_oneshot", "b12x.comm.roce_ring.roce_oneshot",
                 "b12x.comm.roce_twoshot.roce_oneshot")


def enabled():
    return _env("DSV41_L2_PREFETCH", "0").strip() in ("1", "on", "true")


def _sub(name):
    """v3 sub-gate DSV41_L2_PREFETCH_<name> (off by default, needs the main gate)."""
    return enabled() and _env(f"DSV41_L2_PREFETCH_{name}", "0").strip() in ("1", "on", "true")


def _installed(name=None):
    """Install-time gate: always os.environ. Under DSV41_AB_VARIANTS it holds the union of every
    variant (adapter/ab_variant.py), so code a later variant needs is installed; the per-call checks
    (enabled() / _sub()) then pick the variant being captured or replayed."""
    on = os.environ.get("DSV41_L2_PREFETCH", "0").strip() in ("1", "on", "true")
    if name is None or not on:
        return on
    return os.environ.get(f"DSV41_L2_PREFETCH_{name}", "0").strip() in ("1", "on", "true")


def _mb(name, default):
    return int(float(_env(name, default)) * (1 << 20))


def _budget():
    return _mb("DSV41_L2_PREFETCH_MB", "6")


def _shpf():
    """v5 SHPF budget in bytes for the variant being learned / captured / replayed (0 = off)."""
    return _mb("DSV41_L2_PREFETCH_SHPF_MB", "0") if enabled() else 0


def _shpf_el():
    """v5 SHPF window policy for the variant: evict_last through l2pf_el (default) or the plain bulk prefetch."""
    return _env("DSV41_L2_PREFETCH_SHPF_EL", "1").strip() not in ("0", "off", "false", "no")


def _wob2():
    """v6 W2 budget in bytes for the variant being learned / captured / replayed (0 = off)."""
    return _mb("DSV41_L2_PREFETCH_WOB2_MB", "0") if _sub("WOA") else 0


def _lib():
    if _state["lib"] is None:
        cache = Path(os.environ.get("B12X_COMPILE_CACHE_DIR", "/tmp")) / "dsv41_l2pf"
        cache.mkdir(parents=True, exist_ok=True)
        so = cache / "libdsv41_l2pf_v4.so"  # older cached builds lack l2pf_el
        if not so.exists():
            cu = cache / "l2pf.cu"
            cu.write_text(_SRC)
            tmp = cache / f"libdsv41_l2pf_v4.{os.getpid()}.so"
            subprocess.run(["nvcc", "-O3", "-arch=sm_121a", "-shared", "-Xcompiler", "-fPIC",
                            "-o", str(tmp), str(cu)], check=True)
            os.replace(tmp, so)
        lib = C.CDLL(str(so))
        lib.dsv41_l2pf.argtypes = [C.c_void_p, C.c_int, C.c_longlong, C.c_void_p]
        lib.dsv41_l2pf_el.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_longlong, C.c_void_p]
        _state["lib"] = lib
    return _state["lib"]


# -- forward bracketing -------------------------------------------------------------------------

def _begin_forward(sig):
    """Every eager decode/verify forward of a shape that has not been captured yet re-learns its
    plan. The first eager forward of a shape can be a dummy/profiling run outside the graph-capture
    communicator state, whose all-reduces go through NCCL and never reach RoCE (seen 2026-09-25:
    the bs16 verify shape learned 0 collectives and ran uncaptured-plan). The two warm-up forwards
    SGLang runs right before each capture are what the capture replays, so the latest eager
    observation wins, and a shape is frozen once captured."""
    _state["sig"], _state["idx"], _state["lin"], _state["eng"], _state["moe"] = sig, 0, 0, 0, 0
    _state["woa"], _state["woa_forks"] = 0, 0
    _state["hc"], _state["hc_forks"] = 0, 0
    _state["stats"][1] = 0
    if _sub("WOA"):
        _ensure_wo_a()
    if sig not in _state["frozen"] and not torch.cuda.is_current_stream_capturing():
        _state["learning"][sig] = []


def _join():
    pending = _state["pending"]
    if pending is not None:
        torch.cuda.current_stream().wait_stream(pending)
        _state["pending"] = None


def _join2():
    pending = _state["pending2"]
    if pending is not None:
        torch.cuda.current_stream().wait_stream(pending)
        _state["pending2"] = None


def _side_stream(second=False):
    pool = _state["side2" if second else "side"]
    dev = torch.cuda.current_device()
    if dev not in pool:
        pool[dev] = torch.cuda.Stream(device=dev)
    return pool[dev]


def _counts():
    # (idx, lin, eng, moe, woa, hc): the v6 W2 wo_a-call count stays at [4] (v2.7.3), the v5 SHPF hc-entry count is
    # appended at [5]; a learned plan matches a capture only if all six agree
    return (_state["idx"], _state["lin"], _state["eng"], _state["moe"], _state["woa"], _state["hc"])


def _end_forward():
    _join()
    _join2()
    sig = _state["sig"]
    capturing = torch.cuda.is_current_stream_capturing()
    rec = _state["learning"].pop(sig, None) if sig is not None else None
    if rec is not None and not capturing:
        n = _state["idx"]
        counts = _counts()
        key = (counts, tuple(rec))
        old = _state["plans"].get(sig)
        if n == 0:
            # no RoCE collective seen (NCCL path): keep any earlier plan, never learn "nothing"
            if old is None and ("zero",) + sig[1:] not in _state["logged"]:
                _state["logged"].add(("zero",) + sig[1:])
                logger.warning("DSV41_L2_PREFETCH eager forward tokens=%d mode=%s saw no RoCE collective "
                               "(NCCL path); not learned, waiting for the capture warm-up", sig[1], sig[2])
        elif old is None or old[2] != key:
            plan = _make_plan(rec)
            _state["plans"][sig] = (n, plan, key, counts)
            _side_stream()  # created eagerly, never first inside a graph capture
            _side_stream(True)
            nbytes = sum(int(it[0][1::2].sum()) for it in plan.values())
            extra = sum(1 for k in plan if k[0] != "c")
            hk = [k for k in plan if k[0] == "h"]
            shpf = (f" shpf={len(hk)}/{sum(int(plan[k][0][1::2].sum()) for k in hk) / 2**20:.1f} MB"
                    if _shpf() > 0 else "")
            wk = [k for k in plan if k[0] == "W"]
            wob2 = (f" wob2={len(wk)}/{sum(int(plan[k][0][1::2].sum()) for k in wk) / 2**20:.1f} MB"
                    if _wob2() > 0 else "")
            logger.warning("DSV41_L2_PREFETCH plan tokens=%d mode=%s collectives=%d windows=%d "
                           "prefetch=%.1f MB/forward%s%s%s%s", sig[1], sig[2], n, len(plan), nbytes / 2**20,
                           f" (of which {extra} linear/engram windows)" if extra else "",
                           # receipt order: " shpf=" (v5), then " wob2=" (v6)
                           "" if old is None else f" (relearned; was {old[0]} collectives)", shpf, wob2)
    elif sig is not None and capturing:
        _state["frozen"].add(sig)
        key = ("capture",) + sig[1:]
        if key not in _state["logged"]:
            _state["logged"].add(key)
            entry = _state["plans"].get(sig)
            learned = entry[0] if entry else 0
            ok = (entry is not None and learned == _state["idx"] and entry[3] == _counts()
                  and _state["stats"][1] > 0)
            logger.warning("DSV41_L2_PREFETCH captured tokens=%d mode=%s branches=%d collectives=%d "
                           "(learned %d)%s%s%s", sig[1], sig[2], _state["stats"][1], _state["idx"], learned,
                           f" shpf={_state['hc_forks']}" if _shpf() > 0 else "",
                           f" wob2={_state['woa_forks']}" if _wob2() > 0 else "",
                           "" if ok else " MISMATCH: this graph runs without prefetch")
    _state["sig"] = None


def _take(q, start, budget):
    """Bytes of queue q ([(ptr, nbytes)] in consumption order) from position start=(i, off), at most
    budget -> (segments, end position). From (0, 0) this is the v2 rule: whole weights in order,
    the last one cut at the budget, every piece rounded down to 16 bytes."""
    out, left = [], budget
    i, off = start
    while i < len(q) and left >= 16:
        ptr, nbytes = q[i][:2]
        avail = nbytes - off
        take = min(avail, left) & ~15
        if take > 0:
            out.append((ptr + off, take))
        if avail > left:
            return out, (i, off + take)
        left -= take
        i, off = i + 1, 0
    return out, (i, off)


def _skip_n():
    return int(float(_env("DSV41_L2_PREFETCH_SKIP_N", "0") or 0))


def _windows(events):
    """Event sequence -> [(key, budget, queue, look_ahead[, ("W", n)])] in forward order. Queue items are
    (ptr, nbytes, rows or None, kind) with kind "r" (linear / router) or "w" (wo_a). A v6 ("W", n) event
    (the n-th wo_a call) marks the wo_a window it falls in: _make_plan then plans the W2 continuation
    for that window under the key ("W", n)."""
    woa, ahead = _sub("WOA"), _sub("AHEAD")
    windows, cur, last_lin = [], None, None
    for ev in events:
        kind = ev[0]
        if kind in ("h", "S", "s"):  # v5 SHPF markers: planned in _shpf_plan, never a window or a weight
            continue
        if kind == "W":  # v6 W2 fork point: attaches to the current wo_a window, never a window of its own
            if cur is not None and cur[0][0] == "l" and len(cur) == 4:
                cur.append(("W", ev[1]))
            continue
        if kind == "m":  # v4 MOE mark: plans against the windows that follow (see _make_plan)
            windows.append([("m", ev[1]), _mb("DSV41_L2_PREFETCH_MOE_MB", "10"), [], False])
            continue
        if kind == "c":
            cur = [("c", ev[1]), _budget(), [], ahead]
            windows.append(cur)
            last_lin = None
        elif kind == "e":
            cur = [("e", ev[1]), _mb("DSV41_L2_PREFETCH_ENGRAM_MB", "12"), [], True]
            windows.append(cur)
            last_lin = None
        elif kind == "L":
            if cur is not None:
                last_lin = (cur, len(cur[2]), ev[1])
        elif kind == "w":
            if woa and last_lin is not None and last_lin[0] is cur:
                owner, pos, lin = last_lin
                tail = owner[2][pos:]
                del owner[2][pos:]
                cur = [("l", lin), _mb("DSV41_L2_PREFETCH_WOA_MB", "6"), tail, ahead]
                windows.append(cur)
                last_lin = None
            if cur is not None:
                cur[2].append((ev[1], ev[2], None, "w"))
        elif cur is not None:  # "r": a weight read, MXFP8 linears with their output rows N
            cur[2].append((ev[1], ev[2], ev[3] if len(ev) > 3 else None, "r"))
    skip = _skip_n()
    if skip > 0:
        # v4 SKIP_N: MXFP8 weights with N <= skip (the wqkv_a slice: a latency-bound GEMM that
        # prefetching does not speed up) leave the plan; a window left with nothing of its own
        # reaches into the next window's weights (post-MoE AR -> wq_b)
        for w in windows:
            kept = [it for it in w[2] if it[2] is None or it[2] > skip]
            if w[2] and not kept:
                w[3] = True
            w[2] = kept
    return windows


def _make_plan(events):
    """events: the forward's sequence of collectives / windows / weight reads -> {window key:
    device segment table}. Without the v3 sub-gates the windows are the collectives and each takes
    the first budget bytes of the weights read before the next collective (the v2 plan)."""
    windows = _windows(events)
    plan = {}
    start = (0, 0)
    wob = _mb("DSV41_L2_PREFETCH_WOB_MB", "0")
    wob2 = _wob2()
    cover = {}            # window index -> queue position already held (evict_last) by a MOE window
    moe_items = []
    for j, w in enumerate(windows):
        key, budget, queue, look_ahead = w[:4]
        w2key = w[4] if len(w) > 4 else None   # v6: this wo_a window's W2 fork point ("W", n)
        if key[0] == "m":
            # v4 MOE: during the MoE phase (DRAM at ~157 of ~250 GB/s) prefetch the next layer's dense
            # weights with evict_last, up to the next MOE mark; later windows start past them
            segs, left = [], budget
            for k in range(j + 1, len(windows)):
                if windows[k][0][0] == "m" or left < 16:
                    break
                more, end = _take(windows[k][2], (0, 0), left)
                segs += more
                left -= sum(x[1] for x in more)
                cover[k] = end
            if segs:
                t = torch.tensor([v for x in segs for v in x[:2]], dtype=torch.int64, device="cuda")
                _state["keepalive"].append(t)
                moe_items.append((key, t, len(segs)))
            continue
        if cover.get(j, (0, 0)) > start:
            start = cover[j]
        if key[0] == "l" and wob > 0:
            # v4 WOB: the attention-core window takes WOA_MB of wo_a, then WOB_MB of wo_b (the
            # weights after wo_a, up to the post-attention AR) instead of the rest of wo_a
            n_w = next((i for i, it in enumerate(queue) if it[3] != "w"), len(queue))
            segs, _ = _take(queue[:n_w], (0, 0), budget)
            more, end_r = _take(queue[n_w:], (0, 0), wob)
            segs += more
            start = (0, 0)
            if segs:
                t = torch.tensor([v for s in segs for v in s[:2]], dtype=torch.int64, device="cuda")
                plan[key] = (t, len(segs))
                _state["keepalive"].append(t)
            if w2key is not None and wob2 > 0:
                _plan_w2(plan, w2key, queue[n_w:], end_r, wob2)
            continue
        segs, end = _take(queue, start, budget)
        start = (0, 0)
        used = sum(s[1] for s in segs)
        if key[0] == "l" and w2key is not None and wob2 > 0:
            # v6 W2 without WOB: the wo_b items start after the wo_a items; continue past what the
            # WOA take reached into them (nothing, at the production budgets)
            n_w = next((i for i, it in enumerate(queue) if it[3] != "w"), len(queue))
            end_r = (end[0] - n_w, end[1]) if end[0] >= n_w else (0, 0)
            _plan_w2(plan, w2key, queue[n_w:], end_r, wob2)
        if look_ahead and j + 1 < len(windows) and end[0] >= len(queue) and budget - used >= 16:
            more, start = _take(windows[j + 1][2], cover.get(j + 1, (0, 0)), budget - used)
            segs += more
        if segs:
            t = torch.tensor([v for s in segs for v in s], dtype=torch.int64, device="cuda")
            plan[key] = (t, len(segs))
            _state["keepalive"].append(t)  # a captured graph keeps reading this address
    for i, (key, t, n) in enumerate(moe_items):
        _, pt, pn = moe_items[i - 1]  # demote the previous MOE window's lines (wraps to the last)
        plan[key] = (t, n, "el", pt if len(moe_items) > 1 else t, pn if len(moe_items) > 1 else 0)
    shpf = _shpf()
    if shpf > 0:
        # v5 SHPF last: the shared-expert bytes no other window (c / l / W / m / e) already holds
        covered = [(int(v[0]), int(v[1])) for item in plan.values()
                   for v in item[0].view(-1, 2).tolist()]
        items = []
        for key, segs in _shpf_plan(events, covered, shpf).items():
            t = torch.tensor([v for sg in segs for v in sg], dtype=torch.int64, device="cuda")
            _state["keepalive"].append(t)
            items.append((key, t, len(segs)))
        el = _shpf_el()
        for i, (key, t, n) in enumerate(items):
            if el:  # evict_last; demote the previous SHPF window's lines first (wraps to the last, as MOE)
                _, pt, pn = items[i - 1]
                plan[key] = (t, n, "el", pt if len(items) > 1 else t, pn if len(items) > 1 else 0)
            else:
                plan[key] = (t, n)
    return plan


def _subtract(pieces, covered):
    """[(ptr, nbytes)] minus the byte ranges in covered, order kept, every piece trimmed to 16-byte bounds."""
    for c0, cn in sorted(covered):
        c1, out = c0 + cn, []
        for p0, pn in pieces:
            p1 = p0 + pn
            if c1 <= p0 or c0 >= p1:
                out.append((p0, pn))
                continue
            if c0 > p0:
                out.append((p0, c0 - p0))
            if c1 < p1:
                out.append((c1, p1 - c1))
        pieces = out
    res = []
    for p0, pn in pieces:
        a = (p0 + 15) & ~15
        n = (p0 + pn - a) & ~15
        if n > 0:
            res.append((a, n))
    return res


def _shpf_plan(events, covered, budget):
    """v5 SHPF: every ("h", n) marker -> the bytes of the next shared-expert call (the weights recorded between
    the next ("S",) / ("s",) pair) that no other window covers, in consumption order, cut at budget."""
    out, i, n_ev = {}, 0, len(events)
    while i < n_ev:
        ev = events[i]
        i += 1
        if ev[0] != "h":
            continue
        j = i
        while j < n_ev and events[j][0] not in ("S", "h"):
            j += 1
        if j >= n_ev or events[j][0] != "S":
            continue
        k, queue = j + 1, []
        while k < n_ev and events[k][0] != "s":
            if events[k][0] == "r":
                queue.append((events[k][1], events[k][2]))
            k += 1
        if not queue:
            continue
        segs, _ = _take(_subtract(queue, covered), (0, 0), budget)
        if segs:
            out[("h", ev[1])] = segs
    return out


def _plan_w2(plan, w2key, items, start, budget):
    """v6 W2: the wo_b bytes (weight, then scales) from position start of the wo_a window's non-wo_a items,
    at most budget, under key ("W", n). Nothing is planned when the earlier take already holds them all."""
    segs, _ = _take(items, start, budget)
    if segs:
        t = torch.tensor([v for s in segs for v in s[:2]], dtype=torch.int64, device="cuda")
        plan[w2key] = (t, len(segs))
        _state["keepalive"].append(t)


# -- hooks --------------------------------------------------------------------------------------

def _rec(ev):
    sig = _state["sig"]
    if sig is not None and sig in _state["learning"]:
        _state["learning"][sig].append(ev)


def _record_weight(*tensors):
    sig = _state["sig"]
    if sig is None or sig not in _state["learning"]:
        return
    rec = _state["learning"][sig]
    for t in tensors:
        if t is not None and t.is_cuda:
            rec.append(("r", t.data_ptr(), t.numel() * t.element_size()))


def _record_linear(weight, weight_scale):
    """MXFP8 linear: weight and scales, tagged with the weight's output rows N (for SKIP_N)."""
    sig = _state["sig"]
    if sig is None or sig not in _state["learning"]:
        return
    rec = _state["learning"][sig]
    rows = int(weight.shape[0]) if getattr(weight, "dim", lambda: 0)() == 2 else None
    for t in (weight, weight_scale):
        if t is not None and t.is_cuda:
            rec.append(("r", t.data_ptr(), t.numel() * t.element_size(), rows))


def _fork(item, stream=None, second=False):
    t, n = item[:2]
    main = stream if isinstance(stream, torch.cuda.Stream) else torch.cuda.current_stream()
    side = _side_stream(second)
    side.wait_stream(main)
    if len(item) > 2:  # MOE: (segments, n, "el", previous MOE segments, their count)
        rc = _lib().dsv41_l2pf_el(t.data_ptr(), n, item[3].data_ptr(), item[4], _CHUNK, side.cuda_stream)
    else:
        rc = _lib().dsv41_l2pf(t.data_ptr(), n, _CHUNK, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH launch failed: cuda error {rc}")
    _state["pending2" if second else "pending"] = side
    _state["stats"][0] += 1
    _state["stats"][1] += 1


def _before_collective(kind, stream):
    sig = _state["sig"]
    if sig is None:
        return
    _join()
    _state["idx"] += 1
    idx = _state["idx"]
    _rec(("c", idx))
    if kind == "ag" and _env("DSV41_L2_PREFETCH_AG", "1") == "0":
        return
    entry = _state["plans"].get(sig)
    if entry is None:
        return
    if idx > entry[0]:
        return  # more collectives than learned: sequence differs, launch nothing past the plan
    item = entry[1].get(("c", idx))
    if item is not None:
        _fork(item, stream)


def _after_linear():
    """WOA: numbered MXFP8 linear boundary; forks the wo_a window planned after this linear."""
    sig = _state["sig"]
    if sig is None:
        return
    _state["lin"] += 1
    lin = _state["lin"]
    _rec(("L", lin))
    entry = _state["plans"].get(sig)
    if entry is None or lin > entry[3][1] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("l", lin))
    if item is not None:
        _fork(item, second=True)


def _before_wo_a():
    """v6 W2: numbered wo_a call (per bracketed forward); forks the wo_b continuation planned for this
    call on the second side stream, right before the wo_a kernel launches. No-op (and no event) when
    DSV41_L2_PREFETCH_WOB2_MB is 0 for the variant, so off is the v4 event stream bit for bit."""
    sig = _state["sig"]
    if sig is None or _wob2() <= 0:
        return
    _state["woa"] += 1
    n = _state["woa"]
    _rec(("W", n))
    entry = _state["plans"].get(sig)
    if entry is None or len(entry[3]) < 5 or n > entry[3][4] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("W", n))
    if item is not None:
        _fork(item, second=True)
        _state["woa_forks"] += 1


def _after_router():
    """v4 MOE: the MoE phase starts after the router GEMM; fork the evict_last prefetch there."""
    _state["moe"] += 1
    n = _state["moe"]
    _rec(("m", n))
    entry = _state["plans"].get(_state["sig"])
    if entry is None or n > entry[3][3] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("m", n))
    if item is not None:
        _fork(item, second=True)


def _before_hc_stats():
    """v5 SHPF: the attention-side mix statistics (the FFN-side hyper-connection entry) are about to be recorded;
    fork the shared-expert prefetch planned for this boundary. No-op (and no event) while SHPF_MB is 0."""
    sig = _state["sig"]
    if sig is None or _shpf() <= 0:
        return
    _state["hc"] += 1
    n = _state["hc"]
    _rec(("h", n))
    entry = _state["plans"].get(sig)
    if entry is None or len(entry[3]) < 6 or n > entry[3][5] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("h", n))
    if item is not None:
        _fork(item, second=True)
        _state["hc_forks"] += 1


def _shared_mark(kind):
    """v5 SHPF: ("S",) / ("s",) bracket the shared expert's weight reads in the learned event sequence."""
    if _state["sig"] is not None and _shpf() > 0:
        _rec((kind,))


def _before_engram():
    sig = _state["sig"]
    if sig is None or not _sub("ENGRAM"):
        return
    _state["eng"] += 1
    n = _state["eng"]
    _rec(("e", n))
    entry = _state["plans"].get(sig)
    if entry is None or n > entry[3][2] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("e", n))
    if item is not None:
        _fork(item, second=True)


def _record_wo_a(o, wo_a):
    """The bytes the wo_a dispatch will read: the fp8 twin (adapter/wo_a_w8.py) when one exists
    for this weight (2+ rows, or always once the bf16 copy was dropped), else the bf16 weight."""
    sig = _state["sig"]
    if (sig is None or sig not in _state["learning"] or not getattr(wo_a, "is_cuda", False)
            or not _sub("WOA")):
        return
    w8 = sys.modules.get("wo_a_w8")
    twin = None
    if w8 is not None and getattr(w8, "ENABLED", False):
        ptr = wo_a.data_ptr()
        twin = w8._TWINS.get(ptr)
        if twin is not None and o.shape[0] < 2 and ptr not in w8._DROPPED:
            twin = None  # single-row decode keeps the bf16 GEMV
    # twin = (e4m3 [2,1024,4096], exponents [2,32,128]): the 8 KB of exponents first, every tile needs them
    tensors = twin[::-1] if twin is not None else ((wo_a,) if 0 not in wo_a.stride() else ())
    rec = _state["learning"][sig]
    for t in tensors:
        rec.append(("w", t.data_ptr(), t.numel() * t.element_size()))


_WOA_MODULES = ("sglang.srt.models.deepseek_v4", "sglang.srt.models.deepseek_v4_dspark")


def _wrap_wo_a(inner):
    def _apply_wo_a_bf16_matmul(o, wo_a, *args, **kwargs):
        if _state["woa_depth"] == 0:  # outermost wrapper only: one record per call
            _record_wo_a(o, wo_a)
            _before_wo_a()
        _state["woa_depth"] += 1
        try:
            return inner(o, wo_a, *args, **kwargs)
        finally:
            _state["woa_depth"] -= 1

    _apply_wo_a_bf16_matmul.__wrapped__ = inner  # inspect.unwrap reaches the inner dispatch (engine pins)
    _apply_wo_a_bf16_matmul._dsv41_l2_prefetch = True
    return _apply_wo_a_bf16_matmul


def _ensure_wo_a(modules=None):
    """Keep the wo_a recorder outermost on the dispatch the target and the draft call by name.
    Adapters installed after this one (DSV41_FUSE_QUANT) wrap _apply_wo_a_bf16_matmul again and
    serve M 2..8 without calling the inner function, so it is re-checked at every bracketed
    forward (the draft module holds its own reference, imported by name)."""
    for mod in modules or [sys.modules.get(n) for n in _WOA_MODULES]:
        fn = getattr(mod, "_apply_wo_a_bf16_matmul", None) if mod is not None else None
        if fn is not None and not getattr(fn, "_dsv41_l2_prefetch", False):
            mod._apply_wo_a_bf16_matmul = _wrap_wo_a(fn)


def install_fp8_utils(module):
    """Wrap the MXFP8 dense linear (after mxfp8_b12x) so each call records its weight + scales."""
    if not _installed() or ("fp8", id(module)) in _state["installed"]:
        return
    _state["installed"].add(("fp8", id(module)))
    original = module.flashinfer_mxfp8_blockscaled_linear
    woa = _installed("WOA")

    def linear(input, weight, weight_scale, *args, **kwargs):
        _record_linear(weight, weight_scale)
        out = original(input, weight, weight_scale, *args, **kwargs)
        if woa and _sub("WOA"):
            _after_linear()
        return out

    linear.__wrapped__ = original
    module.flashinfer_mxfp8_blockscaled_linear = linear
    logger.warning("DSV41_L2_PREFETCH: MXFP8 linears recorded (budget %.1f MB/window)", _budget() / 2**20)


def install_router(module):
    """Record the bf16 router weight (tiny_gemm_bf16 in deepseek_v2, 3.9 MB, read after the
    post-attention all-reduce at ~140 GB/s)."""
    if not _installed() or ("router", id(module)) in _state["installed"]:
        return
    _state["installed"].add(("router", id(module)))
    _install_shared_marks(module)
    if not hasattr(module, "tiny_gemm_bf16"):
        return
    original = module.tiny_gemm_bf16

    def tiny_gemm_bf16(hidden_states, weight, *args, **kwargs):
        _record_weight(weight)
        out = original(hidden_states, weight, *args, **kwargs)
        if _state["sig"] is not None and _sub("MOE"):
            _after_router()
        return out

    tiny_gemm_bf16.__wrapped__ = original
    module.tiny_gemm_bf16 = tiny_gemm_bf16


def _install_shared_marks(module):
    """v5 SHPF: DeepseekV2MoE._forward_shared_experts brackets the shared expert's weight reads with ("S",) /
    ("s",) (recorded only while the variant's SHPF budget is > 0)."""
    moe = getattr(module, "DeepseekV2MoE", None)
    fse = getattr(moe, "_forward_shared_experts", None) if moe is not None else None
    if fse is None or getattr(fse, "_dsv41_l2_prefetch", False):
        return

    def _forward_shared_experts(self, *args, **kwargs):
        _shared_mark("S")
        try:
            return fse(self, *args, **kwargs)
        finally:
            _shared_mark("s")

    # inspect.unwrap() reaches the engine's own function: the adapters that hash-pin it (prefill_sp's
    # _EXPECTED_FP8_MOE, checked at its install on deepseek_v4, after this deepseek_v2 hook) see the stock source.
    _forward_shared_experts.__wrapped__ = fse
    _forward_shared_experts._dsv41_l2_prefetch = True
    moe._forward_shared_experts = _forward_shared_experts


def install_roce(module):
    """Patch b12x.comm.roce(_ring|_twoshot).roce_oneshot: launch the prefetch right before each collective."""
    if not _installed():
        return
    cls = module.RoceOneshotAllReduce
    if getattr(cls, "_dsv41_l2_prefetch", False):
        return
    cls._dsv41_l2_prefetch = True
    ar, ag = cls.all_reduce, cls.all_gather

    def all_reduce(self, inp, *args, **kwargs):
        if self.should_allreduce(inp):
            _before_collective("ar", kwargs.get("stream"))
        return ar(self, inp, *args, **kwargs)

    def all_gather(self, inp, *args, **kwargs):
        _before_collective("ag", kwargs.get("stream"))
        return ag(self, inp, *args, **kwargs)

    all_reduce.__wrapped__, all_gather.__wrapped__ = ar, ag
    cls.all_reduce, cls.all_gather = all_reduce, all_gather
    _lib()
    logger.warning("DSV41_L2_PREFETCH: RoCE collectives prefetch the next weights into L2")


def _bracket(cls, lm_head=False, gate=None):
    """Wrap cls.forward(self, input_ids, positions, forward_batch, ...): decode / target-verify /
    draft-extend forwards are learned and prefetched, prefill (compute-bound) is left alone."""
    original = cls.forward

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        decode_like = mode is not None and any(
            getattr(mode, f, lambda: False)() for f in ("is_decode", "is_target_verify", "is_draft_extend"))
        if (not decode_like or _state["sig"] is not None or not enabled()
                or (gate is not None and not _sub(gate))):  # per-variant under DSV41_AB_VARIANTS
            return original(self, input_ids, positions, forward_batch, *args, **kwargs)
        sig = (id(self), int(input_ids.shape[0]), str(mode)) + _ab_tag()
        _begin_forward(sig)
        try:
            out = original(self, input_ids, positions, forward_batch, *args, **kwargs)
            if lm_head and _sub("LMHEAD"):
                _record_weight(_state["lm_head"].get(id(self)))
            return out
        finally:
            _end_forward()

    # inspect.unwrap reaches the engine's forward (prefill_sp pins v4.DeepseekV4Model.forward)
    forward.__wrapped__ = original
    cls.forward = forward


def _install_hc_entry(module):
    """v5 SHPF: DeepseekV4DecoderLayer._hc_mix_stats(hc_attn_fn) is the FFN-side hyper-connection entry; the wrapper
    forks the planned shared-expert window there (no-op unless SHPF_MB > 0). True when it installed."""
    layer = getattr(module, "DeepseekV4DecoderLayer", None)
    stats = getattr(layer, "_hc_mix_stats", None) if layer is not None else None
    if stats is None or getattr(stats, "_dsv41_l2_prefetch", False):
        return False

    def _hc_mix_stats(self, x, hc_fn, *args, **kwargs):
        if hc_fn is getattr(self, "hc_attn_fn", None):
            _before_hc_stats()
        return stats(self, x, hc_fn, *args, **kwargs)

    # inspect.unwrap() reaches the engine's own function (prefill_sp's _EXPECTED pins it; prefill_sp's own
    # wrappers keep __wrapped__ the same way)
    _hc_mix_stats.__wrapped__ = stats
    _hc_mix_stats._dsv41_l2_prefetch = True
    layer._hc_mix_stats = _hc_mix_stats
    return True


def install_model(module):
    """Bracket DeepseekV4Model.forward (decode / target-verify only; prefill is compute-bound)."""
    if not _installed():
        return
    cls = module.DeepseekV4Model
    if getattr(cls, "_dsv41_l2_prefetch", False):
        return
    cls._dsv41_l2_prefetch = True
    for name in _ROCE_MODULES:
        if name in sys.modules:  # imported before the finder saw it
            install_roce(sys.modules[name])
    lm_head = _installed("LMHEAD")
    _bracket(cls, lm_head=lm_head)
    extras = []
    if _install_hc_entry(module):
        extras.append("shared-expert hc windows (SHPF, per variant)")
    if _installed("WOA"):
        _ensure_wo_a([module])
        extras.append(f"wo_a windows ({_mb('DSV41_L2_PREFETCH_WOA_MB', '6') / 2**20:.1f} MB)")
        if os.environ.get("DSV41_L2_PREFETCH_WOB2_MB", "0").strip() not in ("", "0"):
            extras.append(f"W2 wo_b windows at the wo_a call ({_mb('DSV41_L2_PREFETCH_WOB2_MB', '0') / 2**20:.1f} MB)")
    if lm_head:
        causal = module.DeepseekV4ForCausalLM
        orig_causal = causal.forward

        def causal_forward(self, *args, **kwargs):
            head = getattr(getattr(self, "lm_head", None), "weight", None)
            if head is not None and id(self.model) not in _state["lm_head"]:
                _state["lm_head"][id(self.model)] = head
            return orig_causal(self, *args, **kwargs)

        causal_forward.__wrapped__ = orig_causal
        causal.forward = causal_forward
        extras.append("LM head")
    if _installed("AHEAD"):
        extras.append("look-ahead")
    logger.warning("DSV41_L2_PREFETCH: model forward bracketed (decode/verify only)%s",
                   f"; v3: {', '.join(extras)}" if extras else "")


def install_draft(module):
    """DSV41_L2_PREFETCH_DRAFT: bracket the DSpark draft forward (TARGET_VERIFY mode, its own
    CUDA graphs) the same way, so its collectives prefetch the next draft stage's weights."""
    if not _installed("DRAFT"):
        return
    cls = module.DeepseekV4ForCausalLMDSpark
    if getattr(cls, "_dsv41_l2_prefetch", False):
        return
    cls._dsv41_l2_prefetch = True
    for name in _ROCE_MODULES:
        if name in sys.modules:  # imported before the finder saw it
            install_roce(sys.modules[name])
    _bracket(cls, gate="DRAFT")
    logger.warning("DSV41_L2_PREFETCH: DSpark draft forward bracketed")


def install_engram(module):
    """DSV41_L2_PREFETCH_ENGRAM: the Engram row join (EngramEmbedding._owned_rows, which waits for
    the host-side row lookups) forks a prefetch of the engram.wkv slice read right after it."""
    if not _installed("ENGRAM"):
        return
    emb = module.EngramEmbedding
    if getattr(emb, "_dsv41_l2_prefetch", False):
        return
    emb._dsv41_l2_prefetch = True
    original = emb._owned_rows

    def _owned_rows(self, indices, *args, **kwargs):
        _before_engram()
        return original(self, indices, *args, **kwargs)

    _owned_rows.__wrapped__ = original
    emb._owned_rows = _owned_rows
    logger.warning("DSV41_L2_PREFETCH: Engram row joins prefetch %.1f MB of engram.wkv",
                   _mb("DSV41_L2_PREFETCH_ENGRAM_MB", "12") / 2**20)
