"""Prefetch the next dense weights into L2 while a RoCE collective runs (decode only).

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

Which weights follow which collective is learned, not hard-coded: during the eager warm-up
forwards that SGLang runs before capturing each CUDA graph, the collectives are numbered per
forward and every MXFP8 dense linear records itself under the last collective. The capture pass
(the same Python sequence) then forks the prefetch right before the collective, so it becomes a
graph branch. A forward whose signature was never learned prefetches nothing; weights are static,
so a stale plan can only waste bandwidth, never change a result.

Gate ``DSV41_L2_PREFETCH=1`` (off by default). ``DSV41_L2_PREFETCH_MB`` budget per window
(default 6), ``DSV41_L2_PREFETCH_AG=0`` skips all-gather windows.

PROTOTYPE, not wired into sitecustomize.py yet. Wiring (all three gated by the env flag):
  sglang.srt.layers.quantization.fp8_utils  -> l2_prefetch.install_fp8_utils(module)  (after mxfp8_b12x)
  b12x.comm.roce.roce_oneshot               -> l2_prefetch.install_roce(module)
  sglang.srt.models.deepseek_v4             -> l2_prefetch.install_model(module)
  sglang.srt.models.deepseek_v2             -> l2_prefetch.install_router(module)
Microbench and go/no-go: sparks/diagnostics/dsv41-l2-pdl/RESULTS.md.
"""
import ctypes as C
import logging
import os
import subprocess
from pathlib import Path

import torch

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
"""

_state = {"lib": None, "learning": {}, "plans": {}, "sig": None, "idx": 0, "side": {},
          "pending": None, "stats": [0, 0], "installed": set(), "logged": set(), "frozen": set(),
          "keepalive": []}
_CHUNK = 16384


def enabled():
    return os.environ.get("DSV41_L2_PREFETCH", "0").strip() in ("1", "on", "true")


def _budget():
    return int(float(os.environ.get("DSV41_L2_PREFETCH_MB", "6")) * (1 << 20))


def _lib():
    if _state["lib"] is None:
        cache = Path(os.environ.get("B12X_COMPILE_CACHE_DIR", "/tmp")) / "dsv41_l2pf"
        cache.mkdir(parents=True, exist_ok=True)
        so = cache / "libdsv41_l2pf.so"
        if not so.exists():
            cu = cache / "l2pf.cu"
            cu.write_text(_SRC)
            tmp = cache / f"libdsv41_l2pf.{os.getpid()}.so"
            subprocess.run(["nvcc", "-O3", "-arch=sm_121a", "-shared", "-Xcompiler", "-fPIC",
                            "-o", str(tmp), str(cu)], check=True)
            os.replace(tmp, so)
        lib = C.CDLL(str(so))
        lib.dsv41_l2pf.argtypes = [C.c_void_p, C.c_int, C.c_longlong, C.c_void_p]
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
    _state["sig"], _state["idx"] = sig, 0
    _state["stats"][1] = 0
    if sig not in _state["frozen"] and not torch.cuda.is_current_stream_capturing():
        _state["learning"][sig] = {}


def _join():
    pending = _state["pending"]
    if pending is not None:
        torch.cuda.current_stream().wait_stream(pending)
        _state["pending"] = None


def _side_stream():
    dev = torch.cuda.current_device()
    if dev not in _state["side"]:
        _state["side"][dev] = torch.cuda.Stream(device=dev)
    return _state["side"][dev]


def _end_forward():
    _join()
    sig = _state["sig"]
    capturing = torch.cuda.is_current_stream_capturing()
    rec = _state["learning"].pop(sig, None) if sig is not None else None
    if rec is not None and not capturing:
        n = _state["idx"]
        key = (n, tuple((i, tuple(v)) for i, v in sorted(rec.items())))
        old = _state["plans"].get(sig)
        if n == 0:
            # no RoCE collective seen (NCCL path): keep any earlier plan, never learn "nothing"
            if old is None and ("zero",) + sig[1:] not in _state["logged"]:
                _state["logged"].add(("zero",) + sig[1:])
                logger.warning("DSV41_L2_PREFETCH eager forward tokens=%d mode=%s saw no RoCE collective "
                               "(NCCL path); not learned, waiting for the capture warm-up", sig[1], sig[2])
        elif old is None or old[2] != key:
            plan = _make_plan(rec)
            _state["plans"][sig] = (n, plan, key)
            _side_stream()  # created eagerly, never first inside a graph capture
            nbytes = sum(int(t[1::2].sum()) for t, _ in plan.values())
            logger.warning("DSV41_L2_PREFETCH plan tokens=%d mode=%s collectives=%d windows=%d "
                           "prefetch=%.1f MB/forward%s", sig[1], sig[2], n, len(plan), nbytes / 2**20,
                           "" if old is None else f" (relearned; was {old[0]} collectives)")
    elif sig is not None and capturing:
        _state["frozen"].add(sig)
        key = ("capture",) + sig[1:]
        if key not in _state["logged"]:
            _state["logged"].add(key)
            entry = _state["plans"].get(sig)
            learned = entry[0] if entry else 0
            ok = entry is not None and learned == _state["idx"] and _state["stats"][1] > 0
            logger.warning("DSV41_L2_PREFETCH captured tokens=%d mode=%s branches=%d collectives=%d "
                           "(learned %d)%s", sig[1], sig[2], _state["stats"][1], _state["idx"], learned,
                           "" if ok else " MISMATCH: this graph runs without prefetch")
    _state["sig"] = None


def _make_plan(rec):
    """rec: {collective idx: [(ptr, bytes), ...] in consumption order} -> device segment tables."""
    budget = _budget()
    plan = {}
    for idx, segs in rec.items():
        out, left = [], budget
        for ptr, nbytes in segs:
            if left <= 0:
                break
            take = min(nbytes, left) & ~15
            if take > 0:
                out.append((ptr, take))
                left -= take
        if out:
            t = torch.tensor([v for s in out for v in s], dtype=torch.int64, device="cuda")
            plan[idx] = (t, len(out))
            _state["keepalive"].append(t)  # a captured graph keeps reading this address
    return plan


# -- hooks --------------------------------------------------------------------------------------

def _record_weight(*tensors):
    sig = _state["sig"]
    if sig is None or sig not in _state["learning"]:
        return
    rec = _state["learning"][sig].setdefault(_state["idx"], [])
    for t in tensors:
        if t is not None and t.is_cuda:
            rec.append((t.data_ptr(), t.numel() * t.element_size()))


def _before_collective(kind, stream):
    sig = _state["sig"]
    if sig is None:
        return
    _join()
    _state["idx"] += 1
    idx = _state["idx"]
    if kind == "ag" and os.environ.get("DSV41_L2_PREFETCH_AG", "1") == "0":
        return
    entry = _state["plans"].get(sig)
    if entry is None:
        return
    if idx > entry[0]:
        return  # more collectives than learned: sequence differs, launch nothing past the plan
    plan = entry[1]
    item = plan.get(idx)
    if item is None:
        return
    t, n = item
    main = stream if isinstance(stream, torch.cuda.Stream) else torch.cuda.current_stream()
    side = _side_stream()
    side.wait_stream(main)
    rc = _lib().dsv41_l2pf(t.data_ptr(), n, _CHUNK, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH launch failed: cuda error {rc}")
    _state["pending"] = side
    _state["stats"][0] += 1
    _state["stats"][1] += 1


def install_fp8_utils(module):
    """Wrap the MXFP8 dense linear (after mxfp8_b12x) so each call records its weight + scales."""
    if not enabled() or ("fp8", id(module)) in _state["installed"]:
        return
    _state["installed"].add(("fp8", id(module)))
    original = module.flashinfer_mxfp8_blockscaled_linear

    def linear(input, weight, weight_scale, *args, **kwargs):
        _record_weight(weight, weight_scale)
        return original(input, weight, weight_scale, *args, **kwargs)

    module.flashinfer_mxfp8_blockscaled_linear = linear
    logger.warning("DSV41_L2_PREFETCH: MXFP8 linears recorded (budget %.1f MB/window)", _budget() / 2**20)


def install_router(module):
    """Record the bf16 router weight (tiny_gemm_bf16 in deepseek_v2, 3.9 MB, read after the
    post-attention all-reduce at ~140 GB/s)."""
    if not enabled() or not hasattr(module, "tiny_gemm_bf16") or ("router", id(module)) in _state["installed"]:
        return
    _state["installed"].add(("router", id(module)))
    original = module.tiny_gemm_bf16

    def tiny_gemm_bf16(hidden_states, weight, *args, **kwargs):
        _record_weight(weight)
        return original(hidden_states, weight, *args, **kwargs)

    module.tiny_gemm_bf16 = tiny_gemm_bf16


def install_roce(module):
    """Patch b12x.comm.roce(_ring).roce_oneshot: launch the prefetch right before each collective."""
    if not enabled():
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

    cls.all_reduce, cls.all_gather = all_reduce, all_gather
    _lib()
    logger.warning("DSV41_L2_PREFETCH: RoCE collectives prefetch the next weights into L2")


def install_model(module):
    """Bracket DeepseekV4Model.forward (decode / target-verify only; prefill is compute-bound)."""
    if not enabled():
        return
    cls = module.DeepseekV4Model
    if getattr(cls, "_dsv41_l2_prefetch", False):
        return
    cls._dsv41_l2_prefetch = True
    import sys
    for name in ("b12x.comm.roce.roce_oneshot", "b12x.comm.roce_ring.roce_oneshot"):
        if name in sys.modules:  # imported before the finder saw it
            install_roce(sys.modules[name])
    original = cls.forward

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        decode_like = mode is not None and any(
            getattr(mode, f, lambda: False)() for f in ("is_decode", "is_target_verify", "is_draft_extend"))
        if not decode_like:
            return original(self, input_ids, positions, forward_batch, *args, **kwargs)
        sig = (id(self), int(input_ids.shape[0]), str(mode))
        _begin_forward(sig)
        try:
            return original(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            _end_forward()

    cls.forward = forward
    logger.warning("DSV41_L2_PREFETCH: model forward bracketed (decode/verify only)")
