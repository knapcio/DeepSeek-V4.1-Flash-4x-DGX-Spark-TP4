"""Engram rows fetched on a side stream as soon as the hash ids exist.

Measured 2026-09-18 with the live profiler at c1: 2.2-2.4 ms of every 48 ms step was GPU idle
before the two `_engram_gather_kernel` launches on real text (1.0 ms on sparkDash prose). The
stall is NVMe latency: the host callback that serves the lookup reads every cache miss
(O_DIRECT 8 KB pread, ~200 us) while the graph waits, once per Engram layer, and the all-reduce
then waits for the slowest rank.

The hash ids for both Engram layers are computed at the start of the forward, ~1 ms of GPU work
before layer 1 needs them and ~15 ms before layer 14 does. This adapter forks a side stream
right after the hasher returns, runs the ids copy, the host lookup and the row copies for every
Engram layer there, and joins the side stream only when the layer's gather runs. Inside a CUDA
graph that is an ordinary fork/join branch with a host node on it; the model, the hasher and
the row store are unchanged, and the gather reads the same staging buffers as before.

Gate ``DSV41_ENGRAM_PREFETCH=1``. ``DSV41_ENGRAM_PREFETCH_CHECK=1`` additionally runs the old
synchronous lookup after each prefetched gather and accumulates the number of gathers whose
output differed (logged every DSV41_STATS_SECONDS; expected 0).
"""
import ctypes as C
import logging
import os
import threading
import time

import torch

import engram_backend as eb

logger = logging.getLogger(__name__)
_EMBEDS = []
_state = {"disabled": False}


def enabled():
    return os.environ.get("DSV41_ENGRAM_PREFETCH", "0").strip() in ("1", "on", "true")


def _checking():
    return os.environ.get("DSV41_ENGRAM_PREFETCH_CHECK", "0").strip() == "1"


def _staging(self, indices, count):
    """Same staging layout as engram_backend.owned, created here when the prefetch is first."""
    capacity = 1 << (count - 1).bit_length()
    key = (indices.device.index, capacity)
    if key not in self._staging:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"Engram staging {key} must be warmed before graph capture")
        ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        w = torch.empty((capacity, 256), dtype=torch.uint8, pin_memory=True)
        s = torch.empty((capacity, 8), dtype=torch.uint8, pin_memory=True)
        dw, ds = w.to(indices.device), s.to(indices.device)
        sequential = torch.arange(capacity, dtype=torch.int64, device=indices.device)
        self._staging[key] = (ids, w, s, dw, ds, sequential)
    ids, w, s, dw, ds, sequential = self._staging[key]
    work_key = (indices.device.index, count)
    if work_key not in self._works:
        self._works[work_key] = eb.Work(self._store, ids.data_ptr(), w.data_ptr(), s.data_ptr(), count)
    return ids, w, s, dw, ds, sequential, self._works[work_key]


def _prefetch(self, indices):
    count = indices.numel()
    self._pending = None
    if not count or self.rows == 0:
        return
    if self._side is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram prefetch stream must exist before graph capture")
        self._side = torch.cuda.Stream(device=indices.device)
    ids, w, s, dw, ds, sequential, work = _staging(self, indices, count)
    main = torch.cuda.current_stream(indices.device)
    self._side.wait_stream(main)  # fork: the hash ids are ready on main
    with torch.cuda.stream(self._side):
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        error = eb._cuda.cudaLaunchHostFunc(self._side.cuda_stream,
                                            C.cast(eb._lib.row_store_lookup, eb.P), C.addressof(work))
        if error:
            raise RuntimeError(f"CUDA Engram host callback failed: {error}")
        dw[:count].copy_(w[:count], non_blocking=True)
        ds[:count].copy_(s[:count], non_blocking=True)
    self._pending = (indices.data_ptr(), tuple(indices.shape), count, dw, ds, sequential)


_WARNED = set()     # once-per-site guard for the WARNING line of the hash_ids shape fall-through


def _prefetch_all(hash_ids):
    # Override site: hash_ids that are not 3-D skip the side-stream prefetch, and the Engram
    # lookup runs synchronously. The per-batch fall-through is kept and logged once as WARNING.
    if hash_ids.dim() != 3 and "dim" not in _WARNED:
        _WARNED.add("dim")
        logger.warning("WARNING DSV41_ENGRAM_PREFETCH skips hash_ids with %d dims (expects 3); "
                       "the Engram lookup runs synchronously", hash_ids.dim())
    if _state["disabled"] or hash_ids.dim() != 3 or hash_ids.shape[0] == 0:
        return
    for emb in _EMBEDS:
        if emb._hash_index is None:
            continue
        _prefetch(emb, hash_ids[:, emb._hash_index])


def install(module):
    if not enabled():
        return
    Emb, Eng, Hasher = module.EngramEmbedding, module.Engram, module.EngramHasher
    for name in ("_staging", "_works", "_store", "rows"):
        pass  # set by engram_backend.install's __init__
    orig_init = Emb.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._hash_index = None
        self._pending = None
        self._side = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._check_acc = None
        _EMBEDS.append(self)

    Emb.__init__ = init
    orig_eng_init = Eng.__init__

    def eng_init(self, *args, **kwargs):
        orig_eng_init(self, *args, **kwargs)
        self.embed._hash_index = self.layer_hash_index

    Eng.__init__ = eng_init
    orig_owned = Emb._owned_rows
    check = _checking()

    def owned(self, indices):
        p = self._pending
        if not p or p[0] != indices.data_ptr() or p[1] != tuple(indices.shape) or p[2] != indices.numel():
            return orig_owned(self, indices)
        self._pending = None
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        _, _, count, dw, ds, sequential = p
        torch.cuda.current_stream(indices.device).wait_stream(self._side)  # join
        out = self._empty(indices)
        engram_gather(dw.data_ptr(), ds.data_ptr(), sequential[:count], out.view(-1, 256), 256, 32)
        if check:
            reference = orig_owned(self, indices)
            if self._check_acc is None:
                self._check_acc = torch.zeros((), dtype=torch.int64, device=indices.device)
            self._check_acc += (out != reference).any().to(torch.int64)
        return out

    Emb._owned_rows = owned
    orig_forward = Hasher.forward

    def forward(self, input_ids, forward_batch):
        hash_ids = orig_forward(self, input_ids, forward_batch)
        # Override site: an error in the side-stream prefetch used to disable DSV41_ENGRAM_PREFETCH for
        # the rest of the process and fall back to the synchronous lookup. The user-set gate that
        # stops taking effect now raises.
        try:
            _prefetch_all(hash_ids)
        except Exception as exc:
            for emb in _EMBEDS:
                emb._pending = None
            raise RuntimeError(
                f"DSV41_ENGRAM_PREFETCH={os.environ.get('DSV41_ENGRAM_PREFETCH', '')!r} is set but the "
                f"Engram side-stream prefetch failed: {exc!r}") from exc
        return hash_ids

    Hasher.forward = forward
    logger.warning("DSV41 Engram prefetch ARMED: rows fetched on a side stream after the hasher%s",
                   " (CHECK mode: old path re-run and compared)" if check else "")
    if check:
        period = float(os.getenv("DSV41_STATS_SECONDS", "60"))

        def loop():
            while True:
                time.sleep(period)
                counts = [int(e._check_acc.item()) if e._check_acc is not None else -1 for e in _EMBEDS]
                logger.warning("DSV41 Engram prefetch CHECK: gathers differing from the synchronous path per layer = %s", counts)

        threading.Thread(target=loop, daemon=True, name="engram-prefetch-check").start()
