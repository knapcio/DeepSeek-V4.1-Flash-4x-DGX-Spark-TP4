"""Faster checkpoint loading without changing the bytes that reach the model.

Measured on the 4x Spark TP4/EP2 fleet (2026-09-17): the target load_weight took 230-290 s while
every rank copied only ~130 GB of the 510 GB checkpoint. The stock loader hands the model
mmap-backed safetensors tensors, and the 24-thread copy pool page-faults them in 4-128 KB pieces
(0.5 GB/s on the ranks with moe_tp_rank 0, ~2 GB/s on the others) while the same NVMe delivers
5-6 GB/s to threaded ``pread``. Warming the page cache ahead of the copies does not survive on
GB10: GPU memory is system memory, so the cache is squeezed while the weights land and the
warmed pages are evicted before the copy touches them. ``posix_fadvise(WILLNEED)`` is capped at
the 128 KB readahead window anyway.

What this does, gated by ``DSV41_FAST_LOAD=1``:

1. When the loader opens a shard, the tensors *this rank* will copy are read eagerly into
   pinned host memory (torch's caching host allocator; anonymous mmaps without CUDA) by a
   16-thread ``pread`` pool (every tensor except the routed experts
   another EP rank owns and the Engram tables, which the row store serves straight from NVMe).
   ``get_tensor`` for those names returns the resident tensor; everything else stays the stock
   mmap tensor. Same file offsets, same dtype, same shape: the values are identical by
   construction and a wrong ownership guess only costs I/O.
2. The model's ``load_weights`` submits every copy to a thread pool and never waits, so the
   shard enumeration would run far ahead and keep whole shards resident. ``maybe_executor_submit``
   is wrapped with a byte budget (``DSV41_FAST_LOAD_INFLIGHT_GB``, default 6): the enumeration,
   and with it the loader's window and the eager reads, stays just ahead of the copies. A copy
   whose source is a view into a slab charges the whole slab, once, until the last queued copy
   from that slab finishes (one slow small copy keeps its whole slab alive); other sources charge
   their own bytes. Host memory in flight is bounded by that budget plus one slab plus the loader
   window (``--model-loader-extra-config {"num_threads":1}`` = 2 shards, ~3.4 GB each on TP4/EP2).
3. The DSpark draft is loaded from the same 48 shards but consumes only ``mtp.*`` tensors (see
   ``_remap_dspark_weight_name``). During that load, shards without any ``mtp.*`` key are handed
   back empty so the loader does not open and enumerate 45 files for nothing.
4. Expert tensor parallelism (``moe_ep_size == 1``, ``moe_tp_size > 1``; knob
   ``DSV41_FAST_LOAD_TP_SLICE``, default ``auto``): every rank needs a 1/moe_tp slice of every
   routed expert, so a routed ``w1``/``w3`` (weight and scale) is read as the rank's contiguous row
   block and a ``w2`` as whole rows (every 4 KB page holds needed bytes, so the device reads it all
   anyway) from which a per-thread bounce buffer keeps the rank's columns. ``get_tensor`` returns a
   full-shape stand-in (``_tp_slice_cls``) whose ``narrow`` along the sliced dim returns the resident
   slice; any other access raises. EP2 is untouched unless ``DSV41_FAST_LOAD_TP_SLICE=1``.
5. Slabs (``DSV41_FAST_LOAD_SLAB_MB``, default 256; 0 = one pinned buffer per tensor as before):
   a shard's eager tensors are packed, in the order the loader yields them (sorted names), into
   power-of-two pinned slabs, and every tensor is a view into its slab. Tensors larger than a
   slab get their own buffer. Full slabs share one size bin of torch's caching host allocator, so
   a freed slab is reused by the next shard: a load makes tens of ``cudaHostAlloc`` calls instead
   of thousands (EP1 used to request 92k small pinned blocks, each rounded up to a power of two).
   A slab goes back to the allocator when the last view into it dies; ``_release_all`` checks with
   storage weak references that none is still alive (an escaped view would pin a whole slab) and
   names the tensors if one is. After the target load an escaped slab is an error (it would sit in
   the KV pool's budget); after the draft load it is a warning. ``DSV41_FAST_LOAD_MEMLOG=1`` logs
   the host-memory counters, the GPU processes and the pinned-allocation counts at each phase end.

Everything is released before the KV pool is sized, so the head's memory budget is unchanged.
"""
import concurrent.futures
import json
import logging
import os
import struct
import threading

logger = logging.getLogger(__name__)

_DRAFT_PREFIX = "mtp."
_TARGET_SKIP_PREFIXES = (_DRAFT_PREFIX,)
_ENGRAM_TABLE = ".engram.embed."
_CHUNK = 8 << 20
_state = {"phase": "target", "armed": False, "ep_warned": False, "layout_logged": False,
          "bytes": 0, "files": 0, "skipped": 0, "resident": 0, "sliced": 0, "tp_logged": False,
          "open_failures": 0, "read_failures": 0,
          # host buffers requested by this module in the current phase (slabs + standalone tensors)
          "host_allocs": 0, "host_alloc_bytes": 0, "slabs": 0, "slab_peak_live": 0, "slab_peak_bytes": 0}
_SLAB_ALIGN = 4096            # every tensor starts on a page boundary inside its slab
_slab_refs: list = []         # [StorageWeakRef, nbytes, path, members, base ptr] of every slab of this phase
_slab_size: dict = {}         # storage base ptr -> (nbytes, StorageWeakRef) of this phase's slabs (pacing)
_load_start: dict = {}        # MemAvailable / cuda_reserved when the target load began (memlog)
_lock = threading.Lock()
_header_cache: dict = {}
_pool = None
# Routed-expert projections the FusedMoE loader narrows by moe_tp_size, and along which dim
# (FusedMoE._weight_loader_impl: SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}; the block
# scales take the same dim through _load_model_weight_or_group_weight_scale).
_TP_SLICE_DIM = {"w1": 0, "w3": 0, "w2": 1}
_TP_SLICE_KINDS = ("weight", "scale")
_tls = threading.local()
_bounces: list = []

_DTYPES = {
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu",
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64", "BOOL": "bool",
}


def enabled() -> bool:
    return os.environ.get("DSV41_FAST_LOAD", "0").strip() in ("1", "on", "true")


def observing() -> bool:
    """``DSV41_FAST_LOAD=observe``: stock loader, but the same memory snapshot after each load."""
    return enabled() or os.environ.get("DSV41_FAST_LOAD", "0").strip() == "observe"


def _parse_header(path):
    with _lock:
        hit = _header_cache.get(path)
    if hit is not None:
        return hit
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    entry = (8 + n, header)
    with _lock:
        _header_cache[path] = entry
    return entry


def _ep_info():
    """(ep_rank, ep_size) of this process, or None when unknown."""
    try:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        return int(par.moe_ep_rank), int(par.moe_ep_size)
    except Exception as exc:
        try:
            from sglang.srt.distributed import parallel_state as ps

            return ps.get_moe_expert_parallel_rank(), ps.get_moe_expert_parallel_world_size()
        except Exception as exc2:
            # Last resort: SGLang assigns moe_ep_rank = tp_rank // (tp_size // ep_size).
            try:
                ep_size = int(os.environ["DSV41_FAST_LOAD_EP_SIZE"])
                from sglang.srt.distributed import parallel_state as ps

                tp_rank, tp_size = ps.get_tensor_model_parallel_rank(), ps.get_tensor_model_parallel_world_size()
                return tp_rank // (tp_size // ep_size), ep_size
            except Exception as exc3:
                # Override site: every EP lookup failed, so the routed experts stay on the stock mmap
                # path. A user-set DSV41_FAST_LOAD_EP_SIZE that cannot be applied raises; without it
                # the fall-through to non-expert eager reads is kept and logged once as WARNING.
                ep_env = os.environ.get("DSV41_FAST_LOAD_EP_SIZE", "").strip()
                if ep_env:
                    raise RuntimeError(
                        f"DSV41 fast load: DSV41_FAST_LOAD_EP_SIZE={ep_env!r} is set but the EP layout "
                        f"cannot be derived from it ({exc3!r}; runtime_context: {exc!r}; parallel_state: {exc2!r})")
                if not _state["ep_warned"]:
                    _state["ep_warned"] = True
                    logger.warning("WARNING DSV41 fast load: EP layout unknown (%r / %r / %r); routed experts "
                                   "fall back to the stock mmap path, eager reads for non-expert tensors only",
                                   exc, exc2, exc3)
                return None


def _moe_tp_info():
    """(moe_tp_rank, moe_tp_size) as FusedMoE reads them (``get_parallel()``), or None when unknown."""
    try:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        return int(par.moe_tp_rank), int(par.moe_tp_size)
    except Exception:
        try:
            from sglang.srt.distributed import parallel_state as ps

            return ps.get_moe_tensor_parallel_rank(), ps.get_moe_tensor_parallel_world_size()
        except Exception:
            # Last resort: the MoE-TP groups are contiguous TP ranks (initialize_model_parallel),
            # moe_tp_size = tp_size // ep_size and moe_tp_rank = tp_rank % moe_tp_size.
            try:
                ep_size = int(os.environ["DSV41_FAST_LOAD_EP_SIZE"])
                from sglang.srt.distributed import parallel_state as ps

                tp_rank, tp_size = ps.get_tensor_model_parallel_rank(), ps.get_tensor_model_parallel_world_size()
                moe_tp = tp_size // ep_size
                return tp_rank % moe_tp, moe_tp
            except Exception as exc3:
                # Override site: every MoE-TP lookup failed, so routed experts are read whole (no TP
                # slicing). A user-set DSV41_FAST_LOAD_EP_SIZE that cannot be applied raises; without
                # it the whole-expert fall-through is kept and logged once as WARNING.
                ep_env = os.environ.get("DSV41_FAST_LOAD_EP_SIZE", "").strip()
                if ep_env:
                    raise RuntimeError(
                        f"DSV41 fast load: DSV41_FAST_LOAD_EP_SIZE={ep_env!r} is set but the MoE-TP layout "
                        f"cannot be derived from it ({exc3!r})")
                if not _state.get("moe_tp_warned"):
                    _state["moe_tp_warned"] = True
                    logger.warning("WARNING DSV41 fast load: MoE-TP layout unknown (%r); routed experts are "
                                   "read whole, no TP slicing", exc3)
                return None


def _tp_slice_mode():
    v = os.environ.get("DSV41_FAST_LOAD_TP_SLICE", "auto").strip().lower()
    if v in ("0", "off", "false"):
        return "off"
    if v in ("1", "on", "true", "force"):
        return "on"
    # Override site: a DSV41_FAST_LOAD_TP_SLICE value outside the known set used to be read as
    # "auto" silently. A user-set unknown value now raises; unset or empty keeps "auto".
    if v not in ("auto", ""):
        raise RuntimeError(
            f"DSV41 fast load: DSV41_FAST_LOAD_TP_SLICE={v!r} is not one of 0/off/false, 1/on/true/force, auto")
    return "auto"


def tp_slicing(ep, tp, mode=None):
    """The (moe_tp_rank, moe_tp_size) to slice routed experts by, or None to read them whole.

    ``auto`` slices only at moe_ep_size == 1 (every rank holds a slice of every expert), so EP>1
    keeps today's whole-expert reads; ``on`` also slices the owned experts under EP>1.
    """
    mode = mode or _tp_slice_mode()
    # Override site: with DSV41_FAST_LOAD_TP_SLICE forced on, an unknown MoE-TP layout used to
    # read the routed experts whole without a message. The forced gate that cannot take effect
    # now raises.
    if mode == "on" and tp is None:
        raise RuntimeError(
            "DSV41 fast load: DSV41_FAST_LOAD_TP_SLICE is set on but the MoE-TP layout is unknown, so "
            "routed experts cannot be TP-sliced")
    if mode == "off" or tp is None or tp[1] <= 1:
        return None
    if mode == "auto" and (ep is None or ep[1] != 1):
        return None
    return tp


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    return None


def _n_routed_experts(path):
    """Routed expert count from config.json next to the shards (nested under text_config for V4.1)."""
    cfg = os.path.join(os.path.dirname(path), "config.json")
    env = os.environ.get("DSV41_FAST_LOAD_N_EXPERTS", "").strip()
    try:
        with open(cfg) as f:
            hit = _find_key(json.load(f), "n_routed_experts")
    except Exception:
        hit = None
    if hit is not None:
        # Override site: config.json n_routed_experts wins over DSV41_FAST_LOAD_N_EXPERTS. A
        # user-set value that disagrees with config.json used to be ignored silently; it now raises.
        if env and int(env) != int(hit):
            raise RuntimeError(
                f"DSV41 fast load: DSV41_FAST_LOAD_N_EXPERTS={env!r} conflicts with n_routed_experts={hit} "
                f"in {cfg}")
        return int(hit)
    return int(env) if env else None


def _expert_id(name):
    if ".experts." not in name:
        return None
    try:
        return int(name.split(".experts.", 1)[1].split(".", 1)[0])
    except ValueError:
        return None


def needed_names(path, phase, ep=None, n_routed=None):
    """Tensor names this rank will copy out of ``path`` in ``phase``."""
    _, header = _parse_header(path)
    if phase == "draft":
        return [k for k in header if k.startswith(_DRAFT_PREFIX)]
    if ep is None or n_routed is None or n_routed % ep[1] != 0:
        return [k for k in header if not k.startswith(_TARGET_SKIP_PREFIXES) and _ENGRAM_TABLE not in k
                and _expert_id(k) is None]
    per_rank = n_routed // ep[1]
    lo, hi = ep[0] * per_rank, (ep[0] + 1) * per_rank
    out = []
    for k in header:
        if k.startswith(_TARGET_SKIP_PREFIXES) or _ENGRAM_TABLE in k:
            continue
        eid = _expert_id(k)
        if eid is None or lo <= eid < hi:
            out.append(k)
    return out


def needed_ranges(path, phase, ep=None, n_routed=None):
    """Merged absolute byte ranges of ``needed_names`` (for tests and accounting)."""
    base, header = _parse_header(path)
    ranges = sorted((base + a, base + b) for k in needed_names(path, phase, ep, n_routed)
                    for a, b in [header[k]["data_offsets"]] if b > a)
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def slice_spec(name, info, tp):
    """(dim, start, length) of this rank's slice of a routed expert tensor, or None to read it whole."""
    if tp is None or _expert_id(name) is None:
        return None
    parts = name.rsplit(".", 2)
    if len(parts) != 3 or parts[1] not in _TP_SLICE_DIM or parts[2] not in _TP_SLICE_KINDS:
        return None
    dim, shape = _TP_SLICE_DIM[parts[1]], info["shape"]
    if len(shape) <= dim or shape[dim] % tp[1] != 0:
        return None
    length = shape[dim] // tp[1]
    return dim, tp[0] * length, length


def read_plan(path, phase, ep=None, n_routed=None, tp=None, experts=True):
    """{name: slice spec or None} of what this rank reads eagerly out of ``path``.

    ``tp`` is ``tp_slicing(...)``'s result; ``experts=False`` leaves the routed experts to the stock
    mmap tensors (used at moe_ep_size == 1 when they cannot be sliced).
    """
    _, header = _parse_header(path)
    names = needed_names(path, phase, ep, n_routed)
    if not experts:
        names = [k for k in names if _expert_id(k) is None]
    return {k: slice_spec(k, header[k], tp) for k in names}


def _itemsize(dtype_name):
    return {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2}.get(dtype_name, 1)


def _slice_geometry(info, spec):
    """(outer, row_bytes, inner_bytes): the tensor as [outer, shape[dim], inner] with byte rows."""
    dim = spec[0]
    shape = info["shape"]
    outer = 1
    for s in shape[:dim]:
        outer *= s
    inner = _itemsize(info["dtype"])
    for s in shape[dim + 1:]:
        inner *= s
    return outer, shape[dim] * inner, inner


def spec_bytes(info, spec):
    """(bytes read from the file, bytes kept resident) for one planned tensor."""
    a, b = info["data_offsets"]
    if spec is None:
        return b - a, b - a
    outer, row, inner = _slice_geometry(info, spec)
    kept = outer * spec[2] * inner
    return (kept if outer == 1 else outer * row), kept


def plan_ranges(path, plan):
    """Merged absolute byte ranges a read plan touches (sliced w1/w3 are row blocks, w2 whole)."""
    base, header = _parse_header(path)
    ranges = []
    for k, spec in plan.items():
        a, b = header[k]["data_offsets"]
        if b <= a:
            continue
        if spec is not None:
            outer, _, inner = _slice_geometry(header[k], spec)
            if outer == 1:
                a, b = a + spec[1] * inner, a + (spec[1] + spec[2]) * inner
        ranges.append((base + a, base + b))
    ranges.sort()
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def slab_bytes():
    """Slab size from ``DSV41_FAST_LOAD_SLAB_MB`` (default 256), rounded down to a power of two
    (torch's caching host allocator rounds every block up to one); 0 = one buffer per tensor."""
    raw = os.environ.get("DSV41_FAST_LOAD_SLAB_MB", "").strip()
    # Override site: a non-numeric DSV41_FAST_LOAD_SLAB_MB used to fall back to 256 silently. A
    # user-set value that does not parse now raises; unset or empty keeps the default 256.
    try:
        mb = float(raw) if raw else 256.0
    except ValueError:
        raise RuntimeError(f"DSV41 fast load: DSV41_FAST_LOAD_SLAB_MB={raw!r} is not a number") from None
    if mb == 0:
        return 0
    # Override site: a negative DSV41_FAST_LOAD_SLAB_MB used to mean 0 (one buffer per tensor)
    # silently. A user-set negative value now raises.
    if mb < 0:
        raise RuntimeError(f"DSV41 fast load: DSV41_FAST_LOAD_SLAB_MB={raw!r} is negative; use 0 for one "
                           f"buffer per tensor")
    n = max(1 << 20, int(mb * 2**20))
    slab = 1 << (n.bit_length() - 1)
    # Override site: the slab size is rounded down to a power of two of at least 1 MB, because
    # torch's caching host allocator rounds every block up to one. A user-set value that the
    # rounding changes now raises instead of being replaced silently.
    if raw and slab != mb * 2**20:
        raise RuntimeError(
            f"DSV41 fast load: DSV41_FAST_LOAD_SLAB_MB={raw!r} would be rounded to {slab >> 20} MB; set a "
            f"power of two of at least 1 MB")
    return slab


def _align(n, a=_SLAB_ALIGN):
    return (n + a - 1) // a * a


def slab_layout(header, plan, slab):
    """Pack a shard's eager tensors into slabs, in the order the loader consumes them.

    Returns ``[(nbytes, [(name, offset, kept_bytes), ...]), ...]``. Names go in sorted order (the
    order ``buffered_multi_thread_safetensors_weights_iterator`` yields them), each at a 4 KiB
    aligned offset, so slabs free progressively as the copies drain. A slab holds at most ``slab``
    bytes; a tensor larger than that is a slab of its own. Empty tensors are left out.
    """
    out = []
    cur, used = [], 0
    for name in sorted(plan):
        kept = spec_bytes(header[name], plan[name])[1]
        if kept <= 0:
            continue
        if kept > slab:
            out.append((kept, [(name, 0, kept)]))
            continue
        if cur and used + kept > slab:
            out.append((used, cur))
            cur, used = [], 0
        cur.append((name, used, kept))
        used = _align(used + kept)
    if cur:
        out.append((min(used, slab), cur))
    return out


_TP_SLICE_CLS = []


def _tp_slice_cls():
    """Full-shape stand-in for a checkpoint tensor of which only this rank's slice is resident.

    The FusedMoE loader only reads ``shape``/``size``/``dim``/``device`` and calls
    ``narrow(shard_dim, moe_tp_rank * n, n)``; that returns the resident slice (a narrow inside it
    returns the matching sub-slice). Every other operation, and any narrow reaching outside the
    slice or along another dim, raises instead of returning bytes that were never read.
    """
    if _TP_SLICE_CLS:
        return _TP_SLICE_CLS[0]
    import torch

    T = torch.Tensor
    meta = (T.shape.__get__, T.dtype.__get__, T.device.__get__, T.ndim.__get__, T.layout.__get__,
            T.is_cuda.__get__, T.is_cpu.__get__, T.requires_grad.__get__, T.dim, T.size, T.numel,
            T.nelement, T.element_size, T.__len__, T.__hash__)

    class TpSliceTensor(T):
        @staticmethod
        def __new__(cls, part, full_shape, dim, start, name):
            r = T._make_wrapper_subclass(cls, tuple(full_shape), dtype=part.dtype, device=part.device)
            r._dsv41_part = part
            r._dsv41_dim = dim
            r._dsv41_start = start
            r._dsv41_name = name
            r._dsv41_resident_nbytes = part.numel() * part.element_size()
            return r

        def __repr__(self):
            return (f"TpSliceTensor({self._dsv41_name}, shape={tuple(self.shape)}, dtype={self.dtype}, "
                    f"resident dim {self._dsv41_dim} [{self._dsv41_start}, "
                    f"{self._dsv41_start + self._dsv41_part.shape[self._dsv41_dim]}))")

        def _narrow(self, dim, start, length):
            dim, start, length = int(dim), int(start), int(length)
            if dim < 0:
                dim += self.dim()
            lo = self._dsv41_start
            hi = lo + self._dsv41_part.shape[self._dsv41_dim]
            if dim != self._dsv41_dim or start < lo or start + length > hi or length < 0:
                raise RuntimeError(
                    f"DSV41 fast load: {self._dsv41_name}: narrow(dim={dim}, start={start}, length={length}) "
                    f"reaches outside the resident slice (dim {self._dsv41_dim} [{lo}, {hi})); the engine's "
                    f"moe_tp layout differs from the one fast_load read for. Boot with "
                    f"DSV41_FAST_LOAD_TP_SLICE=0 (or DSV41_FAST_LOAD=0) and report this.")
            part = self._dsv41_part
            if start == lo and length == part.shape[dim]:
                return part
            return part.narrow(dim, start - lo, length)

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is T.narrow or func is torch.narrow:
                self = args[0]
                rest = list(args[1:])
                for key in ("dim", "start", "length"):
                    if key in kwargs:
                        rest.append(kwargs[key])
                return self._narrow(*rest)
            if func in meta:
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)
            name = next((a._dsv41_name for a in args if isinstance(a, cls)), "?")
            raise RuntimeError(f"DSV41 fast load: {name} holds only this rank's moe_tp slice; "
                               f"{getattr(func, '__name__', func)} is not supported on it (only narrow "
                               f"to the slice). Boot with DSV41_FAST_LOAD_TP_SLICE=0 and report this.")

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError(f"DSV41 fast load: {func} reached a TP-sliced stand-in tensor")

    with _lock:
        if not _TP_SLICE_CLS:
            _TP_SLICE_CLS.append(TpSliceTensor)
        return _TP_SLICE_CLS[0]


def _executor():
    global _pool
    with _lock:
        if _pool is None:
            _pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=int(os.environ.get("DSV41_FAST_LOAD_THREADS", "16")), thread_name_prefix="dsv41-read")
        return _pool


def _read_into(fd, mv, off):
    got = 0
    n = len(mv)
    while got < n:
        r = os.preadv(fd, [mv[got:]], off + got)
        if r <= 0:
            raise IOError(f"short read at {off + got}")
        got += r


def _alloc_host(n):
    """A 1-D uint8 host buffer of ``n`` bytes: pinned, or an anonymous mmap without CUDA.

    Pinned source: the H2D copy is a plain DMA. Pageable sources cost more than the copy: the
    driver keeps a staging pool behind after bursts of concurrent pageable copies (measured ~0.4 GB
    per 3 GB burst on a worker, ~1.5 GB after a full load on the head), and that memory is gone
    from MemAvailable when the KV pool is sized. Pinned blocks come from torch's caching host
    allocator and are returned to the driver by _release_all. The fallback is an anonymous private
    mmap (like safetensors' own tensors), not a malloc allocation: glibc keeps freed multi-MB chunks
    in its arenas once the dynamic mmap threshold has grown, and a first version of this loader left
    ~11 GB resident on the head after the load, which cost 5x of the KV pool. Either way the memory
    goes back when the last tensor viewing it is released.
    """
    import mmap

    import torch

    if _pinned_ok():
        flat = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    else:
        flat = torch.frombuffer(mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS), dtype=torch.uint8)
    with _lock:
        _state["host_allocs"] += 1
        _state["host_alloc_bytes"] += n
    return flat


def _new_slab(n, path, members):
    """Allocate one slab and register a storage weak reference to it (expired = all views gone).

    ``members``: the ``slab_layout`` entries ``(name, offset, kept)`` placed in it.
    """
    flat = _alloc_host(n)
    try:
        from torch.multiprocessing.reductions import StorageWeakRef

        ref = StorageWeakRef(flat.untyped_storage())
    except Exception as exc:
        ref = None
        logger.warning("DSV41 fast load: no weak reference to a slab of %s (%r); its release is not checked",
                       os.path.basename(path), exc)
    with _lock:
        _slab_refs.append([ref, n, path, list(members), flat.data_ptr()])
        _slab_size[flat.data_ptr()] = (n, ref)
        _state["slabs"] += 1
        live = [r for r in _slab_refs if r[0] is not None and not r[0].expired()]
        _state["slab_peak_live"] = max(_state["slab_peak_live"], len(live))
        _state["slab_peak_bytes"] = max(_state["slab_peak_bytes"], sum(r[1] for r in live))
    return flat


def _read_tensor(fd, base, info, dst=None):
    """Read one safetensors entry into a contiguous torch tensor.

    ``dst``: the tensor's region of a slab (uint8, exactly its bytes); None allocates a buffer of
    its own (``_alloc_host``). The returned tensor is a view of that buffer.
    """
    import torch

    dtype = getattr(torch, _DTYPES[info["dtype"]])
    a, b = info["data_offsets"]
    n = b - a
    if n == 0:
        return torch.empty(info["shape"], dtype=dtype)
    flat = _alloc_host(n) if dst is None else dst
    assert flat.numel() == n, (flat.numel(), n)
    mv = memoryview(flat.numpy())
    # Sequential in this thread (a nested pool submit would starve the pool); the pool's
    # parallelism comes from the ~1200 tensors of a shard being read concurrently.
    try:
        for c in range(0, n, _CHUNK):
            _read_into(fd, mv[c:c + _CHUNK], base + a + c)
    finally:
        mv.release()
    return flat.view(dtype).reshape(info["shape"])


def _bounce(n):
    """This reader thread's staging buffer (anonymous mmap, closed by _release_all)."""
    import mmap

    buf = getattr(_tls, "buf", None)
    if buf is None or len(buf) < n:
        buf = mmap.mmap(-1, max(n, _CHUNK), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        _tls.buf = buf
        with _lock:
            _bounces.append(buf)
    return buf


def _read_slice(fd, base, info, spec, dst=None):
    """Read this rank's slice (dim, start, length) of one safetensors entry into a compact tensor.

    ``dst`` as in ``_read_tensor`` (the slice's kept bytes).

    Leading-dim slices (w1/w3 rows) are one contiguous pread straight into the buffer. Other dims
    (w2 columns: 288 of every 1152 bytes per row at TP4) read whole rows through a per-thread bounce
    buffer and keep the slice: a 4 KB page holds ~3.5 such rows, so the device delivers every byte
    either way, and one pread plus a strided copy costs ~0.35 ms per w2 against ~2.6 ms for 5120
    per-row preads (GIL-bound Python loop).
    """
    import torch

    dtype = getattr(torch, _DTYPES[info["dtype"]])
    dim, start, length = spec
    shape = list(info["shape"])
    part_shape = shape[:dim] + [length] + shape[dim + 1:]
    outer, row, inner = _slice_geometry(info, spec)
    n = outer * length * inner
    if n == 0:
        return torch.empty(part_shape, dtype=dtype)
    a = info["data_offsets"][0]
    flat = _alloc_host(n) if dst is None else dst
    assert flat.numel() == n, (flat.numel(), n)
    if outer == 1:
        mv = memoryview(flat.numpy())
        off = base + a + start * inner
        for c in range(0, n, _CHUNK):
            _read_into(fd, mv[c:c + _CHUNK], off + c)
        mv.release()
    else:
        rows = max(1, _CHUNK // row)
        buf = _bounce(min(outer, rows) * row)
        dst = flat.view(outer, length, inner)
        mv = memoryview(buf)
        src_all = torch.frombuffer(buf, dtype=torch.uint8)
        try:
            for r0 in range(0, outer, rows):
                k = min(rows, outer - r0)
                _read_into(fd, mv[:k * row], base + a + r0 * row)
                src = src_all[:k * row].view(k, shape[dim], inner)
                dst[r0:r0 + k].copy_(src[:, start:start + length, :])
        finally:
            del src_all
            mv.release()
    return flat.view(dtype).reshape(part_shape)


def _pinned_ok():
    hit = _state.get("pinned")
    if hit is None:
        raw = os.environ.get("DSV41_FAST_LOAD_PINNED", "").strip()
        # Override site: a DSV41_FAST_LOAD_PINNED value other than 0 or 1 used to mean "not pinned"
        # silently. A user-set unknown value now raises.
        if raw not in ("", "0", "1"):
            raise RuntimeError(f"DSV41 fast load: DSV41_FAST_LOAD_PINNED={raw!r} is not 0 or 1")
        err = None
        try:
            import torch

            hit = bool(torch.cuda.is_available()) and raw != "0"
            if hit:
                torch.empty(1, dtype=torch.uint8, pin_memory=True)
            elif raw != "0":
                err = "torch.cuda.is_available() is False"
        except Exception as exc:
            hit = False
            err = repr(exc)
        # Override site: without CUDA, or when the pinned allocation fails, the eager buffers fall
        # back to anonymous mmaps. A user-set DSV41_FAST_LOAD_PINNED=1 that cannot take effect now
        # raises; with the variable unset the fall-back is kept and logged as WARNING.
        if err is not None:
            if raw == "1":
                raise RuntimeError(f"DSV41 fast load: DSV41_FAST_LOAD_PINNED=1 is set but pinned host memory "
                                   f"is unavailable ({err})")
            logger.warning("WARNING DSV41 fast load: pinned host memory unavailable (%s); eager buffers fall "
                           "back to anonymous mmaps", err)
        _state["pinned"] = hit
        logger.warning("DSV41 fast load: eager buffers are %s", "pinned host memory" if hit else "anonymous mmaps")
    return hit


class _EmptyShard:
    """What the loader sees for a shard the current load cannot use."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return []

    def offset_keys(self):
        return []

    def metadata(self):
        return {}

    def get_tensor(self, name):
        raise KeyError(name)


class _EagerShard:
    """Stock safe_open handle whose needed tensors are read eagerly by the thread pool."""

    def __init__(self, inner, path, names):
        """``names``: a list (every tensor read whole) or a ``read_plan`` dict {name: slice spec}."""
        self._inner = inner
        self._path = path
        base, header = _parse_header(path)
        plan = names if isinstance(names, dict) else dict.fromkeys(names)
        self._specs = {k: s for k, s in plan.items() if s is not None}
        # Destination of every tensor: its region of a slab (the job holds the region, and so the
        # slab, until the read is done; afterwards only the returned view does).
        dst = {}
        slab = slab_bytes()
        if slab:
            for nbytes, members in slab_layout(header, plan, slab):
                flat = _new_slab(nbytes, path, members)
                for name, off, kept in members:
                    dst[name] = flat.narrow(0, off, kept)
                del flat
        self._fd = os.open(path, os.O_RDONLY)
        ex = _executor()
        self._futures = {k: (ex.submit(_read_tensor, self._fd, base, header[k], dst.pop(k, None)) if s is None
                             else ex.submit(_read_slice, self._fd, base, header[k], s, dst.pop(k, None)))
                         for k, s in plan.items()}
        total = kept = 0
        for k, s in plan.items():
            r, h = spec_bytes(header[k], s)
            total += r
            kept += h
        with _lock:
            _state["bytes"] += total
            _state["resident"] += kept
            _state["sliced"] += len(self._specs)
            _state["files"] += 1

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            # Futures nobody asked for would keep their tensor, and with it the whole slab, alive for
            # as long as this handle is referenced: cancel the queued ones, let the running ones
            # finish (they read through self._fd) and drop every reference.
            pending = list(self._futures.values())
            self._futures.clear()
            running = [f for f in pending if not f.cancel()]
            if running:
                concurrent.futures.wait(running)
            del pending, running
            try:
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            os.close(self._fd)
        finally:
            ret = self._inner.__exit__(*exc)
        return ret

    def keys(self):
        return self._inner.keys()

    def offset_keys(self):
        return self._inner.offset_keys()

    def metadata(self):
        return self._inner.metadata()

    def get_slice(self, name):
        return self._inner.get_slice(name)

    def get_tensor(self, name):
        fut = self._futures.pop(name, None)
        if fut is None:
            return self._inner.get_tensor(name)
        try:
            t = fut.result()
            spec = self._specs.get(name)
            if spec is None:
                return t
            _, header = _parse_header(self._path)
            return _tp_slice_cls()(t, header[name]["shape"], spec[0], spec[1], name)
        except Exception as exc:
            with _lock:
                _state["read_failures"] += 1
                n = _state["read_failures"]
            # Override site: a failed eager read used to fall back to the stock mmap tensor with a
            # rate-limited log line. DSV41_FAST_LOAD is set on whenever this handle exists, so the
            # gate that does not take effect for this tensor now raises.
            raise RuntimeError(
                f"DSV41 fast load: DSV41_FAST_LOAD={os.environ.get('DSV41_FAST_LOAD', '')!r} is set but the "
                f"eager read of {name} in {self._path} failed ({exc!r}; {n} eager-read failures so far)") from exc


def _wrap_open(orig, filename, args, kwargs):
    phase = _state["phase"]
    if not (isinstance(filename, (str, os.PathLike)) and str(filename).endswith(".safetensors")):
        return orig(filename, *args, **kwargs)
    path = os.fspath(filename)
    ep = _ep_info() if phase == "target" else None
    n_routed = _n_routed_experts(path) if ep else None
    mode = _tp_slice_mode()
    # Expert TP layout, for both phases (the draft's FusedMoE layers use the same moe_tp groups).
    lay_ep = ep if phase == "target" else (_ep_info() if mode != "off" else None)
    moe_tp = _moe_tp_info() if (lay_ep is not None or mode == "on") else None
    tp = tp_slicing(lay_ep, moe_tp, mode)
    # moe_ep_size == 1 without slicing would read every expert whole on every rank (~2x the EP2
    # bytes, and a pinned window twice EP2's): leave the experts to the stock mmap tensors then.
    experts = not (tp is None and lay_ep is not None and lay_ep[1] == 1 and (moe_tp is None or moe_tp[1] > 1))
    if phase == "target" and not _state["layout_logged"]:
        _state["layout_logged"] = True
        logger.warning("DSV41 fast load: target layout ep=%s n_routed_experts=%s moe_tp=%s tp_slice=%s "
                       "(%s)%s first shard %s", ep, n_routed, moe_tp, tp, mode,
                       "" if experts else "; routed experts left to mmap", path)
    names = needed_names(path, phase, ep, n_routed)
    if phase == "draft" and not names:
        with _lock:
            _state["skipped"] += 1
        return _EmptyShard()
    inner = orig(filename, *args, **kwargs)
    plan = read_plan(path, phase, ep, n_routed, tp, experts) if (tp is not None or not experts) else names
    if not plan:
        return inner
    # Override site: a safe_open call with a non-CPU device gets the stock handle without eager
    # reads. The fall-through is kept and logged once as WARNING.
    if kwargs.get("device", "cpu") not in ("cpu", None) or (len(args) > 1 and args[1] != "cpu"):
        if not _state.get("device_warned"):
            _state["device_warned"] = True
            logger.warning("WARNING DSV41 fast load: safe_open(%s) asks for device %r; the shard uses the "
                           "stock loader without eager reads", path, kwargs.get("device", args[1:2]))
        return inner
    return _EagerShard(inner, path, plan)


def install_weight_utils(module):
    """Wrap ``safetensors.safe_open`` as used by ``weight_utils``."""
    if not enabled() or _state["armed"]:
        return
    import safetensors

    orig = safetensors.safe_open

    def safe_open(filename, *args, **kwargs):
        try:
            return _wrap_open(orig, filename, args, kwargs)
        except Exception as exc:
            with _lock:
                _state["open_failures"] += 1
            # Override site: an error in the fast path used to hand the shard to the stock loader
            # with a log line. DSV41_FAST_LOAD is set on whenever this wrapper is installed, so the
            # gate that does not take effect for this shard now raises (this also lets the
            # RuntimeErrors of the checks above reach the engine start).
            raise RuntimeError(
                f"DSV41 fast load: DSV41_FAST_LOAD={os.environ.get('DSV41_FAST_LOAD', '')!r} is set but shard "
                f"{filename} cannot use the fast path ({exc!r})") from exc

    safetensors.safe_open = safe_open
    if getattr(module, "safetensors", None) is not None:
        module.safetensors.safe_open = safe_open
    _state["armed"] = True
    logger.warning("DSV41 fast load ARMED: eager threaded reads of this rank's tensors; draft load opens only mtp shards")


def _malloc_trim():
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def _release_all():
    """After a load: drop the reader pool, header cache and malloc arenas; report what is still mapped."""
    global _pool
    import gc
    import mmap
    import warnings

    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None
    _tls.__dict__.pop("buf", None)
    with _lock:
        bounces = list(_bounces)
        _bounces.clear()
    for buf in bounces:
        try:
            buf.close()
        except BufferError:  # still exported by a live tensor; unmapped when that goes
            pass
    with _lock:
        paths = list(_header_cache)
        _header_cache.clear()
    gc.collect()
    with warnings.catch_warnings():  # isinstance() over every object trips deprecation shims
        warnings.simplefilter("ignore")
        live = [o for o in gc.get_objects() if isinstance(o, mmap.mmap) and not o.closed]
        escaped = _escaped_slabs()
    _malloc_trim()
    try:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        pass
    # The KV pool is sized from what CUDA reports free, and on GB10 that is the kernel's
    # MemFree: page cache counts as used. Drop every shard's cached pages now (the stock
    # handles of the last window are closed by this point).
    for d in {os.path.dirname(p) for p in paths}:
        try:
            for name in os.listdir(d):
                if name.endswith(".safetensors"):
                    fd = os.open(os.path.join(d, name), os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
        except OSError:
            pass
    return len(live), sum(len(o) for o in live), _rss_mb(), escaped


def _escaped_slabs():
    """Slabs of this phase still alive after the load (call after gc.collect()), then forget them.

    Returns ``[(nbytes, path, [names of the tensors still viewing it])]``. A slab can only stay
    alive through a view of one of its tensors that the model kept (the loader copies into its
    parameters and drops the source); that would hold the whole slab out of the KV pool budget,
    so it is reported by name. Found by the data pointers of live tensors (``gc`` tracks them).
    """
    import bisect
    import gc

    with _lock:
        refs = list(_slab_refs)
        _slab_refs.clear()
        _slab_size.clear()
    alive = [r for r in refs if r[0] is not None and not r[0].expired()]
    if not alive:
        return []
    found = {id(r): set() for r in alive}
    try:
        import torch

        spans = sorted((r[4], r[4] + r[1], r) for r in alive)
        starts = [s[0] for s in spans]
        for o in gc.get_objects():
            if not isinstance(o, torch.Tensor):
                continue
            try:
                parts = [getattr(o, "_dsv41_part", None), o]
                p = next(x for x in parts if x is not None and x.device.type == "cpu" and x.numel())
                ptr = p.data_ptr()
            except Exception:
                continue
            i = bisect.bisect_right(starts, ptr) - 1
            if i < 0 or ptr >= spans[i][1]:
                continue
            r = spans[i][2]
            offs = [m[1] for m in r[3]]
            j = bisect.bisect_right(offs, ptr - r[4]) - 1
            found[id(r)].add(r[3][j][0] if j >= 0 else f"+{ptr - r[4]}")
    except Exception as exc:  # diagnostics only
        return [(r[1], r[2], [f"lookup failed: {exc!r}"]) for r in alive]
    return [(r[1], r[2], sorted(found[id(r)]) or ["no live tensor object found (C++-held view)"]) for r in alive]


def _meminfo_gb(key):
    try:
        for line in open("/proc/meminfo"):
            if line.startswith(key):
                return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return -1.0


def _snapshot():
    out = {k: round(_meminfo_gb(k), 2) for k in ("MemFree", "MemAvailable", "Cached", "Shmem", "AnonPages", "Mapped",
                                                   "Unevictable", "Mlocked", "Slab", "SUnreclaim", "KReclaimable",
                                                   "PageTables", "VmallocUsed", "Percpu", "HugePages_Total")}
    try:
        for line in open("/proc/self/smaps_rollup"):
            if line.startswith(("Rss", "Anonymous", "Shared_Clean", "Private_Clean")):
                out["self_" + line.split(":")[0]] = round(int(line.split()[1]) / 2**20, 2)
    except Exception:
        pass
    try:
        import torch

        out["cuda_free"] = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
        out["cuda_reserved"] = round(torch.cuda.memory_reserved() / 2**30, 2)
        out["cuda_allocated"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        st = torch.cuda.memory_stats()
        out["cuda_inactive_split"] = round(st.get("inactive_split_bytes.all.current", 0) / 2**30, 2)
    except Exception:
        pass
    try:  # biggest mappings of this process (file or anonymous), MB
        big = []
        cur = None
        for line in open("/proc/self/smaps"):
            if line[0] in "0123456789abcdef" and "-" in line.split()[0]:
                parts = line.split()
                cur = parts[5] if len(parts) > 5 else "[anon]"
            elif line.startswith("Rss:"):
                kb = int(line.split()[1])
                if kb >= 64 * 1024:
                    big.append((kb // 1024, cur))
        big.sort(reverse=True)
        out["big_maps_mb"] = big[:8]
    except Exception:
        pass
    return out


def _log_phase(phase):
    """Release and report the phase; returns the escaped slabs (see ``_escaped_slabs``)."""
    with _lock:
        b, f, s = _state["bytes"], _state["files"], _state["skipped"]
        kept, sliced = _state["resident"], _state["sliced"]
        fails = _state["open_failures"], _state["read_failures"]
        allocs = {k: _state[k] for k in ("host_allocs", "host_alloc_bytes", "slabs", "slab_peak_live", "slab_peak_bytes")}
        for k in ("bytes", "files", "skipped", "resident", "sliced", "open_failures", "read_failures") + tuple(allocs):
            _state[k] = 0
    before = _snapshot()
    escaped = []
    if enabled():
        live, live_bytes, rss, escaped = _release_all()
        extra = f" ({sliced} expert tensors TP-sliced, {kept / 1e9:.1f} GB kept)" if sliced else ""
        logger.warning("DSV41 fast load: phase=%s read %.1f GB eagerly over %d shards, %d shards skipped%s; "
                       "host buffers: %d requested (%.1f GB), %d slabs of %d MB, peak %d live (%.2f GB), "
                       "%d alive after release; after release: %d anonymous maps alive (%.2f GB), RSS %d MB",
                       phase, b / 1e9, f, s, extra, allocs["host_allocs"], allocs["host_alloc_bytes"] / 1e9,
                       allocs["slabs"], slab_bytes() >> 20, allocs["slab_peak_live"], allocs["slab_peak_bytes"] / 1e9,
                       len(escaped), live, live_bytes / 1e9, rss)
        if any(fails):
            logger.warning("DSV41 fast load: phase=%s %d shards fell back to the stock loader, %d eager reads fell "
                           "back to mmap (see the warnings above)", phase, *fails)
        for nbytes, path, names in escaped:
            logger.warning("DSV41 fast load: phase=%s slab of %.1f MB from %s still alive after the load, held by "
                           "a view the model kept: %s. It stays pinned and out of the KV pool budget; report this "
                           "(DSV41_FAST_LOAD_SLAB_MB=0 restores one buffer per tensor).",
                           phase, nbytes / 2**20, os.path.basename(path), names[:8])
    logger.warning("DSV41 fast load: phase=%s memory before release %s | after %s", phase, before, _snapshot())
    if memlog_on():
        logger.warning("DSV41 fast load memlog: phase=%s %s", phase, _memlog(allocs, len(escaped)))
    return escaped


def memlog_on() -> bool:
    return os.environ.get("DSV41_FAST_LOAD_MEMLOG", "0").strip() in ("1", "on", "true")


def _meminfo_all():
    """/proc/meminfo in GiB, one read."""
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            parts = v.split()
            out[k] = int(parts[0]) / 2**20 if len(parts) > 1 else int(parts[0])
    except Exception:
        pass
    return out


def _cuda_mem():
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda.memory_reserved() / 2**30, torch.cuda.memory_allocated() / 2**30
    except Exception:
        pass
    return None, None


def _gpu_procs():
    """NVML: [(pid, MiB)] of the processes on the GPU and the device's used GiB (None if not readable).

    PIDs are the host's (they differ from os.getpid() inside a container); GB10 may report N/A."""
    try:
        import pynvml
    except Exception:
        return None, None
    procs, used = [], None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None, None
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                try:
                    for p in getattr(pynvml, fn)(h):
                        mb = getattr(p, "usedGpuMemory", None)
                        procs.append((int(p.pid), None if mb is None else round(mb / 2**20)))
                except Exception:
                    pass
            try:
                used = round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30, 2)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return sorted(set(procs)), used


def _torch_host_stats():
    """Process-wide counters of torch's pinned (caching host) allocator, when it has any."""
    try:
        import torch

        st = torch.cuda.host_memory_stats()
    except Exception:
        return None
    keep = ("num_host_alloc", "num_host_free", "allocations.current", "allocations.peak",
            "allocated_bytes.current", "allocated_bytes.peak", "host_alloc_time.total", "host_free_time.total")
    return {k: (round(st[k] / 2**30, 2) if "bytes" in k else st[k]) for k in keep if k in st} or None


def _memlog(allocs, n_escaped):
    """One line for the phase end: host counters, GPU processes, pinned allocations, and two derived
    numbers. ``unaccounted_gib`` = MemTotal - (MemFree + Buffers + Cached + AnonPages + Slab +
    KernelStack + PageTables + SecPageTables + Percpu) - cuda_reserved: memory no counter shows.
    ``nontorch_since_load_start_gib`` = MemAvailable lost since the target's load_weights began,
    minus what the CUDA caching allocator reserved in that time."""
    m = _meminfo_all()
    reserved, allocated = _cuda_mem()
    out = {k: round(m[k], 2) for k in ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Shmem",
                                        "AnonPages", "Mapped", "Slab", "KReclaimable", "KernelStack",
                                        "PageTables", "SecPageTables", "Percpu", "VmallocUsed", "Unevictable")
           if k in m}
    out["cuda_reserved_gib"] = None if reserved is None else round(reserved, 2)
    out["cuda_allocated_gib"] = None if allocated is None else round(allocated, 2)
    if "MemTotal" in m and reserved is not None:
        counted = sum(m.get(k, 0.0) for k in ("MemFree", "Buffers", "Cached", "AnonPages", "Slab", "KernelStack",
                                               "PageTables", "SecPageTables", "Percpu"))
        out["unaccounted_gib"] = round(m["MemTotal"] - counted - reserved, 2)
    if _load_start and "MemAvailable" in m and reserved is not None:
        out["nontorch_since_load_start_gib"] = round(
            (_load_start["avail"] - m["MemAvailable"]) - (reserved - _load_start["reserved"]), 2)
        out["load_start"] = dict(_load_start)
    procs, used = _gpu_procs()
    out["gpu_procs_mib"] = procs
    out["nvml_used_gib"] = used
    out["self_pid"] = os.getpid()
    out["self_rss_mb"] = _rss_mb()
    out["pinned_requests"] = allocs["host_allocs"]
    out["pinned_request_gb"] = round(allocs["host_alloc_bytes"] / 1e9, 2)
    out["slabs"] = allocs["slabs"]
    out["slab_mb"] = slab_bytes() >> 20
    out["slab_peak_live"] = allocs["slab_peak_live"]
    out["slab_peak_gb"] = round(allocs["slab_peak_bytes"] / 1e9, 2)
    out["slabs_alive_after_release"] = n_escaped
    out["torch_host_alloc"] = _torch_host_stats()
    out["alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF")
    return out


def _mark_load_start():
    """Memlog: MemAvailable and CUDA reserved when the target's load_weights begins (the model's
    parameters are allocated by then, the checkpoint not yet read)."""
    if not memlog_on():
        return
    m = _meminfo_all()
    reserved, _ = _cuda_mem()
    if "MemAvailable" in m:
        _load_start.clear()
        _load_start.update(avail=round(m["MemAvailable"], 2), reserved=round(reserved or 0.0, 2))


def _tensor_bytes(func_args):
    for a in func_args:
        try:
            import torch

            if isinstance(a, torch.Tensor) and not isinstance(a, torch.nn.Parameter):
                kept = getattr(a, "_dsv41_resident_nbytes", None)  # TP-sliced stand-in: its slice
                return kept if kept is not None else a.numel() * a.element_size()
        except Exception:
            pass
    return 0


def _charge(func_args):
    """(key, bytes) a copy charges against the pacing budget.

    The source tensor (first non-Parameter tensor argument; a TP-sliced stand-in's resident slice)
    is looked up by its storage: a view into a slab of this phase charges the whole slab under the
    key ``("slab", base ptr)``, once for all its copies in flight, because any one of them keeps
    the whole slab alive. Anything else charges its own bytes under a key of its own (None).
    """
    try:
        import torch

        for a in func_args:
            if isinstance(a, torch.Tensor) and not isinstance(a, torch.nn.Parameter):
                part = getattr(a, "_dsv41_part", None)
                src = part if part is not None else a
                with torch._C.DisableTorchFunctionSubclass():
                    ptr = src.untyped_storage().data_ptr()
                with _lock:
                    hit = _slab_size.get(ptr)
                # an expired entry is a freed slab whose address now backs another buffer
                if hit is not None and (hit[1] is None or not hit[1].expired()):
                    return ("slab", ptr), hit[0]
                break
    except Exception:
        pass
    return None, _tensor_bytes(func_args)


def install_deepseek_v4(module):
    """Pace the model's async weight copies so the eager reads stay just ahead of consumption."""
    if not enabled():
        return
    orig = module.maybe_executor_submit
    budget = int(float(os.environ.get("DSV41_FAST_LOAD_INFLIGHT_GB", "6")) * 2**30)
    cv = threading.Condition()
    inflight = [0]
    held = {}       # slab key -> [copies in flight, bytes charged]

    def paced_submit(*, executor, futures, use_async, func, func_args=(), func_kwargs=None):
        if not use_async:
            return orig(executor=executor, futures=futures, use_async=use_async, func=func,
                        func_args=func_args, func_kwargs=func_kwargs)
        key, nbytes = _charge(func_args)
        size = max(1, nbytes)
        with cv:
            while not (key is not None and key in held) and inflight[0] > 0 and inflight[0] + size > budget:
                cv.wait(timeout=1.0)
            if key is not None and key in held:
                held[key][0] += 1           # the slab is already charged (and alive)
            else:
                inflight[0] += size
                if key is not None:
                    held[key] = [1, size]
        before = len(futures)

        def release(_f=None):
            with cv:
                if key is None:
                    inflight[0] -= size
                else:
                    h = held[key]
                    h[0] -= 1
                    if h[0] == 0:
                        inflight[0] -= h[1]
                        del held[key]
                cv.notify_all()

        try:
            orig(executor=executor, futures=futures, use_async=use_async, func=func,
                 func_args=func_args, func_kwargs=func_kwargs)
        except BaseException:
            release()
            raise
        if len(futures) > before:
            futures[-1].add_done_callback(release)
        else:
            release()

    module.maybe_executor_submit = paced_submit
    logger.warning("DSV41 fast load: weight copies paced to %.1f GB in flight", budget / 2**30)
    _wrap_target_load(module)


def _wrap_target_load(module):
    """Release everything the moment the target's copies are done, before the engine measures
    its memory: SGLang derives the KV pool from what is free after the weights landed."""
    cls = getattr(module, "DeepseekV4ForCausalLM", None)
    # Override site: without DeepseekV4ForCausalLM the release before KV-pool sizing used to be
    # skipped silently, so the eager buffers would sit in the KV budget. DSV41_FAST_LOAD is set on
    # whenever this runs, so the missing hook now raises.
    if cls is None:
        raise RuntimeError(
            f"DSV41 fast load: DSV41_FAST_LOAD={os.environ.get('DSV41_FAST_LOAD', '')!r} is set but "
            f"{getattr(module, '__name__', module)} has no DeepseekV4ForCausalLM to wrap")
    if getattr(cls.load_weights, "_dsv41_fast_load", False):
        return
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        target = _state["phase"] == "target"
        if target:
            _mark_load_start()
        escaped = []
        try:
            out = orig(self, weights, *args, **kwargs)
        finally:
            if target:
                escaped = _log_phase("target")
        if escaped:
            # an escaped slab stays pinned outside the KV pool budget the engine sizes next
            raise RuntimeError(
                f"DSV41 fast load: {len(escaped)} slab(s) ({sum(e[0] for e in escaped) / 2**20:.0f} MiB) still "
                f"alive after the target load, held by {sorted({n for e in escaped for n in e[2]})[:8]}; boot with "
                f"DSV41_FAST_LOAD_SLAB_MB=0 (one buffer per tensor) and report this")
        return out

    load_weights._dsv41_fast_load = True
    cls.load_weights = load_weights


def _ps_top(n=6):
    try:
        rows = []
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                rss = 0
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmRSS"):
                        rss = int(line.split()[1]) // 1024
                        break
                comm = open(f"/proc/{pid}/comm").read().strip()
                rows.append((rss, pid, comm))
            except Exception:
                pass
        rows.sort(reverse=True)
        return rows[:n]
    except Exception:
        return []


def _schedule_late_snapshots():
    """Diagnostics: how the host memory settles after the loads (other processes still starting)."""
    import time

    def run():
        for delay in (30, 60, 120, 240):
            time.sleep(delay if delay == 30 else delay - prev[0])
            prev[0] = delay
            snap = _snapshot()
            snap.pop("big_maps_mb", None)
            logger.warning("DSV41 fast load: +%ds after draft load: %s | top rss MB %s", delay, snap, _ps_top())

    prev = [0]
    threading.Thread(target=run, name="dsv41-late-snap", daemon=True).start()


def install_dspark(module):
    """Mark the DSpark draft load so shard filtering and mtp-only eager reads apply."""
    if not observing():
        return
    cls = module.DeepseekV4ForCausalLMDSpark
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        _state["phase"] = "draft"
        try:
            return orig(self, weights, *args, **kwargs)
        finally:
            _log_phase("draft")
            _state["phase"] = "target"
            if os.environ.get("DSV41_FAST_LOAD_DEBUG") == "1":
                _schedule_late_snapshots()

    cls.load_weights = load_weights
    if not enabled():
        # observe mode: the target hook is otherwise installed by install_deepseek_v4
        import sys

        mod = sys.modules.get("sglang.srt.models.deepseek_v4")
        if mod is not None:
            _wrap_target_load(mod)


if __name__ == "__main__":  # self-test: python3 fast_load.py <shard> <ep_rank> <ep_size> [<moe_tp_rank> <moe_tp_size>]
    import sys, time

    path, ep = sys.argv[1], (int(sys.argv[2]), int(sys.argv[3]))
    tp = tp_slicing(ep, (int(sys.argv[4]), int(sys.argv[5])), "on") if len(sys.argv) > 5 else None
    plan = read_plan(path, "target", ep, _n_routed_experts(path) or 384, tp)
    base, header = _parse_header(path)
    total = sum(spec_bytes(header[k], sp)[0] for k, sp in plan.items())
    kept = sum(spec_bytes(header[k], sp)[1] for k, sp in plan.items())
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    t = time.time()
    futs = [_executor().submit(_read_tensor, fd, base, header[k]) if sp is None
            else _executor().submit(_read_slice, fd, base, header[k], sp) for k, sp in plan.items()]
    tensors = [f.result() for f in futs]
    dt = time.time() - t
    print(f"{os.path.basename(path)}: read {total/1e9:.2f} GB ({kept/1e9:.2f} GB kept) in {len(plan)} tensors "
          f"in {dt:.1f}s = {total/1e9/dt:.2f} GB/s")
