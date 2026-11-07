"""Bounded exact file-backed replacement for EngramEmbedding's owned-row gather.

The original hash, gating, projections, TP all-reduce and model remain unchanged.
Native callback executes IO inside CUDA graphs without calling the CUDA API.
"""
import ctypes as C
import glob
import json
import logging
import os
from pathlib import Path
import struct

import torch

P, U = C.c_void_p, C.c_uint64
_lib = C.CDLL(str(Path(__file__).with_name('librow_store.so')))
_lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
_lib.row_store_open.restype = P
_lib.row_store_range.argtypes = [P, U, U]
_lib.row_store_stats.argtypes = [P, C.POINTER(U)]
_lib.row_store_attach_packed.argtypes = [P, C.c_char_p, U]
_lib.row_store_attach_packed.restype = C.c_int
_lib.row_store_next_cache_on_drm.argtypes = [C.c_char_p, U]
_DRM_USED = [False]

class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]

_STORES = []

def _stats(store):
    out = (U * 9)()
    _lib.row_store_stats(store, out)
    hits, misses, reads, cache_bytes, slots, scale_bytes, ways, threads, packed = out
    return dict(hits=hits, misses=misses, reads=reads, cache_bytes=cache_bytes,
                slots=slots, scale_bytes=scale_bytes, ways=ways, threads=threads,
                packed=bool(packed))

def _report():
    """Miss rate and read volume decide decode latency here; log them."""
    import threading
    import time
    period = float(os.getenv('DSV41_STATS_SECONDS', '60'))
    if period <= 0:
        return
    log = logging.getLogger(__name__)
    def loop():
        previous = {}
        while True:
            time.sleep(period)
            for store, layer_id in _STORES:
                now = _stats(store)
                was = previous.get(layer_id)
                previous[layer_id] = now
                if was is None:
                    continue
                hits = now['hits'] - was['hits']
                misses = now['misses'] - was['misses']
                total = hits + misses
                if not total:
                    continue
                log.info('Engram layer=%s lookups=%s hit_rate=%.1f%% reads=%s '
                         'cache=%.1fGiB(%s-way) scales=%.1fGiB threads=%s packed=%s',
                         layer_id, total, 100.0*hits/total, now['reads']-was['reads'],
                         now['cache_bytes']/2**30, now['ways'],
                         now['scale_bytes']/2**30, now['threads'], now['packed'])
    threading.Thread(target=loop, daemon=True, name='engram-stats').start()

def _register_store(store, layer_id):
    _STORES.append((store, layer_id))
    if len(_STORES) == 1:
        _report()

def _load_cudart():
    paths = []
    paths += glob.glob('/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*')
    paths += glob.glob('/usr/local/cuda*/targets/x86_64-linux/lib/libcudart.so*')
    paths += glob.glob('/usr/local/cuda/lib64/libcudart.so*')
    paths += glob.glob('/usr/local/lib/python*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*')
    paths += glob.glob('/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*')
    seen, errors = set(), []
    for path in paths:
        if path in seen or path.endswith('.a') or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            lib = C.CDLL(path)
            lib.cudaLaunchHostFunc.argtypes = [P, P, P]
            lib.cudaLaunchHostFunc.restype = C.c_int
            return lib
        except OSError as exc:
            errors.append(f'{path}: {exc}')
    raise RuntimeError('libcudart.so not found (need CUDA runtime for Engram host callbacks). Tried: '
                       + '; '.join(errors or paths or ['<none>']))

_cuda = _load_cudart()

def install(module):
    cls = module.EngramEmbedding
    def init(self, num_embeddings, dim, layer_id):
        torch.nn.Module.__init__(self)
        assert dim == 256
        self.dim, self.tp_size = dim, module.get_parallel().tp_size
        rank = module.get_parallel().tp_rank
        self.row_start = num_embeddings * rank // self.tp_size
        end = num_embeddings * (rank + 1) // self.tp_size
        self.rows, self.host_table = end - self.row_start, None
        root = Path(os.environ['DSV41_SOURCE'])
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        prefix = f'layers.{layer_id}.engram.embed.'
        filename = root/index[prefix+'weight']
        assert index[prefix+'weight'] == index[prefix+'scale']
        with filename.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(length))
        w, s = header[prefix+'weight'], header[prefix+'scale']
        assert w['shape'] == [num_embeddings, 256] and w['dtype'] == 'F8_E4M3'
        assert s['shape'] == [num_embeddings, 8] and s['dtype'] == 'F8_E8M0'
        # DSV41_CACHE_GIB is the per-host budget, split across the Engram layers
        # and the ranks that share that host's RAM. Upstream ran every TP rank in
        # one box, so it divided by tp_size; on the Spark triangle each node runs
        # a single rank and dividing by tp_size silently gave each node a third of
        # the cache it was asked for.
        ranks_per_host = max(1, self.tp_size // max(1, int(os.getenv('NNODES', '1'))))
        budget = int(float(os.getenv('DSV41_CACHE_GIB', '64')) * 2**30) // (2*ranks_per_host)
        # DSV41_ENGRAM_DRM_NODE=/dev/dri/card0: one Engram layer (DSV41_ENGRAM_DRM_LAYER, default the
        # first opened) keeps its row cache in the GB10 display reservation, outside MemAvailable,
        # sized to DSV41_ENGRAM_DRM_MIB (<= 2032, multiple of 16). Its keys stay in ordinary memory.
        drm_node = os.getenv('DSV41_ENGRAM_DRM_NODE', '')
        drm_layer = os.getenv('DSV41_ENGRAM_DRM_LAYER', '')
        # Override site: a set DSV41_ENGRAM_DRM_NODE whose device path does not exist used to leave
        # every row cache in ordinary memory silently. The user-set gate that cannot take effect raises.
        if drm_node and not os.path.exists(drm_node):
            raise RuntimeError(f'DSV41_ENGRAM_DRM_NODE={drm_node!r} is set but the device path does not exist')
        if (drm_node and os.path.exists(drm_node) and not _DRM_USED[0]
                and (not drm_layer or int(drm_layer) == layer_id)):
            pool = int(os.getenv('DSV41_ENGRAM_DRM_MIB', '1792')) << 20
            # Override site: the DRM layer's row cache is capped at the DSV41_ENGRAM_DRM_MIB pool, so
            # its DSV41_CACHE_GIB share shrinks when the pool is smaller. The pool size is the
            # documented size of that layer's cache, so the cap is kept and logged as WARNING.
            if pool * 272 // 264 < budget:
                logging.getLogger(__name__).warning(
                    'WARNING DSV41_CACHE_GIB share of Engram layer %s is capped from %d to %d bytes by '
                    'DSV41_ENGRAM_DRM_MIB=%d', layer_id, budget, pool * 272 // 264, pool >> 20)
            budget = min(budget, pool * 272 // 264)          # slots * 264 B of rows fit the pool
            _lib.row_store_next_cache_on_drm(drm_node.encode(), pool)
            _DRM_USED[0] = True
        self._store = _lib.row_store_open(str(filename).encode(), num_embeddings,
            8+length+w['data_offsets'][0], 8+length+s['data_offsets'][0], budget)
        if not self._store:
            raise RuntimeError(f'Could not open Engram backing shard: {filename}')
        _lib.row_store_range(self._store, self.row_start, end)
        # A repacked local shard, when present, makes every miss a single local
        # read; without it we fall back to the checkpoint (NFS on a worker).
        packed_dir = os.environ.get('DSV41_PACKED_DIR', '')
        if packed_dir:
            shard = Path(packed_dir)/f'engram-l{layer_id}-r{rank}of{self.tp_size}.bin'
            # Override site: a missing packed shard under DSV41_PACKED_DIR makes every miss read the
            # checkpoint instead of the local shard. start.sh always sets DSV41_PACKED_DIR, so the
            # fall-through is kept and logged as WARNING once per Engram layer.
            if shard.exists():
                _lib.row_store_attach_packed(self._store, str(shard).encode(), layer_id)
            else:
                logging.getLogger(__name__).warning(
                    'WARNING DSV41_PACKED_DIR=%s has no %s; Engram layer %s misses read the checkpoint',
                    packed_dir, shard.name, layer_id)
        _register_store(self._store, layer_id)
        self._staging = {}
        self._works = {}
        # The loader sees names but never allocates/copies the complete tables.
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e8m0fnu), requires_grad=False)
        def validate_weight(param, source):
            expected = w if param is self.weight else s
            if list(source.shape) != expected['shape']:
                raise ValueError('Engram checkpoint shape mismatch')
        self.weight.weight_loader = validate_weight
        self.scale.weight_loader = validate_weight
        stats = _stats(self._store)
        logging.getLogger(__name__).info(
            'Exact %s Engram layer=%s rank=%s rows=[%s,%s) cache=%.1fGiB '
            '(%s slots, %s-way, %.1f%% of owned rows) scales=%.1fGiB io_threads=%s '
            'packed=%s',
            os.getenv('OFFLOAD_MODE', 'nvme'), layer_id, rank, self.row_start, end,
            stats['cache_bytes']/2**30, stats['slots'], stats['ways'],
            100.0*stats['slots']/max(1, self.rows), stats['scale_bytes']/2**30,
            stats['threads'], stats['packed'])

    def owned(self, indices):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather
        count = indices.numel()
        if not count:
            return self._empty(indices)
        capacity = 1 << (count - 1).bit_length()
        key = (indices.device.index, capacity)
        if key not in self._staging:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'Engram staging {key} must be warmed before graph capture')
            ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            w = torch.empty((capacity, 256), dtype=torch.uint8, pin_memory=True)
            s = torch.empty((capacity, 8), dtype=torch.uint8, pin_memory=True)
            dw, ds = w.to(indices.device), s.to(indices.device)
            sequential = torch.arange(capacity, dtype=torch.int64, device=indices.device)
            self._staging[key] = (ids, w, s, dw, ds, sequential)
        ids, w, s, dw, ds, sequential = self._staging[key]
        work_key = (indices.device.index, count)
        if work_key not in self._works:
            self._works[work_key] = Work(self._store, ids.data_ptr(), w.data_ptr(), s.data_ptr(), count)
        work = self._works[work_key]
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        error = _cuda.cudaLaunchHostFunc(torch.cuda.current_stream().cuda_stream,
            C.cast(_lib.row_store_lookup, P), C.addressof(work))
        if error:
            raise RuntimeError(f'CUDA Engram host callback failed: {error}')
        dw[:count].copy_(w[:count], non_blocking=True)
        ds[:count].copy_(s[:count], non_blocking=True)
        out = self._empty(indices)
        engram_gather(dw.data_ptr(), ds.data_ptr(), sequential[:count], out.view(-1, 256), 256, 32)
        return out
    cls.__init__ = init
    cls._owned_rows = owned
