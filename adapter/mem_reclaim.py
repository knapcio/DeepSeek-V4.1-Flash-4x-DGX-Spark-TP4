"""Host-memory reclaim for TP4 serving on GB10 unified memory: the tokenizer-heap switch of ds41
perf/mem-reclaim-20261029 @ 13e68a5, taken alone (the branch's other switches are not part of this release; the code
below is 13e68a5's byte for byte, tests/test_tokenizer_heap.py checks it against a pinned digest). Off by default.

``DSV41_TOKENIZER_HEAP=trim|mmap|both``
    launch_server only (the tokenizer manager, head). Tokenizing 200K-950K-token prompts leaves glibc's main heap at
    its high-water mark: 2026-10-30 C1 on production, launch_server RssAnon 1.15 -> 3.91 GiB in a 30-min issue-16
    soak, all in [heap], never returned (idle, /flush_cache). ``trim``: a daemon thread calls ``malloc_trim(0)``
    when at least DSV41_TOKENIZER_TRIM_MIN_MIB (256) are free inside the heap (mallinfo2 fordblks), checked every
    DSV41_TOKENIZER_TRIM_S (2) s. ``mmap``: ``mallopt`` fixes the mmap threshold at DSV41_TOKENIZER_MMAP_KIB (1024)
    so multi-MB tokenization buffers are mapped and unmapped instead of raising the heap (process-local: the spawned
    scheduler / detokenizer keep glibc's defaults). Allocation only; no token, request or scheduling change.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

_OFF = ('0', 'off', 'false', '', 'no')


# ---------------------------------------------------------------------------------------------------------------------
# DSV41_TOKENIZER_HEAP
# ---------------------------------------------------------------------------------------------------------------------
class _Mallinfo2(ctypes.Structure):
    _fields_ = [(k, ctypes.c_size_t) for k in
                ('arena', 'ordblks', 'smblks', 'hblks', 'hblkhd', 'usmblks', 'fsmblks', 'uordblks', 'fordblks', 'keepcost')]


M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3
HEAP_MODES = ('trim', 'mmap', 'both')
_HEAP = {}


def server_role(argv, orig_argv) -> bool:
    """True only for the launch_server interpreter (the HTTP server + tokenizer manager). Its spawned scheduler and
    detokenizer children run 'python -c from multiprocessing.spawn import spawn_main ...' and are excluded."""
    if argv and os.path.basename(str(argv[0])) == 'launch_server.py':
        return True
    o = [str(a) for a in (orig_argv or ())]
    return any(a == '-m' and i + 1 < len(o) and o[i + 1] == 'sglang.launch_server' for i, a in enumerate(o))


def heap_mode(env=None, argv=None, orig_argv=None) -> str:
    """'' (off: the default, any other process, or an unknown value), or one of HEAP_MODES."""
    env = os.environ if env is None else env
    mode = env.get('DSV41_TOKENIZER_HEAP', '').strip().lower()
    if mode in _OFF or mode not in HEAP_MODES:
        return ''
    argv = sys.argv if argv is None else argv
    orig_argv = getattr(sys, 'orig_argv', None) if orig_argv is None else orig_argv
    return mode if server_role(argv, orig_argv) else ''


def _libc():
    import ctypes.util
    if 'libc' not in _HEAP:
        lib = ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6', use_errno=True)
        lib.malloc_trim.argtypes, lib.malloc_trim.restype = [ctypes.c_size_t], ctypes.c_int
        lib.mallopt.argtypes, lib.mallopt.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
        lib.mallinfo2.argtypes, lib.mallinfo2.restype = [], _Mallinfo2
        _HEAP['libc'] = lib
    return _HEAP['libc']


def heap_free_bytes() -> int:
    """Bytes glibc holds free inside its arenas (mallinfo2 fordblks): what malloc_trim(0) can give back at most."""
    return int(_libc().mallinfo2().fordblks)


def rss_anon_kb(path='/proc/self/status') -> int:
    try:
        for line in open(path):
            if line.startswith('RssAnon:'):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def trim_now() -> tuple:
    """malloc_trim(0) (every arena, free pages inside the heap too); -> (seconds, RssAnon kB before, after). ctypes
    releases the GIL during the call."""
    a = rss_anon_kb()
    t0 = time.perf_counter()
    _libc().malloc_trim(0)
    return time.perf_counter() - t0, a, rss_anon_kb()


def set_mmap_threshold(nbytes: int, trim_bytes: int) -> bool:
    """Fixed mmap threshold (disables glibc's dynamic raise to 32 MiB): blocks >= nbytes are mmap'd and unmapped on
    free; trim threshold for the heap top."""
    lib = _libc()
    return bool(lib.mallopt(M_MMAP_THRESHOLD, int(nbytes))) and bool(lib.mallopt(M_TRIM_THRESHOLD, int(trim_bytes)))


def trim_due(free_bytes: int, min_free_bytes: int) -> bool:
    return free_bytes >= min_free_bytes


def _trim_loop(period_s, min_free_bytes, stop):
    while not stop.wait(period_s):
        try:
            free = heap_free_bytes()
            if trim_due(free, min_free_bytes):
                dt, a, b = trim_now()
                _HEAP['trims'] = _HEAP.get('trims', 0) + 1
                logger.info('DSV41_TOKENIZER_HEAP: malloc_trim(0) %.1f ms, %d MiB free in the heap, RssAnon %d -> %d MiB',
                            dt * 1e3, free >> 20, a >> 10, b >> 10)
        except Exception as exc:  # noqa: BLE001
            logger.warning('DSV41_TOKENIZER_HEAP: trim loop stopped: %r', exc)
            return


def install_tokenizer_heap(env=None, argv=None, orig_argv=None) -> str:
    """Called from sitecustomize in every interpreter; acts only in launch_server and only when switched on.
    mmap: blocks >= DSV41_TOKENIZER_MMAP_KIB (1024) are mmap'd (returned to the OS on free), heap top trimmed above
    DSV41_TOKENIZER_TRIM_TOP_MIB (64). trim: a daemon thread calls malloc_trim(0) every DSV41_TOKENIZER_TRIM_S (2) when
    at least DSV41_TOKENIZER_TRIM_MIN_MIB (256) are free inside the heap. both: the two together."""
    env = os.environ if env is None else env
    mode = heap_mode(env, argv, orig_argv)
    if not mode or _HEAP.get('installed'):
        return mode
    _HEAP['installed'] = mode
    if mode in ('mmap', 'both'):
        ok = set_mmap_threshold(int(env.get('DSV41_TOKENIZER_MMAP_KIB', '1024')) << 10,
                                int(env.get('DSV41_TOKENIZER_TRIM_TOP_MIB', '64')) << 20)
        print(f'DSV41_TOKENIZER_HEAP: mmap threshold {env.get("DSV41_TOKENIZER_MMAP_KIB", "1024")} KiB '
              f'({"set" if ok else "mallopt FAILED"})', flush=True)
    if mode in ('trim', 'both'):
        stop = threading.Event()
        _HEAP['stop'] = stop
        t = threading.Thread(target=_trim_loop, name='dsv41-heap-trim', daemon=True,
                             args=(float(env.get('DSV41_TOKENIZER_TRIM_S', '2')),
                                   int(env.get('DSV41_TOKENIZER_TRIM_MIN_MIB', '256')) << 20, stop))
        t.start()
        print(f'DSV41_TOKENIZER_HEAP: malloc_trim(0) when >= {env.get("DSV41_TOKENIZER_TRIM_MIN_MIB", "256")} MiB are '
              f'free in the heap, checked every {env.get("DSV41_TOKENIZER_TRIM_S", "2")} s', flush=True)
    return mode
