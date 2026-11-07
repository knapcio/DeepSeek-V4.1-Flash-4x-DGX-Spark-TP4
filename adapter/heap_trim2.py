"""DSV41_TOKENIZER_HEAP=trim2: give the tokenizer manager's glibc heap back to the OS, rate-limited, in tokenizer gaps.

launch_server only (the HTTP server + tokenizer manager, head); off by default; any other value of
DSV41_TOKENIZER_HEAP (``trim`` / ``mmap`` / ``both`` / ``0`` / unset) does nothing in this release.

What grows. In a long-prompt soak (the issue-16 warm loop: 190K-900K-token prompts back to back, 1 and 4 streams in
alternate minutes, plus a chat companion) launch_server's RssAnon grows 1.9 -> 5-7 GiB in 15 minutes and 1.2 -> 8.8 GiB
in 30 (fleet soaks 2026-10-30 / 2026-11-05) while the schedulers stay flat. It is free memory glibc keeps in the main
arena ([heap]): one malloc_trim(0) returns it (fleet: 8.77 -> 0.99 GiB). An offline reproduction of the request path
(diagnostics/ds-trim-20261106/offline: JSON body -> chat render -> parallel encode -> pickle, 1 / 4 streams) grows the
same way, only in the 4-stream minutes, and not at all with a fixed 1 MiB mmap threshold: the multi-MB per-request
buffers that glibc's dynamic mmap threshold (raised up to 32 MiB by the first large free) serves from the main heap,
interleaved across concurrent requests, keep extending its top. The growth has no bound in sight on the fleet (still
linear at the end of every soak). A fixed mmap threshold avoids it but maps, faults and unmaps every large buffer of
every request (+10-16 % encode time offline); trimming costs only when it runs.

What a trim costs (the reason for every limit below). malloc_trim(0) locks each arena in turn and madvise()s every
free page; whatever the heap reuses afterwards is faulted in again. On the fleet the 2026-10-30 ``trim`` (every 2 s:
mallinfo2 keeps counting released chunks as free, so it re-trimmed on every check) released 227 GiB in a 37-minute
soak at ~0.33-0.4 s/GiB of lock hold and cost the long-prompt stream 17 % of its requests (~1.3-1.5 s of lost work per
GiB released). Offline, the volume released falls steeply with the interval (2 / 10 / 20 / 60 s: 53 / 21 / 14 / 7 GiB
in 4 minutes) while the peak between trims rises (1.5 / 2.4 / 3.0 / 6.4 GiB). trim2 therefore trims seldom:

  1. Growth trigger: only when the heap holds >= DSV41_TOKENIZER_TRIM2_MIN_MIB (256) free AND, since the last trim,
     RssAnon grew by >= that much or the free bytes grew by >= that much (no re-trim of released pages). mallinfo2
     (a walk of every bin under the arena locks: up to ~95 ms on a fragmented 6 GiB heap offline) is read only when
     RssAnon grew by MIN_MIB or every DSV41_TOKENIZER_TRIM2_INFO_EVERY_S (30) s, and once per trim.
  2. Rate limit: at most one trim per DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S (120) s, whatever the gates say: a steady
     long-prompt stream pays one release + refault per interval.
  3. Idle gate: normally a trim waits until no tokenization is in flight and none ended in the last
     DSV41_TOKENIZER_TRIM2_IDLE_S (1.5) s. Tokenization = every OpenAI-style request conversion
     (``_convert_to_internal_request`` of every OpenAIServingBase subclass: chat-template render + prompt encode, incl.
     the parallel encoder), the chat renderer ``_process_messages`` and TokenizerManager._tokenize_texts (native
     /generate text). A counter and a monotonic last-activity time under one lock.
  4. Busy fallback: back-to-back long prompts never leave an idle gap, and the heap would grow until the host runs out
     (the 2026-11-05 ladder controls). When RssAnon has grown by MIN_MIB for DSV41_TOKENIZER_TRIM2_MAX_BUSY_S (30) s
     without an idle gap, the trim is due: it runs at the next gap between two prompts, on the thread that just
     finished a tokenization (the in-flight count's 1 -> 0 edge; no prompt is being encoded then), or, if no gap comes
     within another MAX_BUSY_S, from the trimmer thread. Its receipt says BUSY. 0 = idle gaps only.
  5. Hold budget (idle path): a trim predicted to hold the locks longer than DSV41_TOKENIZER_TRIM2_BUDGET_MS (50)
     also waits for an idle request stream (no detokenized output received for IDLE_S); after
     DSV41_TOKENIZER_TRIM2_MAX_DEFER_S (60) it runs anyway (FORCED). The per-GiB cost starts at
     DSV41_TOKENIZER_TRIM2_MS_PER_GIB (40) and is re-learned from every receipt.
  6. Abort: the gate is re-checked immediately before the call; activity that started in between skips the trim.

Receipts: ``DSV41_TOKENIZER_HEAP=trim2: armed ...`` at start, ``... hooks <cumulative list>`` when the SGLang hooks land
(after each hooked module and after each OpenAIServingBase subclass wrapped later: the last line names them all), one line per
trim (at most one per DSV41_TOKENIZER_TRIM2_LOG_MIN_S (10) s, so with the default interval every trim; over-budget,
forced and busy trims always log; suppressed lines are counted) and a summary every DSV41_TOKENIZER_TRIM2_SUMMARY_S
(600) s. Allocation only: no token, request, sampling or scheduling decision reads any of it, so outputs are
bit-identical with it on or off; the switch and its knobs are volatile for DSV41_AUTOTUNE_KEEP.
"""
from __future__ import annotations

import ctypes
import functools
import importlib.abc
import os
import sys
import threading
import time
import weakref

MODE = 'trim2'
_STATE = {}
_TARGETS = ('sglang.srt.entrypoints.openai.serving_base', 'sglang.srt.managers.tokenizer_manager')
_MARK = '_dsv41_trim2'
# OpenAIServingBase subclass methods that render and encode prompts: every endpoint's conversion, and the chat
# renderer itself, which Anthropic's /v1/messages/count_tokens calls directly (not through the conversion)
_SERVING_METHODS = ('_convert_to_internal_request', '_process_messages')


def _log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# process role and glibc
# ---------------------------------------------------------------------------------------------------------------------
def server_role(argv, orig_argv) -> bool:
    """True only for the launch_server interpreter (the HTTP server + tokenizer manager). Its spawned scheduler and
    detokenizer children run 'python -c from multiprocessing.spawn import spawn_main ...' and are excluded."""
    if argv and os.path.basename(str(argv[0])) == 'launch_server.py':
        return True
    o = [str(a) for a in (orig_argv or ())]
    return any(a == '-m' and i + 1 < len(o) and o[i + 1] == 'sglang.launch_server' for i, a in enumerate(o))


def enabled(env=None, argv=None, orig_argv=None) -> bool:
    """True only for DSV41_TOKENIZER_HEAP=trim2 in the launch_server interpreter."""
    env = os.environ if env is None else env
    if env.get('DSV41_TOKENIZER_HEAP', '').strip().lower() != MODE:
        return False
    argv = sys.argv if argv is None else argv
    orig_argv = getattr(sys, 'orig_argv', None) if orig_argv is None else orig_argv
    return server_role(argv, orig_argv)


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(k, ctypes.c_size_t) for k in
                ('arena', 'ordblks', 'smblks', 'hblks', 'hblkhd', 'usmblks', 'fsmblks', 'uordblks', 'fordblks', 'keepcost')]


def _libc():
    if 'libc' not in _STATE:
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6', use_errno=True)
        lib.malloc_trim.argtypes, lib.malloc_trim.restype = [ctypes.c_size_t], ctypes.c_int
        lib.mallinfo2.argtypes, lib.mallinfo2.restype = [], _Mallinfo2
        _STATE['libc'] = lib
    return _STATE['libc']


def heap_free_bytes() -> int:
    """Bytes glibc holds free inside its arenas (mallinfo2 fordblks; released chunks stay counted)."""
    return int(_libc().mallinfo2().fordblks)


def rss_anon_kb(path='/proc/self/status') -> int:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith('RssAnon:'):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def malloc_trim():
    """malloc_trim(0): every arena, free pages inside the heap too. ctypes releases the GIL during the call."""
    return _libc().malloc_trim(0)


# ---------------------------------------------------------------------------------------------------------------------
# configuration, activity, the trimmer
# ---------------------------------------------------------------------------------------------------------------------
class Config:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        f = lambda k, d: float(env.get('DSV41_TOKENIZER_TRIM2_' + k, d))  # noqa: E731
        self.idle_s = f('IDLE_S', '1.5')
        self.poll_s = f('POLL_S', '1.0')
        self.min_bytes = int(f('MIN_MIB', '256')) << 20
        self.min_interval_s = f('MIN_INTERVAL_S', '120')
        self.max_busy_s = f('MAX_BUSY_S', '30')
        self.info_every_s = f('INFO_EVERY_S', '30')
        self.budget_ms = f('BUDGET_MS', '50')
        self.max_defer_s = f('MAX_DEFER_S', '60')
        self.log_min_s = f('LOG_MIN_S', '10')
        self.summary_s = f('SUMMARY_S', '600')
        self.ms_per_gib = f('MS_PER_GIB', '40')       # initial cost model; re-learned from each trim
        bad = [k for k, v in (('IDLE_S', self.idle_s), ('POLL_S', self.poll_s), ('MIN_INTERVAL_S', self.min_interval_s),
                              ('MAX_BUSY_S', self.max_busy_s), ('INFO_EVERY_S', self.info_every_s), ('BUDGET_MS', self.budget_ms),
                              ('MAX_DEFER_S', self.max_defer_s), ('MS_PER_GIB', self.ms_per_gib))
               if not (v == v and 0 <= v < float('inf'))]
        if bad or self.poll_s <= 0 or self.min_bytes <= 0:
            raise ValueError(f'DSV41_TOKENIZER_TRIM2_ settings out of range: {bad or "POLL_S / MIN_MIB"}')

    def describe(self):
        return (f'idle {self.idle_s:g} s, poll {self.poll_s:g} s, min {self.min_bytes >> 20} MiB new, at most one trim '
                f'per {self.min_interval_s:g} s, busy fallback after {self.max_busy_s:g} s, budget {self.budget_ms:g} ms '
                f'(deep-idle wait, forced after {self.max_defer_s:g} s), summary every {self.summary_s:g} s')


class Activity:
    """Tokenization in flight (counter) and the monotonic time the last one entered or left, under one lock.
    ``outputs_quiet`` reads TokenizerManager.last_receive_tstamp (wall time, set by its handle_loop per batch)."""

    def __init__(self, clock=time.monotonic, wall=time.time):
        self._lock = threading.Lock()
        self._clock, self._wall = clock, wall
        self.inflight = 0
        self.entries = 0
        self.last = clock()
        self.hooks = []
        self._tm = None
        self.on_gap = None                           # called (outside the lock) when the in-flight count drops to 0

    def enter(self):
        with self._lock:
            self.inflight += 1
            self.entries += 1
            self.last = self._clock()

    def exit(self):
        with self._lock:
            self.inflight -= 1
            self.last = self._clock()
            gap = self.inflight == 0
        cb = self.on_gap
        if gap and cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001  (never into the request path)
                pass

    def quiet(self, idle_s, now=None) -> bool:
        with self._lock:
            return self.inflight == 0 and (self._clock() if now is None else now) - self.last >= idle_s

    def between(self) -> bool:
        """No tokenization in flight at this instant (the busy fallback's gate: a gap between two prompts)."""
        with self._lock:
            return self.inflight == 0

    def set_tokenizer_manager(self, tm):
        self._tm = weakref.ref(tm)

    def outputs_quiet(self, idle_s) -> bool:
        tm = self._tm() if self._tm is not None else None
        ts = getattr(tm, 'last_receive_tstamp', None) if tm is not None else None
        return ts is None or self._wall() - float(ts) >= idle_s


def wrap(fn, act):
    """``fn`` (sync or async) counted as tokenization activity for its whole duration (exceptions, cancellation)."""
    if getattr(fn, _MARK, False):
        return fn
    import inspect
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapped(*a, **k):
            act.enter()
            try:
                return await fn(*a, **k)
            finally:
                act.exit()
    else:
        @functools.wraps(fn)
        def wrapped(*a, **k):
            act.enter()
            try:
                return fn(*a, **k)
            finally:
                act.exit()
    setattr(wrapped, _MARK, True)
    return wrapped


class Trimmer:
    """One decision per ``step()``; everything injectable for the CPU tests."""

    def __init__(self, cfg, act, *, clock=time.monotonic, perf=time.perf_counter, rss_kb=None, free_bytes=None,
                 trim=None, log=_log):
        self.cfg, self.act, self.clock, self.perf, self.log = cfg, act, clock, perf, log
        self.rss_kb = rss_kb or rss_anon_kb
        self.free_bytes = free_bytes or heap_free_bytes
        self.trim = trim or malloc_trim
        self.ms_per_gib = cfg.ms_per_gib
        self.r0, self.f0 = self.rss_kb(), 0          # RssAnon kB / mallinfo2 free bytes after the last trim
        self.last_trim = None                        # monotonic time of the last trim (the rate limit)
        self.last_info = None                        # monotonic time of the last mallinfo2 read on the idle path
        self.deferred_since = None
        self.busy_since = None
        self.due_since = None                        # a busy trim is due: the next tokenization gap runs it
        self._lock = threading.Lock()                # one trim at a time (poll thread or a gap)
        act.on_gap = self.gap
        self.n = dict(trims=0, forced=0, busy=0, over_budget=0, aborted=0, deferred=0, busy_waits=0, suppressed=0)
        self.returned_kb = 0
        self.hold_ms = []                            # last 4096 calls
        self.hold_max = 0.0
        self.info_ms_max = 0.0
        self.last_line = None
        self.last_summary = clock()

    def _grown(self, r, f):
        return max((r - self.r0) << 10, f - self.f0)

    def step(self) -> str:
        cfg, act = self.cfg, self.act
        if not act.hooks:
            return 'nohook'
        now = self.clock()
        in_interval = self.last_trim is not None and now - self.last_trim < cfg.min_interval_s
        if not act.quiet(cfg.idle_s):
            # busy: only the RSS (one /proc read) is looked at; mallinfo2 walks the bins under the arena locks. The
            # busy wait counts from the first poll that saw the growth, inside the rate-limit interval too, so a
            # steady stream is trimmed about every max(MIN_INTERVAL_S, growth time + MAX_BUSY_S) s
            if cfg.max_busy_s <= 0 or (self.rss_kb() - self.r0) << 10 < cfg.min_bytes:
                self.busy_since = None
                return 'interval' if in_interval else 'busy'
            if self.busy_since is None:
                self.busy_since = now
                self.n['busy_waits'] += 1
            if in_interval:
                return 'interval'
            if now - self.busy_since < cfg.max_busy_s:
                return 'busy'
            # due: the next gap between two prompts runs it, on the thread that just finished a tokenization
            # (Activity.exit -> gap); a gap that never comes within another MAX_BUSY_S: this thread runs it
            if self.due_since is None:
                self.due_since = now
            if now - self.due_since < cfg.max_busy_s:
                return 'due'
            with self._lock:
                return self._trim(est_ms=None, how='busytrim', wait_gap=False)
        self.busy_since = None
        self.due_since = None
        if in_interval:
            return 'interval'
        r = self.rss_kb()
        # mallinfo2 walks every bin under each arena lock: ~0.2 ms on a small heap, ~95 ms on a fragmented 6 GiB one
        # (offline). Read it only when RSS grew by MIN_MIB, or every INFO_EVERY_S to catch free bytes growing without
        # RSS (a live request that ended); otherwise this poll costs one /proc read.
        if (r - self.r0) << 10 < cfg.min_bytes and self.last_info is not None and now - self.last_info < cfg.info_every_s:
            self.deferred_since = None
            return 'small'
        self.last_info = now
        t0 = self.perf()
        f = self.free_bytes()
        self.info_ms_max = max(self.info_ms_max, (self.perf() - t0) * 1e3)
        grown = self._grown(r, f)
        if f < cfg.min_bytes or grown < cfg.min_bytes:
            self.deferred_since = None
            return 'small'
        est_ms = self.ms_per_gib * min(f, grown) / (1 << 30)
        how = 'trim'
        if est_ms > cfg.budget_ms and not act.outputs_quiet(cfg.idle_s):
            if self.deferred_since is None:
                self.deferred_since = now
                self.n['deferred'] += 1
            if cfg.max_defer_s <= 0 or now - self.deferred_since < cfg.max_defer_s:
                return 'defer'
            how = 'forced'
        if not act.quiet(cfg.idle_s):                # activity started since the gate: skip this trim
            self.n['aborted'] += 1
            return 'abort'
        with self._lock:
            return self._trim(est_ms=est_ms, how=how, r=r, f=f)

    def gap(self):
        """Activity.exit at the in-flight 1 -> 0 edge: run a due busy trim now, between two prompts."""
        if self.due_since is None or not self._lock.acquire(blocking=False):
            return
        try:
            if self.due_since is not None:
                self._trim(est_ms=None, how='busytrim', wait_gap=True)
        finally:
            self._lock.release()

    def _trim(self, est_ms, how, r=None, f=None, wait_gap=True) -> str:
        cfg = self.cfg
        if r is None:
            r = self.rss_kb()
            f = self.free_bytes()
            est_ms = self.ms_per_gib * min(f, self._grown(r, f)) / (1 << 30)
        if how == 'busytrim':
            if f < cfg.min_bytes:                        # RSS grew with live data (prompts in flight), not free pages
                self.busy_since = self.due_since = None
                return 'small'
            if wait_gap and not self.act.between():      # a prompt started during the reads: the next gap
                self.n['aborted'] += 1
                return 'abort'
        t0 = self.perf()
        self.trim()
        ms = (self.perf() - t0) * 1e3
        r1 = self.rss_kb()
        now = self.clock()
        # a trim leaves mallinfo2's free count as it was (released chunks stay in their bins): f, no second walk
        self.r0, self.f0, self.deferred_since, self.busy_since, self.last_trim = r1, f, None, None, now
        self.due_since = None
        back_kb = max(0, r - r1)
        self.returned_kb += back_kb
        self.n['trims'] += 1
        self.n['forced'] += how == 'forced'
        self.n['busy'] += how == 'busytrim'
        over = ms > cfg.budget_ms
        self.n['over_budget'] += over
        self.hold_ms.append(ms)
        del self.hold_ms[:-4096]
        self.hold_max = max(self.hold_max, ms)
        if back_kb >= (64 << 10):                    # re-learn ms per GiB from trims that returned >= 64 MiB
            self.ms_per_gib = 0.5 * self.ms_per_gib + 0.5 * ms / (back_kb / (1 << 20))
        if over or how != 'trim' or self.last_line is None or now - self.last_line >= cfg.log_min_s:
            self.log(f'DSV41_TOKENIZER_HEAP=trim2: trim #{self.n["trims"]} {ms:.1f} ms, returned {back_kb >> 10} MiB '
                     f'(RssAnon {r >> 10} -> {r1 >> 10} MiB, {f >> 20} MiB free in the heap, predicted {est_ms:.0f} ms)'
                     + (' OVER BUDGET' if over else '') + (' FORCED after deep-idle wait' if how == 'forced' else '')
                     + (' BUSY after no idle gap' if how == 'busytrim' else '')
                     + (f', {self.n["suppressed"]} lines suppressed' if self.n['suppressed'] else ''))
            self.last_line, self.n['suppressed'] = now, 0
        else:
            self.n['suppressed'] += 1
        return how

    def summary(self) -> str:
        h = sorted(self.hold_ms)
        p50 = h[len(h) // 2] if h else 0.0
        n = self.n
        return (f'DSV41_TOKENIZER_HEAP=trim2: summary trims {n["trims"]} (forced {n["forced"]}, busy {n["busy"]}, over '
                f'budget {n["over_budget"]}), aborted {n["aborted"]}, deferred {n["deferred"]}, busy waits '
                f'{n["busy_waits"]}, returned {self.returned_kb >> 10} MiB, call ms p50 {p50:.1f} max {self.hold_max:.1f}, '
                f'tokenizations {self.act.entries}, RssAnon {self.rss_kb() >> 10} MiB, model {self.ms_per_gib:.1f} '
                f'ms/GiB, mallinfo2 max {self.info_ms_max:.1f} ms')

    def maybe_summary(self):
        now = self.clock()
        if now - self.last_summary >= self.cfg.summary_s:
            self.last_summary = now
            self.log(self.summary())

    def run(self, stop):
        while not stop.wait(self.cfg.poll_s):
            try:
                self.step()
                self.maybe_summary()
            except Exception as exc:  # noqa: BLE001
                self.log(f'DSV41_TOKENIZER_HEAP=trim2: trim loop stopped: {exc!r}')
                return


# ---------------------------------------------------------------------------------------------------------------------
# SGLang hooks (installed on import of the two modules, only when trim2 is armed)
# ---------------------------------------------------------------------------------------------------------------------
def _wrap_serving(cls, act) -> int:
    n = 0
    for name in _SERVING_METHODS:
        fn = cls.__dict__.get(name)
        if fn is None or getattr(fn, '__isabstractmethod__', False) or getattr(fn, _MARK, False):
            continue
        setattr(cls, name, wrap(fn, act))
        act.hooks.append(f'{cls.__name__}.{name}')
        n += 1
    return n


def hooks_receipt(act):
    """The cumulative hook list. Printed after each module's install and again whenever a later OpenAIServingBase
    subclass (serving_chat, serving_completions, ... are imported after serving_base) gets wrapped, so the LAST
    ``hooks`` line of a boot names every hooked method."""
    _log(f'DSV41_TOKENIZER_HEAP=trim2: hooks {", ".join(act.hooks)}')


def install_serving_base(module, act=None):
    """Every OpenAIServingBase subclass (defined now or later) counts its _SERVING_METHODS (nesting is counted, so a
    conversion that calls _process_messages is in flight once or twice, never zero)."""
    act = act or _STATE['activity']
    base = module.OpenAIServingBase
    if base.__dict__.get(_MARK):
        return
    setattr(base, _MARK, True)
    prev = base.__dict__.get('__init_subclass__')

    def __init_subclass__(cls, **kw):
        if prev is not None:
            prev.__func__(cls, **kw)
        else:
            super(base, cls).__init_subclass__(**kw)
        if _wrap_serving(cls, act):
            hooks_receipt(act)

    base.__init_subclass__ = classmethod(__init_subclass__)
    todo = list(base.__subclasses__())
    while todo:
        c = todo.pop()
        _wrap_serving(c, act)
        todo.extend(c.__subclasses__())
    act.hooks.append('OpenAIServingBase.__init_subclass__')


def install_tokenizer_manager(module, act=None):
    """TokenizerManager._tokenize_texts counts as activity; __init__ hands the instance to the output-idle check."""
    act = act or _STATE['activity']
    cls = module.TokenizerManager
    if getattr(cls.__init__, _MARK, False):
        return
    cls._tokenize_texts = wrap(cls._tokenize_texts, act)
    init = cls.__init__

    @functools.wraps(init)
    def __init__(self, *a, **k):
        init(self, *a, **k)
        act.set_tokenizer_manager(self)

    setattr(__init__, _MARK, True)
    cls.__init__ = __init__
    act.hooks.append('TokenizerManager._tokenize_texts')


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        try:
            (install_serving_base if module.__name__ == _TARGETS[0] else install_tokenizer_manager)(module)
            hooks_receipt(_STATE['activity'])
        except Exception as exc:  # noqa: BLE001
            _log(f'DSV41_TOKENIZER_HEAP=trim2: hook on {module.__name__} NOT installed ({exc!r})')


class Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader that the rest of sys.meta_path picks (sitecustomize's EngramFinder included), so its patches
    run first and the trim2 hooks after them."""
    _busy = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS or getattr(self._busy, 'on', False):
            return None
        self._busy.on = True
        try:
            spec = None
            for f in sys.meta_path:
                if f is self or not hasattr(f, 'find_spec'):
                    continue
                spec = f.find_spec(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._busy.on = False
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install(env=None, argv=None, orig_argv=None) -> bool:
    """Called from sitecustomize in every interpreter; arms only for trim2 in launch_server."""
    env = os.environ if env is None else env
    if not enabled(env, argv, orig_argv) or _STATE.get('installed'):
        return False
    try:
        heap_free_bytes()                             # glibc mallinfo2 + malloc_trim must exist
        _libc().malloc_trim
    except Exception as exc:  # noqa: BLE001
        _log(f'DSV41_TOKENIZER_HEAP=trim2: not armed (no glibc malloc_trim / mallinfo2: {exc!r})')
        return False
    try:
        cfg = Config(env)
    except ValueError as exc:
        _log(f'DSV41_TOKENIZER_HEAP=trim2: not armed ({exc})')
        return False
    act = Activity()
    trimmer = Trimmer(cfg, act)
    stop = threading.Event()
    _STATE.update(installed=True, activity=act, trimmer=trimmer, stop=stop, config=cfg)
    if not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    for name in _TARGETS:                             # already imported (not the case under sitecustomize)
        if name in sys.modules:
            (install_serving_base if name == _TARGETS[0] else install_tokenizer_manager)(sys.modules[name], act)
    threading.Thread(target=trimmer.run, name='dsv41-heap-trim2', daemon=True, args=(stop,)).start()
    _log(f'DSV41_TOKENIZER_HEAP=trim2: armed in launch_server pid {os.getpid()}: malloc_trim(0) after heap growth, '
         f'in tokenizer gaps; {cfg.describe()}; RssAnon {trimmer.r0 >> 10} MiB')
    return True
