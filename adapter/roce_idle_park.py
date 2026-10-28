"""Idle parking for the RoCE all-reduce proxy threads (DSV41_ROCE_IDLE_PARK=1, off by default).

Each TP rank runs one RDMA proxy thread (b12x.comm.roce_twoshot, one runtime for the TP group).
After its last doorbell it spins ROCE_IDLE_SPINS polls and then polls once per nanosleep(20 us)
for as long as the engine is idle: ~10^4 wakeups a second per rank, a few percent of a core and
no deep idle for that core. With the gate on, the proxy blocks on a futex instead, and only
while the engine itself says it is idle:

  Scheduler.on_idle (the idle branch of both event loops, before --sleep-on-idle's poll)
      -> proxy.park_allow() once per busy->idle transition
  Scheduler.process_input_requests with requests, and Scheduler.run_batch (before every forward,
  hence before its first collective)
      -> proxy.park_wake() once per idle->busy transition (one futex wake)

The proxy parks only when the engine has said idle AND no doorbell rang for
DSV41_ROCE_IDLE_PARK_MS (default 1000). A decode never parks: run_batch clears the idle mark
before the first step, and on_idle is reached only with no batch to run (between the steps of
one request, and during drafting and verify, the loop always has a batch). The parked proxy
re-checks its doorbell every DSV41_ROCE_IDLE_PARK_POLL_MS (default 20); a doorbell found that
way is a "late doorbell" (a missed wake): it costs at most that latency, never correctness,
because the proxy's catch-up posts every pending op in order and the kernels wait ~20 s.

Rank safety: every rank parks and wakes its own proxy from its own scheduler. A collective needs
every rank's kernel, every rank launches its kernels from run_batch, and run_batch wakes that
rank's proxy first, so no rank can wait on a peer whose proxy stays parked.

Needs DSV41_ROCE_TWOSHOT=1 (the v2.6 transport, b12x.comm.roce_twoshot); with b12x.comm.roce or
b12x.comm.roce_ring the engine refuses to start. Gate unset: nothing here is imported and the
proxy library is built exactly as before (same command, same cache name, same code).

Receipts (logged): "DSV41_ROCE_IDLE_PARK: armed ..." at install, "DSV41_ROCE_IDLE_PARK rank=R
proxy_tid=T" once per rank, and "DSV41_ROCE_IDLE_PARK_STATS rank=R ..." at each busy->idle
transition after a park.
"""
import importlib
import logging
import os
import weakref

logger = logging.getLogger(__name__)

GATE = "DSV41_ROCE_IDLE_PARK"
KEY_MS = "DSV41_ROCE_IDLE_PARK_MS"
KEY_POLL = "DSV41_ROCE_IDLE_PARK_POLL_MS"
DEFAULT_MS = 1000
DEFAULT_POLL_MS = 20
MODULE = "b12x.comm.roce_twoshot.roce_oneshot"
OTHER_MODULES = ("b12x.comm.roce.roce_oneshot", "b12x.comm.roce_ring.roce_oneshot")


def enabled(environ=os.environ):
    return environ.get(GATE, "0").strip() == "1"


def config(environ=os.environ):
    """(after_ms, poll_ms), validated: 1..3 600 000 ms and 1..1000 ms."""
    out = []
    for key, default, hi in ((KEY_MS, DEFAULT_MS, 3_600_000), (KEY_POLL, DEFAULT_POLL_MS, 1000)):
        raw = (environ.get(key, "") or "").strip() or str(default)
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{key}={raw!r}: expected an integer number of milliseconds") from None
        if not 1 <= value <= hi:
            raise ValueError(f"{key}={value}: must be between 1 and {hi} ms")
        out.append(value)
    return tuple(out)


class _Entry:
    """One runtime's park state as the scheduler sees it (one per rank in TP4)."""

    __slots__ = ("runtime", "allowed", "tid_logged", "parks_logged")

    def __init__(self, runtime):
        self.runtime = weakref.ref(runtime)
        self.allowed = False
        self.tid_logged = False
        self.parks_logged = 0

    def proxy(self):
        rt = self.runtime()
        return None if rt is None else getattr(rt, "_proxy", None)


_ENTRIES = []
# True while some proxy holds an idle mark: run_batch's only cost when busy is this lookup.
_STATE = {"allowed": False}


def _idle():
    """Busy -> idle: let every proxy park (once per transition) and log the last episode."""
    if _STATE["allowed"]:
        return
    for e in _ENTRIES:
        proxy = e.proxy()
        if proxy is None:
            continue
        stats = proxy.park_stats()
        rank = getattr(e.runtime(), "rank", -1)
        if not e.tid_logged and stats["tid"]:
            logger.warning("DSV41_ROCE_IDLE_PARK rank=%d proxy_tid=%d", rank, stats["tid"])
            e.tid_logged = True
        if stats["parks"] != e.parks_logged:
            e.parks_logged = stats["parks"]
            logger.warning(
                "DSV41_ROCE_IDLE_PARK_STATS rank=%d parks=%d wakes=%d timeouts=%d late_doorbells=%d "
                "parked_s=%.3f last_wake_us=%.1f max_wake_us=%.1f last_lead_us=%.1f",
                rank, stats["parks"], stats["wakes"], stats["timeouts"], stats["late_doorbells"],
                stats["parked_ns"] / 1e9, stats["last_wake_ns"] / 1e3, stats["max_wake_ns"] / 1e3,
                stats["last_lead_ns"] / 1e3)
        proxy.park_allow()
        e.allowed = True
    _STATE["allowed"] = True


def _wake():
    """Idle -> busy: clear the idle mark and wake every parked proxy before any GPU work."""
    _STATE["allowed"] = False
    for e in _ENTRIES:
        if e.allowed:
            e.allowed = False
            proxy = e.proxy()
            if proxy is not None:
                proxy.park_wake()


def install_runtime(module):
    """On import of the RoCE runtime module: build the parking proxy, configure every runtime."""
    if module.__name__ != MODULE:
        if os.environ.get("DSV41_ROCE_TWOSHOT", "0").strip() == "1":
            return  # imported, but not the transport (pynccl builds from roce_twoshot)
        raise RuntimeError(
            f"{GATE}=1 needs the two-shot transport b12x.comm.roce_twoshot (DSV41_ROCE_TWOSHOT=1, "
            f"switched fabric); {module.__name__} has no parking proxy. Unset {GATE}.")
    cls = module.RoceOneshotAllReduce
    if getattr(cls, "_dsv41_idle_park", False):
        return
    after_ms, poll_ms = config()
    proxy_mod = importlib.import_module(module.__name__.rsplit(".", 1)[0] + "._proxy")
    if proxy_mod._LIB is not None and not proxy_mod.IDLE_PARK:
        raise RuntimeError(f"{GATE}=1: the RoCE proxy library was loaded before the gate was installed")
    proxy_mod.IDLE_PARK = True
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # the thread is already running; until this call its after_ms is 0 (never parks)
        self._proxy.park_configure(after_ms, poll_ms)
        _ENTRIES.append(_Entry(self))

    __init__.__wrapped__ = original
    cls.__init__ = __init__
    cls._dsv41_idle_park = True
    logger.warning(
        "DSV41_ROCE_IDLE_PARK: armed after_ms=%d poll_ms=%d (b12x.comm.roce_twoshot proxy built with "
        "ROCE_IDLE_PARK; parks only while the scheduler is idle, woken before every forward)",
        after_ms, poll_ms)


def install_scheduler(module):
    """On import of sglang.srt.managers.scheduler: mark idle in on_idle, wake before work."""
    cls = module.Scheduler
    if getattr(cls, "_dsv41_idle_park", False):
        return
    idle_name = next((n for n in ("on_idle", "self_check_during_idle") if hasattr(cls, n)), None)
    if idle_name is None or not hasattr(cls, "run_batch"):
        raise RuntimeError(f"{GATE}=1: Scheduler has no on_idle / run_batch to hook; unset {GATE}")
    on_idle = getattr(cls, idle_name)
    run_batch = cls.run_batch

    def idle_hook(self, *args, **kwargs):
        _idle()  # before --sleep-on-idle's poll, which is inside on_idle
        return on_idle(self, *args, **kwargs)

    def run_batch_hook(self, *args, **kwargs):
        if _STATE["allowed"]:
            _wake()
        return run_batch(self, *args, **kwargs)

    idle_hook.__wrapped__ = on_idle
    run_batch_hook.__wrapped__ = run_batch
    setattr(cls, idle_name, idle_hook)
    cls.run_batch = run_batch_hook
    process = getattr(cls, "process_input_requests", None)
    if process is not None:
        # earlier wake: requests arrived (every rank receives the broadcast); run_batch still wakes
        def process_hook(self, recv_reqs, *args, **kwargs):
            if recv_reqs and _STATE["allowed"]:
                _wake()
            return process(self, recv_reqs, *args, **kwargs)

        process_hook.__wrapped__ = process
        cls.process_input_requests = process_hook
    cls._dsv41_idle_park = True
    logger.warning("DSV41_ROCE_IDLE_PARK: scheduler hooks on %s / run_batch%s", idle_name,
                   " / process_input_requests" if process is not None else "")
