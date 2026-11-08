"""Host-sync audit of eager extend batches (DSV41_EXTEND_SYNC_AUDIT=1, TEST ONLY, default OFF). Logging only.

Why. Under the overlap scheduler a new request's extend is launched while the previous decode step (a CUDA graph,
~80 ms at c4, ~190 ms at c16) is still on the GPU. One host sync anywhere in that extend (``.item()``, a
``repeat_interleave`` without ``output_size``, a pageable H2D copy, ``nonzero``) makes the host wait for the whole
decode step before it launches the rest of the extend, so the extend's host-paced launch (~160 ms) runs after the
decode step instead of under it. This module lists every such site so they can be removed (adapter/extend_nosync.py).

What it does, per rank, for every ``Scheduler.run_batch`` whose batch is an extend (never decode / verify):

* ``torch.cuda.set_sync_debug_mode("warn")`` for the duration of the call; every synchronizing CUDA call then raises a
  Python warning, which is caught and attributed to the innermost engine / adapter frames (torch internals and this
  module are skipped). Sites are deduplicated; the first full stack of each is kept.
* ``torch.cuda.current_stream().query()`` at entry (non-blocking): whether earlier GPU work (the in-flight decode step)
  was still queued when the extend's launch started. That is the condition under which a sync costs a decode step.
* the host span of the call (perf_counter), bucketed by the batch's new-token count.

Every ``DSV41_EXTEND_SYNC_AUDIT_EVERY`` (64) extends, and at the first extend that shows a new site, a process logs one
line ``DSV41_EXTEND_SYNC_AUDIT {json}``. Nothing is written to the batch, the scheduler or the forward; the sync debug
mode only adds a check to calls that synchronize anyway, so the GPU work and its order are unchanged.

Install: sitecustomize.py's ``sglang.srt.managers.scheduler`` branch, gate checked before the import.
"""
import json
import os
import sys
import time
import traceback
import warnings

_OFF = ("", "0", "off", "false", "no")
_TAG = "DSV41_EXTEND_SYNC_AUDIT"
_SKIP = ("/torch/", "extend_sync_audit", "/contextlib.py", "/warnings.py", "<frozen")
_STATE = {"installed": False, "extends": 0, "with_sync": 0, "inflight": 0, "sites": {}, "spans": {},
          "new_site": False, "every": 64}


def enabled(environ=None):
    environ = os.environ if environ is None else environ
    return str(environ.get(_TAG, "0")).strip().lower() not in _OFF


def site_key(stack, depth=4):
    """Innermost ``depth`` engine / adapter frames of a traceback.extract_stack() list, as 'file:line func'."""
    frames = [f for f in stack if not any(s in f.filename for s in _SKIP)]
    out = []
    for f in reversed(frames[-depth:]):
        name = f.filename
        for marker in ("/sglang/", "/dsv41/", "/adapter/"):
            if marker in name:
                name = name[name.rindex(marker) + 1:]
                break
        out.append(f"{name}:{f.lineno} {f.name}")
    return tuple(out)


def _bucket(tokens):
    for edge in (1, 8, 32, 64, 128, 256, 512, 1024, 4096):
        if tokens <= edge:
            return f"<={edge}"
    return ">4096"


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    except Exception:
        return -1


def record(stack_sites, tokens, span_s, inflight):
    """Account one audited extend (pure bookkeeping; unit-tested without CUDA)."""
    st = _STATE
    st["extends"] += 1
    st["inflight"] += int(bool(inflight))
    if stack_sites:
        st["with_sync"] += 1
    for key, stack in stack_sites:
        site = st["sites"].get(key)
        if site is None:
            st["sites"][key] = {"n": 1, "inflight": int(bool(inflight)), "stack": stack}
            st["new_site"] = True
        else:
            site["n"] += 1
            site["inflight"] += int(bool(inflight))
    b = st["spans"].setdefault(_bucket(tokens), {"n": 0, "sum_ms": 0.0, "max_ms": 0.0, "syncs": 0})
    b["n"] += 1
    b["sum_ms"] += span_s * 1e3
    b["max_ms"] = max(b["max_ms"], span_s * 1e3)
    b["syncs"] += len(stack_sites)


def summary():
    st = _STATE
    return {"rank": _rank(), "extends": st["extends"], "extends_with_sync": st["with_sync"],
            "launched_behind_gpu_work": st["inflight"],
            "spans": {k: {"n": v["n"], "mean_ms": round(v["sum_ms"] / max(v["n"], 1), 2),
                          "max_ms": round(v["max_ms"], 2), "syncs": v["syncs"]} for k, v in st["spans"].items()},
            "sites": [{"site": list(k), "n": v["n"], "inflight": v["inflight"], "stack": v["stack"]}
                      for k, v in sorted(st["sites"].items(), key=lambda kv: -kv[1]["n"])]}


def maybe_log(force=False):
    st = _STATE
    if force or st["new_site"] or st["extends"] % st["every"] == 0:
        st["new_site"] = False
        print(f"{_TAG} {json.dumps(summary(), sort_keys=True)}", flush=True)


def install(scheduler_module):
    if not enabled() or _STATE["installed"]:
        return False
    import torch

    cls = scheduler_module.Scheduler
    orig = cls.run_batch
    try:
        _STATE["every"] = max(1, int(os.environ.get(f"{_TAG}_EVERY", "64")))
    except ValueError:
        _STATE["every"] = 64

    def run_batch(self, batch, *a, **kw):
        mode = getattr(batch, "forward_mode", None)
        if mode is None or not mode.is_extend() or mode.is_target_verify() or not torch.cuda.is_available():
            return orig(self, batch, *a, **kw)
        sites = {}
        inflight = not torch.cuda.current_stream().query()
        prev_mode = torch.cuda.get_sync_debug_mode()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            prev_show = warnings.showwarning

            def show(message, category, filename, lineno, file=None, line=None):
                if "synchroniz" not in str(message):
                    return prev_show(message, category, filename, lineno, file, line)
                stack = traceback.extract_stack()[:-1]
                key = site_key(stack)
                if key not in sites:
                    sites[key] = [f"{f.filename}:{f.lineno} {f.name}" for f in stack[-12:]]

            warnings.showwarning = show
            torch.cuda.set_sync_debug_mode("warn")
            try:
                out = orig(self, batch, *a, **kw)
            finally:
                torch.cuda.set_sync_debug_mode(prev_mode)
                warnings.showwarning = prev_show
        span = time.perf_counter() - t0
        tokens = int(getattr(batch, "extend_num_tokens", 0) or 0)
        record(list(sites.items()), tokens, span, inflight)
        maybe_log()
        return out

    run_batch._dsv41_extend_sync_audit = True
    cls.run_batch = run_batch
    _STATE["installed"] = True
    print(f"[{_TAG}] armed (TEST ONLY): extend batches run with torch sync debug mode 'warn'; logging only",
          file=sys.stderr, flush=True)
    return True
