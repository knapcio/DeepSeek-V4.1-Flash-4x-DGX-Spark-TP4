"""DSV41_EXTEND_SYNC_AUDIT: `launched_behind_gpu_work` must look at the stream the forward runs on.

Scheduler.run_batch is entered from the event loop under StreamContext(self.schedule_stream)
(scheduler.py, the `with self.device_module.StreamContext(self.schedule_stream)` around
dispatch_event_loop); the in-flight decode step runs on self.forward_stream (run_batch's
`with self.forward_stream_ctx`). After the previous run_batch, _apply_war_barrier makes the
schedule stream wait only on that forward's shared-read-done event, which the decode graph runner
publishes before / early in its replay. So at an extend's entry the schedule stream is typically
idle while the decode step is still running, and `torch.cuda.current_stream().query()` reports
"no GPU work queued". DESIGN.md section 5 kills the nosync line on `launched_behind_gpu_work ~ 0`.

The check: the schedule (current) stream is idle, the forward stream is busy; the audit must count
the extend as launched behind GPU work. CPU only (torch.cuda is stubbed).

usage: PYTHONPATH=adapter python3 tests/test_extend_sync_audit_stream.py
"""
import os
import sys
import types

import torch

os.environ.pop("DSV41_EXTEND_SYNC_AUDIT", None)
import extend_sync_audit as A  # noqa: E402


class FakeStream:
    def __init__(self, done):
        self.done = done

    def query(self):
        return self.done


class FakeMode:
    def is_extend(self):
        return True

    def is_target_verify(self):
        return False


class FakeScheduler:
    def __init__(self):
        self.schedule_stream = FakeStream(done=True)   # current stream at run_batch entry: idle
        self.forward_stream = FakeStream(done=False)   # the decode step D(k+1) still on the GPU

    def run_batch(self, batch):
        return "ran"


saved = {k: getattr(torch.cuda, k) for k in ("is_available", "current_stream", "get_sync_debug_mode",
                                              "set_sync_debug_mode")}
os.environ["DSV41_EXTEND_SYNC_AUDIT"] = "1"
A._STATE.update(installed=False, extends=0, with_sync=0, inflight=0, sites={}, spans={}, new_site=False)
sch = FakeScheduler()
try:
    torch.cuda.is_available = lambda: True
    torch.cuda.current_stream = lambda *a, **k: sch.schedule_stream
    torch.cuda.get_sync_debug_mode = lambda: 0
    torch.cuda.set_sync_debug_mode = lambda mode: None
    assert A.install(types.SimpleNamespace(Scheduler=FakeScheduler))
    out = sch.run_batch(types.SimpleNamespace(forward_mode=FakeMode(), extend_num_tokens=28))
finally:
    for k, v in saved.items():
        setattr(torch.cuda, k, v)
    os.environ.pop("DSV41_EXTEND_SYNC_AUDIT", None)

s = A.summary()
ok = out == "ran" and s["extends"] == 1 and s["launched_behind_gpu_work"] == 1
print(("[OK]   " if ok else "[FAIL] ") + "audit counts an extend entered while the forward stream is busy "
      f"(extends={s['extends']}, launched_behind_gpu_work={s['launched_behind_gpu_work']})")
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
