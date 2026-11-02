"""Terminal receipts of adapter/block_verify_fold.py (CPU, no GPU, no Triton launch).

flush() logs the per-path totals (and, in check mode, the device counters) once per busy -> idle transition that saw a
sampled call; install_scheduler() calls it from Scheduler.on_idle; sitecustomize installs it only when
DSV41_BLOCK_VERIFY_FOLD is set. The periodic LOG_EVERY cadence is unchanged.
    PYTHONPATH=adapter python3 tests/test_block_verify_fold_receipts.py
"""
import contextlib
import io
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

import block_verify_fold as bvf  # noqa: E402

FAILS = []
RECEIPT = re.compile(r"^\[block_verify_fold\] receipt rank=(\d+) calls=(\d+) torch=(\d+) fold=(\d+) check=(\d+) rows=(\d+)( at=idle)?$")
CHECK = re.compile(r"^\[block_verify_fold\] check: rank=(\d+) calls (\d+) rows (\d+) tau mismatches (\d+) "
                   r"token mismatches (\d+)( at=idle)?$")


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    if not ok:
        FAILS.append(name)


def reset(log_every):
    bvf._R.update(torch=0, fold=0, check=0, rows=0, calls=0)
    bvf._C.update(counters=None, check_calls=0)
    bvf._LOGGED["calls"] = 0
    bvf.LOG_EVERY = log_every


def capture(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a)
    return [l for l in buf.getvalue().splitlines() if l]


def test_flush_reports_the_tail():
    reset(2000)
    out = []
    for i in range(2345):
        out += capture(bvf.receipt, "fold", 3)
    check("periodic: one line at 2000 calls, none for the 345-call tail", len(out) == 1 and "calls=2000 " in out[0], out)
    fl = capture(bvf.flush)
    m = RECEIPT.match(fl[0]) if len(fl) == 1 else None
    check("flush: one terminal receipt with the full totals and at=idle",
          m is not None and m.groups() == ("0", "2345", "0", "2345", "0", str(3 * 2345), " at=idle"), fl)
    check("flush: nothing new on the next idle loop", capture(bvf.flush) == [])
    capture(bvf.receipt, "fold", 1)
    fl = capture(bvf.flush)
    check("flush: the next busy -> idle transition logs again", len(fl) == 1 and "calls=2346 " in fl[0], fl)
    check("the periodic line keeps its format (no at=)", RECEIPT.match(out[0]) and not out[0].endswith("at=idle"))


def test_flush_check_counters():
    reset(200)
    t = torch.tensor([1, 2, 3])
    c = torch.tensor([5, 6, 7])
    lines = []
    for i in range(250):
        lines += capture(bvf.check_counts, (t, c), (t.clone(), c.clone()))
        lines += capture(bvf.receipt, "check", 3)
    bad = (torch.tensor([1, 0, 3]), c)
    lines += capture(bvf.check_counts, bad, (t, c))
    lines += capture(bvf.receipt, "check", 3)
    fl = capture(bvf.flush)
    rm = RECEIPT.match(fl[0]) if len(fl) == 2 else None
    cm = CHECK.match(fl[1]) if len(fl) == 2 else None
    check("check mode: terminal receipt + counters cover the 51 calls after the last periodic line (incl. the mismatch)",
          rm and cm and rm.group(2) == "251" and rm.group(5) == "251" and cm.groups()[1:] == ("251", "753", "1", "0", " at=idle"), fl)
    check("check mode: the receipt's check calls equal the counters' calls, rows equal", rm and cm and rm.group(5) == cm.group(2)
          and rm.group(6) == cm.group(3), fl)
    check("check mode: the periodic check line at 200 calls showed no mismatch yet",
          any(CHECK.match(l) and " calls 200 " in l and "tau mismatches 0" in l for l in lines), lines[-4:])


def test_install_scheduler():
    reset(2000)
    calls = []

    class Scheduler:
        def on_idle(self, x=1):
            calls.append(x)
            return "orig"
    mod = types.SimpleNamespace(Scheduler=Scheduler)
    out = capture(bvf.install_scheduler, mod)
    check("install: hooks on_idle and says so", out == ["[block_verify_fold] terminal receipts on Scheduler.on_idle"], out)
    check("install: idempotent", capture(bvf.install_scheduler, mod) == [])
    capture(bvf.receipt, "fold", 2)
    s = Scheduler()
    lines = []
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        r = s.on_idle(7)
        s.on_idle(8)
    lines = [l for l in buf.getvalue().splitlines() if l]
    check("hook: the original on_idle runs with its arguments and result", r == "orig" and calls == [7, 8])
    check("hook: one terminal receipt for the transition", len(lines) == 1 and lines[0].endswith("at=idle"), lines)

    class NoIdle:
        pass
    out = capture(bvf.install_scheduler, types.SimpleNamespace(Scheduler=NoIdle))
    check("install: no idle hook -> logged, class untouched (the gate then sees no terminal receipt)",
          out and "no on_idle hook" in out[0] and not hasattr(NoIdle, "_dsv41_fold_receipt"), out)


def test_sitecustomize_gate():
    src = open(os.path.join(HERE, "..", "adapter", "sitecustomize.py")).read()
    i = src.index("elif module.__name__ == 'sglang.srt.managers.scheduler':")
    j = src.index("elif module.__name__ ==", i + 10)
    block = src[i:j]
    check("sitecustomize: the scheduler hook is installed only under DSV41_BLOCK_VERIFY_FOLD (same off values as the module)",
          "os.environ.get('DSV41_BLOCK_VERIFY_FOLD', '0').strip().lower() not in ('0', '', 'off', 'false', 'no')" in block
          and "install_block_verify_fold_scheduler(module)" in block)
    check("block_verify_fold: ENABLED uses the same off values", 'ENABLED = MODE not in ("0", "", "off", "false", "no")'
          in open(os.path.join(HERE, "..", "adapter", "block_verify_fold.py")).read())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
