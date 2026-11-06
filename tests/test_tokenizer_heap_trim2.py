"""CPU checks of DSV41_TOKENIZER_HEAP=trim2 (adapter/heap_trim2.py).

Gate and role (launch_server only; every other DSV41_TOKENIZER_HEAP value inert), the idle gate on a fake clock, real
threads entering and leaving tokenization while the trimmer polls, the growth trigger (no re-trim storm), the rate
limit, the busy fallback (a stream of back-to-back long prompts), abort, the hold budget's deep-idle wait, receipts,
the settings' range check, the autotune fingerprint (the switch is volatile), a soak-shaped simulation, the SGLang
hooks on stand-in classes, and sitecustomize end to end through a stand-in ``sglang`` package run as
``python -m sglang.launch_server``.

Run: PYTHONPATH=adapter python3 tests/test_tokenizer_heap_trim2.py   (the glibc cases run in a Linux image)"""
import abc
import asyncio
import ctypes
import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parent / 'adapter'
sys.path.insert(0, str(ADAPTER))
import heap_trim2 as ht  # noqa: E402

IS_GLIBC = sys.platform.startswith('linux') and platform.libc_ver()[0] == 'glibc'
SERVER = (['/sgl-workspace/sglang/python/sglang/launch_server.py', '--model-path', '/models/x'],
          ['/opt/sglang/bin/python3', '-m', 'sglang.launch_server', '--model-path', '/models/x'])
OTHERS = ((['-c'], ['/opt/sglang/bin/python3', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=9)']),
          (['/opt/dsv41/boot.py', 'run'], ['/opt/sglang/bin/python3', '-u', '/opt/dsv41/boot.py', 'run']),
          (['-c'], ['/opt/sglang/bin/python3', '-c', 'from multiprocessing.resource_tracker import main;main(53)']),
          ([], []))
MIB = 1 << 20


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeHeap:
    """RssAnon (kB) and mallinfo2 free bytes that a trim releases like glibc: RSS drops by the resident free bytes,
    the free bytes stay counted (the chunks stay in their bins)."""

    def __init__(self, rss_mib=1024, free_mib=0):
        self.rss_kb, self.free, self.resident_free, self.calls = rss_mib << 10, free_mib * MIB, 0, []

    def dirty_and_free(self, mib):          # a request's buffers: allocated (RSS up), then freed (free up)
        self.rss_kb += mib << 10
        self.free += mib * MIB
        self.resident_free += mib * MIB

    def trim(self):
        self.calls.append(self.resident_free)
        self.rss_kb -= self.resident_free >> 10
        self.resident_free = 0


def make(cfg_env=None, heap=None, clock=None, perf_step=0.0, log=None):
    clock = clock or Clock()
    heap = heap or FakeHeap()
    act = ht.Activity(clock=clock, wall=lambda: clock() + 1e9)
    act.hooks.append('test')
    perf = {'t': 0.0}

    def perf_fn():
        perf['t'] += perf_step
        return perf['t']
    lines = [] if log is None else log
    env = {'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '0'}       # the pre-rate-limit cases; RateLimit sets its own
    env.update(cfg_env or {})
    tr = ht.Trimmer(ht.Config(env), act, clock=clock, perf=perf_fn, rss_kb=lambda: heap.rss_kb,
                    free_bytes=lambda: heap.free, trim=heap.trim, log=lines.append)
    return tr, act, heap, clock, lines


class GateAndRole(unittest.TestCase):
    def test_enabled_only_for_trim2_in_launch_server(self):
        self.assertTrue(ht.enabled({'DSV41_TOKENIZER_HEAP': 'trim2'}, *SERVER))
        self.assertTrue(ht.enabled({'DSV41_TOKENIZER_HEAP': ' TRIM2 '}, *SERVER))
        for v in ('', '0', 'off', 'trim', 'mmap', 'both', 'trim3', 'bogus'):
            self.assertFalse(ht.enabled({'DSV41_TOKENIZER_HEAP': v}, *SERVER), v)
        self.assertFalse(ht.enabled({}, *SERVER))
        for proc in OTHERS:                  # scheduler / detokenizer spawn children, boot.py, resource tracker
            self.assertFalse(ht.enabled({'DSV41_TOKENIZER_HEAP': 'trim2'}, *proc), proc)

    def test_install_refuses_outside_launch_server(self):
        saved = dict(ht._STATE)
        try:
            ht._STATE.clear()
            for proc in OTHERS:
                self.assertFalse(ht.install({'DSV41_TOKENIZER_HEAP': 'trim2'}, *proc))
            self.assertFalse(ht.install({'DSV41_TOKENIZER_HEAP': 'trim'}, *SERVER))
            self.assertNotIn('installed', ht._STATE)
        finally:
            ht._STATE.clear(); ht._STATE.update(saved)


class OtherValuesInert(unittest.TestCase):
    def test_no_v1_trim_code_and_one_exact_sitecustomize_block(self):
        self.assertFalse((ADAPTER / 'mem_reclaim.py').exists())       # v2.6.3 dropped it; this release has none
        site = (ADAPTER / 'sitecustomize.py').read_text()
        self.assertNotIn('mem_reclaim', site)
        self.assertEqual(site.count("os.environ.get('DSV41_TOKENIZER_HEAP', '').strip().lower() == 'trim2':"), 1)
        self.assertEqual(site.count('import heap_trim2'), 1)
        self.assertEqual(site.count('DSV41_TOKENIZER_HEAP'), 3)         # the comment, the gate, the not-installed line

    def test_server_role(self):
        self.assertTrue(ht.server_role(*SERVER))
        self.assertTrue(ht.server_role(SERVER[0], []))           # .../launch_server.py as argv[0]
        self.assertTrue(ht.server_role([], SERVER[1]))           # python -m sglang.launch_server
        for proc in OTHERS:
            self.assertFalse(ht.server_role(*proc))


class IdleGate(unittest.TestCase):
    def test_no_hooks_no_trim(self):
        tr, act, heap, clock, _ = make()
        act.hooks.clear()
        heap.dirty_and_free(512)
        clock.t += 10
        self.assertEqual(tr.step(), 'nohook')
        self.assertEqual(heap.calls, [])

    def test_in_flight_and_idle_window(self):
        tr, act, heap, clock, _ = make()
        heap.dirty_and_free(512)
        act.enter()
        clock.t += 100
        self.assertEqual(tr.step(), 'busy')              # in flight, however long
        act.exit()
        clock.t += 1.49
        self.assertEqual(tr.step(), 'busy')              # 1.49 s after the last exit
        clock.t += 0.01
        self.assertEqual(tr.step(), 'trim')
        self.assertEqual(heap.calls, [512 * MIB])
        self.assertEqual(tr.n['trims'], 1)

    def test_two_overlapping_tokenizations(self):
        tr, act, heap, clock, _ = make()
        heap.dirty_and_free(512)
        act.enter(); act.enter(); act.exit()
        clock.t += 5
        self.assertEqual(tr.step(), 'busy')              # one still in flight
        act.exit()
        self.assertEqual(tr.step(), 'busy')
        clock.t += 1.5
        self.assertEqual(tr.step(), 'trim')

    def test_custom_idle(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_IDLE_S': '0.2'})
        heap.dirty_and_free(300)
        act.enter(); act.exit()
        clock.t += 0.2
        self.assertEqual(tr.step(), 'trim')


class GrowthTrigger(unittest.TestCase):
    def test_below_min_no_trim(self):
        tr, act, heap, clock, _ = make()
        heap.dirty_and_free(255)
        clock.t += 10
        self.assertEqual(tr.step(), 'small')
        heap.dirty_and_free(1)
        self.assertEqual(tr.step(), 'trim')

    def test_no_retrim_storm(self):
        """The free bytes stay counted after a trim: v2.6.3's trim_due() fires on every check, trim2 does not."""
        tr, act, heap, clock, _ = make()
        heap.dirty_and_free(1024)
        clock.t += 2
        self.assertEqual(tr.step(), 'trim')
        storm = 0
        for _ in range(100):
            clock.t += 2
            self.assertEqual(tr.step(), 'small')
            storm += heap.free >= 256 * MIB              # v2.6.3 trim's condition
        self.assertEqual(storm, 100)                     # what mode trim would have done
        self.assertEqual(len(heap.calls), 1)

    def test_free_growth_without_rss_growth(self):
        """mallinfo2 cannot tell released free bytes from resident ones. A live request that grows RSS by >= min
        while >= min (already released) bytes are free costs one trim that returns nothing (a walk of the free
        lists, ~2 ms/GiB measured), never a second one. When it ends, its buffers become free without any RSS
        change, and the free-bytes growth triggers the trim that returns them."""
        tr, act, heap, clock, _ = make()
        heap.dirty_and_free(300)
        clock.t += 2
        self.assertEqual(tr.step(), 'trim')
        heap.rss_kb += 600 << 10                          # a live 600 MiB request
        clock.t += 2
        self.assertEqual(tr.step(), 'trim')
        self.assertEqual(heap.calls[-1], 0)
        for _ in range(10):
            clock.t += 2
            self.assertEqual(tr.step(), 'small')
        heap.free += 600 * MIB; heap.resident_free += 600 * MIB   # it ends
        clock.t += 2
        self.assertEqual(tr.step(), 'small')              # no RSS growth: mallinfo2 is read every INFO_EVERY_S only
        clock.t += 30
        self.assertEqual(tr.step(), 'trim')
        self.assertEqual(heap.calls[-1], 600 * MIB)
        self.assertEqual(len(heap.calls), 3)

    def test_mallinfo2_read_only_on_growth_or_every_info_every_s(self):
        reads = []
        tr, act, heap, clock, _ = make()
        orig = tr.free_bytes
        tr.free_bytes = lambda: (reads.append(clock.t), orig())[1]
        clock.t += 1                                      # past the idle gate's 1.5 s (with the first +1 below)
        for _ in range(60):                               # idle, nothing grows: one read, then one per 30 s
            clock.t += 1
            self.assertEqual(tr.step(), 'small')
        self.assertEqual(len(reads), 2)
        heap.rss_kb += 300 << 10                          # RSS growth: read at once
        heap.free += 300 * MIB; heap.resident_free += 300 * MIB
        clock.t += 1
        self.assertEqual(tr.step(), 'trim')
        self.assertEqual(len(reads), 3)                   # and no second walk after the call (f0 = the pre-trim count)
        self.assertEqual(tr.f0, 300 * MIB)

    def test_rss_growth_reuse(self):
        """Released pages reused and freed again: free bytes unchanged, RSS grew."""
        tr, act, heap, clock, _ = make()
        heap.dirty_and_free(512)
        clock.t += 2
        tr.step()
        heap.rss_kb += 400 << 10; heap.resident_free += 400 * MIB
        clock.t += 2
        self.assertEqual(tr.step(), 'trim')


class AbortAndBudget(unittest.TestCase):
    def test_activity_after_gate_aborts(self):
        heap = FakeHeap()
        tr, act, heap, clock, _ = make(heap=heap)
        heap.dirty_and_free(512)
        clock.t += 2
        orig = tr.free_bytes
        tr.free_bytes = lambda: (act.enter(), orig())[1]   # a request arrives while mallinfo2 runs
        self.assertEqual(tr.step(), 'abort')
        self.assertEqual(heap.calls, [])
        self.assertEqual(tr.n['aborted'], 1)
        act.exit(); tr.free_bytes = orig
        clock.t += 1.5
        self.assertEqual(tr.step(), 'trim')

    def _busy_outputs(self, tr, act, clock):
        tm = _TM()
        act.set_tokenizer_manager(tm)
        act._wall = lambda: clock()
        tm.last_receive_tstamp = clock()
        return tm

    def test_predicted_over_budget_waits_for_idle_outputs(self):
        tr, act, heap, clock, lines = make()
        tm = self._busy_outputs(tr, act, clock)
        heap.dirty_and_free(4096)                         # 40 ms/GiB x 4 GiB = 160 ms > 50 ms
        clock.t += 2
        tm.last_receive_tstamp = clock() - 0.1           # a decode stream is running
        self.assertEqual(tr.step(), 'defer')
        self.assertEqual(heap.calls, [])
        clock.t += 1
        tm.last_receive_tstamp = clock() - 1.5           # stream idle for 1.5 s
        self.assertEqual(tr.step(), 'trim')
        self.assertEqual(tr.n['deferred'], 1)
        self.assertEqual(tr.n['forced'], 0)

    def test_under_budget_ignores_outputs(self):
        tr, act, heap, clock, _ = make()
        tm = self._busy_outputs(tr, act, clock)
        heap.dirty_and_free(1024)                         # 40 ms predicted
        clock.t += 2
        tm.last_receive_tstamp = clock()
        self.assertEqual(tr.step(), 'trim')

    def test_forced_after_max_defer(self):
        tr, act, heap, clock, lines = make({'DSV41_TOKENIZER_TRIM2_MAX_DEFER_S': '60'})
        tm = self._busy_outputs(tr, act, clock)
        heap.dirty_and_free(4096)
        clock.t += 2
        for _ in range(60):
            tm.last_receive_tstamp = clock()
            self.assertEqual(tr.step(), 'defer')
            clock.t += 1
        tm.last_receive_tstamp = clock()
        self.assertEqual(tr.step(), 'forced')
        self.assertEqual(tr.n['forced'], 1)
        self.assertIn('FORCED', lines[-1])

    def test_never_forced_with_zero(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MAX_DEFER_S': '0'})
        tm = self._busy_outputs(tr, act, clock)
        heap.dirty_and_free(4096)
        for _ in range(500):
            clock.t += 2
            tm.last_receive_tstamp = clock()
            self.assertEqual(tr.step(), 'defer')

    def test_budget_zero_means_deep_idle_only(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_BUDGET_MS': '0', 'DSV41_TOKENIZER_TRIM2_MAX_DEFER_S': '0'})
        tm = self._busy_outputs(tr, act, clock)
        heap.dirty_and_free(300)
        clock.t += 2
        tm.last_receive_tstamp = clock()
        self.assertEqual(tr.step(), 'defer')
        tm.last_receive_tstamp = clock() - 2
        self.assertEqual(tr.step(), 'trim')

    def test_cost_model_relearned(self):
        tr, act, heap, clock, lines = make(perf_step=0.040)   # each trim call measures 40 ms
        heap.dirty_and_free(512)                               # -> 80 ms/GiB observed
        clock.t += 2
        tr.step()
        self.assertAlmostEqual(tr.ms_per_gib, 0.5 * 40 + 0.5 * 80, places=6)
        self.assertEqual(tr.n['over_budget'], 0)

    def test_no_tokenizer_manager_counts_as_idle_outputs(self):
        act = ht.Activity()
        self.assertTrue(act.outputs_quiet(1.5))


class RateLimit(unittest.TestCase):
    def test_at_most_one_trim_per_interval_idle_path(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '60'})
        heap.dirty_and_free(512)
        clock.t += 2
        self.assertEqual(tr.step(), 'trim')
        for _ in range(29):                               # idle, growing: still inside the interval
            heap.dirty_and_free(300)
            clock.t += 2
            self.assertEqual(tr.step(), 'interval')
        clock.t += 2                                      # 60 s after the first trim
        self.assertEqual(tr.step(), 'trim')
        self.assertEqual(len(heap.calls), 2)
        self.assertEqual(heap.calls[1], 29 * 300 * MIB)

    def test_first_trim_is_not_delayed(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '600'})
        heap.dirty_and_free(300)
        clock.t += 2
        self.assertEqual(tr.step(), 'trim')

    def test_defaults(self):
        c = ht.Config({})
        self.assertEqual((c.min_interval_s, c.max_busy_s, c.idle_s, c.min_bytes, c.poll_s), (120.0, 30.0, 1.5, 256 << 20, 1.0))
        self.assertEqual(ht.Config({}).max_busy_s, 30.0)


class BusyFallback(unittest.TestCase):
    """A stream of back-to-back long prompts (the issue-16 warm loop: 1 or 4 streams, a new prompt as soon as the last
    answer arrives) never leaves the 1.5 s idle gap. A due trim runs at the next gap between two prompts, on the thread
    whose tokenization just ended (Activity.exit), so no prompt is being encoded during the call."""

    def _stream(self, tr, act, heap, clock, seconds, grow_mib_per_s=7, prompt_s=1.0, gap_s=0.5):
        """prompt_s in flight, gap_s between prompts, a poll every 0.5 s -> (step results, [(time, inflight) per trim])."""
        calls, orig = [], heap.trim

        def trim():
            calls.append((clock.t, act.inflight))
            orig()
        tr.trim = trim
        out, t_end = [], clock.t + seconds
        while clock.t < t_end:
            act.enter()
            for _ in range(int(prompt_s / 0.5)):
                clock.t += 0.5
                heap.dirty_and_free(int(grow_mib_per_s * 0.5) or 1)
                out.append(tr.step())
            act.exit()                                    # the gap edge: a due trim runs here
            for _ in range(int(gap_s / 0.5)):
                clock.t += 0.5
                out.append(tr.step())
        return out, calls

    def test_trims_between_prompts_after_max_busy(self):
        tr, act, heap, clock, lines = make({'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '60',
                                            'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '30'})
        t0 = clock.t
        res, calls = self._stream(tr, act, heap, clock, 600)
        self.assertTrue(calls)
        self.assertNotIn('trim', res)                     # never the idle path: no idle gap exists
        self.assertEqual({i for _, i in calls}, {0})      # every call between two prompts
        # 3 MiB per in-flight poll, 2 of every 3 polls: 4 MiB/s, 256 MiB after ~64 s, then 30 s of busy wait, then the
        # next gap; later the interval (60 s) dominates
        self.assertGreaterEqual(calls[0][0] - t0, 30 + 62)
        self.assertLessEqual(calls[0][0] - t0, 30 + 66)
        gaps = [b[0] - a[0] for a, b in zip(calls, calls[1:])]
        self.assertTrue(all(g >= 60 for g in gaps), gaps)
        self.assertLessEqual(len(calls), 600 // 60)
        self.assertTrue(any('BUSY after no idle gap' in l for l in lines))
        self.assertEqual(tr.n['busy'], len(calls))
        self.assertIn('due', res)

    def test_zero_disables_the_fallback(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '0'})
        res, calls = self._stream(tr, act, heap, clock, 600, grow_mib_per_s=50)
        self.assertEqual(calls, [])
        self.assertEqual(set(res), {'busy'})

    def test_no_growth_no_busy_wait(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '1'})
        act.enter()
        for _ in range(100):
            clock.t += 1
            heap.rss_kb += 1 << 10                        # 1 MiB/s: 100 MiB < 256 MiB
            self.assertEqual(tr.step(), 'busy')
        self.assertIsNone(tr.busy_since)
        self.assertIsNone(tr.due_since)
        self.assertEqual(tr.n['busy_waits'], 0)

    def test_busy_wait_runs_inside_the_interval(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '60',
                                        'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '30'})
        heap.dirty_and_free(300)
        clock.t += 2
        self.assertEqual(tr.step(), 'trim')               # idle trim at t0
        t0 = clock.t
        act.enter()
        heap.dirty_and_free(300)                          # growth seen at t0 + 5: the busy wait starts there
        clock.t += 5
        self.assertEqual(tr.step(), 'interval')
        clock.t = t0 + 59
        self.assertEqual(tr.step(), 'interval')
        clock.t = t0 + 60
        self.assertEqual(tr.step(), 'due')                # waited 55 s; a prompt is in flight: the next gap runs it
        self.assertEqual(len(heap.calls), 1)
        act.exit()                                        # a gap between two prompts (not an idle gap)
        self.assertEqual(len(heap.calls), 2)
        self.assertEqual(tr.n['busy'], 1)
        self.assertIsNone(tr.due_since)

    def test_no_gap_at_all_falls_back_to_the_trimmer_thread(self):
        tr, act, heap, clock, lines = make({'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '30'})
        act.enter()                                       # one tokenization that never ends (or overlapping ones)
        res = []
        for _ in range(100):
            clock.t += 1
            heap.dirty_and_free(10)
            res.append(tr.step())
        self.assertIn('busytrim', res)
        first = res.index('busytrim') + 1                 # seconds since the start
        self.assertGreaterEqual(first, 26 + 30 + 30)      # 256 MiB at 26 s, + the busy wait + the gap wait
        self.assertLessEqual(first, 26 + 30 + 30 + 1)

    def test_live_growth_is_not_trimmed(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '5'})
        act.enter()
        for _ in range(10):
            clock.t += 1
            heap.rss_kb += 100 << 10                      # 1 GiB of live data, nothing freed
            tr.step()
        self.assertIsNotNone(tr.due_since)
        act.exit()                                        # the gap: mallinfo2 has < 256 MiB free: no call
        self.assertEqual(heap.calls, [])
        self.assertIsNone(tr.due_since)
        self.assertIsNone(tr.busy_since)

    def test_prompt_starting_during_the_reads_aborts(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '1'})
        act.enter(); act.enter()                          # two overlapping tokenizations
        heap.dirty_and_free(512)
        clock.t += 1
        self.assertEqual(tr.step(), 'busy')
        clock.t += 1
        self.assertEqual(tr.step(), 'due')
        orig = tr.free_bytes
        tr.free_bytes = lambda: (act.enter(), orig())[1]  # a prompt arrives while mallinfo2 runs
        act.exit(); act.exit()                            # the gap edge
        self.assertEqual(heap.calls, [])
        self.assertEqual(tr.n['aborted'], 1)
        tr.free_bytes = orig
        act.exit()                                        # the next gap
        self.assertEqual(len(heap.calls), 1)

    def test_a_failing_trim_never_reaches_the_request(self):
        tr, act, heap, clock, _ = make({'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '1'})
        act.enter()
        heap.dirty_and_free(512)
        for _ in range(2):
            clock.t += 1
            tr.step()
        self.assertIsNotNone(tr.due_since)
        tr.trim = lambda: 1 / 0
        act.exit()                                        # swallowed: the conversion's caller sees nothing
        self.assertTrue(tr._lock.acquire(blocking=False))
        tr._lock.release()

    def test_real_threads_trim_only_in_gaps(self):
        heap = FakeHeap()
        act = ht.Activity()
        act.hooks.append('test')
        seen = []

        def trim():
            seen.append(act.inflight)
            heap.trim()
        cfg = ht.Config({'DSV41_TOKENIZER_TRIM2_IDLE_S': '5', 'DSV41_TOKENIZER_TRIM2_POLL_S': '0.005',
                         'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '0.05', 'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '0.1'})
        tr = ht.Trimmer(cfg, act, rss_kb=lambda: heap.rss_kb, free_bytes=lambda: heap.free, trim=trim, log=lambda m: None)
        stop = threading.Event()
        runner = threading.Thread(target=tr.run, args=(stop,)); runner.start()
        end = time.monotonic() + 1.0
        lock = threading.Lock()

        def tokenizer():                                  # one event-loop-like thread: prompts back to back
            while time.monotonic() < end:
                act.enter()
                with lock:
                    heap.dirty_and_free(64)
                time.sleep(0.02)
                act.exit()
        t = threading.Thread(target=tokenizer); t.start(); t.join()
        stop.set(); runner.join(5)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {0}, seen)


class ConfigRange(unittest.TestCase):
    def test_rejects_negative_nan_and_zero_poll(self):
        for k, v in (('MIN_INTERVAL_S', '-1'), ('MAX_BUSY_S', 'nan'), ('IDLE_S', 'inf'), ('POLL_S', '0'),
                     ('MIN_MIB', '0'), ('BUDGET_MS', '-5')):
            with self.assertRaises(ValueError, msg=k):
                ht.Config({'DSV41_TOKENIZER_TRIM2_' + k: v})

    def test_install_refuses_bad_settings(self):
        saved = dict(ht._STATE)
        lines = []
        orig = ht._log
        try:
            ht._STATE.clear()
            ht._log = lines.append
            if IS_GLIBC:
                self.assertFalse(ht.install({'DSV41_TOKENIZER_HEAP': 'trim2', 'DSV41_TOKENIZER_TRIM2_POLL_S': '0'},
                                            *SERVER))
                self.assertIn('not armed', lines[-1])
            self.assertNotIn('installed', ht._STATE)
        finally:
            ht._log = orig
            ht._STATE.clear(); ht._STATE.update(saved)


class AutotuneFingerprint(unittest.TestCase):
    """The switch and its knobs act in launch_server only: arms with and without it share one MoE tactic table."""

    def test_switch_and_knobs_are_volatile(self):
        import importlib
        sys.path.insert(0, str(ADAPTER))
        ak = importlib.import_module('autotune_keep')
        keys = ['DSV41_TOKENIZER_HEAP'] + ['DSV41_TOKENIZER_TRIM2_' + k for k in (
            'IDLE_S', 'POLL_S', 'MIN_MIB', 'MIN_INTERVAL_S', 'MAX_BUSY_S', 'INFO_EVERY_S', 'BUDGET_MS', 'MAX_DEFER_S',
            'LOG_MIN_S', 'SUMMARY_S', 'MS_PER_GIB')]
        for k in keys:
            self.assertIn(k, ak._VOLATILE, k)
        src = (ADAPTER / 'heap_trim2.py').read_text()
        import re
        used = set(re.findall(r"f\('([A-Z_]+)', '", src))
        self.assertEqual({'DSV41_TOKENIZER_TRIM2_' + k for k in used}, set(keys[1:]))
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            a = ak.launch_fingerprint()
            os.environ.update({k: '1' for k in keys})
            self.assertEqual(ak.launch_fingerprint(), a)
        finally:
            for k in keys:
                os.environ.pop(k, None)
                if saved[k] is not None:
                    os.environ[k] = saved[k]


class SoakSimulation(unittest.TestCase):
    """The 2026-11-05 R0b warm loop on the fake heap: minutes alternating 1 and 4 back-to-back streams, prompts of
    1-7 s with ~0.4 GiB/min of retained growth (R0b: launch_server 1.86 -> 6.79 GiB in ~11.5 min) and 250-450 MiB of
    pages re-touched after every trim (the 2026-10-30 trim receipts); then the soak ends (idle)."""

    def test_bounded_and_rate_limited(self):
        tr, act, heap, clock, lines = make({'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '60',
                                            'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '30'})
        import random
        rnd = random.Random(20261106)
        rss0 = heap.rss_kb
        peak, t_end = 0, clock.t + 1800
        while clock.t < t_end:
            conc = 1 if int((clock.t - 1000) // 60) % 2 == 0 else 4
            dur = rnd.uniform(1.0, 7.0) / conc                     # tokenizations back to back on one loop
            act.enter()
            steps = max(1, int(dur / 0.5))
            for _ in range(steps):
                clock.t += 0.5
                heap.dirty_and_free(int(0.4 * 1024 / 60 * 0.5) + 1)  # retained growth ~0.4 GiB/min
                tr.step()
                peak = max(peak, heap.rss_kb)
            act.exit()                                              # a due busy trim runs at this gap
            clock.t += 0.05                                         # the gap between two prompts
            tr.step()
        n_busy = tr.n['busy']
        self.assertGreaterEqual(n_busy, 1800 // 90)
        self.assertLessEqual(tr.n['trims'], 1800 // 60 + 1)
        self.assertLessEqual((peak - rss0) >> 10, int(0.42 * 1024 * 1.5) + 64)   # <= ~1.5 min of growth unreturned
        for _ in range(4):                                         # the soak ends: idle
            clock.t += 30
            tr.step()
        self.assertLessEqual((heap.rss_kb - rss0) >> 10, 256)


class _TM:
    last_receive_tstamp = 0.0


class Receipts(unittest.TestCase):
    def test_rate_limited_per_trim_lines_and_summary(self):
        tr, act, heap, clock, lines = make(perf_step=0.005)
        for i in range(5):                                # 5 trims 2 s apart: one line, then 10 s silence
            heap.dirty_and_free(300)
            clock.t += 2
            self.assertEqual(tr.step(), 'trim')
        trim_lines = [l for l in lines if ' trim #' in l]
        self.assertEqual(len(trim_lines), 1)
        self.assertIn('trim #1 5.0 ms, returned 300 MiB', trim_lines[0])
        heap.dirty_and_free(300)
        clock.t += 10
        tr.step()
        self.assertIn('trim #6', lines[-1])
        self.assertIn('4 lines suppressed', lines[-1])
        s = tr.summary()
        self.assertIn('summary trims 6 (forced 0, busy 0, over budget 0)', s)
        self.assertIn('returned 1800 MiB', s)
        self.assertIn('call ms p50 5.0 max 5.0', s)

    def test_over_budget_always_logged(self):
        tr, act, heap, clock, lines = make(perf_step=0.060)
        heap.dirty_and_free(300); clock.t += 2; tr.step()
        heap.dirty_and_free(300); clock.t += 2; tr.step()
        self.assertEqual(sum('OVER BUDGET' in l for l in lines), 2)

    def test_periodic_summary(self):
        tr, act, heap, clock, lines = make({'DSV41_TOKENIZER_TRIM2_SUMMARY_S': '60'})
        tr.maybe_summary()
        self.assertEqual(lines, [])
        clock.t += 60
        tr.maybe_summary()
        self.assertTrue(lines and 'summary' in lines[-1])

    def test_loop_survives_nothing_but_logs_its_death(self):
        tr, act, heap, clock, lines = make({'DSV41_TOKENIZER_TRIM2_POLL_S': '0.001'})
        heap.dirty_and_free(300); clock.t += 2
        tr.trim = lambda: 1 / 0
        stop = threading.Event()
        t = threading.Thread(target=tr.run, args=(stop,)); t.start(); t.join(5)
        self.assertFalse(t.is_alive())
        self.assertIn('trim loop stopped', lines[-1])


class ConcurrentActivity(unittest.TestCase):
    """Real threads and the real monotonic clock."""

    def test_counter_is_consistent_under_contention(self):
        act = ht.Activity()
        def worker():
            for _ in range(5000):
                act.enter(); act.exit()
        ts = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(act.inflight, 0)
        self.assertEqual(act.entries, 40000)

    def test_never_trims_inside_a_busy_period(self):
        heap = FakeHeap()
        act = ht.Activity()
        act.hooks.append('test')
        seen = []
        def trim():
            seen.append((act.inflight, time.monotonic() - act.last))
            heap.trim()
        lines = []
        cfg = ht.Config({'DSV41_TOKENIZER_TRIM2_IDLE_S': '0.15', 'DSV41_TOKENIZER_TRIM2_POLL_S': '0.005'})
        tr = ht.Trimmer(cfg, act, rss_kb=lambda: heap.rss_kb, free_bytes=lambda: heap.free, trim=trim,
                        log=lines.append)
        stop = threading.Event()
        runner = threading.Thread(target=tr.run, args=(stop,)); runner.start()
        busy_end = time.monotonic() + 1.2
        def tokenizer(k):                                 # 4 tokenizers, gaps < idle_s, heap growing throughout
            while time.monotonic() < busy_end:
                act.enter(); heap.dirty_and_free(64); time.sleep(0.02 + 0.01 * k); act.exit(); time.sleep(0.03)
        ts = [threading.Thread(target=tokenizer, args=(k,)) for k in range(4)]
        [t.start() for t in ts]; [t.join() for t in ts]
        during = len(seen)
        time.sleep(0.6)
        stop.set(); runner.join(5)
        self.assertEqual(during, 0, seen)
        self.assertEqual(len(seen), 1, seen)              # one trim once the tokenizers went quiet
        self.assertEqual(seen[0][0], 0)
        self.assertGreaterEqual(seen[0][1], 0.15)


class Hooks(unittest.TestCase):
    def _serving_base(self):
        m = types.ModuleType('fake_serving_base')

        class OpenAIServingBase(abc.ABC):
            @abc.abstractmethod
            def _convert_to_internal_request(self, request, raw_request=None):
                pass
        m.OpenAIServingBase = OpenAIServingBase
        return m

    def test_serving_base_subclasses_before_and_after(self):
        act = ht.Activity()
        m = self._serving_base()
        probe = []

        class Early(m.OpenAIServingBase):
            def _convert_to_internal_request(self, request, raw_request=None):
                probe.append(act.inflight)
                return request, request
        ht.install_serving_base(m, act)
        ht.install_serving_base(m, act)                   # idempotent

        class Chat(m.OpenAIServingBase):
            def _process_messages(self, request, is_multimodal):
                probe.append(('render', act.inflight))
                return request

            def _convert_to_internal_request(self, request, raw_request=None):
                probe.append(act.inflight)
                if request == 'bad':
                    raise ValueError('bad')
                return self._process_messages(request, False), request

        class Responses(Chat):                            # inherits Chat's (already wrapped) method
            pass

        class Tokenize(m.OpenAIServingBase):              # overrides handle_request only
            def _convert_to_internal_request(self, request, raw_request=None):
                return super()._convert_to_internal_request(request, raw_request)
        self.assertEqual(Early()._convert_to_internal_request('a'), ('a', 'a'))
        self.assertEqual(Chat()._convert_to_internal_request('b'), ('b', 'b'))
        self.assertEqual(Responses()._convert_to_internal_request('c'), ('c', 'c'))
        with self.assertRaises(ValueError):
            Chat()._convert_to_internal_request('bad')
        self.assertEqual(probe, [1, 1, ('render', 2), 1, ('render', 2), 1])   # nested render counted, never 0
        self.assertEqual(act.inflight, 0)
        self.assertEqual(act.entries, 6)
        Chat()._process_messages('e', False)              # count_tokens' direct render
        self.assertEqual(probe[-1], ('render', 1))
        self.assertIn('Early._convert_to_internal_request', act.hooks)
        self.assertIn('Chat._convert_to_internal_request', act.hooks)
        self.assertIn('Chat._process_messages', act.hooks)
        self.assertNotIn('Responses._convert_to_internal_request', act.hooks)
        Tokenize()._convert_to_internal_request('d')     # the abstract base method is not wrapped
        self.assertEqual(act.entries, 8)

    def test_receipt_after_each_later_subclass(self):
        act = ht.Activity()
        m = self._serving_base()
        logged = []
        orig, ht._log = ht._log, logged.append
        try:
            ht.install_serving_base(m, act)
            ht.hooks_receipt(act)                         # what the loader prints after serving_base's import
            self.assertEqual(logged, ['DSV41_TOKENIZER_HEAP=trim2: hooks OpenAIServingBase.__init_subclass__'])

            class OpenAIServingChat(m.OpenAIServingBase):
                def _process_messages(self, request, is_multimodal):
                    return request

                def _convert_to_internal_request(self, request, raw_request=None):
                    return request, request

            class Quiet(OpenAIServingChat):               # nothing of its own: no new line
                pass
        finally:
            ht._log = orig
        self.assertEqual(len(logged), 2, logged)
        self.assertEqual(logged[-1], 'DSV41_TOKENIZER_HEAP=trim2: hooks OpenAIServingBase.__init_subclass__, '
                                     'OpenAIServingChat._convert_to_internal_request, OpenAIServingChat._process_messages')

    def test_tokenizer_manager_async_and_cancel(self):
        act = ht.Activity()
        m = types.ModuleType('fake_tm')
        gate = {}

        class TokenizerManager:
            def __init__(self, x):
                self.x = x
                self.last_receive_tstamp = time.time()

            async def _tokenize_texts(self, texts, is_cross_encoder=False):
                gate['inflight'] = act.inflight
                await asyncio.sleep(0.01 if texts != 'slow' else 10)
                return [len(texts)], None
        m.TokenizerManager = TokenizerManager
        ht.install_tokenizer_manager(m, act)
        ht.install_tokenizer_manager(m, act)
        tm = TokenizerManager(7)
        self.assertEqual(tm.x, 7)
        self.assertIs(act._tm(), tm)
        self.assertFalse(act.outputs_quiet(1.5))

        async def go():
            self.assertEqual(await tm._tokenize_texts('abc'), ([3], None))
            task = asyncio.ensure_future(tm._tokenize_texts('slow'))
            await asyncio.sleep(0.02)
            self.assertEqual(act.inflight, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(go())
        self.assertEqual(gate['inflight'], 1)
        self.assertEqual(act.inflight, 0)
        self.assertEqual(act.entries, 2)
        self.assertEqual(act.hooks, ['TokenizerManager._tokenize_texts'])


# a stand-in sglang tree: the two hooked modules (plus serving_chat, which sitecustomize's own finder patches)
_FAKE = {
    'sglang/__init__.py': '',
    'sglang/srt/__init__.py': '',
    'sglang/srt/entrypoints/__init__.py': '',
    'sglang/srt/entrypoints/openai/__init__.py': '',
    'sglang/srt/managers/__init__.py': '',
    'sglang/srt/entrypoints/openai/serving_base.py': '''
        from abc import ABC, abstractmethod
        class OpenAIServingBase(ABC):
            @abstractmethod
            def _convert_to_internal_request(self, request, raw_request=None):
                pass
    ''',
    'sglang/srt/entrypoints/openai/serving_chat.py': '''
        from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
        class OpenAIServingChat(OpenAIServingBase):
            def _process_messages(self, request, is_multimodal):
                return request
            def _convert_to_internal_request(self, request, raw_request=None):
                import heap_trim2
                return heap_trim2._STATE['activity'].inflight, request
    ''',
    'sglang/srt/managers/tokenizer_manager.py': '''
        import time
        class TokenizerManager:
            def __init__(self):
                self.last_receive_tstamp = time.time()
            def _validate_one_request(self, obj, input_ids):
                return None
            async def _tokenize_texts(self, texts, is_cross_encoder=False):
                import heap_trim2
                return heap_trim2._STATE['activity'].inflight, None
    ''',
    'sglang/launch_server.py': '''
        import asyncio, sys, types
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        try:
            import heap_trim2
            st = heap_trim2._STATE
        except ImportError:
            st = {}
        if not st.get('installed'):
            print('RESULT not-armed')
            sys.exit(0)
        req = types.SimpleNamespace(max_tokens=None, max_completion_tokens=None)
        n1, r = OpenAIServingChat()._convert_to_internal_request(req)
        tm = TokenizerManager()
        n2, _ = asyncio.run(tm._tokenize_texts('x'))
        act = st['activity']
        print('RESULT', n1, n2, act.inflight, act.entries, act._tm() is tm, r.max_tokens is not None,
              sorted(act.hooks))
    ''',
}


class SitecustomizeEndToEnd(unittest.TestCase):
    """python -m sglang.launch_server (stand-in) with the adapter's sitecustomize, as the image runs it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        for rel, src in _FAKE.items():
            p = Path(cls.tmp.name) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_py(self, args, heap, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith('DSV41_')}
        env.update(DSV41_SOURCE='1', PYTHONPATH=f'{ADAPTER}{os.pathsep}{self.tmp.name}')
        env.update(extra or {})
        if heap is not None:
            env['DSV41_TOKENIZER_HEAP'] = heap
        r = subprocess.run([sys.executable] + args, env=env, capture_output=True, text=True, timeout=120,
                           cwd=self.tmp.name)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return r.stdout

    def test_trim2_in_launch_server(self):
        out = self.run_py(['-m', 'sglang.launch_server'], 'trim2')
        if not IS_GLIBC:
            self.assertIn('DSV41_TOKENIZER_HEAP=trim2: not armed (no glibc', out)
            self.assertIn('RESULT not-armed', out)
            return
        self.assertIn('DSV41_TOKENIZER_HEAP=trim2: armed in launch_server', out)
        self.assertIn('DSV41_TOKENIZER_HEAP=trim2: hooks', out)
        # the judge reads the LAST hooks line (the 11-06 fleet KILL: it was printed before serving_chat's class existed)
        last = [l for l in out.splitlines() if l.startswith('DSV41_TOKENIZER_HEAP=trim2: hooks ')][-1]
        for h in ('OpenAIServingBase.__init_subclass__', 'OpenAIServingChat._convert_to_internal_request',
                  'OpenAIServingChat._process_messages', 'TokenizerManager._tokenize_texts'):
            self.assertIn(h, last, out)
        self.assertNotIn('DSV41_TOKENIZER_HEAP: malloc_trim(0) when', out)      # mode trim's thread not started
        res = [l for l in out.splitlines() if l.startswith('RESULT')][-1]
        self.assertEqual(res, "RESULT 1 1 0 2 True True ['OpenAIServingBase.__init_subclass__', "
                              "'OpenAIServingChat._convert_to_internal_request', 'OpenAIServingChat._process_messages', "
                              "'TokenizerManager._tokenize_texts']", out)

    def test_every_other_value_is_inert(self):
        for heap in (None, '0', 'trim', 'mmap', 'both', 'trim3'):
            out = self.run_py(['-m', 'sglang.launch_server'], heap)
            self.assertNotIn('trim2', out, heap)
            self.assertNotIn('malloc_trim', out, heap)                  # v2.6.3 dropped the v1 trim; nothing else runs
            self.assertIn('RESULT not-armed', out)

    def test_bad_settings_do_not_arm(self):
        out = self.run_py(['-m', 'sglang.launch_server'], 'trim2', {'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '-1'})
        if IS_GLIBC:
            self.assertIn('DSV41_TOKENIZER_HEAP=trim2: not armed (DSV41_TOKENIZER_TRIM2_ settings out of range', out)
        self.assertIn('RESULT not-armed', out)

    def test_spawn_children_not_armed(self):
        out = self.run_py(['-c', 'import heap_trim2; print("RESULT", heap_trim2._STATE.get("installed"))'], 'trim2')
        self.assertIn('RESULT None', out)
        self.assertNotIn('armed', out)


SGLANG_ROOT = os.environ.get('DSV41_TRIM2_SGLANG_ROOT', '')        # <tree>/python/sglang
# tokenizer calls in the pinned tree's srt/entrypoints that the hooks do not cover, and why: none of them encodes a
# prompt that is then generated from (startup probes, decodes of outputs, the count / tokenize utility endpoints, the
# endpoints this deploy does not serve: Ollama, gRPC, Whisper; the rerank yes / no token ids)
UNHOOKED_OK = {('serving_chat.py', 'OpenAIServingChat', '__init__'),                 # startup probe of encode('')
               ('serving_chat.py', 'OpenAIServingChat', '_probe_prompt_text_round_trip'),  # v3 tree: once, in __init__
               ('serving_completions.py', 'OpenAIServingCompletion', '_get_echo_text'),        # echo decode
               ('serving_completions.py', 'OpenAIServingCompletion', '_prepare_echo_prompts'),  # echo decode
               ('serving_responses.py', 'OpenAIServingResponses', 'create_responses'),  # length of a str prompt
               # its render + encode run in the hooked chat _process_messages; this encode is the multimodal str case
               ('serving.py', 'AnthropicServing', 'handle_count_tokens'),
               # /v1/tokenize and /v1/detokenize: utility endpoints (no generation), handle_request path
               ('serving_tokenize.py', 'OpenAIServingTokenize', '_handle_non_streaming_request'),
               ('serving_tokenize.py', 'OpenAIServingTokenize', '_tokenize_chat_request'),
               ('serving_tokenize.py', 'OpenAIServingDetokenize', '_handle_non_streaming_request'),
               ('serving.py', 'OllamaServing', 'handle_chat'),                  # Ollama API: not served by this deploy
               ('grpc_bridge.py', 'RuntimeHandle', 'tokenize'),                 # gRPC bridge: not served by this deploy
               ('grpc_bridge.py', 'RuntimeHandle', 'detokenize'),
               ('whisper.py', 'WhisperAdapter', '_parse_segments'),             # transcription output decode
               ('serving_rerank.py', 'OpenAIServingRerank', '__init__')}        # startup: two fixed words' ids


_TOK_ATTRS = ('encode', 'decode', 'apply_chat_template', 'encode_batch', 'batch_decode', 'encode_special', 'encode_text')


def _tree_files(root):
    base = os.path.join(root, 'srt', 'entrypoints')
    for dp, _, fs in sorted(os.walk(base)):
        for f in sorted(fs):
            if f.endswith('.py'):
                yield os.path.relpath(os.path.join(dp, f), root)


def _encoders(root):
    """Module-level functions of srt/entrypoints that tokenize (directly or through another one), to a fixpoint:
    parallel_prompt_encode, plain_text_encode.encode_marked_prompt, chat_encoding.encode_dsv41_prompt, ..."""
    import ast
    funcs = {}
    for rel in _tree_files(root):
        for n in ast.parse(Path(root, rel).read_text()).body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(n.name, []).append(n)
    enc = {'parallel_prompt_encode', 'encode_prompt_segments'}
    while True:
        more = {name for name, defs in funcs.items() if name not in enc
                and any(_tok_call(c, enc) for d in defs for c in ast.walk(d) if isinstance(c, ast.Call))}
        if not more:
            return enc
        enc |= more


def _tok_call(n, enc):
    """A call that tokenizes: <...tokenizer... / ...backend...>.encode / decode / apply_chat_template / ..., or a call
    of a tokenizing module function (by name or as module.attr)."""
    import ast
    fn = n.func
    if isinstance(fn, ast.Name):
        return fn.id in enc
    if isinstance(fn, ast.Attribute):
        if fn.attr in enc:
            return True
        recv = ast.unparse(fn.value).lower()
        return fn.attr in _TOK_ATTRS and ('tokenizer' in recv or 'backend' in recv)
    return False


def _encode_sites(root):
    """(file, class, method, reachable from a hooked method through self.<m>() calls) for every class method (and
    every module function outside the tokenizing helpers themselves) of srt/entrypoints that tokenizes."""
    import ast
    enc = _encoders(root)
    out = []
    for rel in _tree_files(root):
        tree = ast.parse(Path(root, rel).read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in enc:
                if any(_tok_call(c, enc) for c in ast.walk(node) if isinstance(c, ast.Call)):
                    out.append((os.path.basename(rel), '<module>', node.name, False))
            if not isinstance(node, ast.ClassDef) or node.name.startswith('_'):
                continue                                  # private classes: a helper's internals, reached through
            # it (v3 tree: parallel_prompt_encode._SegmentEncoder)
            meths = {f.name: f for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
            edges = {k: {n.func.attr for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                         and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self'} for k, f in meths.items()}
            seen, todo = set(), [m for m in ht._SERVING_METHODS if m in meths]
            while todo:
                m = todo.pop()
                if m not in seen and m in meths:
                    seen.add(m)
                    todo.extend(edges[m])
            for k, f in meths.items():
                if any(_tok_call(c, enc) for c in ast.walk(f) if isinstance(c, ast.Call)):
                    out.append((os.path.basename(rel), node.name, k, k in seen))
    return out


def _serving_classes(root):
    """(module, class, bases, own hooked method names) for every class of srt/entrypoints that derives from
    OpenAIServingBase (through the tree's own classes), in file order."""
    import ast
    rows = []
    for rel in _tree_files(root):
        mod = 'sglang.' + rel[:-3].replace(os.sep, '.')
        for n in ast.parse(Path(root, rel).read_text()).body:
            if isinstance(n, ast.ClassDef):
                bases = [ast.unparse(b).split('.')[-1] for b in n.bases]
                own = [f.name for f in n.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and f.name in ht._SERVING_METHODS]
                rows.append((mod, n.name, bases, own))
    serving, grew = {'OpenAIServingBase'}, True
    while grew:
        grew = False
        for _, name, bases, _ in rows:
            if name not in serving and any(b in serving for b in bases):
                serving.add(name)
                grew = True
    return [r for r in rows if r[1] in serving and r[1] != 'OpenAIServingBase']


@unittest.skipUnless(SGLANG_ROOT and os.path.isdir(SGLANG_ROOT), 'needs DSV41_TRIM2_SGLANG_ROOT=<tree>/python/sglang')
class PinnedSglangSource(unittest.TestCase):
    """The hook targets exist in the SGLang tree the image runs, and every prompt encode there runs inside one."""

    def _cls(self, rel, name):
        import ast
        tree = ast.parse(Path(SGLANG_ROOT, rel).read_text())
        return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)

    def test_hook_targets(self):
        import ast
        base = self._cls('srt/entrypoints/openai/serving_base.py', 'OpenAIServingBase')
        self.assertIn('_convert_to_internal_request', [f.name for f in base.body if isinstance(f, ast.FunctionDef)])
        tm = self._cls('srt/managers/tokenizer_manager.py', 'TokenizerManager')
        meths = {f.name: f for f in tm.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIsInstance(meths['_tokenize_texts'], ast.AsyncFunctionDef)
        self.assertIn('self.last_receive_tstamp = real_time()', ast.unparse(meths['handle_loop']))
        self.assertGreaterEqual(ast.unparse(tm).count('self.last_receive_tstamp = real_time()'), 2)   # + its init
        chat = self._cls('srt/entrypoints/openai/serving_chat.py', 'OpenAIServingChat')
        self.assertIn('_process_messages', [f.name for f in chat.body if isinstance(f, ast.FunctionDef)])

    def test_every_prompt_encode_is_hooked(self):
        sites = _encode_sites(SGLANG_ROOT)
        self.assertTrue(any(r for *_, r in sites))
        self.assertIn(('serving_chat.py', 'OpenAIServingChat', '_process_messages', True), sites)
        ce = Path(SGLANG_ROOT, 'srt/entrypoints/openai/chat_encoding.py')
        if ce.exists() and 'def encode_dsv41_prompt(' in ce.read_text():
            # the production canary tree (f80c91a4 + the fetch script's patches): the DSV4.1 render + parallel /
            # plain-text encode run inside the hooked chat methods
            for m in ('_apply_jinja_template', '_render_and_encode_chat_template'):
                self.assertIn(('serving_chat.py', 'OpenAIServingChat', m, True), sites)
            self.assertIn('encode_dsv41_prompt', _encoders(SGLANG_ROOT))
        missing = [s[:3] for s in sites if not s[3] and s[:3] not in UNHOOKED_OK]
        self.assertEqual(missing, [], sites)
        stale = sorted(UNHOOKED_OK - {s[:3] for s in sites} - {('serving_chat.py', 'OpenAIServingChat',
                                                                   '_probe_prompt_text_round_trip')})
        self.assertEqual(stale, [])                      # every exception still names a real site of this tree

    def test_last_hooks_receipt_names_every_hooked_encode_site(self):
        """The 2026-11-06 fleet A/B judged 'hooks missing OpenAIServingChat._convert_to_internal_request' because the
        receipt was printed when serving_base was imported, before serving_chat defined (and the hook wrapped) the
        class. Stand-ins of this tree's own serving classes (names, bases, hooked methods, one module each, imported
        through the trim2 Finder as the image imports them): the LAST hooks line names every class.method that
        reaches a prompt encode, and every hooked method of the tree."""
        rows = _serving_classes(SGLANG_ROOT)
        where = {name: mod for mod, name, _, _ in rows}
        where['OpenAIServingBase'] = 'sglang.srt.entrypoints.openai.serving_base'
        files = {'sglang.srt.entrypoints.openai.serving_base': [
            'from abc import ABC, abstractmethod',
            'class OpenAIServingBase(ABC):',
            '    @abstractmethod',
            '    def _convert_to_internal_request(self, request, raw_request=None):',
            '        pass']}
        for mod, name, bases, own in rows:
            lines = files.setdefault(mod, [])
            for b in bases:
                if b in where and where[b] != mod:
                    lines.insert(0, f'from {where[b]} import {b}')
            lines.append(f'class {name}({", ".join(b for b in bases if b in where) or "object"}):')
            lines += [f'    def {m}(self, *a, **k):\n        return None' for m in own] or ['    pass']
        with tempfile.TemporaryDirectory() as tmp:
            for mod, lines in files.items():
                path = Path(tmp, *mod.split('.')).with_suffix('.py')
                for d in [path.parent] + list(path.parent.parents):
                    if d == Path(tmp):
                        break
                    d.mkdir(parents=True, exist_ok=True)
                    (d / '__init__.py').touch()
                path.write_text('\n'.join(lines) + '\n')
            code = textwrap.dedent(f'''
                import sys; sys.path[:0] = [{tmp!r}, {str(ADAPTER)!r}]
                import heap_trim2 as ht
                ht._STATE['activity'] = ht.Activity()
                sys.meta_path.insert(0, ht.Finder())
                import importlib
                for m in {sorted(m for m in files if m != 'sglang.srt.entrypoints.openai.serving_base')!r}:
                    importlib.import_module(m)
            ''')
            r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        lines = [l for l in r.stdout.splitlines() if l.startswith('DSV41_TOKENIZER_HEAP=trim2: hooks ')]
        self.assertTrue(lines, r.stdout)
        last = set(lines[-1].split(' hooks ', 1)[1].split(', '))
        want = {f'{name}.{m}' for _, name, _, own in rows for m in own} | {'OpenAIServingBase.__init_subclass__'}
        self.assertEqual(sorted(want - last), [], lines[-1])
        for f, cls, meth, reach in _encode_sites(SGLANG_ROOT):
            if reach:
                own = next(o for _, n, _, o in rows if n == cls)
                self.assertTrue(any(f'{cls}.{m}' in last for m in own), (f, cls, meth))
        self.assertIn('OpenAIServingChat._convert_to_internal_request', last)
        self.assertIn('OpenAIServingChat._process_messages', last)


HAVE_SGLANG = importlib.util.find_spec('sglang') is not None


@unittest.skipUnless(HAVE_SGLANG, 'needs the engine image (SGLang importable)')
class LiveSglang(unittest.TestCase):
    def test_hooks_land_on_the_real_classes(self):
        code = textwrap.dedent(f'''
            import sys; sys.path.insert(0, {str(ADAPTER)!r})
            import heap_trim2 as ht
            act = ht.Activity(); ht._STATE["activity"] = act
            sys.meta_path.insert(0, ht.Finder())
            import sglang.srt.entrypoints.openai.serving_chat, sglang.srt.entrypoints.openai.serving_completions
            import sglang.srt.managers.tokenizer_manager as tm
            print("HOOKS", ",".join(sorted(act.hooks)), getattr(tm.TokenizerManager.__init__, ht._MARK, False))
        ''')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        line = [l for l in r.stdout.splitlines() if l.startswith('HOOKS')][-1]
        for h in ('OpenAIServingBase.__init_subclass__', 'OpenAIServingChat._convert_to_internal_request',
                  'OpenAIServingChat._process_messages', 'OpenAIServingCompletion._convert_to_internal_request',
                  'TokenizerManager._tokenize_texts'):
            self.assertIn(h, line)
        self.assertTrue(line.endswith(' True'), line)


@unittest.skipUnless(IS_GLIBC, 'glibc only (Linux image)')
class RealGlibc(unittest.TestCase):
    def test_returns_a_fragmented_heap_only_when_idle(self):
        code = textwrap.dedent(f'''
            import sys, time, ctypes; sys.path.insert(0, {str(ADAPTER)!r})
            import heap_trim2 as ht
            lib = ht._libc(); lib.malloc.restype = ctypes.c_void_p; lib.free.argtypes = [ctypes.c_void_p]
            env = {{'DSV41_TOKENIZER_TRIM2_IDLE_S': '0.3', 'DSV41_TOKENIZER_TRIM2_MIN_MIB': '128', 'DSV41_TOKENIZER_TRIM2_MAX_BUSY_S': '0',
                   'DSV41_TOKENIZER_TRIM2_MIN_INTERVAL_S': '0'}}
            act = ht.Activity(); act.hooks.append('test'); lines = []
            tr = ht.Trimmer(ht.Config(env), act, log=lines.append)
            a = ht.rss_anon_kb()
            act.enter()
            ps = [lib.malloc(2000) for _ in range(300000)]
            [lib.free(p) for i, p in enumerate(ps) if i % 64]
            b = ht.rss_anon_kb()
            busy = tr.step(); act.exit(); early = tr.step(); time.sleep(0.35); done = tr.step()
            c = ht.rss_anon_kb(); again = tr.step()
            print(busy, early, done, again, a >> 10, b >> 10, c >> 10, round(tr.hold_max, 1))
        ''')
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        busy, early, done, again, a, b, c, ms = r.stdout.split()
        a, b, c = int(a), int(b), int(c)
        self.assertEqual((busy, early, done, again), ('busy', 'busy', 'trim', 'small'))
        self.assertGreater(b - a, 400)
        self.assertLess(c - a, 0.25 * (b - a))
        self.assertLess(float(ms), 200)



@unittest.skipUnless((HERE.parent / '.env.tp4.example').exists(), 'not a repo checkout (in an image)')
class Profile(unittest.TestCase):
    """v2.6.5: the production line of .env.tp4.example ends with DSV41_TOKENIZER_HEAP=trim2 (after the v41n keys), no
    other profile line sets it, no tuning knob of it is in any profile (the code defaults are the measured ones)."""

    def _lines(self):
        return (HERE.parent / '.env.tp4.example').read_text().splitlines()

    def _production_line(self):
        lines = self._lines()
        i = next(n for n, line in enumerate(lines) if line.startswith('# Production settings, together with the line'))
        self.assertTrue(lines[i - 1].startswith('#EXTRA_CONTAINER_ENV="'), lines[i - 1][:60])
        return lines[i - 1]

    def test_production_line_ends_with_trim2_once(self):
        items = self._production_line().split('"')[1].split()
        self.assertEqual(items[-2:], ['DSV41_V41C_READER=native', 'DSV41_TOKENIZER_HEAP=trim2'])
        self.assertEqual(sum(x.split('=')[0] == 'DSV41_TOKENIZER_HEAP' for x in items), 1)

    def test_only_the_production_line_sets_it(self):
        on = [line for line in self._lines() if 'DSV41_TOKENIZER_HEAP=' in line and not line.startswith('# ')]
        self.assertEqual(on, [self._production_line()])
        knobs = [line for line in self._lines() if 'DSV41_TOKENIZER_TRIM2_' in line and not line.startswith('# ')]
        self.assertEqual(knobs, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
