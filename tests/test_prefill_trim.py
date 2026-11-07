#!/usr/bin/env python3
"""DSV41_PREFILL_TRIM (adapter/prefill_empty_cache.py): the working-set-preserving prefill trim, on a stand-in allocator.

CPU only (no GPU, torch optional): a small model of PyTorch's caching allocator (segments of blocks, the default pool
on the current stream, private pools, empty_cache releasing fully free default-pool segments) drives the trim's
decisions: no trim under the working set + DSV41_PREFILL_TRIM_GIB, the largest free blocks covering the working set
held through empty_cache, the 10 s rate limit, private-pool / other-stream / small blocks never held, the hold stops
when the allocator had to grow, working set 0 = plain empty_cache. The hook: off by default = the per-chunk
empty_cache exactly as before (after a long extend only), DSV41_PREFILL_EMPTY_CACHE_TOKENS=0 installs nothing,
bookkeeping errors never fail a forward, the forward's own errors propagate. A chunked prefill whose scratch grows
every chunk: per-chunk path = one cudaMalloc per chunk and nothing cached; trim path = the scratch reused, cached-free
bounded by the working set + 1 GiB + the growth inside one trim interval.
"""
import importlib.util
import itertools
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "adapter"))
if importlib.util.find_spec("torch") is None:          # offline CI: the module only needs the name at import
    sys.modules["torch"] = types.ModuleType("torch")
import prefill_empty_cache as P  # noqa: E402

GIB, MIB = 1 << 30, 1 << 20


class FakeCuda:
    """Segments of blocks; alloc() best-fits an inactive default-pool block on the current stream (whole block, no
    split) or cudaMallocs a new segment; empty_cache() releases fully inactive default-pool segments."""

    STREAM = 7

    def __init__(self):
        self.segments = []
        self.peak = 0
        self.mallocs = 0
        self.empties = 0
        self._ids = itertools.count()

    def add_segment(self, sizes, state="inactive", pool=(0, 0), stream=STREAM):
        seg = {"segment_pool_id": pool, "stream": stream, "id": next(self._ids),
               "blocks": [{"size": s, "state": state} for s in sizes]}
        self.segments.append(seg)
        return seg

    # the torch.cuda surface the trim reads
    def memory_allocated(self):
        return sum(b["size"] for s in self.segments for b in s["blocks"] if b["state"] == "active_allocated")

    def memory_reserved(self):
        return sum(b["size"] for s in self.segments for b in s["blocks"])

    def reset_peak_memory_stats(self):
        self.peak = self.memory_allocated()

    def max_memory_allocated(self):
        return self.peak

    def memory_snapshot(self):
        return [{"segment_pool_id": s["segment_pool_id"], "stream": s["stream"],
                 "blocks": [dict(b) for b in s["blocks"]]} for s in self.segments]

    def current_stream(self):
        return SimpleNamespace(cuda_stream=self.STREAM)

    def empty_cache(self):
        self.empties += 1
        self.segments = [s for s in self.segments if s["segment_pool_id"] != (0, 0)
                         or any(b["state"] != "inactive" for b in s["blocks"])]

    # the model's own allocation API
    def alloc(self, size):
        free = [(b["size"], s["id"], b) for s in self.segments
                if s["segment_pool_id"] == (0, 0) and s["stream"] == self.STREAM
                for b in s["blocks"] if b["state"] == "inactive" and b["size"] >= size]
        if free:
            block = min(free, key=lambda t: (t[0], t[1]))[2]
        else:
            self.mallocs += 1
            block = self.add_segment([size])["blocks"][0]
        block["state"] = "active_allocated"
        self.peak = max(self.peak, self.memory_allocated())
        return block

    def free(self, block):
        block["state"] = "inactive"


class Handle:
    def __init__(self, cuda, size):
        self.cuda, self.block = cuda, cuda.alloc(size)

    def __del__(self):
        self.cuda.free(self.block)


def make_trim(cuda, clock, **kw):
    return P.Trim(cuda=cuda, clock=clock, hold=lambda size: Handle(cuda, size), **kw)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TrimDecisions(unittest.TestCase):
    def setUp(self):
        self.cuda, self.clock = FakeCuda(), Clock()
        self.cuda.add_segment([4 * GIB], state="active_allocated")       # resident weights / pools

    def test_no_trim_within_working_set_plus_slack(self):
        t = make_trim(self.cuda, self.clock)
        t.working = 2 * GIB
        self.cuda.add_segment([2 * GIB + GIB])                          # exactly working + 1 GiB free
        self.assertIsNone(t.after())
        self.assertEqual(self.cuda.empties, 0)

    def test_trim_keeps_largest_blocks_covering_the_working_set(self):
        t = make_trim(self.cuda, self.clock)
        t.working = 2 * GIB
        for _ in range(3):
            self.cuda.add_segment([GIB])
        self.cuda.add_segment([100 * MIB] * 20)                         # mixed-batch waste
        kept = t.after()
        self.assertEqual(kept, 2 * GIB)
        self.assertEqual(self.cuda.empties, 1)
        self.assertEqual(self.cuda.memory_reserved() - self.cuda.memory_allocated(), 2 * GIB)
        self.assertEqual(self.cuda.mallocs, 0)                          # holding reused cached blocks only
        self.assertEqual(t.trims, 1)

    def test_rate_limit(self):
        t = make_trim(self.cuda, self.clock, every_s=10)
        t.working = GIB
        self.cuda.add_segment([GIB] * 4)
        self.assertIsNotNone(t.after())
        self.cuda.add_segment([GIB] * 4)
        self.clock.t += 9.9
        self.assertIsNone(t.after())
        self.clock.t += 0.2
        self.assertIsNotNone(t.after())
        self.assertEqual(t.trims, 2)

    def test_private_pools_other_streams_and_small_blocks_not_held(self):
        t = make_trim(self.cuda, self.clock)
        t.working = 3 * GIB
        self.cuda.add_segment([2 * GIB], pool=(1, 5))                   # a CUDA-graph private pool
        self.cuda.add_segment([2 * GIB], stream=99)                     # a side stream
        self.cuda.add_segment([32 * MIB] * 40)                          # below the 64 MiB hold minimum
        kept = t.after()
        self.assertEqual(kept, 0)
        # empty_cache released the side-stream and small segments; the private pool stays (as in PyTorch)
        self.assertEqual(self.cuda.memory_reserved() - self.cuda.memory_allocated(), 2 * GIB)

    def test_hold_stops_when_the_reserve_grows(self):
        t = P.Trim(cuda=self.cuda, clock=self.clock,
                   hold=lambda size: Handle(self.cuda, size + 1))         # never fits: forces a cudaMalloc
        t.working = 2 * GIB
        self.cuda.add_segment([GIB] * 4)
        self.assertEqual(t.after(), 0)
        self.assertEqual(self.cuda.mallocs, 1)                          # stopped at the first growth

    def test_zero_working_set_is_plain_empty_cache(self):
        t = make_trim(self.cuda, self.clock)
        self.cuda.add_segment([2 * GIB])
        self.assertEqual(t.after(), 0)
        self.assertEqual(self.cuda.memory_reserved(), self.cuda.memory_allocated())

    def test_working_set_is_the_long_forwards_transient_peak(self):
        t = make_trim(self.cuda, self.clock)
        t.before_long()
        a = self.cuda.alloc(3 * GIB)
        b = self.cuda.alloc(GIB)
        self.cuda.free(a)
        self.cuda.free(b)
        t.after_long()
        self.assertEqual(t.working, 4 * GIB)


class _Lens:
    def __init__(self, xs):
        self.xs = xs

    def max(self):
        return max(self.xs)

    def numel(self):
        return len(self.xs)


def fb(extend, lens):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: extend),
                           seq_lens_cpu=_Lens(lens))


class Hook(unittest.TestCase):
    def setUp(self):
        self.cuda = FakeCuda()
        self.cuda.add_segment([4 * GIB], state="active_allocated")
        self.saved = P.torch
        P.torch = SimpleNamespace(cuda=self.cuda, uint8="u8",
                                  empty=lambda size, dtype=None, device=None: Handle(self.cuda, size))

    def tearDown(self):
        P.torch = self.saved

    def _module(self, body=None):
        calls = []

        def forward(self_, forward_batch, *a, **k):
            calls.append(forward_batch)
            if body is not None:
                return body(forward_batch)
            return "out"
        return SimpleNamespace(ModelRunner=type("MR", (), {"forward": forward})), calls

    def test_default_is_the_per_chunk_empty_cache(self):
        mod, calls = self._module()
        orig = mod.ModelRunner.forward
        P.install(mod, env={})
        self.assertIsNot(mod.ModelRunner.forward, orig)
        mr = mod.ModelRunner()
        self.assertEqual(mr.forward(fb(True, [9000, 10])), "out")
        self.assertEqual(self.cuda.empties, 1)
        mr.forward(fb(True, [8191]))                                    # shorter than the threshold
        mr.forward(fb(False, [100000]))                                 # decode / verify
        self.assertEqual(self.cuda.empties, 1)
        self.assertEqual(len(calls), 3)

    def test_trim_values(self):
        for v, on in (("1", True), ("on", True), ("true", True), ("0", False), ("", False), ("2", False)):
            self.assertEqual(P.trim_enabled({"DSV41_PREFILL_TRIM": v}), on, v)
        self.assertFalse(P.trim_enabled({}))

    def test_threshold_zero_installs_nothing(self):
        mod, _ = self._module()
        orig = mod.ModelRunner.forward
        P.install(mod, env={"DSV41_PREFILL_EMPTY_CACHE_TOKENS": "0", "DSV41_PREFILL_TRIM": "1"})
        self.assertIs(mod.ModelRunner.forward, orig)

    def test_trim_hook_never_fails_a_forward(self):
        mod, _ = self._module()
        P.install(mod, env={"DSV41_PREFILL_TRIM": "1"})
        mr = mod.ModelRunner()
        self.cuda.memory_reserved = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        self.cuda.reset_peak_memory_stats = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        with self.assertLogs(P.logger, "WARNING"):
            self.assertEqual(mr.forward(fb(True, [9000])), "out")

    def test_forward_errors_propagate(self):
        def body(_):
            raise ValueError("forward failed")
        for env in ({}, {"DSV41_PREFILL_TRIM": "1"}):
            mod, _ = self._module(body)
            P.install(mod, env=env)
            with self.assertRaises(ValueError):
                mod.ModelRunner().forward(fb(True, [9000]))

    def _chunked_prefill(self, env, chunks=24, chunk_s=1.5):
        """Each chunk's scratch is a bit larger than the last (the prefix grows) and is freed after the forward."""
        cuda = self.cuda
        clock = Clock()

        def body(batch):
            if batch.forward_mode.is_extend_without_speculative():
                n = int(batch.seq_lens_cpu.max()) // 8192
                scratch = [cuda.alloc(GIB + n * 16 * MIB), cuda.alloc(512 * MIB + n * 8 * MIB)]
                for s in scratch:
                    cuda.free(s)
            clock.t += chunk_s
            return "out"
        mod, _ = self._module(body)
        saved = P.time.monotonic
        P.time.monotonic = clock
        try:
            P.install(mod, env=env)
        finally:
            P.time.monotonic = saved
        mr = mod.ModelRunner()
        worst = 0
        for k in range(1, chunks + 1):
            mr.forward(fb(True, [8192 * k]))
            mr.forward(fb(False, [8192 * k]))                            # a decode step in between
            worst = max(worst, cuda.memory_reserved() - cuda.memory_allocated())
        return worst, cuda.mallocs

    def test_chunked_prefill_per_chunk_vs_trim(self):
        stock_free, stock_mallocs = self._chunked_prefill({})
        self.setUp()
        trim_free, trim_mallocs = self._chunked_prefill({"DSV41_PREFILL_TRIM": "1"})
        print(f"\n  24 chunks: per-chunk path {stock_mallocs} cudaMallocs, max cached-free {stock_free / GIB:.2f} GiB; "
              f"trim {trim_mallocs} cudaMallocs, max cached-free {trim_free / GIB:.2f} GiB")
        self.assertEqual(stock_free, 0)
        self.assertEqual(stock_mallocs, 2 * 24)
        self.assertLess(trim_mallocs, stock_mallocs)
        working = GIB + 24 * 16 * MIB + 512 * MIB + 24 * 8 * MIB
        growth = 7 * working                                             # <= one 10 s interval of 1.5 s chunks
        self.assertLessEqual(trim_free, working + GIB + growth)


if __name__ == "__main__":
    unittest.main(verbosity=2)
