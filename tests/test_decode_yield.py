"""DSV41_DECODE_YIELD (adapter/decode_yield.py) on a simulated scheduler whose get_next_batch_to_run,
_should_defer_prefill and _arm_prefill_decode_interval are lifted from the real pinned SGLang source.

The simulator drives the lifted functions round after round like the engine's event loop (one batch per round, model
time advanced by a cost model): a long prompt prefills in chunks while decode streams run; get_new_batch_prefill,
update_running_batch and the batch objects are mocks with the engine's semantics for this path (the next chunk of the
in-flight prompt, the final chunk clearing chunked_req, finished streams filtered out, the last extend batch merged
into the running batch minus its in-flight chunk). Checked:
- off (not installed) and an installed adapter with share 0 give the stock trace round for round;
- the long prompt's chunks (boundaries, order, batch contents) are the same with the yield on;
- with share 50 the running streams get ~50 % of the model time during the long prefill (2 streams, 300K-class
  contexts, a 500K prompt) instead of ~7 %, the prompt's time to its last chunk grows by the predicted factor, and
  no round is idle while a stream is waiting;
- a lone long prompt (no stream running) and a final chunk arm nothing; a yield ends at once when the streams finish
  or the chunked prompt disappears; a batch without a chunked request (a solo short pass) arms nothing;
- the decisions are rank-invariant: four simulated ranks with different clocks, environments and host readings give
  the same trace, and the decision functions never call a clock;
- env parsing, the source pins of the staged tree, refusal on drift (nothing mutated), the sitecustomize gate.
Needs the staged pinned scheduler source (DSV41_DECODE_YIELD_SOURCE_ROOT, else DSV41_PREFILL_BYPASS_SOURCE_ROOT, else
runtime/sglang-canary/python/sglang/srt/managers); the pin / simulator tests skip without it.
"""
import ast
import importlib
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))
_SRC = (os.environ.get('DSV41_DECODE_YIELD_SOURCE_ROOT') or os.environ.get('DSV41_PREFILL_BYPASS_SOURCE_ROOT')
        or str(ROOT / 'runtime/sglang-canary/python/sglang/srt/managers'))
STAGED = Path(_SRC)
SUPPORTED = (STAGED / 'scheduler.py').is_file()
LIFTED = ('get_next_batch_to_run', '_should_defer_prefill', '_arm_prefill_decode_interval')

import decode_yield as dy  # noqa: E402


# ---------------------------------------------------------------- mocks with the engine's semantics on this path
class FM:
    def __init__(self, name):
        self.name = name

    def is_extend(self):
        return self.name in ('EXTEND', 'MIXED', 'TARGET_VERIFY')


EXTEND, DECODE = FM('EXTEND'), FM('DECODE')


class Req:
    def __init__(self, rid, total, max_new=1 << 30, accept=3):
        self.rid = rid
        self.total = total
        self.prefix = 0
        self.extend_range = types.SimpleNamespace(start=0, end=0, length=0)
        self.prefix_indices = []
        self.out = 0
        self.max_new = max_new
        self.accept = accept
        self.first_token_t = None
        self.step_times = []

    def finished(self):
        return self.out >= self.max_new


class Batch:
    def __init__(self, reqs=None, batch_is_full=False, mode=DECODE, chunked_req=None, extend_num_tokens=None):
        self.reqs = list(reqs or [])
        self.batch_is_full = batch_is_full
        self.forward_mode = mode
        self.chunked_req = chunked_req
        self.extend_num_tokens = extend_num_tokens
        self.is_prefill_only = False

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, chunked_req_to_exclude=None):
        ex = set(map(id, chunked_req_to_exclude or []))
        self.reqs = [r for r in self.reqs if id(r) not in ex and not r.finished()]

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


class Plan:
    def __init__(self, batch_to_run, running_batch):
        self.batch_to_run = batch_to_run
        self.running_batch = running_batch


def lift(cls, filename, methods, namespace):
    """Compile the pinned methods at their original line numbers (inspect.getsource then reads the pinned file)."""
    tree = ast.parse(filename.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    for method in methods:
        node = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == method)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],
                                                 level=0), node], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(filename), 'exec'), namespace)
        setattr(cls, method, namespace[method])


def scheduler_class():
    def stage(_name):
        def deco(fn):
            return fn
        return deco

    class Scheduler:
        pass
    namespace = dict(scheduler_stage_method=stage, SCHEDULER_STAGE_GET_NEXT_BATCH='get_next_batch',
                     ScheduleBatch=Batch, NextBatchPlan=Plan, set_schedule_time_batch=lambda b: None, time=time)
    lift(Scheduler, STAGED / 'scheduler.py', LIFTED, namespace)
    return Scheduler


class Sched:
    """Instance-side mocks of the scheduler pieces the lifted functions call."""

    def __init__(self, cls, chunk, interval, min_bs=0, solo_shorts=None):
        self.s = s = cls()
        s.chunk = chunk
        s.prefill_decode_interval = interval
        s._prefill_decode_interval_remaining = 0
        s.prefill_decode_interval_min_bs = min_bs
        s.enable_fpm = False
        s.dllm_config = None
        s.enable_hisparse = False
        s.require_mlp_sync = False
        s.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)
        s.chunked_req = None
        s.waiting = []
        s.shorts = list(solo_shorts or [])
        s.stashes = []
        s.process_pending_chunked_abort = lambda: None
        s._process_hicache_events = lambda: None
        s.dp_attn_adapter = types.SimpleNamespace(maybe_prepare_mlp_sync_batch=lambda b, need_sync=False: b,
                                                  maybe_convert_decode_to_extend=lambda b: b)
        s.ngram_embedding_manager = types.SimpleNamespace(prepare_for_forward=lambda b, chunked_req=None: b)

        def stash(req):
            s.stashes.append((req.rid, req.extend_range.end))
            req.prefix = req.extend_range.end
            req.prefix_indices = [0] * req.prefix
        s.stash_chunked_request = stash

        def update_running_batch(rb):
            # prepare_for_decode: the batch decodes; a batch that was the last prefill keeps its stale chunked_req
            # attribute, as the engine's does (convert_decode_to_extend's comment)
            rb.filter_batch()
            rb.forward_mode = DECODE
            return rb
        s.update_running_batch = update_running_batch

        def get_new_batch_prefill(running_batch):
            # a solo short pass (DSV41_PREFILL_BYPASS_SOLO): one short request alone, the chunk parked, built without
            # the chunked request; then the long prompt resumes
            if s.chunked_req is not None and s.shorts and not getattr(s.chunked_req, '_parked', False):
                short = s.shorts.pop(0)
                short.extend_range = types.SimpleNamespace(start=0, end=short.total, length=short.total)
                s.chunked_req._parked = True
                s.chunked_req.extend_range = types.SimpleNamespace(start=s.chunked_req.prefix,
                                                                   end=s.chunked_req.prefix, length=0)
                return Plan(Batch([short], mode=EXTEND, chunked_req=None, extend_num_tokens=short.total),
                            running_batch)
            req = s.chunked_req
            if req is None:
                if not s.waiting:
                    return Plan(None, running_batch)
                req = s.waiting.pop(0)
            req._parked = False
            start = req.prefix
            end = min(start + s.chunk, req.total)
            req.extend_range = types.SimpleNamespace(start=start, end=end, length=end - start)
            s.chunked_req = req if end < req.total else None
            return Plan(Batch([req], mode=EXTEND, chunked_req=s.chunked_req, extend_num_tokens=end - start),
                        running_batch)
        s.get_new_batch_prefill = get_new_batch_prefill


def true_chunk_us(tokens, prefix):
    return dy.chunk_us(tokens, prefix)


def true_step_us(bs):
    return dy.step_us(bs)


def simulate(cls, *, long_total=500_000, chunk=8192, streams=2, stream_tokens=1 << 30, interval=2, min_bs=0,
             long_rid='L', long_at_us=0, solo_shorts=None, horizon_us=10**9, chunk_cost=true_chunk_us,
             step_cost=true_step_us, abort_long_after_chunks=None):
    """Event loop: one batch per round. Returns a dict with the trace and timings (microseconds of model time)."""
    sim = Sched(cls, chunk, interval, min_bs, solo_shorts)
    s = sim.s
    t = 0
    running = Batch([Req(f'D{i}', 300_000 + i, max_new=stream_tokens) for i in range(streams)])
    for r in running.reqs:
        r.prefix = r.total
    long = Req(long_rid, long_total, max_new=64)
    sent = False
    last = None
    trace, idle_while_waiting, chunks = [], 0, []
    window = None
    decode_us_in_window = 0
    while t < horizon_us:
        if not sent and t >= long_at_us:
            s.waiting.append(long)
            sent = True
            window = [t, None]
        plan = s.get_next_batch_to_run(running, last)
        running = plan.running_batch
        b = plan.batch_to_run
        if b is None:
            if sent and window[1] is None and (s.chunked_req is not None or s.waiting):
                idle_while_waiting += 1
            trace.append(('idle',))
            t += 1000
            last = None
            if running.is_empty() and s.chunked_req is None and not s.waiting and sent and long.finished():
                break
            if running.is_empty() and s.chunked_req is None and not s.waiting and sent and window[1] is not None \
                    and not any(not r.finished() for r in [long]):
                break
            continue
        if b.forward_mode.name == 'EXTEND':
            r0 = b.reqs[0]
            trace.append(('chunk', tuple(r.rid for r in b.reqs), r0.extend_range.start, r0.extend_range.end,
                          getattr(b.chunked_req, 'rid', None)))
            if r0 is long:
                chunks.append((r0.extend_range.start, r0.extend_range.end))
            t += chunk_cost(b.extend_num_tokens, r0.extend_range.start)
            for r in b.reqs:
                if r.extend_range.end >= r.total and r.first_token_t is None:
                    r.first_token_t = t
                    r.out = 1
                    r.prefix = r.total
                    if r is long:
                        window[1] = t
            if abort_long_after_chunks is not None and r0 is long and len(chunks) >= abort_long_after_chunks:
                s.chunked_req = None            # the engine's abort path clears the in-flight chunk
                window[1] = t
                long.max_new = 0
        else:
            bs = b.batch_size()
            trace.append(('decode', bs))
            dt = step_cost(bs)
            t += dt
            if window is not None and window[1] is None and window[0] <= t:
                decode_us_in_window += dt
            for r in b.reqs:
                r.out += r.accept
                r.step_times.append(t)
        last = b
        if sent and long.finished() and window[1] is not None and all(r.finished() for r in running.reqs) \
                and s.chunked_req is None:
            break
        if sent and window[1] is not None and t > window[1] + 5_000_000:
            break
    w = (window[1] - window[0]) if window and window[1] else None
    streams_steps = [len([x for x in r.step_times if window[0] <= x <= (window[1] or t)])
                     for r in running.reqs if r.rid.startswith('D')] if window else []
    return dict(trace=trace, chunks=chunks, ttft_us=w, decode_share=(decode_us_in_window / w) if w else None,
                idle_while_waiting=idle_while_waiting, stream_steps=streams_steps, stashes=s.stashes, sched=s)


def install(cls, value, rid_arms=None):
    dy.STATS = dy._Stats()
    module = types.SimpleNamespace(Scheduler=cls)
    env = {dy.ENV: value}
    if rid_arms is not None:
        env[dy.ENV_RID_ARMS] = rid_arms
    return dy.install(module, env=env)


# ---------------------------------------------------------------- pure functions (no staged source needed)
class PolicyTests(unittest.TestCase):
    def test_parse(self):
        for v in (None, '', '0', 'off', 'false', 'no', ' 0 '):
            self.assertIsNone(dy.parse_pct(v))
        self.assertEqual(dy.parse_pct('50'), 50)
        self.assertEqual(dy.parse_pct(' 1 '), 1)
        self.assertEqual(dy.parse_pct('90'), 90)
        for v in ('91', '100', '-5', 'abc', '0.5', '50%'):
            with self.assertRaises(ValueError):
                dy.parse_pct(v)

    def test_install_refuses_bad_values_without_touching_the_class(self):
        cls = type('S', (), {'get_next_batch_to_run': lambda self, *a: None,
                             '_should_defer_prefill': lambda self, rb: False,
                             '_arm_prefill_decode_interval': lambda self, b: None})
        before = dict(vars(cls))
        for v in ('abc', '95', '50%'):
            self.assertFalse(dy.install(types.SimpleNamespace(Scheduler=cls), env={dy.ENV: v}))
        self.assertFalse(dy.install(types.SimpleNamespace(Scheduler=cls), env={dy.ENV: '50', dy.ENV_RID_ARMS: '2'}))
        self.assertFalse(dy.install(types.SimpleNamespace(Scheduler=cls), env={}))
        # source drift: the toy methods are not the pinned engine
        self.assertFalse(dy.install(types.SimpleNamespace(Scheduler=cls), env={dy.ENV: '50'}))
        self.assertEqual(dict(vars(cls)), before)

    def test_budget_is_integer_and_bounded(self):
        for pct in (1, 33, 50, 67, 90):
            for bs in (1, 2, 4, 8, 16):
                for prefix in (0, 300_000, 999_999):
                    b = dy.yield_budget_us(pct, 8192, prefix, bs, 2)
                    self.assertIsInstance(b, int)
                    self.assertGreaterEqual(b, 2 * dy.step_us(bs))
                    self.assertLessEqual(b, dy.MAX_STEPS * dy.step_us(bs))
        # share 50 = the chunk's own modelled time (ceil)
        self.assertEqual(dy.yield_budget_us(50, 8192, 0, 2, 2), dy.chunk_us(8192, 0))
        self.assertEqual(dy.chunk_us(8192, 0), 1_354_500)
        self.assertEqual(dy.chunk_us(8192, 500_000), 1_354_500 + 243_500)
        self.assertEqual(dy.step_us(2), 56_700)

    def test_share_from_rid_only_with_test_switch(self):
        self.assertEqual(dy.share_for('dyield-s33-arm', 50, True), 33)
        self.assertEqual(dy.share_for('dyield-s0-arm', 50, True), 0)
        self.assertEqual(dy.share_for('dyield-s33-arm', 50, False), 50)
        self.assertEqual(dy.share_for('dyield-s99-arm', 50, True), 50)
        self.assertEqual(dy.share_for('other', 50, True), 50)
        self.assertEqual(dy.share_for(None, 50, True), 50)

    def test_sitecustomize_gate_checked_before_import(self):
        text = (ROOT / 'adapter/sitecustomize.py').read_text()
        i = text.index("os.environ.get('DSV41_DECODE_YIELD', '0').strip().lower() not in ('0', '', 'off', 'false', "
                       "'no')")
        j = text.index('from decode_yield import install as install_decode_yield')
        k = text.index("elif module.__name__ == 'sglang.srt.managers.scheduler':")
        self.assertTrue(k < i < j)
        # after the prefill bypass in the same hook
        self.assertTrue(text.index('install_prefill_bypass(module)') < j)

    def test_no_clock_or_host_reads_in_the_module(self):
        tree = ast.parse((ROOT / 'adapter/decode_yield.py').read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in getattr(n, 'names', [])}
        imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        for bad in ('time', 'random', 'psutil', 'datetime', 'socket', 'torch'):
            self.assertNotIn(bad, imported)
        floats = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
        self.assertEqual(floats, [])
        # os.environ is read only by install()/requested() (once, at import)
        for fn in ('arm_after', 'defer_step', 'yield_budget_us', 'chunk_us', 'step_us', 'share_for'):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
            names = {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}
            self.assertNotIn('os', names, fn)


# ---------------------------------------------------------------- the lifted pinned scheduler
@unittest.skipUnless(SUPPORTED, 'needs the staged pinned scheduler source (DSV41_DECODE_YIELD_SOURCE_ROOT)')
class SimulatedSchedulerTests(unittest.TestCase):
    def fresh(self):
        return scheduler_class()

    def test_pins_match_the_staged_source(self):
        cls = self.fresh()
        self.assertEqual(dy.pin_drift(cls), [])

    def test_drift_refuses_and_mutates_nothing(self):
        cls = self.fresh()
        before = {k: getattr(cls, k) for k in LIFTED}
        pins = dict(dy.PINS)
        pins['Scheduler._should_defer_prefill'] = '0' * 64
        with patch.object(dy, 'PINS', pins):
            self.assertFalse(install(cls, '50'))
        self.assertEqual({k: getattr(cls, k) for k in LIFTED}, before)
        self.assertFalse(dy.STATS.installed)

    def test_off_and_share0_are_the_stock_trace(self):
        stock = simulate(self.fresh(), long_rid='dyield-s0-L')
        cls = self.fresh()
        self.assertTrue(install(cls, '50', rid_arms='1'))
        s0 = simulate(cls, long_rid='dyield-s0-L')
        self.assertEqual(stock['trace'], s0['trace'])
        self.assertEqual(dy.STATS.yields, 0)
        # stock production cadence: exactly `interval` decode rounds between two chunks of the long prompt
        tr = stock['trace']
        firsts = [i for i, x in enumerate(tr) if x[0] == 'chunk']
        gaps = {sum(1 for x in tr[a + 1:b] if x[0] == 'decode') for a, b in zip(firsts, firsts[1:])}
        self.assertEqual(gaps, {2})

    def test_chunks_unchanged_and_share_reached(self):
        stock = simulate(self.fresh())
        cls = self.fresh()
        self.assertTrue(install(cls, '50'))
        on = simulate(cls)
        self.assertEqual(stock['chunks'], on['chunks'])
        self.assertEqual([x for x in stock['trace'] if x[0] == 'chunk'], [x for x in on['trace'] if x[0] == 'chunk'])
        self.assertEqual(stock['stashes'], on['stashes'])
        self.assertEqual(on['idle_while_waiting'], 0)
        self.assertLess(stock['decode_share'], 0.09)
        self.assertGreater(on['decode_share'], 0.48)
        self.assertLess(on['decode_share'], 0.53)
        # streams: >= 6x the decode steps of stock during the long prompt; its TTFT grows by ~1 / (1 - 0.5) of the
        # chunk time (the stock already spends ~7 % on 2 steps per chunk)
        self.assertGreater(min(on['stream_steps']), 6 * max(stock['stream_steps']))
        ratio = on['ttft_us'] / stock['ttft_us']
        self.assertGreater(ratio, 1.80)
        self.assertLess(ratio, 1.90)
        self.assertGreater(dy.STATS.yields, 0)
        self.assertEqual(dy.STATS.yields, len(on['chunks']) - 1)     # every non-final chunk, not the final one

    def test_rid_arms_select_the_share_in_one_boot(self):
        cls = self.fresh()
        self.assertTrue(install(cls, '50', rid_arms='1'))
        s33 = simulate(cls, long_rid='dyield-s33-L')
        s50 = simulate(cls, long_rid='dyield-s50-L')
        self.assertAlmostEqual(s33['decode_share'], 0.33, delta=0.03)
        self.assertAlmostEqual(s50['decode_share'], 0.50, delta=0.03)
        self.assertEqual(s33['chunks'], s50['chunks'])

    def test_lone_long_prompt_is_untouched(self):
        stock = simulate(self.fresh(), streams=0)
        cls = self.fresh()
        self.assertTrue(install(cls, '50'))
        on = simulate(cls, streams=0)
        self.assertEqual(stock['trace'], on['trace'])
        self.assertEqual(dy.STATS.yields, 0)

    def test_yield_ends_when_streams_finish(self):
        # streams finish a few hundred tokens into the long prefill: no idle round while they are gone
        cls = self.fresh()
        self.assertTrue(install(cls, '50'))
        on = simulate(cls, stream_tokens=600)
        self.assertGreater(dy.STATS.cancels, 0)
        # once the streams are gone, the gaps between two chunks are the stock ones (no decode, no extra idle round)
        tr = on['trace']
        firsts = [i for i, x in enumerate(tr) if x[0] == 'chunk' and x[1] == ('L',)]
        gaps = [tr[a + 1:b] for a, b in zip(firsts, firsts[1:])]
        k = next(i for i, g in enumerate(gaps) if not any(x[0] == 'decode' for x in g))
        self.assertGreater(k, 0)
        self.assertTrue(all(not any(x[0] == 'decode' for x in g) for g in gaps[k:]))
        self.assertTrue(all(len(g) <= 2 for g in gaps[k:]))
        stock = simulate(self.fresh(), stream_tokens=600)
        self.assertEqual(stock['chunks'], on['chunks'])
        # the idle rounds left are the stock interval's own (--prefill-decode-interval with min-bs 0 defers prefill
        # by 2 rounds after every chunk even with nothing to decode: production behaviour, untouched here)

    def test_yield_ends_when_the_prompt_is_aborted(self):
        cls = self.fresh()
        self.assertTrue(install(cls, '50'))
        simulate(cls, abort_long_after_chunks=3, horizon_us=20_000_000)
        self.assertGreater(dy.STATS.cancels, 0)

    def test_final_chunk_and_solo_pass_arm_nothing(self):
        cls = self.fresh()
        self.assertTrue(install(cls, '50'))
        shorts = [Req('S1', 200), Req('S2', 150)]
        on = simulate(cls, long_total=5 * 8192 + 100, solo_shorts=shorts)
        tr = on['trace']
        chunk_rows = [x for x in tr if x[0] == 'chunk']
        self.assertEqual([x[1] for x in chunk_rows if x[1] != ('L',)], [('S1',), ('S2',)])
        # yields = non-final chunks of L only (solo passes and the final chunk arm nothing)
        self.assertEqual(dy.STATS.yields, 5)
        # after each solo pass: the stock interval (2 decode rounds), then L's chunk resumes alone
        for i, x in enumerate(tr):
            if x[0] == 'chunk' and x[1] in (('S1',), ('S2',)):
                nxt = [y for y in tr[i + 1:] if y[0] == 'chunk'][0]
                self.assertEqual(nxt[1], ('L',))
                self.assertEqual(sum(1 for y in tr[i + 1:tr.index(nxt, i + 1)] if y[0] == 'decode'), 2)
        # the final chunk arms no yield: after it only the stock interval's counter can be left
        fin = max(i for i, x in enumerate(tr) if x[0] == 'chunk' and x[1] == ('L',))
        self.assertIsNone(tr[fin][4])                         # built without a chunked request (the final chunk)
        self.assertLessEqual(getattr(on['sched'], '_dsv41_yield_left_us', 0), 0)
        self.assertLessEqual(on['sched']._prefill_decode_interval_remaining, 2)

    def test_rank_invariant(self):
        traces = []
        for rank in range(4):
            cls = self.fresh()
            env = {dy.ENV: '50'}
            dy.STATS = dy._Stats()
            self.assertTrue(dy.install(types.SimpleNamespace(Scheduler=cls), env=env))

            def boom(*a, **k):
                raise AssertionError('clock read in a scheduling decision')
            with patch.dict(os.environ, {'DSV41_RANK_NOISE': str(rank), dy.ENV: str(10 + rank)}), \
                    patch.object(time, 'time', boom), patch.object(time, 'monotonic', boom), \
                    patch.object(time, 'perf_counter', boom), patch.object(time, 'time_ns', boom):
                r = simulate(cls, stream_tokens=4000 + 0 * rank)
            traces.append(r['trace'])
        self.assertTrue(all(t == traces[0] for t in traces))

    def test_cadence_table(self):
        """The model's prediction for the kit's workload (printed with -v; numbers asserted loosely)."""
        rows = []
        stock = simulate(self.fresh())
        for pct in (0, 25, 33, 50, 67):
            cls = self.fresh()
            install(cls, '50', rid_arms='1')
            r = simulate(cls, long_rid=f'dyield-s{pct}-L')
            rows.append((pct, r['decode_share'], r['ttft_us'] / 1e6, r['ttft_us'] / stock['ttft_us'],
                         min(r['stream_steps']) / (r['ttft_us'] / 1e6)))
        if os.environ.get('DSV41_DECODE_YIELD_TABLE'):
            print('\nshare  decode-share  long-TTFT-s  x-stock  stream-steps/s (2 streams, 500K prompt)')
            for row in rows:
                print('%5d  %12.3f  %11.1f  %7.2f  %14.2f' % row)
        self.assertEqual(rows[0][3], 1.0)
        self.assertTrue(all(a[4] < b[4] for a, b in zip(rows, rows[1:])))


# ---------------------------------------------------------------- v2.6.4 profile (no staged source needed)
class Profile(unittest.TestCase):
    """v2.6.4: the production line of .env.tp4.example sets DSV41_DECODE_YIELD=33 (owner's share; measured on the
    fleet 2026-11-05); the module stays off when unset; the switch (and its test-only rid arms) is volatile in
    adapter/autotune_keep.py (scheduler-side timing only, so toggling it must reuse the tuned MoE tactics); both canary
    Dockerfiles run this test against the image's own scheduler source."""

    def _production_line(self):
        lines = (ROOT / '.env.tp4.example').read_text().splitlines()
        i = next(n for n, line in enumerate(lines) if line.startswith('# Production settings, together with the line'))
        self.assertTrue(lines[i - 1].startswith('#EXTRA_CONTAINER_ENV="'), lines[i - 1][:60])
        return lines[i - 1]

    def test_production_line_sets_share_33_once_after_swan(self):
        items = self._production_line().split('"')[1].split()
        self.assertEqual([x for x in items if x.startswith('DSV41_DECODE_YIELD')], ['DSV41_DECODE_YIELD=33'])
        i = items.index('DSV41_DECODE_YIELD=33')        # right after swan (v2.6.4's SWA thin / cone keys follow)
        # v2.7: swan is retired (FlashInfer 0.7 decodes the SWA pool's own pages); the yield follows v2.6.3's idle park
        self.assertEqual(items[i - 1:i + 1], ['DSV41_ROCE_IDLE_PARK=1', 'DSV41_DECODE_YIELD=33'])
        self.assertEqual(dy.parse_pct('33'), 33)

    def test_only_the_production_line_sets_it(self):
        text = (ROOT / '.env.tp4.example').read_text()
        on = [line for line in text.splitlines() if line.startswith('#EXTRA_CONTAINER_ENV=') and 'DECODE_YIELD' in line]
        self.assertEqual(on, [self._production_line()])
        on = [line for line in text.splitlines() if line.lstrip('#').startswith(('EXTRA_CONTAINER_ENV=', 'DSV41_'))
              and 'DECODE_YIELD_RID_ARMS' in line]
        self.assertEqual(on, [], 'the rid arms are test only')

    def test_volatile_in_autotune_keep(self):
        tree = ast.parse((ROOT / 'adapter' / 'autotune_keep.py').read_text())
        vol = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == '_VOLATILE')
        self.assertIn('DSV41_DECODE_YIELD', vol)
        self.assertIn('DSV41_DECODE_YIELD_RID_ARMS', vol)

    def test_default_off_in_module(self):
        self.assertFalse(dy.requested({}))
        self.assertFalse(dy.requested({dy.ENV: '0'}))
        self.assertTrue(dy.requested({dy.ENV: '33'}))
        self.assertFalse(install(type('S', (), {}), ''))

    def test_dockerfiles_run_this_test_on_the_image_source(self):
        for name in ('Dockerfile.canary', 'Dockerfile.canary-roce'):
            d = (ROOT / name).read_text()
            self.assertIn('DSV41_DECODE_YIELD_SOURCE_ROOT=/sgl-workspace/sglang/python/sglang/srt/managers '
                          'PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_decode_yield.py', d, name)


if __name__ == '__main__':
    unittest.main()
