"""DSV41_PREFILL_BYPASS_SOLO (solo short pass) on the mocked CPU scheduler of test_prefill_bypass.py, whose
_get_new_batch_prefill_raw and PrefillAdder methods are lifted from the real pinned SGLang source.

A pass simulator drives the pinned scheduler function pass after pass: after each pass the long prompt's computed
range is stashed into its prefix (as get_next_batch_to_run's stash does), admitted short requests leave the queue,
and short requests arrive at chosen passes. Checked:
- the long prompt's chunks (boundaries, order) equal an idle run's, and every pass that carries a chunk carries
  nothing else (computed as on an idle server); a pass that carries short requests carries only them;
- a pass with short requests is followed by a chunk pass (no starvation); the final chunk is parked like any other;
- solo passes leave the long prompt's in-flight accounting alone (inflight_middle_chunks), build the batch without
  the chunked request and mark it as containing its last prefill chunk;
- the decision is rank-invariant (four simulated ranks, different unrelated host values);
- off (unset / 0) is the v2.6.1 adapter's patched function byte for byte (tests/data fixture); an unknown value and
  any anchor drift refuse to arm without mutating anything; the source pins are the v2.6.1 ones.
"""
import hashlib
import importlib
import importlib.util
import marshal
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))
sys.path.insert(0, str(ROOT / 'tests'))
import test_prefill_bypass as base  # noqa: E402

FIXTURE = ROOT / 'tests/data/prefill_bypass_f744d83.py.txt'
FIXTURE_SHA = '0e0cb18efdb241d0f6d889f2ba2eb377d24666c647d8a038a8021d83813024bf'
SUPPORTED = base.EXPECT == 'supported' and (base.STAGED / 'scheduler.py').is_file()


def load_fixture():
    spec = importlib.util.spec_from_file_location('prefill_bypass_f744d83', FIXTURE,
                                                  loader=importlib.machinery.SourceFileLoader(
                                                      'prefill_bypass_f744d83', str(FIXTURE)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arm(module, solo, limit='4', scan='32', adapter=None):
    adapter = adapter or importlib.import_module('prefill_bypass')
    env = {'DSV41_PREFILL_BYPASS': limit, 'DSV41_PREFILL_BYPASS_SCAN': scan}
    if solo is not None:
        env['DSV41_PREFILL_BYPASS_SOLO'] = solo
    with patch.dict(os.environ, env, clear=False):
        if solo is None:
            os.environ.pop('DSV41_PREFILL_BYPASS_SOLO', None)
        return adapter.install(module)


def engine(long_tokens, solo, capacity=10_000, rank_noise=0):
    module, scheduler, running, memory = base.mocked_engine(capacity=capacity)
    seen = {}
    batch_cls = scheduler.__class__._get_new_batch_prefill_raw.__globals__['ScheduleBatch']

    def init_new(cls, reqs, *args, **kwargs):
        b = cls(reqs)
        b.init_chunked_req = kwargs.get('chunked_req')
        return b
    batch_cls.init_new = classmethod(init_new)
    scheduler.unrelated_host_free_bytes = rank_noise
    scheduler.chunked_req = base.Request('long', long_tokens)
    scheduler.waiting_queue = []
    if solo is not False:
        assert arm(module, '1' if solo else None)
    else:
        assert arm(module, None)
    return module, scheduler, running, seen


def simulate(long_tokens=20, arrivals=None, solo=True, passes=40, rank_noise=0, short_tokens=2,
             keep_scanning=False):
    """Run passes until the long prompt is done and the queue is empty. arrivals: {pass index: [names]}.
    Returns per pass: (names in the batch, long extend range or None, batch.init_chunked_req name,
    contains_last_prefill_chunk) and the long prompt's inflight_middle_chunks history."""
    module, scheduler, running, _ = engine(long_tokens, solo, rank_noise=rank_noise)
    if keep_scanning:     # the candidate loop stays open after an admission (the engine's budget_state may stop it)
        module.PrefillAdder.budget_state = lambda self: base.Result.CONTINUE
    long = scheduler.chunked_req
    arrivals = arrivals or {}
    out, inflight = [], []
    for i in range(passes):
        for name in arrivals.get(i, []):
            scheduler.waiting_queue.append(base.Request(name, short_tokens))
        if scheduler.chunked_req is None and not scheduler.waiting_queue:
            break
        batch, running = scheduler._get_new_batch_prefill_raw(None, running)
        if batch is None:
            out.append(((), None, None, None))
            continue
        names = tuple(r.rid for r in batch.reqs)
        rng = None
        if long in batch.reqs:
            rng = (long.extend_range.start, long.extend_range.end)
            long.prefix_indices = [0] * long.extend_range.end          # the stash of get_next_batch_to_run
        else:
            # a parked long prompt has nothing new to stash
            assert long.extend_range.end == len(long.prefix_indices), (i, long.extend_range.end)
        chunked = getattr(batch.init_chunked_req, 'rid', None)
        out.append((names, rng, chunked, getattr(batch, 'contains_last_prefill_chunk', None)))
        inflight.append(long.inflight_middle_chunks)
    return out, inflight, scheduler


def long_ranges(out):
    return [rng for _, rng, _, _ in out if rng is not None]


@unittest.skipUnless(SUPPORTED, 'needs the staged pinned scheduler source (DSV41_PREFILL_BYPASS_SOURCE_ROOT)')
class SoloShortPassTests(unittest.TestCase):
    def test_idle_long_prompt_chunks(self):
        out, inflight, _ = simulate(20, solo=True)
        self.assertEqual(long_ranges(out), [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)])
        self.assertTrue(all(names == ('long',) for names, _, _, _ in out))

    def test_short_arrival_gets_a_solo_pass_and_the_long_prompt_is_unchanged(self):
        idle, _, _ = simulate(20, solo=True)
        out, inflight, sched = simulate(20, arrivals={2: ['s1']}, solo=True)
        self.assertEqual(long_ranges(out), long_ranges(idle))
        passes = [names for names, _, _, _ in out]
        self.assertEqual(passes, [('long',), ('long',), ('s1',), ('long',), ('long',), ('long',)])
        # every pass carrying the long prompt carries nothing else; the short request runs alone
        for names, rng, _, _ in out:
            self.assertTrue(names == ('long',) or 'long' not in names)
        self.assertEqual(sched.waiting_queue, [])

    def test_off_keeps_the_stock_beside_the_chunk_admission(self):
        out, _, _ = simulate(20, arrivals={2: ['s1']}, solo=False)
        self.assertIn(('long', 's1'), [names for names, _, _, _ in out])

    def test_continuous_arrivals_alternate_and_never_starve_the_long_prompt(self):
        idle, _, _ = simulate(20, solo=True)
        arrivals = {i: [f's{i}'] for i in range(0, 12)}
        out, _, _ = simulate(20, arrivals=arrivals, solo=True, passes=60)
        self.assertEqual(long_ranges(out), long_ranges(idle))
        kinds = ['L' if 'long' in names else 'S' for names, _, _, _ in out if names]
        self.assertNotIn('SS', ''.join(kinds[:kinds.index('L', len(kinds) - kinds[::-1].index('L') - 1) + 1]))
        solo_passes = kinds.count('S')
        self.assertGreater(solo_passes, 0)
        # at most one solo pass between two chunk passes while the long prompt is in flight
        last_l = len(kinds) - 1 - kinds[::-1].index('L')
        self.assertLessEqual(kinds[:last_l].count('S'), kinds.count('L'))

    def test_final_chunk_is_parked_like_any_other(self):
        # 18 tokens: the final chunk (16, 18) leaves residual budget, so the stock final-chunk admission would put the
        # short request beside it; solo parks the final chunk instead
        idle, _, _ = simulate(18, solo=True)
        self.assertEqual(long_ranges(idle)[-1], (16, 18))
        off, _, _ = simulate(18, arrivals={4: ['s1']}, solo=False)
        self.assertEqual([names for names, _, _, _ in off][4], ('long', 's1'))
        out, _, _ = simulate(18, arrivals={4: ['s1']}, solo=True)
        self.assertEqual(long_ranges(out), long_ranges(idle))
        self.assertEqual([names for names, _, _, _ in out][4:6], [('s1',), ('long',)])

    def test_solo_pass_leaves_the_in_flight_accounting_and_batch_metadata_clean(self):
        out, inflight, _ = simulate(20, arrivals={2: ['s1']}, solo=True)
        solo = next(i for i, (names, _, _, _) in enumerate(out) if names == ('s1',))
        self.assertEqual(inflight[solo], inflight[solo - 1])          # no increment for the parked prompt
        self.assertIsNone(out[solo][2])                                # batch built without the chunked request
        self.assertTrue(out[solo][3])                                  # contains its last prefill chunk
        chunk_passes = [i for i, (names, _, _, _) in enumerate(out) if names == ('long',)]
        self.assertEqual(out[chunk_passes[0]][2], 'long')

    def test_long_arrivals_never_trigger_a_solo_pass(self):
        out, _, _ = simulate(20, arrivals={1: ['big']}, solo=True, short_tokens=12)
        self.assertTrue(all(names in (('long',),) for names, _, _, _ in out[:5]))

    def test_two_short_arrivals_share_one_solo_pass(self):
        out, _, _ = simulate(20, arrivals={2: ['s1', 's2']}, solo=True, keep_scanning=True)
        self.assertIn(('s1', 's2'), [names for names, _, _, _ in out])
        # with the engine's own budget_state the second waits for the next solo pass, never beside the long prompt
        out, _, _ = simulate(20, arrivals={2: ['s1', 's2']}, solo=True)
        self.assertTrue(all(names == ('long',) or 'long' not in names for names, _, _, _ in out))

    def test_rank_invariant(self):
        arrivals = {1: ['a'], 3: ['b', 'c'], 4: ['d']}
        runs = [simulate(20, arrivals=arrivals, solo=True, rank_noise=r * 10**6)[0] for r in range(4)]
        self.assertTrue(all(r == runs[0] for r in runs))

    def test_no_long_prompt_is_stock(self):
        def run(solo):
            module, scheduler, running, memory = base.mocked_engine()
            scheduler.chunked_req = None
            if solo is not False:
                arm(module, '1')
            batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
            return [r.rid for r in batch.reqs], [r.rid for r in scheduler.waiting_queue], memory.reservations
        self.assertEqual(run(True), run(False))


@unittest.skipUnless(SUPPORTED, 'needs the staged pinned scheduler source (DSV41_PREFILL_BYPASS_SOURCE_ROOT)')
class SoloOffIdentityAndPinTests(unittest.TestCase):
    def patched_code(self, solo, adapter=None):
        module, scheduler, running, memory = base.mocked_engine()
        self.assertTrue(arm(module, solo, adapter=adapter))
        return marshal.dumps(module.Scheduler._get_new_batch_prefill_raw.__code__), module

    def test_fixture_is_the_v2_6_1_adapter(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE_SHA)

    def test_off_is_the_v2_6_1_patched_function_byte_for_byte(self):
        old = load_fixture()
        reference, _ = self.patched_code(None, adapter=old)
        for value in (None, '0', 'off', ''):
            code, module = self.patched_code(value)
            self.assertEqual(code, reference, value)
            self.assertFalse(getattr(module.Scheduler, '_dsv41_prefill_bypass_solo', False))
        on, module = self.patched_code('1')
        self.assertNotEqual(on, reference)
        self.assertTrue(module.Scheduler._dsv41_prefill_bypass_solo)

    def test_bypass_unset_with_solo_set_mutates_nothing(self):
        module, scheduler, running, memory = base.mocked_engine()
        before = (module.Scheduler._get_new_batch_prefill_raw, module.PrefillAdder.add_one_req)
        adapter = importlib.import_module('prefill_bypass')
        with patch.dict(os.environ, {'DSV41_PREFILL_BYPASS': '', 'DSV41_PREFILL_BYPASS_SOLO': '1'}):
            self.assertFalse(adapter.install(module))
        self.assertEqual(before, (module.Scheduler._get_new_batch_prefill_raw, module.PrefillAdder.add_one_req))

    def test_unknown_solo_value_refuses_without_mutation(self):
        module, scheduler, running, memory = base.mocked_engine()
        before = (module.Scheduler._get_new_batch_prefill_raw, module.PrefillAdder.add_one_req)
        self.assertFalse(arm(module, 'maybe'))
        self.assertEqual(before, (module.Scheduler._get_new_batch_prefill_raw, module.PrefillAdder.add_one_req))

    def test_pins_are_the_v2_6_1_pins(self):
        self.assertEqual(importlib.import_module('prefill_bypass').EXPECTED, load_fixture().EXPECTED)

    def test_solo_anchor_drift_refuses_without_mutation(self):
        adapter = importlib.import_module('prefill_bypass')
        for i, (old, new) in enumerate(adapter.SOLO_PATCHES):
            module, scheduler, running, memory = base.mocked_engine()
            before = (module.Scheduler._get_new_batch_prefill_raw, module.PrefillAdder.add_one_req,
                      module.PrefillAdder._select_prefill_admission)
            broken = list(adapter.SOLO_PATCHES)
            broken[i] = (old + '# drifted\n', new)
            with patch.object(adapter, 'SOLO_PATCHES', broken):
                self.assertFalse(arm(module, '1', adapter=adapter), i)
            self.assertEqual(before, (module.Scheduler._get_new_batch_prefill_raw,
                                      module.PrefillAdder.add_one_req,
                                      module.PrefillAdder._select_prefill_admission), i)

    def test_source_drift_refuses_solo_too(self):
        module, scheduler, running, memory = base.mocked_engine()
        module.Scheduler._get_new_batch_prefill_raw = lambda *a: None
        self.assertFalse(arm(module, '1'))


class SoloSwitchTests(unittest.TestCase):
    def test_values(self):
        adapter = importlib.import_module('prefill_bypass')
        for v, want in (('1', True), ('on', True), ('0', False), ('', False), ('off', False), ('2', None)):
            self.assertIs(adapter.solo_requested({'DSV41_PREFILL_BYPASS_SOLO': v}), want, v)
        self.assertIs(adapter.solo_requested({}), False)


if __name__ == '__main__':
    unittest.main()
