#!/usr/bin/env python3
"""DSV41_SWA_PROMPT_TAIL_KEEP (adapter/swa_prompt_tail.py): a request-owned prompt end keeps its SWA through decode.

Offline part (no SGLang): knob parsing, the floor rule on stand-in objects, the sitecustomize gate, the profile
and autotune-key bookkeeping.
Engine part (skipped without SGLang; CPU container with the pinned SGLang tree on PYTHONPATH): a real
UnifiedRadixCache (FULL + SWA, page 4, window 4) and the real ScheduleBatch.maybe_evict_swa / _evict_swa per decode
step. A short prompt (one extend) is decoded and finished, then revisited with a new suffix:
  * with the prompt-end insert (the stock scheduler path) the revisit already matches page_floor(prompt), with or
    without the module, and the module applies no floor at all (exact no-op: same reuse, same SWA ledgers);
  * without the prompt-end insert (the case jjang-ai measured: 0 cached on the 2nd visit) stock reuses 0 and the
    module reuses page_floor(prompt); the SWA the request holds stays <= keep + hold + window + page, and past the
    hold bound the request falls back to stock frees.
"""
import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "adapter"))
import swa_prompt_tail as T  # noqa: E402

HAVE_SGLANG = importlib.util.find_spec("sglang") is not None


# ---------------------------------------------------------------- offline
class Knobs(unittest.TestCase):
    def test_off_values(self):
        for v in (None, "", "0", "off", "false", "no", " 0 "):
            env = {} if v is None else {T.ENV_KEEP: v}
            self.assertEqual(T.config(env), (0, T.DEFAULT_HOLD), v)
            self.assertFalse(T.enabled(env), v)

    def test_values_and_ranges(self):
        self.assertEqual(T.config({T.ENV_KEEP: "1024"}), (1024, 8192))
        self.assertEqual(T.config({T.ENV_KEEP: "1024", T.ENV_HOLD: "0"}), (1024, 0))
        self.assertEqual(T.config({T.ENV_KEEP: "8192", T.ENV_HOLD: "1048576"}), (8192, 1 << 20))
        for bad in ({T.ENV_KEEP: "8193"}, {T.ENV_KEEP: "-4"}, {T.ENV_KEEP: "abc"},
                    {T.ENV_KEEP: "1024", T.ENV_HOLD: "-1"}, {T.ENV_KEEP: "1024", T.ENV_HOLD: "1048577"},
                    {T.ENV_KEEP: "1024", T.ENV_HOLD: "x"}):
            with self.assertRaises(ValueError, msg=bad):
                T.config(bad)
            self.assertTrue(T.enabled(bad), bad)       # a typo is not silently "off": install refuses and logs

    def test_install_refuses_bad_knob_and_off(self):
        mod = types.SimpleNamespace(ScheduleBatch=type("SB", (), {"_evict_swa": lambda self, r, p: None}))
        orig = mod.ScheduleBatch._evict_swa
        with self.assertLogs(T.logger, "WARNING") as cm:
            self.assertFalse(T.install(mod, env={T.ENV_KEEP: "abc"}))
        self.assertIn("not installed", "".join(cm.output))
        self.assertFalse(T.install(mod, env={}))
        self.assertIs(mod.ScheduleBatch._evict_swa, orig)


def _cache(page=4, chunk=False, **tc):
    tree = SimpleNamespace(is_eagle=False, enable_hicache=False, enable_session_radix_cache=False)
    for k, v in tc.items():
        setattr(tree, k, v)
    return SimpleNamespace(page_size=page, is_chunk_cache=lambda: chunk, tree_core=tree, is_mamba_enabled=False)


def _req(prompt, out, protected):
    return SimpleNamespace(origin_input_ids=[0] * prompt, seqlen=prompt + out,
                           kv=SimpleNamespace(cache_protected_len=protected))


class FloorRule(unittest.TestCase):
    def setUp(self):
        T.STATE.released = 0

    def test_floor_when_prompt_end_is_request_owned(self):
        c = _cache(page=256)
        self.assertEqual(T.prompt_tail_floor(c, _req(6000, 10, 0), 1024, 8192), 5888 - 1024)
        self.assertEqual(T.prompt_tail_floor(c, _req(6000, 10, 4096), 1024, 8192), 5888 - 1024)
        self.assertEqual(T.prompt_tail_floor(c, _req(700, 10, 0), 1024, 8192), 0)    # floor clamps at 0

    def test_no_floor_when_the_tree_owns_the_prompt_end(self):
        c = _cache(page=256)
        self.assertIsNone(T.prompt_tail_floor(c, _req(6000, 10, 5888), 1024, 8192))
        self.assertIsNone(T.prompt_tail_floor(c, _req(6000, 10, 6144), 1024, 8192))

    def test_no_floor_outside_scope(self):
        r = _req(6000, 10, 0)
        self.assertIsNone(T.prompt_tail_floor(_cache(page=256, chunk=True), r, 1024, 8192))
        self.assertIsNone(T.prompt_tail_floor(_cache(page=256, is_eagle=True), r, 1024, 8192))
        self.assertIsNone(T.prompt_tail_floor(_cache(page=256, enable_hicache=True), r, 1024, 8192))
        self.assertIsNone(T.prompt_tail_floor(_cache(page=256, enable_session_radix_cache=True), r, 1024, 8192))
        m = _cache(page=256)
        m.is_mamba_enabled = True
        self.assertIsNone(T.prompt_tail_floor(m, r, 1024, 8192))
        self.assertIsNone(T.prompt_tail_floor(_cache(page=256), _req(200, 10, 0), 1024, 8192))   # < one page
        self.assertIsNone(T.prompt_tail_floor(SimpleNamespace(is_chunk_cache=lambda: False), r, 1024, 8192))

    def test_hold_bound(self):
        c = _cache(page=256)
        floor = 5888 - 1024
        r = _req(6000, 0, 0)
        r.seqlen = floor + 1024 + 8192
        self.assertEqual(T.prompt_tail_floor(c, r, 1024, 8192), floor)          # at the bound: still held
        setattr(r, T.REQ_ATTR, "held")
        r.seqlen += 1
        self.assertIsNone(T.prompt_tail_floor(c, r, 1024, 8192))                # past it: stock
        self.assertEqual(getattr(r, T.REQ_ATTR), "released")
        self.assertEqual(T.STATE.released, 1)
        self.assertIsNone(T.prompt_tail_floor(c, r, 1024, 8192))
        self.assertEqual(T.STATE.released, 1)                                    # counted once


class Wiring(unittest.TestCase):
    def test_sitecustomize_gate_before_import(self):
        src = (ROOT / "adapter" / "sitecustomize.py").read_text()
        i = src.index("from swa_prompt_tail import install")
        gate = src.rfind("if os.environ.get('DSV41_SWA_PROMPT_TAIL_KEEP'", 0, i)
        self.assertGreater(gate, 0)
        branch = src.rfind("elif module.__name__ == ", 0, gate)
        self.assertIn("'sglang.srt.managers.schedule_batch'", src[branch:branch + 80])
        self.assertEqual(src.count("from swa_prompt_tail import"), 1)
        ast.parse(src)

    def test_off_in_every_profile_line_and_volatile(self):
        env_text = (ROOT / ".env.tp4.example").read_text()
        for line in env_text.splitlines():
            if line.lstrip("#").startswith("EXTRA_CONTAINER_ENV="):
                self.assertNotIn("DSV41_SWA_PROMPT_TAIL", line)
        self.assertIn("DSV41_SWA_PROMPT_TAIL_KEEP", env_text)                   # documented
        ak = (ROOT / "adapter" / "autotune_keep.py").read_text()
        for k in ("DSV41_SWA_PROMPT_TAIL_KEEP", "DSV41_SWA_PROMPT_TAIL_HOLD"):
            self.assertIn(f'"{k}"', ak)

    def test_dockerfile_and_ci_run_this_test(self):
        self.assertIn("tests/test_swa_prompt_tail.py", (ROOT / "Dockerfile.canary-roce").read_text())
        self.assertIn("tests/test_swa_prompt_tail.py", (ROOT / ".github/workflows/offline-tests.yml").read_text())

    def test_pins_are_hex(self):
        self.assertEqual(len(T.PINS), 7)
        for k, v in T.PINS.items():
            self.assertRegex(v, r"^[0-9a-f]{64}$", k)


# ---------------------------------------------------------------- engine
@unittest.skipUnless(HAVE_SGLANG, "needs the pinned SGLang tree")
class Engine(unittest.TestCase):
    PAGE, WIN = 4, 4

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        # the decode frees every SGLANG_SWA_EVICTION_INTERVAL tokens (128 at window 128 / page 256); scaled to window 4
        os.environ["SGLANG_SWA_EVICTION_INTERVAL"] = "4"
        import torch  # noqa: F401
        from sglang.srt.managers import schedule_batch as sb
        from sglang.srt.mem_cache import common
        from sglang.srt.mem_cache import unified_radix_cache as urc
        from sglang.srt.mem_cache.unified_cache.components import swa
        cls.sb, cls.common, cls.urc, cls.swa = sb, common, urc, swa
        cls.stock_evict = staticmethod(sb.ScheduleBatch._evict_swa)

    def setUp(self):
        self.sb.ScheduleBatch._evict_swa = self.stock_evict
        T.STATE.__dict__.update(installed=False, active=False, applied=0, requests=0, released=0, errors=0)

    tearDown = setUp

    def _arm(self, keep, hold=T.DEFAULT_HOLD):
        if keep:
            self.assertTrue(T.install(self.sb, env={T.ENV_KEEP: str(keep), T.ENV_HOLD: str(hold)}))

    def _build(self, swa_size=512):
        import torch
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=self.PAGE))
        full_size = 2048
        pool = ReqToTokenPool(size=16, max_context_len=1024, device="cpu", enable_memory_saver=False)
        kv = SWAKVPool(size=full_size, size_swa=swa_size, page_size=self.PAGE, dtype=torch.bfloat16,
                       head_num=1, head_dim=8, swa_attention_layer_ids=[1],
                       full_attention_layer_ids=[0], device="cpu")
        alloc = SWATokenToKVPoolAllocator(size=full_size, size_swa=swa_size, page_size=self.PAGE,
                                          dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False)
        cache = self.urc.UnifiedRadixCache(params=CacheInitParams(
            req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=self.PAGE,
            disable=False, sliding_window_size=self.WIN,
            tree_components=(ComponentType.FULL, ComponentType.SWA), eviction_policy="lru", is_eagle=False))
        return cache, alloc, pool

    def _alloc(self, alloc, n):
        n2 = -(-n // self.PAGE) * self.PAGE
        full = alloc.full_attn_allocator.alloc(n2)
        swa = alloc.swa_attn_allocator.alloc(n2)
        assert full is not None and swa is not None, "test pool too small"
        alloc.full_to_swa_index_mapping[full] = swa
        return full[:n]

    def _batch(self, cache, alloc, pool, req):
        """The attributes ScheduleBatch.maybe_evict_swa / _evict_swa read, on a stand-in batch of one decode req."""
        b = SimpleNamespace(tree_cache=cache, token_to_kv_pool_allocator=alloc, req_to_token_pool=pool,
                            reqs=[req], forward_mode=SimpleNamespace(is_decode=lambda: True,
                                                                     is_extend=lambda: False))
        b._evict_swa = types.MethodType(self.sb.ScheduleBatch._evict_swa, b)
        return b

    def _visit(self, cache, alloc, pool, rid, prompt, nout, prompt_end_insert=True):
        """One request as the scheduler runs it: match + lock, the prompt in one extend, the first token, the
        prompt-end insert (skipped to reproduce the request-owned prompt end), `nout` decode steps each followed by
        the real maybe_evict_swa, the finished insert. Returns (reused, max request-owned live SWA during decode)."""
        from array import array
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(rid=str(rid), origin_input_text="", origin_input_ids=array("q"),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=nout))
        pool.alloc([req])
        req.origin_input_ids = array("q", prompt)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", prompt)
        req.extra_key = None
        mr = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prompt))))
        m = len(mr.device_indices)
        req.last_node = mr.last_device_node
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        if m:
            pool.write((req.kv.req_pool_idx, slice(0, m)), mr.device_indices)
        req.prefix_indices = mr.device_indices
        req.kv.cache_protected_len = m
        req.swa_branching_seqlen = getattr(mr, "swa_branching_seqlen", None)
        idx = self._alloc(alloc, len(prompt) - m)
        pool.write((req.kv.req_pool_idx, slice(m, len(prompt))), idx)
        req.set_extend_range(m, len(prompt))
        req.kv.kv_committed_len = len(prompt)
        out = [90000 + 1000 * rid + i for i in range(nout)]
        req.output_ids = array("q", out[:1])
        req._refresh_fill_ids()
        if prompt_end_insert:
            cache.cache_unfinished_req(req, chunked=False)
        batch = self._batch(cache, alloc, pool, req)
        held = 0
        for i in range(1, nout):
            pos = len(prompt) + i - 1
            idx = self._alloc(alloc, 1)
            pool.write((req.kv.req_pool_idx, slice(pos, pos + 1)), idx)
            req.output_ids = array("q", out[: i + 1])
            req.kv.kv_committed_len = pos + 1
            req.decode_batch_idx = i
            self.sb.ScheduleBatch.maybe_evict_swa(batch)
            held = max(held, req.seqlen - max(req.kv.swa_evicted_seqlen, req.kv.cache_protected_len))
        kvlen = len(prompt) + nout - 1
        cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=kvlen)
        pool.free(req)
        cache.sanity_check()
        return m, held

    def _run(self, keep, hold=T.DEFAULT_HOLD, prompt_end_insert=True, nout=30, plen=46):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        self.setUp()                                              # every run starts from the stock engine
        self._arm(keep, hold)
        cache, alloc, pool = self._build()
        base = list(range(1000, 1000 + plen))
        v1 = self._visit(cache, alloc, pool, 1, base, nout, prompt_end_insert)
        v2 = self._visit(cache, alloc, pool, 2, base + [7, 8, 9], nout, prompt_end_insert)
        tc = cache.tree_core
        sizes = (dict(tc.component_evictable_size_), dict(tc.component_protected_size_),
                 alloc.swa_available_size(), alloc.full_available_size())
        return v1, v2, sizes, T.STATE.applied

    def test_pins_match_engine(self):
        self.assertEqual(T.pin_drift(T.pin_targets(self.sb, self.common, self.urc, self.swa)), [])

    def test_replacement_mirrors_the_pinned_body(self):
        # the stock body calls free_swa_out_of_window_slots with exactly these keywords; ours passes the same
        import inspect
        call = [n for n in ast.walk(ast.parse(inspect.getsource(self.stock_evict).strip()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "free_swa_out_of_window_slots"]
        self.assertEqual(len(call), 1)
        stock_kw = [k.arg for k in call[0].keywords]
        mine = [n for n in ast.walk(ast.parse((ROOT / "adapter" / "swa_prompt_tail.py").read_text()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "free"]
        self.assertEqual(len(mine), 1)
        self.assertEqual([k.arg for k in mine[0].keywords], stock_kw)
        self.assertEqual(len(call[0].args), 2)
        self.assertEqual(len(mine[0].args), 2)

    def test_stock_path_is_an_exact_noop(self):
        stock = self._run(0)
        armed = self._run(1024)
        armed8 = self._run(8)
        page_floor = 46 // self.PAGE * self.PAGE
        self.assertEqual(stock[1][0], page_floor)            # the revisit already matches the prompt end
        self.assertEqual(stock[:3], armed[:3])                # same reuse, held SWA and ledgers
        self.assertEqual(stock[:3], armed8[:3])
        self.assertEqual(armed[3], 0)                         # and no floor was ever applied
        self.assertEqual(armed8[3], 0)

    def test_request_owned_prompt_end_is_kept(self):
        stock = self._run(0, prompt_end_insert=False)
        armed = self._run(8, prompt_end_insert=False)
        requests = T.STATE.requests
        page_floor = 46 // self.PAGE * self.PAGE
        print(f"\n  request-owned prompt end: stock revisit {stock[1][0]}, keep=8 {armed[1][0]} "
              f"(page_floor {page_floor}); held SWA stock {stock[0][1]} / keep {armed[0][1]}")
        self.assertEqual(stock[1][0], 0)                      # the bug: the revisit reuses nothing
        self.assertEqual(armed[1][0], page_floor)             # kept: it matches the prompt end
        self.assertGreater(armed[3], 0)
        self.assertEqual(requests, 2)
        # the hold bound: keep + hold + (window + page) of the stock trailing slots, here hold 8192 > output
        self.assertLessEqual(armed[0][1], 8 + 30 + self.WIN + self.PAGE)

    def test_hold_bound_falls_back_to_stock(self):
        stock = self._run(0, prompt_end_insert=False)
        capped = self._run(8, hold=8, prompt_end_insert=False)
        released = T.STATE.released
        print(f"\n  hold 8: revisit {capped[1][0]}, max held SWA {capped[0][1]}, released {released}")
        self.assertEqual(capped[1][0], stock[1][0])           # 30-token outputs pass the bound: stock outcome
        self.assertEqual(released, 2)
        self.assertLessEqual(capped[0][1], 8 + 8 + self.WIN + 2 * self.PAGE)
        roomy = self._run(8, hold=64, prompt_end_insert=False)
        self.assertEqual(roomy[1][0], 46 // self.PAGE * self.PAGE)

    def test_deterministic(self):
        self.assertEqual(self._run(8, prompt_end_insert=False)[:3], self._run(8, prompt_end_insert=False)[:3])

    def test_sitecustomize_installs_the_hook(self):
        # The engine's import hook (active with DSV41_SOURCE, as boot.py sets it) next to the v2.6.5 scheduler-side
        # switches (thin, cone, bypass, decode yield); off, it never imports the module.
        import subprocess
        adapter = str(ROOT / "adapter")
        probe = ("import sglang.srt.managers.scheduler as S, sglang.srt.managers.schedule_batch as SB,"
                 " sglang.srt.mem_cache.unified_radix_cache as U, sys\n"
                 "print('HOOKS', getattr(SB.ScheduleBatch._evict_swa, '_dsv41_prompt_tail_hook', False),"
                 " 'swa_prompt_tail' in sys.modules,"
                 " getattr(SB.Req.init_next_round_input, '_dsv41_swa_cone_hook', False),"
                 " getattr(U.UnifiedRadixCache.cache_unfinished_req, '_dsv41_swa_thin', False))\n")
        for keep, want, log in (("1024", "True True True True", "DSV41_SWA_PROMPT_TAIL installed: keep=1024"),
                                ("0", "False False True True", None),
                                ("abc", "False True True True", "DSV41_SWA_PROMPT_TAIL not installed")):
            env = dict(os.environ, DSV41_SOURCE="/models/x", DSV41_SWA_PROMPT_TAIL_KEEP=keep, DSV41_SWA_THIN="1",
                       DSV41_SWA_CONE_REPLAY="1", DSV41_SWA_CONE_TOKENS="8192", DSV41_PREFILL_BYPASS="256",
                       DSV41_PREFILL_BYPASS_SOLO="1", DSV41_DECODE_YIELD="33", CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join([adapter, os.environ.get("PYTHONPATH", "")]))
            out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=600)
            line = [x for x in out.stdout.splitlines() if x.startswith("HOOKS")]
            self.assertEqual(line, ["HOOKS " + want], out.stdout[-2000:] + out.stderr[-2000:])
            if log:
                self.assertIn(log, out.stdout + out.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
