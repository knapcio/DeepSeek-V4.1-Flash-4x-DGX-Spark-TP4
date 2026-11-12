#!/usr/bin/env python3
"""DSV41_SWA_PROMPT_TAIL_KEEP (adapter/swa_prompt_tail.py): the prompt-end insert keeps the last N prompt tokens' SWA.

Offline part (no SGLang): knob parsing, the floor rules on stand-in objects, the sitecustomize gate, the profile and
autotune-key bookkeeping.
Engine part (skipped without SGLang; CPU container with the pinned SGLang tree on PYTHONPATH): a real
UnifiedRadixCache (FULL + SWA, page 4, window 4) and the real ScheduleBatch.maybe_evict_swa per decode step:
  * a document + a new question (the common prefix ends before the last page of the previous prompt): stock reuses 0
    on the 2nd visit and hits from the 3rd (jjang-ai's measured pattern); with the module the 2nd visit reuses the
    whole document; decode frees are identical (no hold), the tree holds at most N more SWA slots;
  * a revisit that continues after the prompt end: unchanged; chunked inserts: the tree identical; with DSV41_SWA_THIN
    on as well: the same reuse;
  * a request-owned prompt end (no prompt-end insert): stock reuses 0, the module the prompt end; the decode hold
    stays <= N + hold, past the bound the request falls back to stock frees.
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


# v2.7.5's production-line tail (scale codes, then the evict-first experts + SHPF window)
V275_TAIL = ["DSV41_MOE_SCALE_CODES=1", "DSV41_MOE_SCALE_CODES_MIN_ROWS=2", "DSV41_MOE_EVICT_FIRST=1",
             "DSV41_L2_PREFETCH_SHPF_MB=6", "DSV41_L2_PREFETCH_SHPF_EL=0"]


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

    def test_prompt_end_insert_keeps_the_tail(self):
        c = _cache(page=256)
        for phase in ("prompt_end", None):        # rule 1 (the insert) and rule 2 (decode, request-owned)
            self.assertEqual(T.prompt_tail_floor(c, _req(6000, 1, 0), 1024, 8192, phase), 5888 - 1024)
            self.assertEqual(T.prompt_tail_floor(c, _req(6000, 1, 4096), 1024, 8192, phase), 5888 - 1024)
            self.assertEqual(T.prompt_tail_floor(c, _req(700, 1, 0), 1024, 8192, phase), 0)   # clamps at 0

    def test_chunked_inserts_are_stock(self):
        c = _cache(page=256)
        self.assertIsNone(T.prompt_tail_floor(c, _req(6000, 0, 0), 1024, 8192, "chunked"))

    def test_no_floor_once_the_tree_owns_the_prompt_end(self):
        c = _cache(page=256)
        for phase in ("prompt_end", None):
            self.assertIsNone(T.prompt_tail_floor(c, _req(6000, 10, 5888), 1024, 8192, phase))
            self.assertIsNone(T.prompt_tail_floor(c, _req(6000, 10, 6144), 1024, 8192, phase))

    def test_no_floor_outside_scope(self):
        r = _req(6000, 10, 0)
        for phase in ("prompt_end", None):
            self.assertIsNone(T.prompt_tail_floor(_cache(page=256, chunk=True), r, 1024, 8192, phase))
            self.assertIsNone(T.prompt_tail_floor(_cache(page=256, is_eagle=True), r, 1024, 8192, phase))
            self.assertIsNone(T.prompt_tail_floor(_cache(page=256, enable_hicache=True), r, 1024, 8192, phase))
            self.assertIsNone(T.prompt_tail_floor(_cache(page=256, enable_session_radix_cache=True), r, 1024, 8192,
                                                  phase))
            m = _cache(page=256)
            m.is_mamba_enabled = True
            self.assertIsNone(T.prompt_tail_floor(m, r, 1024, 8192, phase))
            self.assertIsNone(T.prompt_tail_floor(_cache(page=256), _req(200, 10, 0), 1024, 8192, phase))  # < 1 page
            self.assertIsNone(T.prompt_tail_floor(SimpleNamespace(is_chunk_cache=lambda: False), r, 1024, 8192,
                                                  phase))

    def test_decode_hold_bound(self):
        c = _cache(page=256)
        floor = 5888 - 1024
        r = _req(6000, 0, 0)
        r.seqlen = floor + 1024 + 8192
        self.assertEqual(T.prompt_tail_floor(c, r, 1024, 8192, None), floor)          # at the bound: still held
        setattr(r, T.REQ_ATTR, "held")
        r.seqlen += 1
        self.assertIsNone(T.prompt_tail_floor(c, r, 1024, 8192, None))                # past it: stock
        self.assertEqual(getattr(r, T.REQ_ATTR), "released")
        self.assertEqual(T.STATE.released, 1)
        self.assertIsNone(T.prompt_tail_floor(c, r, 1024, 8192, None))
        self.assertEqual(T.STATE.released, 1)                                          # counted once
        self.assertEqual(T.prompt_tail_floor(c, r, 1024, 8192, "prompt_end"), floor)  # the insert has no bound


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

    def test_only_the_production_line_sets_it_and_volatile(self):
        # v2.6.6: the production line ends with DSV41_SWA_PROMPT_TAIL_KEEP=1024 (once); the hold stays at its code
        # default (no profile line sets it); no other profile line sets either key
        if not (ROOT / ".env.tp4.example").exists():
            self.skipTest("no .env.tp4.example in this tree")
        lines = (ROOT / ".env.tp4.example").read_text().splitlines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        prod = lines[i - 1]
        self.assertTrue(prod.startswith('#EXTRA_CONTAINER_ENV="'), prod[:60])
        items = prod.split('"')[1].split()
        # v2.7.5's keys come last (V275_TAIL): checked here, then the v2.7.4 line's checks below run on the rest
        self.assertEqual(items[-len(V275_TAIL):], V275_TAIL)
        items = items[:-len(V275_TAIL)]
        # v2.7.1: the seven new keys follow it; v2.7.2: the CED skip; v2.7.3: stage 2b, the GEMM tactic and the W2 /
        # evict-first decode keys come last
        self.assertEqual(items[-16], "DSV41_SWA_PROMPT_TAIL_KEEP=1024")
        self.assertTrue(all(x.startswith(("DSV41_PREFILL_GEMM_ROW_SPLIT", "DSV41_L2_PREFETCH_WO", "DSV41_EXTEND_NOSYNC",
                                          "DSV41_SAMPLED_")) for x in items[-15:-8]), items[-15:-8])
        self.assertEqual(items[-8:], ["DSV41_PREFILL_CED_SKIP=1", "DSV41_PREFILL_SP_FP8_MOE=1", "DSV41_PREFILL_GEMM_TACTIC=1",
                                      "DSV41_L2_PREFETCH_WOB2_MB=10", "DSV41_WO_A_W8_EVICT_FIRST=1", "DSV41_DENSE_TILE_TABLE=1", "DSV41_PREFILL_MOE_FOLD=1", "DSV41_WO_A_CFG_TABLE=1"])
        self.assertEqual(sum(x.startswith("DSV41_SWA_PROMPT_TAIL") for x in items), 1)
        on = [line for line in lines if "DSV41_SWA_PROMPT_TAIL" in line and not line.startswith("# ")]
        self.assertEqual(on, [prod])
        self.assertEqual(T.config({T.ENV_KEEP: "1024"}), (1024, T.DEFAULT_HOLD))
        self.assertEqual(T.DEFAULT_HOLD, 8192)
        ak = (ROOT / "adapter" / "autotune_keep.py").read_text()
        for k in ("DSV41_SWA_PROMPT_TAIL_KEEP", "DSV41_SWA_PROMPT_TAIL_HOLD"):
            self.assertIn(f'"{k}"', ak)

    def test_dockerfile_and_ci_run_this_test(self):
        self.assertIn("tests/test_swa_prompt_tail.py", (ROOT / "Dockerfile.canary-roce").read_text())
        ci = ROOT / ".github/workflows/offline-tests.yml"
        if not ci.exists():  # the image ships /opt/dsv41 without .github
            self.skipTest("no .github in this tree (image)")
        self.assertIn("tests/test_swa_prompt_tail.py", ci.read_text())

    def test_pins_are_hex(self):
        self.assertEqual(len(T.PINS), 9)
        for k, v in T.PINS.items():
            self.assertRegex(v, r"^[0-9a-f]{64}$", k)


# ---------------------------------------------------------------- engine
@unittest.skipUnless(HAVE_SGLANG, "needs the pinned SGLang tree")
class Engine(unittest.TestCase):
    PAGE, WIN, CHUNK = 4, 4, 16

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
        U = urc.UnifiedRadixCache
        cls.stock = {"swa_retain_floor": U.swa_retain_floor, "cache_unfinished_req": U.cache_unfinished_req}

    def setUp(self):
        for k, v in self.stock.items():
            setattr(self.urc.UnifiedRadixCache, k, v)
        T.STATE.__dict__.update(installed=False, active=False, prompt_end=0, decode=0, requests=0, released=0,
                                errors=0)

    tearDown = setUp

    def _arm(self, keep, hold=T.DEFAULT_HOLD, thin=False):
        if thin:
            import swa_thin
            self.assertTrue(swa_thin.install(self.urc, env={swa_thin.ENV: "1"}))
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

    def _swa_held(self, cache):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        tc = cache.tree_core
        return tc.component_evictable_size_[ComponentType.SWA] + tc.component_protected_size_[ComponentType.SWA]

    def _visit(self, cache, alloc, pool, rid, prompt, nout, prompt_end_insert=True, chunk=None, after_chunks=None):
        """One request as the scheduler runs it: match + lock, the prompt in extends of `chunk` tokens (chunked inserts
        in between), the first token, the prompt-end insert (skipped to reproduce a request-owned prompt end), `nout`
        decode steps each followed by the real maybe_evict_swa, the finished insert. Returns (reused, max SWA the
        request held past cache_protected_len during decode)."""
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
        pos = m
        step = chunk or len(prompt)
        while pos < len(prompt):
            end = min(pos + step, len(prompt))
            idx = self._alloc(alloc, end - pos)
            pool.write((req.kv.req_pool_idx, slice(pos, end)), idx)
            req.set_extend_range(pos, end)
            req.kv.kv_committed_len = end
            if end < len(prompt):
                cache.cache_unfinished_req(req, chunked=True)
            pos = end
        if after_chunks is not None:
            after_chunks(cache)
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

    def _turns(self, keep, prompts, hold=T.DEFAULT_HOLD, prompt_end_insert=True, nout=30, thin=False, chunk=None):
        """Fresh cache, the prompts in order; returns [(reused, held)], the tree's SWA after the first request, and
        the applied counters."""
        self.setUp()                                              # every run starts from the stock engine
        self._arm(keep, hold, thin)
        cache, alloc, pool = self._build()
        out, first = [], None
        for i, p in enumerate(prompts):
            out.append(self._visit(cache, alloc, pool, i + 1, p, nout, prompt_end_insert, chunk))
            if first is None:
                first = self._swa_held(cache)
        return out, first, (T.STATE.prompt_end, T.STATE.decode, T.STATE.requests, T.STATE.released)

    DOC = list(range(1000, 1040))                                 # 40 tokens: ends on a page boundary

    def _q(self, k):
        return [5000 + 10 * k + j for j in range(6)]               # a 6-token question: prompt 46, page_floor 44

    def test_pins_match_engine(self):
        self.assertEqual(T.pin_drift(T.pin_targets(self.urc, self.swa, self.common, self.sb)), [])

    def test_divergence_before_the_last_page(self):
        # visit k = DOC + question k: the common prefix ends at 40 < page_floor(46) = 44
        prompts = [self.DOC + self._q(k) for k in range(3)]
        stock, s_tree, _ = self._turns(0, prompts)
        armed, a_tree, cnt = self._turns(8, prompts)
        print(f"\n  DOC + new question: stock reuse {[m for m, _ in stock]}, keep=8 {[m for m, _ in armed]}; "
              f"tree SWA after visit 1 stock {s_tree} / keep {a_tree}; counters {cnt}")
        self.assertEqual([m for m, _ in stock], [0, 0, 40])       # their pattern: 2nd visit 0, hits from the 3rd
        self.assertEqual([m for m, _ in armed], [0, 40, 40])      # kept: the 2nd visit matches the whole DOC
        self.assertEqual([h for _, h in armed], [h for _, h in stock])   # decode frees stay stock
        self.assertEqual(cnt[1:], (0, 0, 0))                      # no decode-time floor: the tree owned the prompt end
        self.assertEqual(cnt[0], 3)                               # one prompt-end insert per visit
        self.assertGreater(a_tree, s_tree)                        # the cost: the kept tail in the tree
        self.assertLessEqual(a_tree - s_tree, 8)

    def test_revisit_after_the_prompt_end_is_unchanged(self):
        base = self.DOC + self._q(0)
        prompts = [base, base + [7, 8, 9]]
        stock, _, _ = self._turns(0, prompts)
        armed, _, cnt = self._turns(8, prompts)
        self.assertEqual(stock[1][0], 44)
        self.assertEqual([m for m, _ in armed], [m for m, _ in stock])
        self.assertEqual([h for _, h in armed], [h for _, h in stock])
        self.assertEqual(cnt[1], 0)

    def test_chunked_inserts_are_stock(self):
        # a 58-token prompt in 16-token chunks (inserts at 16, 32, 48; page_floor 56): the tree after the chunked
        # inserts is identical with the module, and only the prompt-end insert uses the floor
        prompt = list(range(3000, 3058))
        snaps = []
        for keep in (0, 8):
            self.setUp()
            self._arm(keep)
            cache, alloc, pool = self._build()
            seen = []

            def snap(c):
                tc = c.tree_core
                seen.append((dict(tc.component_evictable_size_), dict(tc.component_protected_size_),
                             alloc.swa_available_size(), alloc.full_available_size()))
            self._visit(cache, alloc, pool, 1, prompt, 30, chunk=self.CHUNK, after_chunks=snap)
            snaps.append(seen[0])
            if keep:
                self.assertEqual(T.STATE.prompt_end, 1)          # only the prompt-end insert used the floor
        self.assertEqual(snaps[0], snaps[1])

    def test_composes_with_swa_thin(self):
        prompts = [self.DOC + self._q(k) for k in range(3)]
        armed, _, _ = self._turns(8, prompts)
        both, _, _ = self._turns(8, prompts, thin=True)
        import swa_thin
        self.assertTrue(swa_thin.STATS.installed)
        self.assertEqual([m for m, _ in both], [m for m, _ in armed])
        self.assertTrue(getattr(self.urc.UnifiedRadixCache.cache_unfinished_req, "_dsv41_swa_thin", False))

    def test_request_owned_prompt_end_is_kept(self):
        base = self.DOC + self._q(0)
        prompts = [base, base + [7, 8, 9]]
        stock, _, _ = self._turns(0, prompts, prompt_end_insert=False)
        armed, _, cnt = self._turns(8, prompts, prompt_end_insert=False)
        print(f"\n  request-owned prompt end: stock revisit {stock[1][0]}, keep=8 {armed[1][0]}; "
              f"held SWA stock {stock[0][1]} / keep {armed[0][1]}; counters {cnt}")
        self.assertEqual(stock[1][0], 0)                          # their docstring's case: nothing reusable
        self.assertEqual(armed[1][0], 44)                         # kept: it matches the prompt end
        self.assertEqual(cnt[2], 2)
        self.assertLessEqual(armed[0][1], 8 + 30 + self.WIN + self.PAGE)

    def test_hold_bound_falls_back_to_stock(self):
        base = self.DOC + self._q(0)
        prompts = [base, base + [7, 8, 9]]
        stock, _, _ = self._turns(0, prompts, prompt_end_insert=False)
        capped, _, cnt = self._turns(8, prompts, hold=8, prompt_end_insert=False)
        print(f"\n  hold 8: revisit {capped[1][0]}, max held SWA {capped[0][1]}, counters {cnt}")
        self.assertEqual(capped[1][0], stock[1][0])               # 30-token outputs pass the bound: stock outcome
        self.assertEqual(cnt[3], 2)
        self.assertLessEqual(capped[0][1], 8 + 8 + self.WIN + 2 * self.PAGE)
        roomy, _, _ = self._turns(8, prompts, hold=64, prompt_end_insert=False)
        self.assertEqual(roomy[1][0], 44)

    def test_deterministic(self):
        prompts = [self.DOC + self._q(k) for k in range(3)]
        self.assertEqual(self._turns(8, prompts)[:2], self._turns(8, prompts)[:2])

    def test_sitecustomize_installs_the_hook(self):
        # The engine's import hook (active with DSV41_SOURCE, as boot.py sets it) next to the v2.6.5 scheduler-side
        # switches (thin, cone, bypass, decode yield); off, it never imports the module.
        import subprocess
        adapter = str(ROOT / "adapter")
        probe = ("import sglang.srt.managers.scheduler as S, sglang.srt.managers.schedule_batch as SB,"
                 " sglang.srt.mem_cache.unified_radix_cache as U, sys\n"
                 "thin = sys.modules.get('swa_thin')\n"
                 "print('HOOKS', getattr(U.UnifiedRadixCache.swa_retain_floor, '_dsv41_prompt_tail_hook', False),"
                 " 'swa_prompt_tail' in sys.modules,"
                 " getattr(SB.Req.init_next_round_input, '_dsv41_swa_cone_hook', False),"
                 " getattr(U.UnifiedRadixCache.cache_unfinished_req, '_dsv41_swa_thin', False),"
                 " bool(thin and thin.STATS.installed and thin.STATS.active))\n")
        for keep, want, log in (("1024", "True True True True True", "DSV41_SWA_PROMPT_TAIL installed: keep=1024"),
                                ("0", "False False True True True", None),
                                ("abc", "False True True True True", "DSV41_SWA_PROMPT_TAIL not installed")):
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
