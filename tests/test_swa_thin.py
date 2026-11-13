#!/usr/bin/env python3
"""DSV41_SWA_THIN (adapter/swa_thin.py): superseded chunk-boundary SWA checkpoints go first.

Offline part (no SGLang): flag parsing and the demotion rules on a minimal tree / LRU model.
Engine part (skipped without SGLang; run in a CPU container with the pinned SGLang tree on
PYTHONPATH): a real UnifiedRadixCache (FULL + SWA, page 4, window 4, the same 2-page
checkpoint geometry as page 256 / window 128) replays the issue16 pattern: documents A and
B are prefilled in chunks, then a long document C, then SWA pressure. Stock LRU evicts A's
and B's last chunk checkpoints (a request on A + another suffix reuses 0 tokens); with the
hook A and B keep them (reuse = last chunk boundary inside the document) and C keeps its
own last chunk checkpoint. The tree's sanity check, sizes and pin hashes are checked too.
"""
import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
import swa_thin as T  # noqa: E402

HAVE_SGLANG = importlib.util.find_spec("sglang") is not None


# ---------------------------------------------------------------- offline model
class _CD:
    def __init__(self, value=True, lock_ref=0):
        self.value = object() if value else None
        self.lock_ref = lock_ref


class _Node:
    def __init__(self, parent, key_len, swa=True, lock_ref=0):
        self.parent = parent
        self.key = [0] * key_len
        self.children = {}
        self.component_data = {"SWA": _CD(swa, lock_ref)}
        self.lru_prev, self.lru_next = {0: None}, {0: None}
        if parent is not None:
            parent.children[id(self)] = self


class _LRU:
    """Order-only stand-in with the attributes the hook touches."""

    def __init__(self):
        self._pt = 0
        self.mid = None
        self.head, self.tail = _Node(None, 0), _Node(None, 0)
        self.head.lru_next[0], self.tail.lru_prev[0] = self.tail, self.head
        self.cache = {}

    def _add_node_after(self, prev, node):
        node.lru_prev[0], node.lru_next[0] = prev, prev.lru_next[0]
        prev.lru_next[0].lru_prev[0] = node
        prev.lru_next[0] = node

    def _remove_node(self, node):
        node.lru_prev[0].lru_next[0] = node.lru_next[0]
        node.lru_next[0].lru_prev[0] = node.lru_prev[0]
        node.lru_prev[0] = node.lru_next[0] = None

    def insert_mru(self, node):
        self.cache[id(node)] = node
        self._add_node_after(self.head, node)

    def in_list(self, node):
        return node is not None and id(node) in self.cache

    def order(self):  # MRU -> LRU
        out, x = [], self.head.lru_next[0]
        while x is not self.tail:
            out.append(x)
            x = x.lru_next[0]
        return out


def _offline_tree():
    root = _Node(None, 0, swa=False)
    lru = _LRU()
    tc = SimpleNamespace(root_node=root, page_size=4, lru_lists={"SWA": lru},
                         enable_session_radix_cache=False, enable_hicache=False)
    cache = SimpleNamespace(tree_core=tc, components={"SWA": SimpleNamespace(sliding_window_size=4)},
                            is_mamba_enabled=False)
    return root, lru, cache


class Offline(unittest.TestCase):
    def test_flag(self):
        self.assertFalse(T.enabled({}))
        for v in ("0", "", "off", "false", "no"):
            self.assertFalse(T.enabled({T.ENV: v}))
            self.assertIsNone(T.mode({T.ENV: v}))
        for v in ("1", "on", "yes", "all"):
            self.assertTrue(T.enabled({T.ENV: v}))
            self.assertEqual(T.mode({T.ENV: v}), "all")
        self.assertEqual(T.mode({T.ENV: "chunk"}), "chunk")
        self.assertEqual(T.mode({T.ENV: " CHUNK "}), "chunk")
        self.assertFalse(T.install(SimpleNamespace(), env={}))  # off: touches nothing

    def test_demotes_superseded_checkpoint_only(self):
        root, lru, cache = _offline_tree()
        body1 = _Node(root, 8, swa=False)
        ck1 = _Node(body1, 8)          # chunk boundary 1 (2 pages of SWA)
        body2 = _Node(ck1, 8, swa=False)
        ck2 = _Node(body2, 8)          # chunk boundary 2 (the request's new position)
        other = _Node(root, 16)        # another document's resume point
        for n in (other, ck1, ck2):
            lru.insert_mru(n)
        self.assertEqual(lru.order(), [ck2, ck1, other])
        self.assertEqual(T.demote_superseded(cache, ck1, ck2, "SWA"), 1)
        self.assertEqual(lru.order(), [ck2, other, ck1])   # ck1 evicted first now
        self.assertEqual(set(lru.order()), {ck1, ck2, other})

    def test_keeps_locked_branch_unrelated_and_root(self):
        root, lru, cache = _offline_tree()
        ck1 = _Node(root, 8)
        ck2 = _Node(_Node(ck1, 8, swa=False), 8)
        side = _Node(ck1, 8)           # second child: ck1 is a branch point
        for n in (ck1, ck2, side):
            lru.insert_mru(n)
        before = lru.order()
        self.assertEqual(T.demote_superseded(cache, ck1, ck2, "SWA"), 0)   # branch point
        ck1.children.pop(id(side))
        ck1.component_data["SWA"].lock_ref = 1
        self.assertEqual(T.demote_superseded(cache, ck1, ck2, "SWA"), 0)   # locked
        ck1.component_data["SWA"].lock_ref = 0
        self.assertEqual(T.demote_superseded(cache, side, ck2, "SWA"), 0)  # not an ancestor
        self.assertEqual(T.demote_superseded(cache, root, ck2, "SWA"), 0)  # root
        self.assertEqual(T.demote_superseded(cache, ck2, ck2, "SWA"), 0)   # no move
        self.assertEqual(lru.order(), before)
        cache.tree_core.enable_session_radix_cache = True
        self.assertEqual(T.demote_superseded(cache, ck1, ck2, "SWA"), 0)   # sessions: no-op

    def test_window_span_bounds_the_walk(self):
        root, lru, cache = _offline_tree()
        far = _Node(root, 4)           # beyond window + page (4 + 4) above ck
        mid = _Node(far, 4)
        ck = _Node(mid, 4)
        new = _Node(_Node(ck, 8, swa=False), 8)
        for n in (far, mid, ck, new):
            lru.insert_mru(n)
        self.assertEqual(T.demote_superseded(cache, ck, new, "SWA"), 2)    # ck, mid
        self.assertEqual(lru.order()[-2:], [ck, mid])
        self.assertIn(far, lru.order()[:2])


# ---------------------------------------------------------------- engine part
@unittest.skipUnless(HAVE_SGLANG, "needs the pinned SGLang tree")
class Engine(unittest.TestCase):
    PAGE, WIN, CHUNK = 4, 4, 16

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import torch  # noqa: F401
        from array import array  # noqa: F401
        from sglang.srt.mem_cache import unified_radix_cache as urc
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedLRUList
        cls.urc = urc
        cls.LRU = UnifiedLRUList

    def _build(self, swa_size):
        import torch
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=self.PAGE))
        full_size = 1024
        pool = ReqToTokenPool(size=16, max_context_len=512, device="cpu", enable_memory_saver=False)
        kv = SWAKVPool(size=full_size, size_swa=swa_size, page_size=self.PAGE, dtype=torch.bfloat16,
                       head_num=1, head_dim=8, swa_attention_layer_ids=[1],
                       full_attention_layer_ids=[0], device="cpu")
        alloc = SWATokenToKVPoolAllocator(size=full_size, size_swa=swa_size, page_size=self.PAGE,
                                          dtype=torch.bfloat16, device="cpu", kvcache=kv,
                                          need_sort=False)
        cache = self.urc.UnifiedRadixCache(params=CacheInitParams(
            req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=self.PAGE,
            disable=False, sliding_window_size=self.WIN,
            tree_components=(ComponentType.FULL, ComponentType.SWA), eviction_policy="lru",
            is_eagle=False))
        return cache, alloc, pool

    def _alloc(self, alloc, n):
        assert n % self.PAGE == 0
        full = alloc.full_attn_allocator.alloc(n)
        swa = alloc.swa_attn_allocator.alloc(n)
        assert full is not None and swa is not None, "test pool too small"
        alloc.full_to_swa_index_mapping[full] = swa
        return full

    def _prefill(self, cache, alloc, pool, rid, prompt, out_len=4, stop_at=None):
        """Chunked prefill of `prompt` (CHUNK tokens per step), then out_len decode tokens."""
        from array import array
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q"),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=out_len))
        pool.alloc([req])
        req.origin_input_ids = array("q", prompt)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", prompt)
        req.last_node = cache.root_node_handle()
        req.kv.cache_protected_len = 0
        req.lock_receipt = DecLockRefParams()
        req.extra_key = None
        pos = 0
        while pos < len(prompt):
            end = min(pos + self.CHUNK, len(prompt))
            idx = self._alloc(alloc, end - pos)
            pool.write((req.kv.req_pool_idx, slice(pos, end)), idx)
            req.set_extend_range(pos, end)
            req.kv.kv_committed_len = end
            cache.cache_unfinished_req(req, chunked=end < len(prompt))
            pos = end
            if stop_at is not None and pos >= stop_at:
                return req
        out = list(range(90000 + rid * 100, 90000 + rid * 100 + out_len))
        idx = self._alloc(alloc, out_len)
        pool.write((req.kv.req_pool_idx, slice(len(prompt), len(prompt) + out_len)), idx)
        req.output_ids = array("q", out)
        req.kv.kv_committed_len = len(prompt) + out_len
        cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=len(prompt) + out_len)
        pool.free(req)
        return req

    def _turn(self, cache, alloc, pool, rid, prompt, out):
        """One request the way the scheduler runs it: prefix match + lock, chunked prefill of the
        rest, the prompt-end insert (chunked=False), decode `out`, the finished insert.
        Returns the node the finished insert ended on (the request's resume point)."""
        from array import array
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q"),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=len(out)))
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
        while pos < len(prompt):
            end = min(pos + self.CHUNK, len(prompt))
            idx = self._alloc(alloc, -(-(end - pos) // self.PAGE) * self.PAGE)[: end - pos]
            pool.write((req.kv.req_pool_idx, slice(pos, end)), idx)
            req.set_extend_range(pos, end)
            req.kv.kv_committed_len = end
            cache.cache_unfinished_req(req, chunked=end < len(prompt))
            pos = end
        idx = self._alloc(alloc, -(-len(out) // self.PAGE) * self.PAGE)[: len(out)]
        pool.write((req.kv.req_pool_idx, slice(len(prompt), len(prompt) + len(out))), idx)
        req.output_ids = array("q", out)
        req.kv.kv_committed_len = len(prompt) + len(out)
        cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=len(prompt) + len(out))
        pool.free(req)
        return req.last_node, m

    def _sessions(self, mode):
        """Agent sessions: S and three others take turns (each prompt = the previous prompt + its
        output + a few new tokens); then S idles while the others take more turns. Returns how
        many SWA tokens can be evicted (LRU order, one page at a time, nothing matched meanwhile)
        before S's resume point loses its SWA window, the per-turn reuse, and the ledgers."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        cls = self.urc.UnifiedRadixCache
        saved = cls.cache_unfinished_req
        try:
            if mode:
                self.assertTrue(T.install(self.urc, env={T.ENV: mode}))
            cache, alloc, pool = self._build(swa_size=512)
            hist = {n: list(range(1000 * (k + 1), 1000 * (k + 1) + 24)) for k, n in enumerate("SOPQ")}
            reuse, rid, resume = [], [0], {}

            def turn(n, new):
                rid[0] += 1
                prompt = hist[n] + new
                out = list(range(90000 + 10 * rid[0], 90000 + 10 * rid[0] + 4))
                node, m = self._turn(cache, alloc, pool, rid[0], prompt, out)
                reuse.append((n, m, len(hist[n])))
                hist[n] = prompt + out
                resume[n] = node
                cache.sanity_check()

            for r in range(3):                       # everyone active
                for k, n in enumerate("SOPQ"):
                    turn(n, list(range(5000 + 100 * k + 10 * r, 5004 + 100 * k + 10 * r)))
            for r in range(3, 9):                    # S idle, the others keep going
                for k, n in enumerate("OPQ"):
                    turn(n, list(range(5000 + 100 * (k + 1) + 10 * r, 5004 + 100 * (k + 1) + 10 * r)))
            tc = cache.tree_core
            sizes = (dict(tc.component_evictable_size_), dict(tc.component_protected_size_),
                     alloc.swa_available_size(), alloc.full_available_size())
            s_node = tc.node_by_id(resume["S"])
            evicted = 0
            while s_node.component_data[ComponentType.SWA].value is not None:
                if tc.component_evictable_size_[ComponentType.SWA] <= 0:
                    break
                cache.evict(EvictParams(swa_num_tokens=self.PAGE))
                evicted += self.PAGE
                cache.sanity_check()
            return evicted, reuse, sizes
        finally:
            cls.cache_unfinished_req = saved

    def test_prompt_end_demotion_keeps_idle_session(self):
        # Turn traffic of other sessions: stock keeps every superseded turn checkpoint in LRU order, so the
        # idle session's resume point is among the first to go; mode "all" evicts the others' superseded
        # match-end checkpoints first.
        stock, r0, z0 = self._sessions(None)
        chunk, r1, z1 = self._sessions("chunk")
        allm, r2, z2 = self._sessions("1")
        print(f"\n  SWA tokens evictable before S loses its resume point: stock {stock}, chunk {chunk}, all {allm}")
        # Same tree, same reuse on every turn, same ledgers: only the LRU order differs before pressure.
        self.assertEqual(r0, r1)
        self.assertEqual(r0, r2)
        self.assertEqual(z0, z1)
        self.assertEqual(z0, z2)
        # every turn after the first reused the whole previous prompt + output (a resume, not a re-prefill)
        for n, m, prev in r0[4:]:
            self.assertEqual(m, prev // self.PAGE * self.PAGE, (n, m, prev))
        self.assertEqual(chunk, stock)               # no chunked steps here: chunk mode changes nothing
        self.assertGreater(allm, stock)

    def _reuse(self, cache, tokens):
        from array import array
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        return len(cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens)))).device_indices)

    def _scenario(self, flag, pressure=0.6, probes=False):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        cls = self.urc.UnifiedRadixCache
        saved = cls.cache_unfinished_req
        try:
            if flag:
                self.assertTrue(T.install(self.urc, env={T.ENV: "1"}))
            cache, alloc, pool = self._build(swa_size=256)
            docs = {
                "A": list(range(1000, 1052)),   # 52 tokens: chunk boundaries 16, 32, 48
                "B": list(range(2000, 2068)),   # 68: 16 .. 64
                "C": list(range(3000, 3164)),   # 164: 16 .. 160 (10 interior checkpoints)
            }
            for i, name in enumerate(("A", "B", "C")):
                suffix = list(range(5000 + 10 * i, 5008 + 10 * i))
                self._prefill(cache, alloc, pool, i, docs[name] + suffix)
                cache.sanity_check()
            tc = cache.tree_core
            swa = tc.component_evictable_size_[ComponentType.SWA]
            sizes = (dict(tc.component_evictable_size_), dict(tc.component_protected_size_),
                     alloc.swa_available_size(), alloc.full_available_size())
            if pressure:
                # SWA pressure: free a share of what the tree holds (another long prefill).
                cache.evict(EvictParams(swa_num_tokens=int(swa * pressure)))
                cache.sanity_check()
            reuse = {n: self._reuse(cache, d + [7]) for n, d in docs.items()}
            if probes:  # divergence right after every page of every document
                for n, d in docs.items():
                    for cut in range(self.PAGE, len(d) + 1, self.PAGE):
                        reuse[(n, cut)] = self._reuse(cache, d[:cut] + [7])
            return reuse, swa, sizes
        finally:
            cls.cache_unfinished_req = saved

    def test_pins_match_engine(self):
        self.assertEqual(T.pin_drift(self.urc.UnifiedRadixCache, self.LRU), [])

    def test_stock_loses_older_documents_hook_keeps_them(self):
        stock, held_stock, _ = self._scenario(flag=False)
        thin, held_thin, _ = self._scenario(flag=True)
        print(f"\n  stock reuse {stock}\n  thin reuse  {thin}\n  SWA held before pressure {held_stock}")
        self.assertEqual(held_stock, held_thin)          # same SWA held before pressure
        # Stock LRU: A's and B's checkpoints were the oldest -> no reuse at all.
        self.assertEqual(stock["A"], 0, stock)
        self.assertEqual(stock["B"], 0, stock)
        # Hook: interior checkpoints went first; every document keeps its last chunk boundary.
        self.assertEqual(thin["A"], 48, thin)
        self.assertEqual(thin["B"], 64, thin)
        self.assertEqual(thin["C"], 160, thin)
        self.assertGreater(T.STATS.demoted, 0)

    def test_chunk_checkpoint_keeps_two_pages(self):
        # The cost model: an unlocked chunk-boundary checkpoint holds max(window, page) rounded
        # to whole pages below the boundary = 2 pages (512 slots at page 256 / window 128).
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        cache, alloc, pool = self._build(swa_size=256)
        self._prefill(cache, alloc, pool, 0, list(range(1000, 1048)), stop_at=32)
        held = cache.tree_core.component_evictable_size_[ComponentType.SWA]
        # Tree: [8,16) live (boundary 16, unlocked), [24,32) live with [28,32) under the request's
        # window lock -> evictable = 2 pages + 1 page.
        self.assertEqual(held, 3 * self.PAGE)
        spans, node, pos = [], cache.tree_core.root_node, 0
        while node.children:
            (node,) = node.children.values()
            if node.component_data[ComponentType.SWA].value is not None:
                spans.append((pos, pos + len(node.key)))
            pos += len(node.key)
        self.assertEqual(spans, [(8, 12), (12, 16), (24, 28), (28, 32)])

    def test_without_pressure_identical_to_stock(self):
        # The hook only reorders the LRU: with no eviction every match, the pool ledgers and the
        # allocators are identical to stock, at every page of every document.
        stock = self._scenario(flag=False, pressure=0, probes=True)
        thin = self._scenario(flag=True, pressure=0, probes=True)
        self.assertEqual(stock[0], thin[0])
        self.assertEqual(stock[1:], thin[1:])
        self.assertEqual(thin[0][("C", 100)], 96)        # interior checkpoints still usable


ROOT = Path(__file__).resolve().parent.parent


class Profile(unittest.TestCase):
    """v2.6.4: the production line of .env.tp4.example sets DSV41_SWA_THIN=1 (after the decode yield) and the
    production EXTRA_SGLANG_ARGS carry --swa-prefix-tails 384 (was 128); the module stays off when unset; the switch is
    volatile in adapter/autotune_keep.py (LRU order after boot only); the canary-roce Dockerfile runs this test."""

    def _production_line(self):
        lines = (ROOT / ".env.tp4.example").read_text().splitlines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        self.assertTrue(lines[i - 1].startswith('#EXTRA_CONTAINER_ENV="'), lines[i - 1][:60])
        return lines[i - 1]

    def _production_args(self):
        lines = (ROOT / ".env.tp4.example").read_text().splitlines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        args = [line for line in lines[i:i + 8] if line.startswith('#EXTRA_SGLANG_ARGS="')]
        self.assertEqual(len(args), 1, lines[i:i + 8])
        return args[0].split('"')[1].split()

    def _volatile(self):
        tree = ast.parse((ROOT / "adapter" / "autotune_keep.py").read_text())
        return next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_VOLATILE")

    def test_production_line_sets_thin_once_after_the_yield(self):
        items = self._production_line().split('"')[1].split()
        self.assertEqual([x for x in items if x.startswith("DSV41_SWA_THIN")], ["DSV41_SWA_THIN=1"])
        i = items.index("DSV41_SWA_THIN=1")
        # v2.7: swan retired from the line (tests/test_v41_pool.py Profile)
        self.assertEqual(items[i - 2:i + 1], ["DSV41_ROCE_IDLE_PARK=1", "DSV41_DECODE_YIELD=33", "DSV41_SWA_THIN=1"])
        self.assertEqual(T.mode({T.ENV: "1"}), "all")

    def test_production_args_tails_384_once(self):
        args = self._production_args()
        idx = [n for n, a in enumerate(args) if a == "--swa-prefix-tails"]
        self.assertEqual(len(idx), 1, args)
        self.assertEqual(args[idx[0] + 1], "384")

    def test_only_the_production_line_sets_it(self):
        text = (ROOT / ".env.tp4.example").read_text()
        on = [line for line in text.splitlines() if line.startswith("#EXTRA_CONTAINER_ENV=") and "SWA_THIN" in line]
        self.assertEqual(on, [self._production_line()])
        self.assertFalse([line for line in text.splitlines() if line.startswith("DSV41_SWA_THIN")])

    def test_volatile_in_autotune_keep(self):
        self.assertIn("DSV41_SWA_THIN", self._volatile())

    def test_default_off_in_module(self):
        self.assertIsNone(T.mode({}))
        self.assertIsNone(T.mode({T.ENV: "0"}))
        self.assertFalse(T.install(SimpleNamespace(), env={}))

    def test_dockerfile_runs_this_test(self):
        d = (ROOT / "Dockerfile.canary-roce").read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES= python3 /opt/dsv41/tests/test_swa_thin.py", d)


if __name__ == "__main__":
    unittest.main()
