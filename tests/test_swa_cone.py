#!/usr/bin/env python3
"""DSV41_SWA_CONE_REPLAY (adapter/swa_cone.py): exact replay of an SWA cone on a lost resume point.

Offline part (pure Python, no torch / SGLang):
  * the replay length rule (production: N >= 3,072, default 4,096) and the window planner;
  * a toy model with the same dependency structure (E early layers with their own sliding-window
    KV of window W, shared "global" latents written by source layers and read at positions <= row,
    a bounded late tail) shows the cone argument: rows >= F after a replay of N >= (W-1)*E rows are
    bit-identical to a cold prefill, and break with N one row shorter or when the replay rows'
    global writes are not masked (the SWA floor is not needed for rows >= F: defense in depth);
  * the admission guard and the batch plan on stand-ins.
Torch part (torch, no SGLang): page copy over page-row buffers (rows per full page 1 and 2), floor
vector, compressor write mask.
Engine part (CPU container + the pinned SGLang tree on PYTHONPATH):
  * every pinned engine source matches;
  * a real UnifiedRadixCache (FULL + SWA, page 4, window 4, decoder bounded replay on) loses a
    document's SWA resume point; the real Req.init_next_round_input with the hook plans the cone
    (prefix = resident [0, s), lock on the node ending at F); the forward is emulated with the real
    _low_ratio_compression_metadata, the real build_causal_swa_page_indices and the hook's copy /
    mask / floor; the real cache_unfinished_req / cache_finished_req then dedup: the tree keeps its
    resident pages, req_to_token re-points to them, no slot leaks, no SWA from inside the cone is
    adopted by the tree, and the next turn resumes from the new prompt end. A first chunk shorter than
    the replay reverts to the stock match. Also: real DeepSeekV4SingleKVPool / DeepSeekV4IndexerPool
    page rows are copied byte for byte.
"""
import ast
import importlib.util
import os
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
import swa_cone as C  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_SGLANG = importlib.util.find_spec("sglang") is not None


# v2.7.5's production-line tail (scale codes, then the evict-first experts + SHPF window)
V275_TAIL = ["DSV41_MOE_SCALE_CODES=1", "DSV41_MOE_SCALE_CODES_MIN_ROWS=2"]


# ---------------------------------------------------------------- toy model (pure Python)
class Toy:
    """Layers 0..E-1 run every row: per-layer window-W attention over their own KV plus dense
    attention over the global latents of the newest source layer at positions <= row. Source
    layers write those latents from their input. Late layers run the last W rows only, window
    floored at the tail start (decoder bounded replay), over their own KV and the last source's
    latents (written for every row by the last source, like layer 20)."""

    def __init__(self, early=3, late=2, window=4, dim=4, sources=(0, 2), seed=7):
        rnd = random.Random(seed)
        self.E, self.L, self.W, self.d, self.sources = early, late, window, dim, sources

        def mat():
            return [[rnd.uniform(-0.6, 0.6) for _ in range(dim)] for _ in range(dim)]
        self.wq = [mat() for _ in range(early + late)]
        self.wk = [mat() for _ in range(early + late)]
        self.wo = [mat() for _ in range(early + late)]
        self.wg = {s: mat() for s in list(sources) + [early]}
        self.emb = {}
        self.rnd = rnd

    def embed(self, tok):
        if tok not in self.emb:
            r = random.Random(tok * 7919 + 1)
            self.emb[tok] = [r.uniform(-1, 1) for _ in range(self.d)]
        return list(self.emb[tok])

    @staticmethod
    def mv(v, m):
        return [sum(v[i] * m[i][j] for i in range(len(v))) for j in range(len(m[0]))]

    def src_of(self, layer):
        return max(s for s in self.sources if s <= layer)

    def attend(self, q, keys):
        import math
        sc = [sum(a * b for a, b in zip(q, k)) for k in keys]
        mx = max(sc)
        ex = [math.exp(x - mx) for x in sc]
        z = sum(ex)
        return [sum(e * k[j] for e, k in zip(ex, keys)) / z for j in range(self.d)]

    def run(self, tokens, start, glob, floor=0, write_from=0, sink=None):
        """Rows [start, len(tokens)). glob[src][pos] holds resident latents; rows >= write_from
        overwrite their own latent (rows below keep the resident one). floor: SWA floor; when
        sink is given, window positions below floor read it instead of being skipped (no floor).
        Returns the late-tail output rows and the early-layer output per row."""
        n = len(tokens)
        glob = {k: dict(v) for k, v in glob.items()}
        h = {r: self.embed(tokens[r]) for r in range(start, n)}
        for layer in range(self.E):
            if layer in self.sources:
                for r in range(start, n):
                    if r >= write_from:
                        glob[layer][r] = [math_tanh(x) for x in self.mv(h[r], self.wg[layer])]
            kv = {r: self.mv(h[r], self.wk[layer]) for r in range(start, n)}
            g = glob[self.src_of(layer)]
            new = {}
            for r in range(start, n):
                win = []
                for j in range(r - self.W + 1, r + 1):
                    if j < 0:
                        continue
                    if j < floor:
                        if sink is not None:
                            win.append(sink)
                        continue
                    win.append(kv[j])
                keys = win + [g[j] for j in range(0, r + 1)]
                out = self.attend(self.mv(h[r], self.wq[layer]), keys)
                new[r] = [a + math_tanh(b) for a, b in zip(h[r], self.mv(out, self.wo[layer]))]
            h = new
        early_out = dict(h)
        # the last source (layer E) writes latents for every extend row, then the tail runs
        for r in range(start, n):
            if r >= write_from:
                glob[self.E][r] = [math_tanh(x) for x in self.mv(h[r], self.wg[self.E])]
        tail = list(range(max(start, n - self.W), n))
        for i in range(self.L):
            layer = self.E + i
            kv = {r: self.mv(h[r], self.wk[layer]) for r in tail}
            g = glob[self.E]
            new = {}
            for r in tail:
                keys = [kv[j] for j in range(max(tail[0], r - self.W + 1), r + 1)] + \
                       [g[j] for j in range(0, r + 1)]
                out = self.attend(self.mv(h[r], self.wq[layer]), keys)
                new[r] = [a + math_tanh(b) for a, b in zip(h[r], self.mv(out, self.wo[layer]))]
            h = {**h, **new}
        return {r: h[r] for r in tail}, early_out, glob


def math_tanh(x):
    import math
    return math.tanh(x)


class Offline(unittest.TestCase):
    def test_length_rule(self):
        # production: window 128, 20 early layers (late_layer_start), page 256
        self.assertEqual((128 - 1) * 20, 2540)                            # cone
        self.assertEqual((128 - 1) * 19 + 256 + 256, 2925)                # first-insert guard
        self.assertEqual(C.min_replay_tokens(128, 20, 256), 3072)
        self.assertGreaterEqual(C.DEFAULT_TOKENS, C.min_replay_tokens(128, 20, 256))
        self.assertEqual(C.min_replay_tokens(4, 3, 4), 16)               # the engine test geometry
        for w, e, p in ((128, 20, 256), (4, 3, 4), (8, 5, 16), (128, 40, 256)):
            n = C.min_replay_tokens(w, e, p)
            self.assertEqual(n % p, 0)
            self.assertGreaterEqual(n, (w - 1) * e)
            self.assertGreaterEqual(n - max(w, p) - p + 1, (w - 1) * (e - 1) + 1)

    def test_plan_window(self):
        P = 256
        self.assertEqual(C.plan_window(0, 300 * 1024, 4096, 8192, P), 300 * 1024 - 4096)
        self.assertIsNone(C.plan_window(0, None, 4096, 8192, P))           # full KV not past SWA
        self.assertIsNone(C.plan_window(0, 4096 + 8192 - P, 4096, 8192, P))  # saves < min_save
        self.assertEqual(C.plan_window(0, 4096 + 8192, 4096, 8192, P), 8192)
        self.assertIsNone(C.plan_window(290 * 1024, 300 * 1024, 4096, 8192, P))  # SWA hit is close
        self.assertIsNone(C.plan_window(0, 300 * 1024 + 1, 4096, 8192, P))  # not page aligned
        self.assertIsNone(C.plan_window(0, 300 * 1024, 4000, 8192, P))

    def _cone(self, n_replay, floor=True, mask=True, seed=7):
        toy = Toy(seed=seed)
        rnd = random.Random(seed)
        prompt = [rnd.randrange(1, 10 ** 6) for _ in range(60)]
        full = prompt + [rnd.randrange(1, 10 ** 6) for _ in range(12)]
        resident = {s: {} for s in list(toy.sources) + [toy.E]}
        # previous turn: cold prefill of `prompt` (its latents become the resident cache)
        _, _, resident = toy.run(prompt, 0, resident)
        F = len(prompt)
        cold_tail, cold_early, _ = toy.run(full, 0, {k: {} for k in resident})
        s = F - n_replay
        cone_tail, cone_early, _ = toy.run(
            full, s, resident, floor=s, write_from=F if mask else s,
            sink=None if floor else [9.0] * toy.d)
        same_tail = all(cone_tail[r] == cold_tail[r] for r in cold_tail)
        same_rows = all(cone_early[r] == cold_early[r] for r in range(F, len(full)))
        return same_tail and same_rows, cone_early, cold_early, s, F

    def test_toy_cone_exact_at_the_rule(self):
        need = (4 - 1) * 3                                   # (W - 1) * E = 9
        for n in (need, need + 3, 20):
            ok, cone, cold, s, F = self._cone(n)
            self.assertTrue(ok, f"N={n}")
            # inside the cone the early rows are NOT exact (they see a floored window)
            self.assertNotEqual(cone[s], cold[s])

    def test_toy_cone_breaks_below_the_rule(self):
        hits = sum(self._cone((4 - 1) * 3 - 1, seed=sd)[0] for sd in range(5))
        self.assertEqual(hits, 0)

    def test_toy_mask_is_load_bearing_floor_is_not(self):
        # Replay rows that overwrite the resident latents with their (inexact) own break rows >= F.
        self.assertFalse(self._cone(12, mask=False)[0])
        # The floor only decides what the first W-1 replay rows read below s (nothing, or a stale
        # slot): those rows are outside the dependency cone of every row >= F once N >= (W-1)*E, so
        # rows >= F are exact either way. The hook floors anyway (no read of the shared sink slot,
        # deterministic replay rows).
        self.assertTrue(self._cone(12, floor=False)[0])
        self.assertFalse(self._cone(8, floor=False)[0])

    def test_guarded_select(self):
        class R:  # AddReqResult stand-in
            NO_TOKEN = "no"
        calls = []

        def orig(adder, req, **kw):
            calls.append(dict(kw))
            ext = len(req.full) - len(req.prefix_indices)
            return SimpleNamespace(extend_len=min(ext, adder.rem), prefix_len=len(req.prefix_indices),
                                   is_chunked=ext > adder.rem)

        req = SimpleNamespace(full=list(range(100)), prefix_indices=list(range(40)), last_node=7,
                              last_host_node=7, best_match_node=7, host_hit_length=0,
                              swa_host_hit_length=0, swa_branching_seqlen=80, kv=SimpleNamespace(cache_protected_len=40))
        stock = C._snapshot(req)
        stock["prefix_indices"] = list(range(10))
        stock["cache_protected_len"] = 10
        stock["last_node"] = 3
        plan = C.Plan(s=40, full_hit=56, src_pages=[1, 2, 3, 4], stock=stock)
        setattr(req, C.REQ_ATTR, plan)
        adder = SimpleNamespace(rem=32)
        res = C.guarded_select(orig, adder, req, (), {"total_tokens": 100}, R)
        self.assertEqual(res.extend_len, 32)                 # a chunk covering the 16 replay rows: kept
        self.assertTrue(res.is_chunked)
        self.assertIs(getattr(req, C.REQ_ATTR), plan)
        whole = dict(vars(req), kv=SimpleNamespace(cache_protected_len=40))
        req2 = SimpleNamespace(**whole)
        adder.rem = 64                                       # the whole prompt (60) in one pass: it may
        res = C.guarded_select(orig, adder, req2, (), {"total_tokens": 100}, R)   # finish at its first token
        self.assertIsNone(getattr(req2, C.REQ_ATTR))         # -> stock match
        self.assertEqual(req2.prefix_indices, list(range(10)))
        self.assertEqual(calls[-1]["total_tokens"], 130)
        adder.rem = 12                                       # first chunk would end inside [s, F)
        res = C.guarded_select(orig, adder, req, (), {"total_tokens": 100}, R)
        self.assertIsNone(getattr(req, C.REQ_ATTR))
        self.assertEqual(req.prefix_indices, list(range(10)))
        self.assertEqual(req.kv.cache_protected_len, 10)
        self.assertEqual(req.last_node, 3)
        self.assertEqual(calls[-1]["total_tokens"], 130)     # + (s - m) rows, as add_one_req would debit
        self.assertEqual(res.prefix_len, 10)

    def test_build_cone_batch(self):
        def req(plan):
            r = SimpleNamespace(rid="r")
            setattr(r, C.REQ_ATTR, plan)
            return r
        a = req(None)
        b = req(C.Plan(s=8, full_hit=24, src_pages=[5, 6, 7, 8]))
        c = req(None)
        cone = C.build_cone_batch([a, b, c], [0, 8, 4], [12, 30, 4], 4)
        self.assertEqual(cone.entries, [(12, 16, 30, 8, [5, 6, 7, 8])])
        self.assertIsNone(getattr(b, C.REQ_ATTR))           # consumed: one forward only
        self.assertIsNone(C.build_cone_batch([a, c], [0, 4], [12, 4], 4))
        d = req(C.Plan(s=8, full_hit=24, src_pages=[5, 6, 7, 8]))
        with self.assertRaises(RuntimeError):
            C.build_cone_batch([d], [8], [12], 4)            # chunk shorter than the replay


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TorchPart(unittest.TestCase):
    def test_copy_full_pages(self):
        import torch
        g = torch.Generator().manual_seed(0)
        a = torch.randint(0, 255, (12, 40), dtype=torch.uint8, generator=g)     # 1 row per page
        b = torch.randint(0, 255, (24, 16), dtype=torch.uint8, generator=g)     # 2 rows per page
        a0, b0 = a.clone(), b.clone()
        src, dst = torch.tensor([3, 4, 9]), torch.tensor([7, 1, 2])
        self.assertEqual(C.copy_full_pages([(a, 1), (b, 2)], src, dst), 2)
        for s, d in zip(src.tolist(), dst.tolist()):
            self.assertTrue(torch.equal(a[d], a0[s]))
            self.assertTrue(torch.equal(b[2 * d:2 * d + 2], b0[2 * s:2 * s + 2]))
        untouched = [p for p in range(12) if p not in dst.tolist()]
        self.assertTrue(torch.equal(a[untouched], a0[untouched]))

    def test_floor_and_mask(self):
        import torch
        cone = C.ConeBatch(entries=[(4, 8, 10, 100, [1, 2])], page=4)
        f = C.floor_vector(cone, 20, "cpu")
        self.assertEqual(f.dtype, torch.int32)
        self.assertEqual(f.tolist(), [0] * 4 + [100] * 10 + [0] * 6)
        core = SimpleNamespace(c1_out_loc=torch.arange(1, 21), c2_out_loc=torch.full((20,), -1))
        C.mask_compressor_writes(core, cone)
        self.assertEqual(core.c1_out_loc.tolist(), list(range(1, 5)) + [0] * 8 + list(range(13, 21)))
        self.assertEqual(core.c2_out_loc[4:12].tolist(), [0] * 8)


# ---------------------------------------------------------------- engine part
@unittest.skipUnless(HAVE_SGLANG, "needs the pinned SGLang tree")
class Engine(unittest.TestCase):
    PAGE, WIN, CHUNK, EARLY = 4, 4, 64, 3
    TOKENS, MIN_SAVE = 16, 8

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from sglang.srt.mem_cache import unified_radix_cache as urc
        from sglang.srt.managers import schedule_batch as sb
        cls.urc, cls.sb = urc, sb

    @staticmethod
    def _reset_state():
        C.STATE.installed = set()
        C.STATE.active = False
        C.STATE.early_layers = None
        for k in ("planned", "reverted", "replayed", "rows", "errors"):
            setattr(C.STATE, k, 0)
        C._TREE_OK.clear()

    def setUp(self):
        self._saved = (self.sb.Req.init_next_round_input, self.sb.ScheduleBatch.prepare_for_extend)
        self._reset_state()

    def tearDown(self):
        self.sb.Req.init_next_round_input, self.sb.ScheduleBatch.prepare_for_extend = self._saved
        self._reset_state()

    def test_pins_match_engine(self):
        self.assertEqual(C.pin_drift(), [])

    def _build(self, swa_size=512):
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
                                          dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False)
        cache = self.urc.UnifiedRadixCache(params=CacheInitParams(
            req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=self.PAGE,
            disable=False, sliding_window_size=self.WIN, decoder_swa_bounded_replay=True,
            tree_components=(ComponentType.FULL, ComponentType.SWA), eviction_policy="lru",
            is_eagle=False))
        return cache, alloc, pool

    def _alloc(self, alloc, n):
        n = -(-n // self.PAGE) * self.PAGE
        full = alloc.full_attn_allocator.alloc(n)
        swa = alloc.swa_attn_allocator.alloc(n)
        assert full is not None and swa is not None, "test pool too small"
        alloc.full_to_swa_index_mapping[full] = swa
        return full

    def _req(self, pool, rid, prompt, out_len):
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
        req.extra_key = None
        req.lock_receipt = DecLockRefParams()
        return req

    def _run(self, cache, alloc, pool, req, prompt, out, chunks=None, forward=None):
        """Admission (lock the matched node), chunked prefill with real inserts, decode `out`, the
        finished insert. `forward(req, start, end, idx)` emulates the model on each chunk."""
        from array import array
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        m = len(req.prefix_indices)
        if m:
            pool.write((req.kv.req_pool_idx, slice(0, m)), req.prefix_indices)
        pos, k = m, 0
        while pos < len(prompt):
            step = (chunks[k] if chunks and k < len(chunks) else self.CHUNK)
            end = min(pos + step, len(prompt))
            idx = self._alloc(alloc, end - pos)[: end - pos]
            pool.write((req.kv.req_pool_idx, slice(pos, end)), idx)
            if forward is not None:
                forward(req, pos, end, idx)
            req.set_extend_range(pos, end)
            req.kv.kv_committed_len = end
            cache.cache_unfinished_req(req, chunked=end < len(prompt))
            pos, k = end, k + 1
        idx = self._alloc(alloc, len(out))[: len(out)]
        pool.write((req.kv.req_pool_idx, slice(len(prompt), len(prompt) + len(out))), idx)
        req.output_ids = array("q", out)
        req.kv.kv_committed_len = len(prompt) + len(out)
        cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=len(prompt) + len(out))
        pool.free(req)

    def _tree_paths(self, cache):
        """[(start, end, full_value, swa_live)] for every node, depth first."""
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        out = []

        def walk(node, pos):
            for child in node.children.values():
                n = len(child.key)
                fv = child.component_data[ComponentType.FULL].value
                sv = child.component_data[ComponentType.SWA].value
                out.append((pos, pos + n, None if fv is None else fv.clone(), sv is not None))
                walk(child, pos + n)
        walk(cache.tree_core.root_node, 0)
        return out

    def _setup_lost_resume(self):
        """A 96-token document (6 chunks of 16) + 4 output tokens, then every evictable SWA slot
        evicted: the document's SWA resume points are gone, most of its full KV is resident."""
        import torch
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        cache, alloc, pool = self._build()
        doc = list(range(1000, 1096))
        out0 = [9001, 9002, 9003, 9004]
        r0 = self._req(pool, 0, doc, 4)
        r0.prefix_indices = torch.empty(0, dtype=torch.int64)
        r0.last_node = cache.root_node_handle()
        r0.kv.cache_protected_len = 0
        self._run(cache, alloc, pool, r0, doc, out0, chunks=[16] * 8)
        cache.sanity_check()
        cache.evict(EvictParams(swa_num_tokens=int(cache.swa_evictable_size())))
        cache.sanity_check()
        return cache, alloc, pool, doc + out0

    def _install(self):
        env = {C.ENV: "1", C.ENV_TOKENS: str(self.TOKENS), C.ENV_MIN_SAVE: str(self.MIN_SAVE)}
        self.assertTrue(C.install_schedule_batch(self.sb, env=env))
        C.STATE.early_layers = self.EARLY

    def test_stock_loses_the_resume_point(self):
        cache, alloc, pool, hist = self._setup_lost_resume()
        prompt = hist + list(range(5000, 5024))                 # 124 tokens
        req = self._req(pool, 1, prompt, 4)
        req.init_next_round_input(cache)
        self.assertEqual(len(req.prefix_indices), 0)            # stock: re-prefill from token 0
        F = req.swa_branching_seqlen                             # ... although the full KV reaches F
        self.assertIsNotNone(F)
        self.assertGreaterEqual(F, 64)
        print(f"\n  stock: reuse 0, full KV resident to {F} of {len(prompt)}")

    def _cone_turn(self, chunks=None):
        """One turn on a lost resume point with the hook; returns the evidence."""
        import torch
        from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import build_causal_swa_page_indices
        from sglang.srt.layers.attention.deepseek_v4_backend import _low_ratio_compression_metadata

        cache, alloc, pool, hist = self._setup_lost_resume()
        self._install()
        capacity_full = alloc.full_available_size() + cache.full_evictable_size() + cache.full_protected_size()
        capacity_swa = alloc.swa_available_size() + cache.swa_evictable_size() + cache.swa_protected_size()
        prompt = hist + list(range(5000, 5024))
        req = self._req(pool, 1, prompt, 4)
        req.init_next_round_input(cache)
        plan = getattr(req, C.REQ_ATTR)
        self.assertIsNotNone(plan)
        F, s = plan.full_hit, plan.s
        self.assertEqual(s, F - self.TOKENS)
        resident = cache.tree_core.collect_full_device_indices(req.last_node, cache.tree_core.root_node.id)
        self.assertEqual(resident.numel(), F)                    # the locked node ends at F
        self.assertTrue(torch.equal(req.prefix_indices, resident[:s]))
        self.assertEqual(req.kv.cache_protected_len, s)
        self.assertEqual(plan.src_pages, (resident[s:F:self.PAGE] // self.PAGE).tolist())

        # Full-page-addressed pool stand-ins with the engine's geometry (slot = loc // ratio):
        # ratio-1 latents (1 row per full page), ratio-2 latents (1 row of PAGE/2 slots per full
        # page), ratio-2 index keys in pages of 1 slot (2 rows per full page).
        P, npages = self.PAGE, 1024 // self.PAGE + 2
        c1 = torch.zeros(npages, P, dtype=torch.int64)
        c2 = torch.zeros(npages, P // 2, dtype=torch.int64)
        i2 = torch.zeros(npages * 2, 1, dtype=torch.int64)
        val = {"c1": lambda p: 10 * p + 1, "c2": lambda p: 10 * p + 2, "i2": lambda p: 10 * p + 3}
        for pos in range(F):                                     # resident content = f(position)
            loc = int(resident[pos])
            c1.view(-1)[loc] = val["c1"](pos)
            if pos % 2 == 1:                                     # odd rows complete a pair
                c2.view(-1)[loc // 2] = val["c2"](pos)
                i2.view(-1)[loc // 2] = val["i2"](pos)
        fake_pool = SimpleNamespace(page_size=P, kv_pools={1: SimpleNamespace(kv_buffer=[c1]),
                                                           2: SimpleNamespace(kv_buffer=[c2])},
                                    index_pools={2: SimpleNamespace(page_size=1,
                                                                    contiguous_page_row_buffers=lambda: [i2])})
        self.assertEqual([p for _, p in C.full_page_buffers(fake_pool)], [1, 1, 2])
        ev = {}

        def holds(loc, pos):
            ok = int(c1.view(-1)[loc]) == val["c1"](pos)
            if pos % 2 == 1:
                ok &= int(c2.view(-1)[loc // 2]) == val["c2"](pos) and int(i2.view(-1)[loc // 2]) == val["i2"](pos)
            return ok

        def forward(r, start, end, out_loc):
            if start != s:
                return
            cone = C.build_cone_batch([r], [start], [end - start], P)
            self.assertIsNotNone(cone)
            C.replay_copy(fake_pool, cone, out_loc)
            ev["copies_ok"] = all(holds(int(out_loc[p - s]), p) for p in range(s, F))
            # compressor writes through the real slot metadata, before and after the hook's mask
            seq = torch.arange(start + 1, end + 1)
            core = SimpleNamespace()
            core.c1_out_loc, _ = _low_ratio_compression_metadata(1, seq, out_loc)
            core.c2_out_loc, _ = _low_ratio_compression_metadata(2, seq, out_loc)
            n = F - s
            ev["unmasked_hits_copies"] = bool((core.c1_out_loc[:n] == out_loc[:n]).all())
            C.mask_compressor_writes(core, cone)
            ev["sink_only"] = bool((core.c1_out_loc[:n] == 0).all()) and bool((core.c2_out_loc[:n] == 0).all())
            w1 = core.c1_out_loc
            w2 = torch.where(core.c2_out_loc >= 0, core.c2_out_loc, torch.zeros_like(core.c2_out_loc))
            for row in range(end - start):
                pos = start + row
                bad = row < n                                    # replay rows compute garbage
                c1.view(-1)[int(w1[row])] = -7 if bad else val["c1"](pos)
                c2.view(-1)[int(w2[row])] = -7 if bad else val["c2"](pos)
                i2.view(-1)[int(w2[row])] = -7 if bad else val["i2"](pos)
            ev["copies_after_writes"] = all(holds(int(out_loc[p - s]), p) for p in range(s, F))
            ev["resident_after_writes"] = all(holds(int(resident[p]), p) for p in range(F))
            ev["new_rows_written"] = all(holds(int(out_loc[p - s]), p) for p in range(F, end))
            # SWA window: floored at s; without the floor the first rows read the tombstone sink
            rt = pool.req_to_token
            rp = torch.full((end - start,), r.kv.req_pool_idx, dtype=torch.int64)
            fl = C.floor_vector(cone, end - start, "cpu")
            kw = dict(req_to_token=rt, full_to_swa_mapping=alloc.full_to_swa_index_mapping,
                      req_pool_indices_repeated=rp, seq_lens_casual=seq, swa_window=self.WIN,
                      page_index_aligned_size=self.WIN)
            with_floor = build_causal_swa_page_indices(swa_replay_start=fl, **kw)
            no_floor = build_causal_swa_page_indices(**kw)
            first = self.WIN - 1
            pos_of = torch.arange(start, end)[:, None] - torch.arange(self.WIN)[None, :]
            below = (pos_of < s) & (pos_of >= 0)
            ev["floored"] = (bool(below.any())
                             and bool((with_floor[below] == -1).all())       # floored: skipped
                             and bool((no_floor[below] == 0).all())          # unfloored: the sink slot
                             and torch.equal(with_floor[first:], no_floor[first:]))

        self._run(cache, alloc, pool, req, prompt, [9101, 9102, 9103, 9104], chunks=chunks, forward=forward)
        cache.sanity_check()
        after = self._tree_paths(cache)
        return dict(cache=cache, alloc=alloc, pool=pool, req=req, plan=plan, resident=resident, ev=ev,
                    after=after, capacity=(capacity_full, capacity_swa), prompt=prompt)

    def _check_turn(self, t):
        import torch
        ev, plan, resident = t["ev"], t["plan"], t["resident"]
        s, F = plan.s, plan.full_hit
        print(f"\n  cone: F={F} s={s} replay={F - s} evidence={ev}")
        for k in ("copies_ok", "unmasked_hits_copies", "sink_only", "copies_after_writes",
                  "resident_after_writes", "new_rows_written", "floored"):
            self.assertTrue(ev[k], k)
        # the tree kept its resident full pages for [0, F)
        spans = sorted((a, b, v) for a, b, v, _ in t["after"] if v is not None and b <= F)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], F)
        self.assertTrue(torch.equal(torch.cat([v for _, _, v in spans]), resident[:F]))
        # no SWA from inside the cone in the tree: every live SWA span ends <= s or starts >= guard
        guard = s + (self.WIN - 1) * (self.EARLY - 1)
        live = [(a, b) for a, b, _v, lv in t["after"] if lv]
        self.assertTrue(live)
        for a, b in live:
            self.assertTrue(b <= s or a >= guard, (a, b, s, guard))
        # no leaked slots: free + tree-held = capacity on both pools
        cache, alloc = t["cache"], t["alloc"]
        cap_full, cap_swa = t["capacity"]
        self.assertEqual(alloc.full_available_size() + cache.full_evictable_size() + cache.full_protected_size(),
                         cap_full)
        self.assertEqual(alloc.swa_available_size() + cache.swa_evictable_size() + cache.swa_protected_size(),
                         cap_swa)
        self.assertEqual(cache.full_protected_size(), 0)
        # the next turn resumes at the new prompt end (+ output), not at 0 or s
        from array import array
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        nxt = t["prompt"] + [9101, 9102, 9103, 9104] + [77] * 8
        m = len(cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", nxt)))).device_indices)
        print(f"  next turn reuses {m} of {len(nxt)}; live SWA spans {live}")
        self.assertGreaterEqual(m, len(t["prompt"]) // self.PAGE * self.PAGE)
        self.assertEqual(C.STATE.planned, 1)

    def test_cone_turn_single_chunk(self):
        self._check_turn(self._cone_turn())

    def test_cone_turn_first_chunk_exactly_the_replay(self):
        # Also the evidence a release gate reads without the test switch: one plan and one copy line per replay.
        with self.assertLogs(C.logger, "WARNING") as cm:
            t = self._cone_turn(chunks=[16, 64])
        self.assertFalse(C.STATE.test)
        plans = [m for m in cm.output if "DSV41_SWA_CONE plan: " in m]
        copies = [m for m in cm.output if "DSV41_SWA_CONE copy: " in m]
        self.assertEqual(len(plans), 1, cm.output)
        self.assertEqual(len(copies), 1, cm.output)
        self.assertIn(f"s={t['plan'].s} F={t['plan'].full_hit} ", plans[0])
        self.assertIn(f"s={t['plan'].s} rows={t['plan'].n} ", copies[0])
        self._check_turn(t)

    def test_finish_at_prefill_would_adopt_cone_swa(self):
        """Why the admission requires a chunked first pass: a cone request whose whole prompt runs in
        one pass and finishes at its first token goes straight to cache_finished_req, which inserts
        without freeing out-of-window SWA first, and the tree adopts the replay rows' approximate
        SWA below the guard. (The hook reverts such admissions; this shows the rule is load-bearing.)"""
        from array import array
        cache, alloc, pool, hist = self._setup_lost_resume()
        self._install()
        prompt = hist + list(range(5000, 5024))
        req = self._req(pool, 1, prompt, 1)
        req.init_next_round_input(cache)
        plan = getattr(req, C.REQ_ATTR)
        self.assertIsNotNone(plan)
        s = plan.s
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        pool.write((req.kv.req_pool_idx, slice(0, s)), req.prefix_indices)
        idx = self._alloc(alloc, len(prompt) + 1 - s)[: len(prompt) + 1 - s]
        pool.write((req.kv.req_pool_idx, slice(s, len(prompt) + 1)), idx)
        req.set_extend_range(s, len(prompt))
        req.output_ids = array("q", [9101])
        req.kv.kv_committed_len = len(prompt) + 1
        self.assertEqual(req.kv.swa_evicted_seqlen, 0)
        cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=len(prompt) + 1)
        pool.free(req)
        cache.sanity_check()
        guard = s + (self.WIN - 1) * (self.EARLY - 1)
        live = [(a, b) for a, b, _v, lv in self._tree_paths(cache) if lv]
        inside = [(a, b) for a, b in live if not (b <= s or a >= guard)]
        print(f"\n  finish at prefill (not admitted by the hook): live SWA spans {live}, inside [{s}, {guard}): {inside}")
        self.assertTrue(inside)

    def test_short_first_chunk_reverts_to_stock(self):
        cache, alloc, pool, hist = self._setup_lost_resume()
        self._install()
        prompt = hist + list(range(5000, 5024))
        req = self._req(pool, 1, prompt, 4)
        req.init_next_round_input(cache)
        plan = getattr(req, C.REQ_ATTR)
        self.assertIsNotNone(plan)

        def orig(adder, r, **kw):
            ext = min(len(r.full_untruncated_fill_ids) - len(r.prefix_indices), adder.rem)
            return SimpleNamespace(extend_len=ext, prefix_len=len(r.prefix_indices), is_chunked=True)
        res = C.guarded_select(orig, SimpleNamespace(rem=8), req, (), {"total_tokens": 10}, type("R", (), {}))
        self.assertEqual(res.prefix_len, 0)                       # the stock match again
        self.assertIsNone(getattr(req, C.REQ_ATTR))
        self.assertEqual(req.last_node, plan.stock["last_node"])
        self.assertEqual(req.kv.cache_protected_len, 0)
        self.assertEqual(req.swa_branching_seqlen, plan.stock["swa_branching_seqlen"])

    def test_off_and_ineligible_requests_unchanged(self):
        cache, alloc, pool, hist = self._setup_lost_resume()
        prompt = hist + list(range(5000, 5024))
        # bounded replay off on the cache -> no plan
        self._install()
        cache.decoder_swa_bounded_replay = False
        r = self._req(pool, 1, prompt, 4)
        r.init_next_round_input(cache)
        self.assertIsNone(getattr(r, C.REQ_ATTR))
        self.assertEqual(len(r.prefix_indices), 0)
        cache.decoder_swa_bounded_replay = True
        # prompt logprobs -> no plan
        r2 = self._req(pool, 2, prompt, 4)
        r2.return_logprob, r2.logprob_start_len = True, 0
        r2.init_next_round_input(cache)
        self.assertIsNone(getattr(r2, C.REQ_ATTR))
        # early-layer count unknown (no target forward yet) -> no plan
        C.STATE.early_layers = None
        r3 = self._req(pool, 3, prompt, 4)
        r3.init_next_round_input(cache)
        self.assertIsNone(getattr(r3, C.REQ_ATTR))

    def test_sitecustomize_installs_every_hook(self):
        # The engine's import hook (adapter/sitecustomize.py, active with DSV41_SOURCE as boot.py sets
        # it) installs all four hooks next to DSV41_SWA_THIN, the prefill bypass and the decode yield (the
        # v2.6.4 production profile's scheduler-side switches), and none when off; the others stay installed.
        import subprocess
        adapter = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter")
        probe = (
            "import sglang.srt.managers.scheduler as S, sglang.srt.model_executor.forward_batch_info as F,"
            " sglang.srt.layers.attention.deepseek_v4_backend as B, sglang.srt.managers.schedule_batch as SB,"
            " sglang.srt.managers.schedule_policy as SP, sglang.srt.mem_cache.unified_radix_cache as U,"
            " swa_cone as C\n"
            "fns = (SB.Req.init_next_round_input, SB.ScheduleBatch.prepare_for_extend,"
            " SP.PrefillAdder._select_prefill_admission, F.ForwardBatch.init_new.__func__,"
            " B.DeepseekV4AttnBackend.init_forward_metadata_prefill)\n"
            "print('HOOKS', sorted(C.STATE.installed), [getattr(f, '_dsv41_swa_cone_hook', False) for f in fns],"
            " getattr(S.Scheduler, '_dsv41_prefill_bypass', False),"
            " getattr(S.Scheduler.get_next_batch_to_run, '_dsv41_decode_yield', False),"
            " getattr(U.UnifiedRadixCache.cache_unfinished_req, '_dsv41_swa_thin', False))\n")
        for flag, want in (("1", "['adder', 'backend', 'forward_batch', 'schedule_batch'] "
                                 "[True, True, True, True, True] True True True"),
                           ("0", "[] [False, False, False, False, False] True True True")):
            env = dict(os.environ, DSV41_SOURCE="/models/x", DSV41_SWA_CONE_REPLAY=flag, DSV41_SWA_THIN="1",
                       DSV41_PREFILL_BYPASS="256", DSV41_PREFILL_BYPASS_SOLO="1", DSV41_DECODE_YIELD="33",
                       CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join([adapter, os.environ.get("PYTHONPATH", "")]))
            out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True,
                                 timeout=600)
            line = [x for x in out.stdout.splitlines() if x.startswith("HOOKS")]
            self.assertEqual(line, ["HOOKS " + want], out.stdout[-2000:] + out.stderr[-2000:])

    def test_real_pool_page_rows_copy(self):
        import torch
        try:
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                DeepSeekV4IndexerPool, DeepSeekV4SingleKVPool, KVLayout)
            page = 256
            kv1 = DeepSeekV4SingleKVPool(size=4 * page, page_size=page, dtype=torch.float8_e4m3fn,
                                         qk_nope_head_dim=448, qk_rope_head_dim=64, layer_num=1,
                                         device="cpu", enable_memory_saver=False, kv_layout=KVLayout.V41)
            kv2 = DeepSeekV4SingleKVPool(size=2 * page, page_size=page // 2, dtype=torch.float8_e4m3fn,
                                         qk_nope_head_dim=448, qk_rope_head_dim=64, layer_num=3,
                                         device="cpu", enable_memory_saver=False, kv_layout=KVLayout.V41)
            ip = DeepSeekV4IndexerPool(size=(4 * page + page) // 2, page_size=64, dtype=torch.float8_e4m3fn,
                                       index_head_dim=128, layer_num=3, device="cpu",
                                       enable_memory_saver=False, use_fp4_indexer=True)
        except Exception as exc:                     # pragma: no cover - layout API drift
            self.skipTest(f"pool construction on CPU: {exc!r}")
        pool = SimpleNamespace(page_size=page, kv_pools={1: kv1, 2: kv2, 4: None}, index_pools={2: ip})
        bufs = C.full_page_buffers(pool)
        self.assertEqual([p for _, p in bufs], [1, 1, 1, 1, 2, 2, 2])
        g = torch.Generator().manual_seed(1)
        for b, _ in bufs:
            b.copy_(torch.randint(0, 255, b.shape, dtype=torch.uint8, generator=g).view(b.dtype))
        before = [b.clone() for b, _ in bufs]
        src, dst = torch.tensor([1, 2]), torch.tensor([3, 4])
        C.copy_full_pages(bufs, src, dst)
        for (b, per), b0 in zip(bufs, before):
            for s_, d_ in zip(src.tolist(), dst.tolist()):
                self.assertTrue(torch.equal(b[d_ * per:(d_ + 1) * per].view(torch.uint8),
                                            b0[s_ * per:(s_ + 1) * per].view(torch.uint8)))
            self.assertTrue(torch.equal(b[:1].view(torch.uint8), b0[:1].view(torch.uint8)))


ROOT = Path(__file__).resolve().parent.parent


class Profile(unittest.TestCase):
    """v2.6.4: the production line of .env.tp4.example ends with DSV41_SWA_CONE_REPLAY=1 DSV41_SWA_CONE_TOKENS=8192
    (after DSV41_SWA_THIN=1); the code default is 8192 too (the production chunk); DSV41_SWA_CONE_TEST is in no
    profile; the module stays off when unset; the keys are volatile in adapter/autotune_keep.py; the canary-roce
    Dockerfile runs this test."""

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

    def test_production_line_ends_with_the_cone(self):
        items = self._production_line().split('"')[1].split()
        # v2.7.5's keys come last (V275_TAIL): checked here, then the v2.7.4 line's checks below run on the rest
        self.assertEqual(items[-len(V275_TAIL):], V275_TAIL)
        items = items[:-len(V275_TAIL)]
        self.assertEqual([x for x in items if x.startswith("DSV41_SWA_CONE")],
                         ["DSV41_SWA_CONE_REPLAY=1", "DSV41_SWA_CONE_TOKENS=8192"])
        i = items.index("DSV41_SWA_THIN=1")
        self.assertEqual(items[i:i + 3], ["DSV41_SWA_THIN=1", "DSV41_SWA_CONE_REPLAY=1", "DSV41_SWA_CONE_TOKENS=8192"])
        # v2.7: only the native V4.1 layout keys, the tokenizer heap trim and (v2.6.6) the SWA prompt tail follow the
        # cone; v2.7.1's seven new keys come after them, then v2.7.2's CED skip, then v2.7.3's stage 2b and GEMM tactic and
        # its W2 / evict-first decode keys (v2.7.3 also flips WOA_MB / WOB_MB in place: 6 / 0)
        self.assertEqual(items[i + 3:i + 7], ["DSV41_KV_LAYOUT=v41", "SGLANG_SM120_V41_PRECISION=bf16",
                                              "DSV41_TOKENIZER_HEAP=trim2", "DSV41_SWA_PROMPT_TAIL_KEEP=1024"])
        self.assertEqual(items[i + 7:], ["DSV41_PREFILL_GEMM_ROW_SPLIT=4096", "DSV41_PREFILL_GEMM_ROW_SPLIT_N=1792,1152", "DSV41_L2_PREFETCH_WOA_MB=6",
                                         "DSV41_L2_PREFETCH_WOB_MB=0", "DSV41_EXTEND_NOSYNC=1", "DSV41_SAMPLED_COMMIT_REUSE=1",
                                         "DSV41_SAMPLED_DRAFT_PROBS_EARLY=1", "DSV41_PREFILL_CED_SKIP=1",
                                         "DSV41_PREFILL_SP_FP8_MOE=1", "DSV41_PREFILL_GEMM_TACTIC=1",
                                         "DSV41_L2_PREFETCH_WOB2_MB=10", "DSV41_WO_A_W8_EVICT_FIRST=1", "DSV41_DENSE_TILE_TABLE=1", "DSV41_PREFILL_MOE_FOLD=1", "DSV41_WO_A_CFG_TABLE=1"])

    def test_code_default_is_the_production_chunk(self):
        self.assertEqual(C.DEFAULT_TOKENS, 8192)
        text = (ROOT / ".env.tp4.example").read_text().splitlines()
        self.assertIn("#CHUNKED_PREFILL_SIZE=8192", text)
        self.assertGreaterEqual(C.DEFAULT_TOKENS, C.min_replay_tokens(128, 20, 256))
        self.assertEqual(C.DEFAULT_TOKENS % 256, 0)
        C.configure({C.ENV: "1"})
        self.assertEqual((C.STATE.tokens, C.STATE.min_save, C.STATE.test), (8192, 8192, False))
        C.configure({})
        self.assertFalse(C.STATE.active)

    def test_test_switch_in_no_profile(self):
        text = (ROOT / ".env.tp4.example").read_text()
        on = [line for line in text.splitlines() if line.lstrip("#").startswith(("EXTRA_CONTAINER_ENV=", "DSV41_"))
              and "SWA_CONE_TEST" in line]
        self.assertEqual(on, [], "DSV41_SWA_CONE_TEST is test only")
        on = [line for line in text.splitlines() if line.startswith("#EXTRA_CONTAINER_ENV=") and "SWA_CONE" in line]
        self.assertEqual(on, [self._production_line()])

    def test_volatile_in_autotune_keep(self):
        vol = self._volatile()
        for k in (C.ENV, C.ENV_TOKENS, C.ENV_MIN_SAVE, C.ENV_TEST):
            self.assertIn(k, vol)

    def test_default_off_in_module(self):
        self.assertFalse(C.enabled({}))
        self.assertFalse(C.enabled({C.ENV: "0"}))
        self.assertTrue(C.enabled({C.ENV: "1"}))

    def test_dockerfile_runs_this_test(self):
        d = (ROOT / "Dockerfile.canary-roce").read_text()
        self.assertIn("RUN CUDA_VISIBLE_DEVICES= python3 /opt/dsv41/tests/test_swa_cone.py", d)


if __name__ == "__main__":
    unittest.main()
