#!/usr/bin/env python3
"""DSV41_SWA_CONE_REPLAY x DSV41_KV_LAYOUT=v41c (v41n): the cone's page copy and its compressor write mask stay exact
when the ratio-1/2 pools hold V4.1 FP4 records (adapter/v41c_pool.py).

The cone (adapter/swa_cone.py) relies on two things that the v41c layout could break:
  1. copy: swa_cone.full_page_buffers lists every buffer the KV pool addresses by FULL page id and copy_full_pages
     copies the resident pages of [s, F) into the request's fresh pages, byte for byte. Under v41c the ratio-1 / ratio-2
     pools are V41_FP4 buffers (73,728 / 36,864 bytes per page: 256 / 128 records of 256 data bytes, then the page's
     32-byte e4m3 scale rows). The copy must cover them, with one row per full page (data and scales of a page live in
     its row, so a row copy carries both);
  2. mask: the replay rows' compressor writes (latent + index key) must go to the padding slot 0. The cone zeroes
     core.c1_out_loc / c2_out_loc on the replay rows; every prefill write of a ratio-1/2 latent goes through
     DeepseekV4AttnBackend._low_ratio_write_group with the slots taken from those vectors, and under v41c it takes that
     function's V41_FP4 branch (pool.set_extra_key_buffer_fused(loc=slots, ..., freqs_cis=...)).

Engine part (torch + the pinned SGLang tree, CPU; the canary-roce image build and ds-b3-cutecheck-style containers):
  * real DeepSeekV4TokenToKVPool._init_compressed_pools with v41c_pool.install_pool on the real pool module (the
    DeepSeek-V4.1-Flash ratio layout, 256-token pages): full_page_buffers == SGLang's own get_contiguous_buf_infos
    (the PD / HiCache list of every full-page-addressed buffer: same pointers, same bytes per full page), on v41c and
    on the stock V4 layout; the FP4 buffers are in it, 2-D, contiguous, one row per full page;
  * the real _low_ratio_compress_torch -> _low_ratio_pair_partners -> _low_ratio_write_group (ratio 1 and ratio 2,
    slots from the real _low_ratio_compression_metadata) on that pool, after the cone's real replay_copy and
    mask_compressor_writes: every write reaches the V41_FP4 branch (freqs_cis passed), the replay rows' latent and
    index-key slots are 0, the fresh pages keep the copied resident bytes in every buffer (FP4 data + scales, index
    keys), only page 0 and the rows past F change; without the mask the same call overwrites the copied pages
    (the mask is load-bearing on v41c as on V4). The device kernels at the leaves (fused_store_cache's V41_FP4 store,
    store_fp4_index_k_cache) are CUDA-only; here they are replaced by CPU writers of the same page ABI
    (tests/v41_fp4_ref.py = the v41w window's reference, checked against SGLang's writer on sm_121a; the index page
    layout is DeepSeekV4IndexerPool.get_index_k_fp4's), so what is tested is WHERE the writes land;
  * both adapters pin the same engine text: swa_cone's pins and v41c_pool's pins of _low_ratio_write_group,
    _low_ratio_compress_torch (cone) and _low_ratio_compress_fused (v41c) match the staged tree, and the pinned
    _low_ratio_write_group stores the latent (both branches) and the index key at `slots` only;
  * the import hooks install together: DSV41_KV_LAYOUT=v41c DSV41_V41C_READER=native DSV41_V41C_ARM=1 with the
    v2.6.4 scheduler profile (thin, cone 8192, decode yield 33, bypass 256 + solo): all four cone hooks, the v41c
    pool / sizing patches, the backend pins and the layer tags.

usage: PYTHONPATH=<SGLang python>:adapter python3 tests/test_swa_cone_v41c.py
"""
import ast
import importlib.util
import inspect
import os
import subprocess
import sys
import textwrap
import types
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "..", "adapter")
sys.path.insert(0, ADAPTER)
sys.path.insert(0, HERE)
import swa_cone as C  # noqa: E402
import v41c_pool as P  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_SGLANG = importlib.util.find_spec("sglang") is not None

# DeepSeek-V4.1-Flash text config (40 layers; tests/test_v41c_pool.py)
RATIOS = [0, 0] + [2] * 18 + [1] * 20
SOURCES = {2: [2, 8, 14], 1: [20]}
PAGE = 256
FULL_PAGES = 12
INDEX_PAGE = 64           # DSV41_INDEX_PAGE_SIZE


@unittest.skipUnless(HAVE_TORCH and HAVE_SGLANG, "needs torch and the pinned SGLang tree")
class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ["SGLANG_SM120_FLASHMLA_BACKEND"] = "flashinfer"
        import torch
        from sglang.srt.layers.attention import deepseek_v4_backend as B
        from sglang.srt.mem_cache import deepseek_v4_memory_pool as M
        cls.torch, cls.B, cls.M = torch, B, M
        # a tuple: plain functions stored on the class would come back as bound methods
        cls.stock = (M.resolve_compressed_kv_layout, M.is_valid_kv_layout_pair, M.DeepSeekV4TokenToKVPool.__init__)

    @classmethod
    def tearDownClass(cls):
        M = cls.M
        M.resolve_compressed_kv_layout, M.is_valid_kv_layout_pair, M.DeepSeekV4TokenToKVPool.__init__ = cls.stock
        M._dsv41_v41c_pool = False

    def _layouts(self, on):
        M = self.M
        M.resolve_compressed_kv_layout, M.is_valid_kv_layout_pair, M.DeepSeekV4TokenToKVPool.__init__ = self.stock
        M._dsv41_v41c_pool = False
        if on:
            P.install_pool(M)            # the real install: pins, layout values, backend check

    def _pool(self, on):
        """The real compressed-pool construction of a DeepSeekV4TokenToKVPool (no SWA part: the cone copies only
        full-page-addressed buffers)."""
        torch, M = self.torch, self.M
        self._layouts(on)
        p = object.__new__(M.DeepSeekV4TokenToKVPool)
        p.kv_layout, p.compressed_kv_layout_option = M.KVLayout.V4, None
        p.compressed_pool_configs, p.full_size, p._unified_kv, p.uniform_fp8 = {}, FULL_PAGES * PAGE, False, False
        p.sources_by_ratio = SOURCES
        p.qk_nope_head_dim, p.qk_rope_head_dim, p.indexer_head_dim = 448, 64, 128
        p.page_size, p.request_window, p.layer_transfer_counter = PAGE, None, None
        p.compression_ratios, p._stage_start, p._stage_end = RATIOS, 0, len(RATIOS)
        p._init_compressed_pools(stage_ratios=RATIOS, page_size=PAGE, dtype=torch.float8_e4m3fn, device="cpu",
                                 enable_memory_saver=False, enable_hisparse=False, kv_pool_cls=M.DeepSeekV4SingleKVPool)
        p._init_compressed_layer_mapping()
        return p

    # ------------------------------------------------------------------ 1. the copy covers every v41c buffer
    def test_full_page_buffers_cover_every_v41c_buffer(self):
        for on in (True, False):
            p = self._pool(on)
            bufs = C.full_page_buffers(p)
            ptrs, lens, items = p.get_contiguous_buf_infos()
            mine = [(b.data_ptr(), b.nbytes, b[0].nbytes * per) for b, per in bufs]
            self.assertEqual(sorted(mine), sorted(zip(ptrs, lens, items)), f"v41c={on}")
            if not on:
                continue
            KV = self.M.KVLayout
            for ratio, page_bytes in ((1, 73728), (2, 36864)):
                q = p.kv_pools[ratio]
                self.assertIs(q.kv_layout, KV.V41_FP4)
                self.assertEqual(q.page_size, PAGE // ratio)
                for b in q.kv_buffer:
                    self.assertIn((b.data_ptr(), b.nbytes, b[0].nbytes), mine)
                    self.assertEqual((b.dim(), b.shape[1], b.stride(), b.is_contiguous()),
                                     (2, page_bytes, (page_bytes, 1), True))
                    self.assertGreaterEqual(b.shape[0], FULL_PAGES + 1)
                # compressed slot of a full token t = t // ratio; its page in this pool = full page of t
                for t in (0, 1, 255, 256, 257, 1023, 1024, FULL_PAGES * PAGE - 1):
                    self.assertEqual((t // ratio) // q.page_size, t // PAGE)
            self.assertIn("r1=v41_fp4:page=256:page_bytes=73728", P.check_pool(p, KV))

    # ------------------------------------------------------------------ 2. writes land on slot 0 under v41c
    def _writers(self, p, log):
        """CPU stand-ins for the CUDA leaf stores, same page ABI: V41_FP4 records (tests/v41_fp4_ref.py) and the fp4
        index-key pages ([page x 64 payload | page x 4 scale bytes], get_index_k_fp4)."""
        torch = self.torch
        import v41_fp4_ref as R

        def fp4_store(pool):
            def set_key_buffer_fused(layer_id, loc, cache_k, freqs_cis=None):
                log.append(("kv", pool.kv_layout, int(pool.page_size), loc.clone(), freqs_cis is not None))
                assert pool.kv_layout is self.M.KVLayout.V41_FP4, pool.kv_layout
                data, scales = R.encode_rows(cache_k.to(torch.bfloat16))
                buf, ps = pool.kv_buffer[layer_id], pool.page_size
                for i, s in enumerate(loc.tolist()):
                    pg, o = divmod(int(s), ps)
                    buf[pg, o * R.DATA:(o + 1) * R.DATA] = data[i]
                    buf[pg, ps * R.DATA + o * R.SCALE: ps * R.DATA + (o + 1) * R.SCALE] = scales[i]
            return set_key_buffer_fused

        def index_store(ipool):
            def set_index_fp4(layer_id, loc, cache_k):
                log.append(("index", None, int(ipool.page_size), loc.clone(), None))
                buf, ps = ipool.index_k_with_scale_buffer[layer_id], ipool.page_size
                raw = (cache_k.float().abs() * 7).to(torch.uint8)
                for i, s in enumerate(loc.tolist()):
                    pg, o = divmod(int(s), ps)
                    buf[pg, o * 64:(o + 1) * 64] = raw[i, :64]
                    buf[pg, ps * 64 + o * 4: ps * 64 + (o + 1) * 4] = raw[i, 64:68]
            return set_index_fp4

        for q in p.kv_pools.values():
            if q is not None:
                q.set_key_buffer_fused = fp4_store(q)
        for ip in p.index_pools.values():
            ip.set_index_fp4 = index_store(ip)

    def _backend(self, p, core):
        B = self.B
        fake = SimpleNamespace(forward_metadata=SimpleNamespace(core_metadata=core), token_to_kv_pool=p)
        for name in ("_low_ratio_compress_torch", "_low_ratio_pair_partners", "_low_ratio_write_group"):
            setattr(fake, name, types.MethodType(getattr(B.DeepseekV4AttnBackend, name), fake))
        return fake

    def _layer(self, layer_id, ratio):
        torch = self.torch
        g = torch.Generator().manual_seed(layer_id)
        comp = SimpleNamespace(finish=lambda pooled: pooled.to(torch.bfloat16),
                               pool_pairs=lambda kv, score: kv.mean(dim=1))
        idx = SimpleNamespace(owns_k=True, index_head_dim=128,
                              index_keys=lambda latent, freqs: latent[:, :128].float())
        freqs = torch.polar(torch.ones(4 * FULL_PAGES * PAGE, 32), torch.rand(4 * FULL_PAGES * PAGE, 32, generator=g))
        return SimpleNamespace(layer_id=layer_id, compress_ratio=ratio, compressor=comp, indexer=idx,
                               freqs_cis=freqs, rope_head_dim=64)

    def _replay(self, mask):
        """One replay: resident full pages SRC hold [s, F) = [3 x 256, 5 x 256); the request's fresh pages DST get
        the copy; the extend = the replay rows + one more page of new rows past F. Returns (pool, snapshots after
        the copy, every buffer after the writes, the write log)."""
        torch, B = self.torch, self.B
        p = self._pool(True)
        g = torch.Generator().manual_seed(7)
        for b, _ in C.full_page_buffers(p):
            b.copy_(torch.randint(0, 255, b.shape, dtype=torch.uint8, generator=g).view(b.dtype))
        SRC, DST, NEXT = [3, 4], [8, 9], 10
        s, F = SRC[0] * PAGE, (SRC[-1] + 1) * PAGE
        n = F - s
        loc = torch.cat([torch.arange(d * PAGE, (d + 1) * PAGE) for d in DST + [NEXT]]).to(torch.int64)
        pos = torch.arange(s, s + loc.numel(), dtype=torch.int64)
        cone = C.ConeBatch(entries=[(0, n, loc.numel(), s, list(SRC))], page=PAGE)
        C.replay_copy(p, cone, loc)
        bufs = C.full_page_buffers(p)
        for b, per in bufs:                       # the copy itself: fresh pages == resident pages, byte for byte
            for sp, dp in zip(SRC, DST):
                self.assertTrue(torch.equal(b[dp * per:(dp + 1) * per].view(torch.uint8),
                                            b[sp * per:(sp + 1) * per].view(torch.uint8)))
        snap = [b.clone() for b, _ in bufs]
        seq = pos + 1
        c1, _ = B._low_ratio_compression_metadata(1, seq, loc)
        c2, _ = B._low_ratio_compression_metadata(2, seq, loc)
        core = SimpleNamespace(raw_out_loc=loc.clone(), c1_out_loc=c1, c2_out_loc=c2)
        if mask:
            C.mask_compressor_writes(core, cone)
            self.assertEqual(int(core.c1_out_loc[:n].abs().sum()), 0)
            self.assertEqual(int(core.c2_out_loc[:n].abs().sum()), 0)
        log = []
        self._writers(p, log)
        fake = self._backend(p, core)
        req = torch.zeros(loc.numel(), dtype=torch.int64)

        class State:
            ring_size = 4

            def translate_from_req_position_to_state_loc(self, r, q):
                return torch.where(q >= 0, r * 4 + q % 4, torch.full_like(q, -1))

            def get_state_by_state_loc(self, sl):
                from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
                z = torch.zeros(sl.numel(), 512)
                return KVAndScore.from_kv_score(kv=z, score=z.clone())

            def set_state_by_state_loc(self, sl, value):
                pass

        p.compress_state_pools = [State() if RATIOS[i] == 2 and i in SOURCES[2] else None for i in range(len(RATIOS))]
        x = torch.randn(loc.numel(), 512, generator=g)
        for lid, ratio in ((SOURCES[1][0], 1), (SOURCES[2][0], 2)):
            fake._low_ratio_compress_torch(self._layer(lid, ratio), x, req, pos,
                                           projected=(x.clone(), torch.randn(loc.numel(), 512, generator=g)))
        # the slots the stock path would write (no cone): ratio 1 = c1_out_loc, ratio 2 = c2_out_loc with -1 -> 0
        want = {1: c1.clone(), 2: torch.where(c2 >= 0, c2, torch.zeros_like(c2))}
        if mask:
            for r in want.values():
                r[:n] = 0
        return p, bufs, snap, log, (SRC, DST, NEXT, n, want)

    def test_replay_writes_go_to_the_padding_slot_under_v41c(self):
        torch = self.torch
        p, bufs, snap, log, (SRC, DST, NEXT, n, want) = self._replay(mask=True)
        # call order: ratio 1 (index key, then latent), ratio 2 (index key, then latent)
        self.assertEqual([(e[0], e[2]) for e in log], [("index", INDEX_PAGE), ("kv", PAGE), ("index", INDEX_PAGE),
                                                       ("kv", PAGE // 2)], [e[:3] for e in log])
        for (kind, layout, ps, loc, rope), ratio in zip(log, (1, 1, 2, 2)):
            if kind == "kv":
                self.assertIs(layout, self.M.KVLayout.V41_FP4)   # the V41_FP4 branch of _low_ratio_write_group
                self.assertTrue(rope)                            # in-kernel RoPE: freqs_cis passed (that branch only)
            self.assertEqual(int(loc[:n].abs().sum()), 0, (kind, ratio))   # replay rows: the padding slot
            self.assertTrue(torch.equal(loc, want[ratio]), (kind, ratio))   # rows past F: the stock slots
            self.assertTrue(bool((loc[n:][loc[n:] > 0] // (PAGE // ratio) == NEXT).all()), (kind, ratio))
        for (b, per), b0 in zip(bufs, snap):
            for dp in DST:                                   # the copied resident bytes stay
                self.assertTrue(torch.equal(b[dp * per:(dp + 1) * per].view(torch.uint8),
                                            b0[dp * per:(dp + 1) * per].view(torch.uint8)))
            changed = {r // per for r in range(b.shape[0])
                       if not torch.equal(b[r].view(torch.uint8), b0[r].view(torch.uint8))}
            self.assertTrue(changed <= {0, NEXT}, changed)  # only the padding page and the new rows' page
        written = {id(q.kv_buffer[0]) for q in p.kv_pools.values() if q is not None}
        self.assertEqual(len(written), 2)

    def test_without_the_mask_the_copy_is_overwritten(self):
        torch = self.torch
        p, bufs, snap, log, (SRC, DST, NEXT, n, want) = self._replay(mask=False)
        fp4 = [(b, per, b0) for (b, per), b0 in zip(bufs, snap) if b.shape[1] in (73728, 36864)]
        self.assertTrue(fp4)
        hit = [any(not torch.equal(b[dp * per:(dp + 1) * per].view(torch.uint8),
                                   b0[dp * per:(dp + 1) * per].view(torch.uint8)) for dp in DST)
               for b, per, b0 in fp4 if b.data_ptr() in (p.kv_pools[1].kv_buffer[0].data_ptr(),
                                                         p.kv_pools[2].kv_buffer[0].data_ptr())]
        self.assertEqual(hit, [True, True])                  # both written layers' copied pages overwritten

    # ------------------------------------------------------------------ 3. one engine text under both adapters
    def test_both_adapters_pin_the_same_write_path(self):
        self.assertEqual(C.pin_drift([n for n in C.PINS if n.startswith("deepseek_v4_backend.")]), [])
        P.check_pins(self.B)                                 # raises on drift
        src = textwrap.dedent(inspect.getsource(inspect.unwrap(self.B.DeepseekV4AttnBackend._low_ratio_write_group)))
        fn = ast.parse(src).body[0]
        self.assertEqual([a.arg for a in fn.args.args][:5], ["self", "layer", "pooled", "slots", "group_pos"])
        stores = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", getattr(n.func, "id", None)) in (
                      "set_extra_key_buffer_fused", "set_index_k_fp4", "rope_fake_quant_pack_indexer")]
        self.assertEqual(len(stores), 4, ast.dump(fn)[:300])
        for c in stores:
            locs = [k.value for k in c.keywords if k.arg in ("loc",)]
            self.assertEqual([getattr(v, "id", None) for v in locs], ["slots"], ast.unparse(c))
        fp4 = [n for n in ast.walk(fn) if isinstance(n, ast.If) and "V41_FP4" in ast.unparse(n.test)]
        self.assertEqual(len(fp4), 1)
        self.assertIn("freqs_cis=freqs", ast.unparse(fp4[0].body[0]))
        # the extend path: _low_ratio_compress_torch hands c1_out_loc / masked c2_out_loc to the write group
        tsrc = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(
            self.B.DeepseekV4AttnBackend._low_ratio_compress_torch))))
        self.assertIn("core.c1_out_loc[:num_tokens]", tsrc)
        self.assertIn("out_loc = core.c2_out_loc[:num_tokens]", tsrc)
        # extend (neither decode nor target verify) and the prefill graph both take _low_ratio_compress_torch
        dsrc = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(self.B.DeepseekV4AttnBackend._low_ratio_compress))))
        self.assertIn("forward_batch.forward_mode.is_decode()", dsrc)
        self.assertIn("is_target_verify()", dsrc)
        self.assertEqual(dsrc.count("self._low_ratio_compress_torch("), 1)


@unittest.skipUnless(HAVE_SGLANG, "needs the pinned SGLang tree")
class Hooks(unittest.TestCase):
    def test_sitecustomize_installs_cone_and_v41c_together(self):
        probe = (
            "import sglang.srt.managers.scheduler as S, sglang.srt.model_executor.forward_batch_info as F,"
            " sglang.srt.layers.attention.deepseek_v4_backend as B, sglang.srt.managers.schedule_batch as SB,"
            " sglang.srt.managers.schedule_policy as SP, sglang.srt.mem_cache.deepseek_v4_memory_pool as M,"
            " sglang.srt.model_executor.pool_configurator as PC, swa_cone as C\n"
            "fns = (SB.Req.init_next_round_input, SB.ScheduleBatch.prepare_for_extend,"
            " SP.PrefillAdder._select_prefill_admission, F.ForwardBatch.init_new.__func__,"
            " B.DeepseekV4AttnBackend.init_forward_metadata_prefill)\n"
            "print('HOOKS', sorted(C.STATE.installed), all(getattr(f, '_dsv41_swa_cone_hook', False) for f in fns),"
            " getattr(B.DeepseekV4AttnBackend._forward_attention, '_dsv41_v41c_tags', False),"
            " getattr(M, '_dsv41_v41c_pool', False), getattr(PC, '_dsv41_v41c_sizing', False),"
            " getattr(S.Scheduler, '_dsv41_prefill_bypass', False))\n")
        for v41c, want in (("v41c", "['adder', 'backend', 'forward_batch', 'schedule_batch'] True True True True True"),
                           ("", "['adder', 'backend', 'forward_batch', 'schedule_batch'] True False False False True")):
            env = dict(os.environ, DSV41_SOURCE="/models/x", DSV41_SWA_CONE_REPLAY="1", DSV41_SWA_CONE_TOKENS="8192",
                       DSV41_SWA_THIN="1", DSV41_PREFILL_BYPASS="256", DSV41_PREFILL_BYPASS_SOLO="1",
                       DSV41_DECODE_YIELD="33", CUDA_VISIBLE_DEVICES="", SGLANG_SM120_FLASHMLA_BACKEND="flashinfer",
                       PYTHONPATH=os.pathsep.join([ADAPTER, os.environ.get("PYTHONPATH", "")]))
            for k in ("DSV41_KV_LAYOUT", "DSV41_V41C_READER", "DSV41_V41C_ARM"):
                env.pop(k, None)
            if v41c:
                env.update(DSV41_KV_LAYOUT=v41c, DSV41_V41C_READER="native", DSV41_V41C_ARM="1")
            out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=600)
            line = [x for x in out.stdout.splitlines() if x.startswith("HOOKS")]
            self.assertEqual(line, ["HOOKS " + want], out.stdout[-2000:] + out.stderr[-2000:])


class Profile(unittest.TestCase):
    def test_dockerfile_runs_this_test(self):
        root = os.path.join(HERE, "..")
        if not os.path.exists(os.path.join(root, "Dockerfile.canary-roce")):
            self.skipTest("not a repo checkout (in an image)")
        with open(os.path.join(root, "Dockerfile.canary-roce")) as f:
            d = f.read()
        self.assertIn("RUN CUDA_VISIBLE_DEVICES= python3 /opt/dsv41/tests/test_swa_cone_v41c.py", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
