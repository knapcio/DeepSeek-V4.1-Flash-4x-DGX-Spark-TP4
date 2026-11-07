"""CPU checks of the v41n eager-prefill page pass (DSV41_V41C_READER=native, runtime/flashinfer_v41c/v41c.py).

dbd4644's one-GB10 run: an 8192-row prefill chunk was +12 % (+48.6 ms over 400.5 ms) with the in-kernel reader, which
converts every record once per (query row, top-k entry): 38 layers x 8192 rows x 512 entries = 160 M conversions on the
math warps' critical path. The page pass converts each touched record once per layer group (O(unique records)) into
compacted V4 64-token sub-pages and runs the stock prefill. What must hold:

- mapping: native_page_view's rewritten indices and scratch bytes equal f092906 / 942f0ec's page mode (plan_page_mode +
  the reference transcode) on the live entries: holes, short lengths, duplicates, pools whose last sub-page is partial;
- rows: every live entry reads its record's V4 row (the IO-warp rule) through the rewritten index;
- kernels: the GPU path's Triton kernels (mark, remap, page transcode; TRITON_INTERPRET=1 runs them on the CPU) write
  the same indices and byte-identical scratch as the torch path;
- routing: auto = page for eager calls with >= DSV41_V41N_PAGE_MIN_ROWS rows, kernel below; never page under CUDA-graph
  capture; DSV41_V41N_PREFILL=kernel / page forces; bad values raise; the scratch cap falls back to the kernel;
- groups: one page pass per layer group (kv_source / index_source), reused by the group's other layers;
- native_prefill: the page path calls FlashInfer's stock prefill with the page view and rewritten indices.

usage: python3 tests/test_v41n_page.py      (torch needed; triton for the kernel cases: ds-b3-cutecheck:1)
"""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")

ROOT = Path(__file__).resolve().parent.parent
try:
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False
try:
    import triton  # noqa: F401
    HAVE_TRITON = HAVE_TORCH
except Exception:  # noqa: BLE001
    HAVE_TRITON = False


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if HAVE_TORCH:
    F = _load("test_v41c_fast_helpers", ROOT / "tests" / "test_v41c_fast.py")
    v = F.v


def _env(**kw):
    for k, val in kw.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    v.reader.cache_clear()
    v.native_prefill_rule.cache_clear()


def _chunk_indices(rows, k, n_tokens, seed, hole=0.1):
    g = torch.Generator().manual_seed(seed)
    eidx = torch.randint(0, n_tokens, (rows, k), generator=g, dtype=torch.int32)
    eidx[torch.rand(rows, k, generator=g) < hole] = -1
    lens = torch.randint(1, k + 1, (rows,), generator=g, dtype=torch.int32)
    return eidx, lens


def _reference(view_fp4, eidx, lens):
    """942f0ec page mode on the live entries: (scratch bytes, rewritten indices)."""
    live_idx = torch.where(v._live(eidx, lens), eidx, torch.full_like(eidx, -1))
    n_pool = int(view_fp4.shape[0]) * int(view_fp4.shape[1])
    tokens, slots, new_idx, n_pages = v.plan_page_mode(live_idx, n_pool)
    buf = torch.zeros(max(n_pages, 1), v.SUB_PAGE_BYTES, dtype=torch.uint8)
    v.transcode_tokens_torch(view_fp4, tokens, buf, slots)
    return buf, new_idx


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class PageView(unittest.TestCase):
    def setUp(self):
        _env(DSV41_V41C_READER="native", DSV41_V41N_PREFILL=None, DSV41_V41N_PAGE_MIN_ROWS=None,
             DSV41_V41N_PAGE_CAP_MB=None)
        v.reset_group()
        for k_ in v.stats:
            v.stats[k_] = 0

    def test_equals_page_mode(self):
        for pbs, pages, rows, k, seed in ((256, 4, 300, 64, 1), (128, 7, 513, 128, 2), (256, 2, 97, 512, 3)):
            view_fp4, _, _ = F._fp4_pool(pages, pbs, 40 + seed)
            eidx, lens = _chunk_indices(rows, k, pages * pbs, seed)
            v.reset_group()
            view, new = v.native_page_view(view_fp4, eidx, lens)
            ref_buf, ref_new = _reference(view_fp4, eidx, lens)
            raw = torch.as_strided(view, (view.shape[0], v.SUB_PAGE_BYTES), (v.SUB_PAGE_BYTES, 1))
            with self.subTest(pbs=pbs, rows=rows, k=k):
                self.assertTrue(torch.equal(new, ref_new))
                self.assertTrue(torch.equal(raw, ref_buf[:raw.shape[0]]))
                self.assertEqual(raw.shape[0], ref_buf.shape[0])

    def test_live_entries_read_their_rows(self):
        view_fp4, d4, s4 = F._fp4_pool(5, 128, 77, tail=True)
        eidx, lens = _chunk_indices(600, 64, 5 * 128, 9, hole=0.2)
        eidx[:5] = eidx[5:10]                                   # duplicated rows
        view, new = v.native_page_view(view_fp4, eidx, lens)
        live = v._live(eidx, lens)
        self.assertTrue(torch.equal(new >= 0, live))
        for r_, j in list(zip(*torch.nonzero(live, as_tuple=True)))[::11]:
            d, s = F._read_rows(view, int(new[r_, j]))
            t = int(eidx[r_, j])
            self.assertTrue(torch.equal(d, d4[t]) and torch.equal(s, s4[t]), (int(r_), int(j)))
        self.assertEqual(view.shape[0], (eidx[live] // 64).unique().numel())

    def test_partial_last_sub_page(self):
        """A pool whose token count is not a multiple of 64 (pbs 128, 1 page = 2 sub-pages, all touched)."""
        view_fp4, d4, s4 = F._fp4_pool(1, 128, 5)
        eidx = torch.tensor([[0, 63, 64, 127] * 16] * 600, dtype=torch.int32)
        view, new = v.native_page_view(view_fp4, eidx, None)
        ref_buf, ref_new = _reference(view_fp4, eidx, None)
        raw = torch.as_strided(view, (view.shape[0], v.SUB_PAGE_BYTES), (v.SUB_PAGE_BYTES, 1))
        self.assertTrue(torch.equal(new, ref_new) and torch.equal(raw, ref_buf))

    def test_cap_returns_none(self):
        _env(DSV41_V41N_PAGE_CAP_MB="0")
        view_fp4, _, _ = F._fp4_pool(4, 256, 3)
        eidx, lens = _chunk_indices(600, 64, 4 * 256, 4)
        self.assertIsNone(v.native_page_view(view_fp4, eidx, lens))
        self.assertEqual(v.stats["native_page_cap"], 1)

    def test_reused_across_group(self):
        view_fp4, _, _ = F._fp4_pool(3, 256, 130)
        eidx = torch.randint(0, 768, (600, 64), dtype=torch.int32)
        v.tag_layer(20, 20)
        a = v.native_page_view(view_fp4, eidx)
        for L in (21, 22, 23):
            v.tag_layer(L, 20)
            b = v.native_page_view(view_fp4, eidx)
            self.assertIs(b[1], a[1])
        v.tag_layer(24, 24)
        v.native_page_view(view_fp4, eidx)
        self.assertEqual((v.stats["plan"], v.stats["reuse"], v.stats["native_page"]), (2, 3, 2))


@unittest.skipUnless(HAVE_TRITON, "needs triton (TRITON_INTERPRET=1 runs the kernels on the CPU)")
class Kernels(unittest.TestCase):
    def test_triton_path_equals_torch_path(self):
        for pbs, pages, rows, k, lens_on in ((256, 3, 70, 64, True), (128, 5, 40, 128, False)):
            view_fp4, _, _ = F._fp4_pool(pages, pbs, 60 + pbs)
            eidx, lens = _chunk_indices(rows, k, pages * pbs, rows)
            lens = lens if lens_on else None
            n_pool = pages * pbs
            n_sub = -(-n_pool // 64)
            mark = v._v41n_page_plan_triton(view_fp4, eidx.contiguous(), lens, n_sub)
            cmap = torch.cumsum(mark, 0, dtype=torch.int32) - 1
            n_pages = int(cmap[-1]) + 1
            buf = torch.zeros(max(n_pages, 1), v.SUB_PAGE_BYTES, dtype=torch.uint8)
            new = v._v41n_page_finish_triton(view_fp4, eidx.contiguous(), lens, mark, cmap, n_pool, buf)
            ref_buf, ref_new = _reference(view_fp4, eidx, lens)
            with self.subTest(pbs=pbs, lens=lens_on):
                self.assertTrue(torch.equal(new, ref_new))
                self.assertTrue(torch.equal(buf, ref_buf))


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Routing(unittest.TestCase):
    def tearDown(self):
        _env(DSV41_V41N_PREFILL=None, DSV41_V41N_PAGE_MIN_ROWS=None, DSV41_V41N_PAGE_CAP_MB=None)

    def test_modes(self):
        _env(DSV41_V41N_PREFILL=None)
        self.assertEqual(v.native_prefill_mode(8192), "page")
        self.assertEqual(v.native_prefill_mode(512), "page")
        self.assertEqual(v.native_prefill_mode(511), "kernel")
        self.assertEqual(v.native_prefill_mode(96), "kernel")         # the c16 verify graphs
        _env(DSV41_V41N_PAGE_MIN_ROWS="97")
        self.assertEqual(v.native_prefill_mode(96), "kernel")
        self.assertEqual(v.native_prefill_mode(97), "page")
        _env(DSV41_V41N_PREFILL="kernel")
        self.assertEqual(v.native_prefill_mode(8192), "kernel")
        _env(DSV41_V41N_PREFILL="page")
        self.assertEqual(v.native_prefill_mode(65), "page")
        _env(DSV41_V41N_PREFILL="bogus")
        with self.assertRaises(ValueError):
            v.native_prefill_mode(100)

    def test_never_page_under_capture(self):
        _env(DSV41_V41N_PREFILL="page")
        saved = v._capturing
        v._capturing = lambda: True
        try:
            self.assertEqual(v.native_prefill_mode(8192), "kernel")
        finally:
            v._capturing = saved

    def test_native_prefill_page_path_calls_stock(self):
        _env(DSV41_V41C_READER="native", DSV41_V41N_PREFILL="page")
        v.reset_group()
        view_fp4, _, _ = F._fp4_pool(3, 256, 11)
        eidx, lens = _chunk_indices(600, 64, 768, 12)
        calls = []

        def stock(q2, kv_64, idx, output, out_lse, scale, **kw):
            calls.append(kw)

        mod = types.ModuleType("flashinfer.mla._sparse_mla_sm120")
        mod._sparse_mla_sm120_paged_attention = stock
        saved = {k: sys.modules.get(k) for k in ("flashinfer", "flashinfer.mla", "flashinfer.mla._sparse_mla_sm120")}
        sys.modules.update({"flashinfer": types.ModuleType("flashinfer"), "flashinfer.mla": types.ModuleType("flashinfer.mla"),
                            "flashinfer.mla._sparse_mla_sm120": mod})
        v.native_page_receipt.cache_clear()
        try:
            q2 = torch.zeros(600, 16, 512, dtype=torch.bfloat16)
            v.native_prefill(q2, None, None, 0.04, topk_length=None, attn_sink=None, extra_fp4=view_fp4, extra_idx=eidx,
                             extra_topk_length=lens, head_dim_v=512)
        finally:
            for k_, m in saved.items():
                sys.modules.pop(k_, None) if m is None else sys.modules.__setitem__(k_, m)
        self.assertEqual(len(calls), 1)
        ref_buf, ref_new = _reference(view_fp4, eidx, lens)
        self.assertTrue(torch.equal(calls[0]["extra_indices"], ref_new))
        self.assertEqual(int(calls[0]["extra_kv_cache"].shape[1]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=1)
