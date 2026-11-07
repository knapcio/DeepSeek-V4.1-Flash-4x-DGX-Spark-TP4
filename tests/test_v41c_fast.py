"""CPU checks of the v41c group reader (DSV41_V41C_READER=group, runtime/flashinfer_v41c/v41c.py; 2026-10-28).

The f092906 reader transcoded every FP4 record once per (query row, 16-head block, layer) inside the decode kernel's IO
warp; the group reader transcodes the records a layer group reads once into a V4 scratch and runs the stock kernels on
it. What must hold, and what is checked here:

- numerics: for every live entry the attention reads exactly the V4 row the IO-warp reader builds in shared memory.
  `cuda_rule_rows` is a numpy port of v41c_store_v4_row (decode_dsv4_kernel.cuh: lane layout, 4-lane amax, __fdiv_rn,
  power-of-two inverse, RN satfinite e4m3 via an exact nearest-even oracle, bf16 rope); it equals the torch reference
  (v4_rows o decode_records) on random, heavy-tail, e2m1-tie and e4m3-scale-tie records. The integer RN-even e4m3
  encoder of the Triton kernels equals the oracle and torch on every value the transcode can produce plus 2^20 random
  values and every e4m3 midpoint.
- kernels: the Triton kernels (run by the Triton interpreter on the CPU: TRITON_INTERPRET=1) write byte-identical
  scratch and identical rewritten indices to the torch planners, with and without dedupe, with holes, short
  extra_topk_length and duplicate tokens. (The f092906 kernel _fp4_to_v4_kernel cannot be checked this way: the
  interpreter's fp32 -> fp8 cast is not RN-even; the new kernels do both roundings in integer arithmetic.)
- plans: every live entry reads its record's V4 row through the rewritten index; dead entries (-1 or past the length)
  are -1; dedupe gives one slot per distinct token of a block, shared by its rows; slots stay inside the block.
- attention: a reference sparse attention (the canary's torch fallback) over v2.6's V4 pool with the original indices
  and over the group scratch with the rewritten indices gives bit-identical outputs and LSE.
- groups: the layer tags reuse the scratch inside a kv_source / index_source group (8 transcodes for layers 2..39) and
  recompute at every group start even when the top-k buffer is refilled in place (same pointer, new content); a
  missing tag, a non-consecutive layer or another tensor recomputes.
- capture safety: output shapes and kernel grids depend on (rows, K) only (holes 0 / 50 / 100 %, any lengths); the torch
  planners run under FakeTensorMode (no data-dependent shape, no .item()); the GPU-path functions contain no host-sync
  call (AST).
- routing: decode() with the group reader calls FlashInfer's stock paged attention with the scratch as the extra cache,
  the rewritten indices and the caller's split-K scratch; DSV41_V41C_READER=io keeps the f092906 runner.
- adapter: group_starts on the DeepSeek-V4.1 config; the _forward_attention wrapper tags / clears around the call.

usage: python3 tests/test_v41c_fast.py        (torch needed; triton for the kernel cases: ds-b3-cutecheck:1)
"""
import ast
import importlib.util
import inspect
import os
import sys
import textwrap
import types
import unittest
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")     # before triton is imported: kernels run on the CPU

ROOT = Path(__file__).resolve().parent.parent
V41C = ROOT / "runtime" / "flashinfer_v41c"
CANARY = ROOT / "runtime" / "flash_mla_sm120.canary.py"

try:
    import numpy as np
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


v = _load("v41c_fast_under_test", V41C / "v41c.py")
P = _load("v41c_pool_under_test", ROOT / "adapter" / "v41c_pool.py")

# DeepSeek-V4.1-Flash text config (config.json): 40 layers + 3 DSpark/MTP entries
RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
KV_SOURCES = [2, 8, 14, 20]
INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]


def _env(**kw):
    for k, val in kw.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    v.reader.cache_clear()
    v.dedup_rule.cache_clear()
    v.enabled.cache_clear()


# ------------------------------------------------------------------------------------------------ numpy oracles
def _e4m3_table():
    vals = []
    for b in range(127):                      # 0x00..0x7E: every non-negative finite e4m3fn value
        e, m = b >> 3, b & 7
        vals.append(m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return np.array(vals, dtype=np.float64)


E4M3 = _e4m3_table() if HAVE_TORCH else None


def e4m3_oracle(a):
    """fp32 array, |a| <= 448 -> e4m3fn bytes: nearest, ties to the even code (exact in float64), sign kept."""
    a = np.asarray(a, dtype=np.float32)
    x = np.abs(a).astype(np.float64)
    hi = np.clip(np.searchsorted(E4M3, x), 1, 126)
    lo = hi - 1
    dl, dh = x - E4M3[lo], E4M3[hi] - x
    pick = np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(x == 0, 0, pick)
    return (pick | (np.signbit(a).astype(np.int64) << 7)).astype(np.uint8)


def e4m3_int_rn(a):
    """numpy port of the Triton kernels' _v41c_e4m3_rn (integer RN-even)."""
    b = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    sign = (b >> 31) & 1
    x = b & 0x7FFFFFFF
    ex = (x >> 23) - 127
    mant = (x & 0x7FFFFF) | 0x800000
    normal = ex >= -6
    sh = np.where(normal, 20, np.minimum(14 - ex, 30))
    q0 = mant >> sh
    rem = mant & ((1 << sh) - 1)
    half = 1 << (sh - 1)
    up = (rem > half) | ((rem == half) & ((q0 & 1) == 1))
    code = q0 + up + np.where(normal, (ex + 6) << 3, 0)
    code = np.where(x == 0, 0, code)
    return ((sign << 7) | code).astype(np.uint8)


def e4m3_value(b):
    b = np.asarray(b, dtype=np.int64)
    mag = E4M3[np.minimum(b & 127, 126)]
    return np.where(b & 128, -mag, mag).astype(np.float32)


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32) if HAVE_TORCH else None


def cuda_rule_rows(data, scales):
    """numpy port of the IO-warp reader (decode_dsv4_kernel.cuh, v41c_load_fp4 + v41c_store_v4_row) for valid records:
    lane L owns values [16L, 16L + 16) (8 code bytes, scale byte L); lanes 0-27 nope (4 lanes per 64-value tile, amax
    shuffled over the 4), lanes 28-31 rope. Returns (data [n, 576] u8, scales [n, 8] u8) = the shared-memory row."""
    data = np.asarray(data, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    n = data.shape[0]
    out_d = np.zeros((n, 576), np.uint8)
    out_s = np.zeros((n, 8), np.uint8)
    lane_v = []
    for lane in range(32):
        byts = data[:, lane * 8:(lane + 1) * 8].astype(np.int64)
        codes = np.stack([(byts[:, i // 2] >> (4 * (i % 2))) & 15 for i in range(16)], axis=1)
        mag = E2M1[codes & 7]
        val = np.where(codes & 8, -mag, mag).astype(np.float32)
        lane_v.append((val * e4m3_value(scales[:, lane])[:, None]).astype(np.float32))
    lane_amax = [np.max(np.abs(x), axis=1) for x in lane_v]
    for lane in range(32):
        g = lane & ~3
        amax = np.maximum.reduce([lane_amax[g + i] for i in range(4)]).astype(np.float32)
        if lane < 28:
            sc = (np.maximum(np.float32(1e-4), amax) / np.float32(448.0)).astype(np.float32)
            u = sc.view(np.int32).astype(np.int64)
            e = ((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0)
            inv = ((254 - e) << 23).astype(np.int32).view(np.float32)
            a = np.clip(lane_v[lane] * inv[:, None], np.float32(-448.0), np.float32(448.0)).astype(np.float32)
            out_d[:, lane * 16:(lane + 1) * 16] = e4m3_oracle(a)
            if lane % 4 == 0:
                out_s[:, lane >> 2] = e.astype(np.uint8)
        else:
            bits = lane_v[lane].view(np.int32).astype(np.int64) & 0xFFFFFFFF
            rb = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
            pair = np.stack([rb & 255, (rb >> 8) & 255], axis=2).reshape(n, 32).astype(np.uint8)
            out_d[:, 448 + (lane - 28) * 32: 448 + (lane - 27) * 32] = pair
    return out_d, out_s


# ------------------------------------------------------------------------------------------------ fixtures
def _ref():
    return _load("v41_fp4_ref_fast", ROOT / "tests" / "v41_fp4_ref.py")


def _fp4_pool(pages, pbs, seed, tail=False):
    """A [pages, pbs, 1, 288] uint8 FP4 pool (SGLang V41_FP4 page ABI) and the V4 row of every token."""
    r = _ref()
    pb = r.page_bytes(pbs)
    x = r.latents(pages * pbs, seed, tail=tail)
    flat = r.expected_pages(x, torch.arange(pages * pbs), pages, pbs)
    view = torch.as_strided(flat, (pages, pbs, 1, 288), (pb, 288, 288, 1))
    d4, s4 = r.transcode_fp4_to_v4(*r.encode_rows(x))
    return view, d4, s4


def _v4_pool(d4, s4, pages, pbs):
    """v2.6's V4 pool for the same tokens: [pages, pbs, 1, 584], 576-byte rows then 8-byte scales per page."""
    pb = -(-pbs * 584 // 576) * 576
    flat = torch.zeros(pages * pb, dtype=torch.uint8)
    for t in range(pages * pbs):
        p, o = divmod(t, pbs)
        flat[p * pb + o * 576: p * pb + (o + 1) * 576] = d4[t]
        flat[p * pb + pbs * 576 + o * 8: p * pb + pbs * 576 + (o + 1) * 8] = s4[t]
    return torch.as_strided(flat, (pages, pbs, 1, 584), (pb, 584, 584, 1))


def _verify_indices(bs, k, n_tokens, seed, rows_per_req=6, share=0.85, hole_frac=0.1, short=True):
    """DSpark-like verify top-k: rows_per_req rows per request share most entries; per-row lengths, holes."""
    g = torch.Generator().manual_seed(seed)
    rows = bs * rows_per_req
    eidx = torch.full((rows, k), -1, dtype=torch.int32)
    lens = torch.empty(rows, dtype=torch.int32)
    for b in range(bs):
        base = torch.randperm(n_tokens, generator=g)[:k].to(torch.int32)
        ln = int(torch.randint(1, k + 1, (1,), generator=g)) if short else k
        for r in range(rows_per_req):
            row = base.clone()
            swap = torch.rand(k, generator=g) > share
            row[swap] = torch.randint(0, n_tokens, (int(swap.sum()),), generator=g, dtype=torch.int32)
            row[torch.rand(k, generator=g) < hole_frac] = -1
            eidx[b * rows_per_req + r] = row
            lens[b * rows_per_req + r] = min(k, ln + r)
    return eidx, lens


def _read_rows(view, idx):
    raw = torch.as_strided(view, (view.shape[0], v.SUB_PAGE_BYTES), (v.SUB_PAGE_BYTES, 1))
    p, o = idx // 64, idx % 64
    return raw[p, o * 576:(o + 1) * 576], raw[p, 64 * 576 + o * 8: 64 * 576 + (o + 1) * 8]


# ------------------------------------------------------------------------------------------------ numerics
@unittest.skipUnless(HAVE_TORCH, "needs torch + numpy")
class Numerics(unittest.TestCase):
    def test_cuda_rule_equals_reference(self):
        """The IO-warp reader's row == the torch reference the group reader (and f092906's token mode) writes."""
        r = _ref()
        sets = {"random": r.latents(512, 1), "tail": r.latents(512, 2, tail=True), "value_ties": r.value_ties(),
                "scale_ties": r.scale_ties()}
        for name, x in sets.items():
            data, scales = r.encode_rows(x)
            d_ref, s_ref = v.v4_rows(v.decode_records(data, scales))
            d_cu, s_cu = cuda_rule_rows(data.numpy(), scales.numpy())
            with self.subTest(set=name):
                self.assertTrue(np.array_equal(d_cu, d_ref.numpy()))
                self.assertTrue(np.array_equal(s_cu, s_ref.numpy()))

    def test_e4m3_encoders(self):
        """Integer RN-even == exact oracle == torch, on every value the transcode can produce and more."""
        e2 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
        prods = (e2[:, None] * E4M3[None, :].astype(np.float32)).reshape(-1)
        vals = [prods * np.float32(2.0 ** k) for k in range(-24, 24)]
        mids = ((E4M3[:-1] + E4M3[1:]) / 2).astype(np.float32)
        g = np.random.default_rng(0)
        vals += [mids, np.nextafter(mids, np.float32(0)), np.nextafter(mids, np.float32(1000)),
                 g.uniform(-448, 448, 1 << 20).astype(np.float32),
                 (g.standard_normal(1 << 18) * 2.0 ** g.integers(-14, 8, 1 << 18)).astype(np.float32)]
        a = np.concatenate([x.astype(np.float32) for x in vals])
        a = np.concatenate([a, -a, np.array([0.0, -0.0, 448.0, -448.0], np.float32)])
        a = np.clip(a, np.float32(-448.0), np.float32(448.0)).astype(np.float32)
        want = e4m3_oracle(a)
        self.assertTrue(np.array_equal(e4m3_int_rn(a), want))
        got_t = torch.from_numpy(a).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        self.assertTrue(np.array_equal(got_t, want))

    def test_triton_encoder_port_is_the_kernel(self):
        """e4m3_int_rn above is the kernel's _v41c_e4m3_rn statement for statement (guard against silent drift)."""
        src = (V41C / "v41c.py").read_text()
        for needle in ("sh = tl.where(normal, 20, tl.minimum(14 - ex, 30))",
                       "up = (rem > half) | ((rem == half) & ((q0 & 1) == 1))",
                       "code = q0 + up.to(tl.int32) + tl.where(normal, (ex + 6) << 3, 0)",
                       "rb = (ub + 0x7FFF + ((ub >> 16) & 1)) >> 16",
                       "sc = tl.math.div_rn(tl.maximum(amax, 1e-4), 448.0)"):
            self.assertIn(needle, src)


# ------------------------------------------------------------------------------------------------ plans (torch)
@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Plans(unittest.TestCase):
    def setUp(self):
        _env(DSV41_V41C_READER=None, DSV41_V41C_DEDUP_ROWS=None, DSV41_V41C_DEDUP_MIN_ROWS=None)
        v.reset_group()
        v._scratch.clear()

    def _check(self, view, new, eidx, lens, d4, s4):
        live = v._live(eidx, lens)
        self.assertTrue(torch.equal(new >= 0, live))
        for r_, j in zip(*torch.nonzero(live, as_tuple=True)):
            d, s = _read_rows(view, int(new[r_, j]))
            t = int(eidx[r_, j])
            self.assertTrue(torch.equal(d, d4[t]) and torch.equal(s, s4[t]), (int(r_), int(j)))

    def test_group_no_dedupe(self):
        for pbs, pages in ((256, 3), (128, 5)):
            view_fp4, d4, s4 = _fp4_pool(pages, pbs, 3)
            eidx, lens = _verify_indices(2, 128, pages * pbs, 5)
            _env(DSV41_V41C_DEDUP_ROWS="0")
            v.reset_group()
            view, new = v.group_view(view_fp4, eidx, lens)
            self.assertEqual(view.shape, (12 * 128 // 64, 64, 1, 584))
            self._check(view, new, eidx, lens, d4, s4)
            ok = new[new >= 0]
            self.assertEqual(ok.unique().numel(), ok.numel())          # one slot per entry
            self.assertTrue(torch.equal(ok, (torch.nonzero(new >= 0)[:, 0] * 128 + torch.nonzero(new >= 0)[:, 1]).int()))

    def test_group_dedupe(self):
        for pbs, pages in ((256, 3), (128, 5)):
            view_fp4, d4, s4 = _fp4_pool(pages, pbs, 4)
            eidx, lens = _verify_indices(4, 128, pages * pbs, 6)
            before = v.stats["dedup"]
            view, new = v.group_view(view_fp4, eidx, lens)            # 24 rows, block 6: dedupe
            self.assertEqual(v.stats["dedup"], before + 1)
            self._check(view, new, eidx, lens, d4, s4)
            live = v._live(eidx, lens)
            for b in range(4):
                blk_new, blk_tok = new[b * 6:(b + 1) * 6], eidx[b * 6:(b + 1) * 6]
                m = live[b * 6:(b + 1) * 6]
                toks = blk_tok[m].tolist()
                slots = blk_new[m].tolist()
                pairs = set(zip(toks, slots))
                self.assertEqual(len(pairs), len(set(toks)))            # same token -> same slot
                self.assertEqual(len(set(slots)), len(set(toks)))       # distinct tokens -> distinct slots
                self.assertTrue(all(b * 6 * 128 <= s < b * 6 * 128 + len(set(toks)) for s in slots))

    def test_dedup_counts_and_lists(self):
        eidx, lens = _verify_indices(3, 64, 900, 9)
        toks, slots, new, cnt = v.plan_dedup_torch(eidx, lens, 6)
        live = v._live(eidx, lens)
        for b in range(3):
            want = sorted(set(eidx[b * 6:(b + 1) * 6][live[b * 6:(b + 1) * 6]].tolist()))
            self.assertEqual(int(cnt[b]), len(want))
            got = toks[b * 384: b * 384 + len(want)].tolist()
            self.assertEqual(got, want)                                  # sorted distinct tokens, rank = slot
            self.assertTrue(bool((toks[b * 384 + len(want): (b + 1) * 384] == -1).all()))

    def test_dedup_rule(self):
        self.assertEqual(v.dedup_block(6, 512), 0)          # c1: below the default 24 rows
        self.assertEqual(v.dedup_block(24, 512), 6)
        self.assertEqual(v.dedup_block(96, 512), 6)
        self.assertEqual(v.dedup_block(25, 512), 0)         # not whole blocks
        self.assertEqual(v.dedup_block(96, 1024), 0)        # 6 x 1024 > one program's 4096
        _env(DSV41_V41C_DEDUP_ROWS="0")
        self.assertEqual(v.dedup_block(96, 512), 0)
        _env(DSV41_V41C_DEDUP_ROWS="6", DSV41_V41C_DEDUP_MIN_ROWS="6")
        self.assertEqual(v.dedup_block(6, 512), 6)
        _env(DSV41_V41C_DEDUP_ROWS="x")
        with self.assertRaises(ValueError):
            v.dedup_rule()
        _env(DSV41_V41C_DEDUP_ROWS=None, DSV41_V41C_DEDUP_MIN_ROWS=None, DSV41_V41C_READER="bogus")
        with self.assertRaises(ValueError):
            v.reader()
        _env(DSV41_V41C_READER="io")
        self.assertEqual(v.reader(), "io")
        _env(DSV41_V41C_READER=None)
        self.assertEqual(v.reader(), "group")

    def test_rows_limit(self):
        view_fp4, _, _ = _fp4_pool(2, 256, 1)
        with self.assertRaises(ValueError):
            v.group_view(view_fp4, torch.zeros(97, 64, dtype=torch.int32))


# ------------------------------------------------------------------------------------------------ attention equality
def _canary_fns():
    src = CANARY.read_text()
    tree = ast.parse(src)
    ns = {"torch": torch, "math": __import__("math")}
    for node in tree.body:                          # module constants (_NOPE_DIM, ...); env-dependent ones fail
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith("_") for t in node.targets):
            try:
                exec(compile(ast.Module([node], []), str(CANARY), "exec"), ns)
            except Exception:  # noqa: BLE001
                pass
    for name in ("_gather_and_dequant", "_sm120_sparse_decode_fwd"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.get_source_segment(src, fn), str(CANARY), "exec"), ns)
    return ns["_sm120_sparse_decode_fwd"]


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Attention(unittest.TestCase):
    def test_v26_pool_vs_group_scratch(self):
        """Same queries, same SWA, extra = v2.6's V4 pool + original indices vs group scratch + rewritten indices."""
        fwd = _canary_fns()
        for dedupe, bs in (("0", 1), ("6", 4)):
            _env(DSV41_V41C_DEDUP_ROWS=dedupe, DSV41_V41C_DEDUP_MIN_ROWS="6")
            v.reset_group()
            v._scratch.clear()
            pages, pbs = 3, 256
            view_fp4, d4, s4 = _fp4_pool(pages, pbs, 11)
            v4pool = _v4_pool(d4, s4, pages, pbs)
            eidx, lens = _verify_indices(bs, 128, pages * pbs, 12)
            rows = eidx.shape[0]
            g = torch.Generator().manual_seed(13)
            q = torch.randn(rows, 1, 8, 512, generator=g).to(torch.bfloat16)
            swa = v4pool[:1]                              # any V4 cache works as the main source
            sidx = torch.randint(0, pbs, (rows, 1, 64), generator=g, dtype=torch.int32)
            sink = torch.randn(8, generator=g)
            ref = fwd(q, swa, sidx, None, sink, 512, 0.04, v4pool, eidx.unsqueeze(1), lens)
            view, new = v.group_view(view_fp4, eidx, lens)
            got = fwd(q, swa, sidx, None, sink, 512, 0.04, view, new.unsqueeze(1), lens)
            with self.subTest(dedupe=dedupe):
                self.assertTrue(torch.equal(ref[0], got[0]))
                self.assertTrue(torch.equal(ref[1], got[1]))
        _env(DSV41_V41C_DEDUP_ROWS=None, DSV41_V41C_DEDUP_MIN_ROWS=None)


# ------------------------------------------------------------------------------------------------ kernels (interpreter)
@unittest.skipUnless(HAVE_TRITON, "needs triton (TRITON_INTERPRET=1 runs the kernels on the CPU)")
class Kernels(unittest.TestCase):
    def setUp(self):
        _env(DSV41_V41C_READER=None, DSV41_V41C_DEDUP_ROWS=None, DSV41_V41C_DEDUP_MIN_ROWS=None)
        v.reset_group()

    def _pair(self, pages, pbs, eidx, lens, g, seed, tail=False):
        view_fp4, d4, s4 = _fp4_pool(pages, pbs, seed, tail=tail)
        rows, k = eidx.shape
        n_pages = -(-rows * k // 64)
        buf_t = torch.zeros(n_pages, v.SUB_PAGE_BYTES, dtype=torch.uint8)
        buf_r = buf_t.clone()
        if g:
            new_t = v._dedup_triton(view_fp4, eidx.contiguous(), lens, buf_t, g)
            toks, slots, new_r, _ = v.plan_dedup_torch(eidx, lens, g)
        else:
            new_t = v._plan_group_triton(view_fp4, eidx.contiguous(), lens, buf_t)
            toks, slots, new_r = v.plan_group_torch(eidx, lens)
        v._transcode_list_torch(view_fp4, toks, buf_r, slots)
        return new_t, new_r, buf_t, buf_r

    def test_plan_kernel(self):
        for pbs, pages, lens_on in ((256, 3, True), (128, 5, False)):
            eidx, lens = _verify_indices(2, 64, pages * pbs, 21)
            new_t, new_r, bt, br = self._pair(pages, pbs, eidx, lens if lens_on else None, 0, 22, tail=True)
            with self.subTest(pbs=pbs):
                self.assertTrue(torch.equal(new_t, new_r))
                self.assertTrue(torch.equal(bt, br))

    def test_dedup_kernel(self):
        for pbs, pages, lens_on in ((256, 3, True), (128, 5, False)):
            eidx, lens = _verify_indices(3, 64, pages * pbs, 31)
            new_t, new_r, bt, br = self._pair(pages, pbs, eidx, lens if lens_on else None, 6, 32)
            with self.subTest(pbs=pbs):
                self.assertTrue(torch.equal(new_t, new_r))
                self.assertTrue(torch.equal(bt, br))

    def test_ties_through_kernel(self):
        """e2m1 value ties and e4m3 scale ties go through the Triton kernel bit-exact (rows 0..n of a pool)."""
        r = _ref()
        x = torch.cat([r.value_ties()[:120], r.scale_ties()[:8]])
        pbs = 128
        flat = r.expected_pages(x, torch.arange(x.shape[0]), 1, pbs)
        pool = torch.as_strided(flat, (1, pbs, 1, 288), (r.page_bytes(pbs), 288, 288, 1))
        eidx = torch.arange(x.shape[0], dtype=torch.int32).reshape(1, -1)
        buf = torch.zeros(-(-x.shape[0] // 64), v.SUB_PAGE_BYTES, dtype=torch.uint8)
        new = v._plan_group_triton(pool, eidx, None, buf)
        d_cu, s_cu = cuda_rule_rows(*[t.numpy() for t in r.encode_rows(x)])
        view = v.sub_pages_view(buf, buf.shape[0])
        for t in range(x.shape[0]):
            d, s = _read_rows(view, int(new[0, t]))
            self.assertTrue(np.array_equal(d.numpy(), d_cu[t]) and np.array_equal(s.numpy(), s_cu[t]), t)

    def test_grids_static(self):
        """Grids and output shapes from (rows, K) only: record every launch for 0 / 50 / 100 % holes, any lengths."""
        seen = []
        orig = (v._v41c_plan_kernel, v._v41c_dedup_kernel, v._v41c_list_kernel)

        class Rec:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def launch(*args, **kw):
                    seen.append((self.name, tuple(grid), tuple(a.shape for a in args if hasattr(a, "shape")),
                                 tuple(sorted((k_, val) for k_, val in kw.items() if not hasattr(val, "shape")))))
                return launch
        try:
            v._v41c_plan_kernel, v._v41c_dedup_kernel, v._v41c_list_kernel = Rec("plan"), Rec("dedup"), Rec("list")
            view_fp4, _, _ = _fp4_pool(2, 256, 41)
            for g in (0, 6):
                runs = []
                for holes in (0.0, 0.5, 1.0):
                    seen.clear()
                    eidx, lens = _verify_indices(4, 512, 512, 42, hole_frac=holes)
                    buf = torch.zeros(24 * 512 // 64, v.SUB_PAGE_BYTES, dtype=torch.uint8)
                    new = (v._dedup_triton(view_fp4, eidx, lens, buf, g) if g
                           else v._plan_group_triton(view_fp4, eidx, lens, buf))
                    runs.append((tuple(seen), tuple(new.shape)))
                self.assertEqual(len(set(runs)), 1, runs)
                self.assertEqual(len(runs[0][0]), 2 if g else 1)
        finally:
            v._v41c_plan_kernel, v._v41c_dedup_kernel, v._v41c_list_kernel = orig


# ------------------------------------------------------------------------------------------------ capture safety
@unittest.skipUnless(HAVE_TORCH, "needs torch")
class CaptureSafety(unittest.TestCase):
    def test_torch_planners_under_fake_tensors(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        eidx, lens = _verify_indices(4, 512, 4096, 51)
        with FakeTensorMode() as mode:
            fe, fl = mode.from_tensor(eidx), mode.from_tensor(lens)
            a = v.plan_group_torch(fe, fl)
            b = v.plan_dedup_torch(fe, fl, 6)
        self.assertEqual([tuple(t.shape) for t in a], [(24 * 512,), (24 * 512,), (24, 512)])
        self.assertEqual([tuple(t.shape) for t in b], [(4 * 3072,), (4 * 3072,), (24, 512), (4,)])

    def test_gpu_path_has_no_host_sync(self):
        """The functions on the captured GPU path call nothing that waits for the device."""
        bad_attr = {"item", "tolist", "nonzero", "cpu", "numpy", "unique", "masked_select", "synchronize", "any", "all"}
        for fn in (v.group_view, v._plan_group_triton, v._dedup_triton, v._group_sig, v._group_hit, v._group_keep,
                   v._rows_2d, v.dedup_block, v._token_scratch, v.sub_pages_view, v.fp4_page_view):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, bad_attr, f"{fn.__name__}: .{node.attr}")
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Compare):
                    self.fail(f"{fn.__name__}: boolean-mask indexing")
            # int() / bool() only on host values (shapes, data_ptr, env)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("int", "bool"):
                    arg = ast.unparse(node.args[0]) if node.args else ""
                    self.assertTrue(arg == "g" or any(s in arg for s in ("data_ptr()", ".shape", ".stride(")),
                                    f"{fn.__name__}: {node.func.id}({arg})")

    def test_scratch_is_static(self):
        v._scratch.clear()
        a = v._token_scratch(torch.device("cpu"), 6 * 512 // 64, 512)
        self.assertEqual(a.shape[0], 96 * 512 // 64)
        self.assertIs(v._token_scratch(torch.device("cpu"), 96 * 512 // 64, 512), a)


# ------------------------------------------------------------------------------------------------ layer groups
class _Layer:
    def __init__(self, lid):
        self.layer_id = lid


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Groups(unittest.TestCase):
    def setUp(self):
        _env(DSV41_V41C_READER=None, DSV41_V41C_DEDUP_ROWS=None, DSV41_V41C_DEDUP_MIN_ROWS=None)
        v.reset_group()
        v._scratch.clear()
        for k_ in v.stats:
            v.stats[k_] = 0

    def test_group_starts(self):
        cfg = types.SimpleNamespace(compress_ratios=RATIOS, kv_source_layer_ids=KV_SOURCES,
                                    index_source_layer_ids=INDEX_SOURCES)
        st = P.group_starts(cfg)
        want = {L: (2 if L < 8 else 8 if L < 14 else 14 if L < 20 else 20 if L < 24 else 4 * (L // 4))
                for L in range(2, 40)}
        self.assertEqual(st, want)
        self.assertEqual(len(set(st.values())), 8)
        self.assertEqual(P.group_starts(types.SimpleNamespace()), {})

    def _backend_module(self, body):
        mod = types.ModuleType("fake_backend")

        class DeepseekV4AttnBackend:
            def __init__(self):
                self.model_runner = types.SimpleNamespace(model_config=types.SimpleNamespace(
                    hf_text_config=types.SimpleNamespace(compress_ratios=RATIOS, kv_source_layer_ids=KV_SOURCES,
                                                         index_source_layer_ids=INDEX_SOURCES)))

            def _forward_attention(self, q, k, v_, layer, forward_batch, *a, **kw):
                return body(layer)
        mod.DeepseekV4AttnBackend = DeepseekV4AttnBackend
        return mod

    def test_wrapper_tags_and_clears(self):
        seen = []
        mod = self._backend_module(lambda layer: seen.append((layer.layer_id, v._tag[0])))
        orig = P._reader_module
        try:
            P._reader_module = lambda: v
            P.install_layer_tags(mod)
            P.install_layer_tags(mod)                                   # idempotent
            self.assertTrue(mod.DeepseekV4AttnBackend._forward_attention._dsv41_v41c_tags)
            be = mod.DeepseekV4AttnBackend()
            for L in (0, 1, 2, 7, 8, 21, 39, 40):
                be._forward_attention(None, None, None, _Layer(L), None)
                self.assertIsNone(v._tag[0])                            # cleared after the call
            self.assertEqual(seen, [(0, None), (1, None), (2, (2, 2)), (7, (7, 2)), (8, (8, 8)), (21, (21, 20)),
                                    (39, (39, 36)), (40, None)])
            boom = self._backend_module(lambda layer: 1 / 0)
            P.install_layer_tags(boom)
            with self.assertRaises(ZeroDivisionError):
                boom.DeepseekV4AttnBackend()._forward_attention(None, None, None, _Layer(5), None)
            self.assertIsNone(v._tag[0])
        finally:
            P._reader_module = orig

    def test_check_backend_respects_reader(self):
        mod = self._backend_module(lambda layer: None)
        calls = []
        orig_pins, orig_install = P.check_pins, P.install_layer_tags
        try:
            P.check_pins = lambda m, pins=None: calls.append("pins")
            P.install_layer_tags = lambda m: calls.append("tags")
            os.environ["DSV41_V41C_READER"] = "io"
            P.check_backend(mod)
            os.environ.pop("DSV41_V41C_READER")
            P.check_backend(mod)
            self.assertEqual(calls, ["pins", "pins", "tags"])
            os.environ["DSV41_V41C_READER"] = "x"
            with self.assertRaises(ValueError):
                P.check_backend(mod)
        finally:
            os.environ.pop("DSV41_V41C_READER", None)
            P.check_pins, P.install_layer_tags = orig_pins, orig_install

    def test_forward_reuses_within_groups_and_follows_refills(self):
        """Layers 2..39 through the wrapper: the ratio-2 pools are three tensors, ratio 1 is one; each ratio's top-k
        buffer is ONE tensor refilled in place at its index sources (same pointer, new content), as in SGLang."""
        pbs = 256
        pools = {s: _fp4_pool(2, pbs, 60 + s) for s in KV_SOURCES}
        buf = {2: torch.empty(24, 128, dtype=torch.int32), 1: torch.empty(24, 128, dtype=torch.int32)}
        lens = {2: torch.full((24,), 100, dtype=torch.int32), 1: torch.full((24,), 128, dtype=torch.int32)}
        checks = []

        def body(layer):
            L = layer.layer_id
            ratio = RATIOS[L]
            if L in INDEX_SOURCES:                                     # the index source refills in place
                e, _ = _verify_indices(4, 128, 2 * pbs, 100 + L)
                buf[ratio].copy_(e)
            src = max(s for s in KV_SOURCES if s <= L and RATIOS[s] == ratio)
            view_fp4, d4, s4 = pools[src]
            view, new = v.group_view(view_fp4, buf[ratio], lens[ratio])
            live = v._live(buf[ratio], lens[ratio])
            self.assertTrue(torch.equal(new >= 0, live))
            for r_, j in list(zip(*torch.nonzero(live, as_tuple=True)))[::97]:
                d, s = _read_rows(view, int(new[r_, j]))
                t = int(buf[ratio][r_, j])
                checks.append(torch.equal(d, d4[t]) and torch.equal(s, s4[t]))

        mod = self._backend_module(body)
        orig = P._reader_module
        try:
            P._reader_module = lambda: v
            P.install_layer_tags(mod)
            be = mod.DeepseekV4AttnBackend()
            for step in range(2):
                for L in range(0, 43):
                    if RATIOS[L] in (1, 2):
                        be._forward_attention(None, None, None, _Layer(L), None)
            self.assertTrue(all(checks) and len(checks) > 100)
            self.assertEqual(v.stats["plan"], 2 * 8)                   # 8 groups per forward
            self.assertEqual(v.stats["reuse"], 2 * 30)
            self.assertEqual(v.stats["dedup"], 2 * 8)                  # 24 rows, block 6
        finally:
            P._reader_module = orig

    def test_no_reuse_without_tag_or_order(self):
        view_fp4, _, _ = _fp4_pool(2, 256, 70)
        eidx, lens = _verify_indices(1, 64, 512, 71)
        v.group_view(view_fp4, eidx, lens)
        v.group_view(view_fp4, eidx, lens)                              # no tags: transcode every call
        self.assertEqual((v.stats["plan"], v.stats["reuse"]), (2, 0))
        v.tag_layer(3, 2)
        v.group_view(view_fp4, eidx, lens)                              # first tagged call: plan
        v.tag_layer(5, 2)
        v.group_view(view_fp4, eidx, lens)                              # not layer 4: plan
        v.tag_layer(6, 2)
        a = v.group_view(view_fp4, eidx, lens)                          # consecutive, same group: reuse
        self.assertEqual((v.stats["plan"], v.stats["reuse"]), (4, 1))
        v.tag_layer(7, 2)
        b = v.group_view(view_fp4, eidx.clone(), lens)                  # another tensor: plan
        v.tag_layer(8, 8)
        v.group_view(view_fp4, eidx, lens)
        v.tag_layer(9, 2)                                               # wrong group start: plan
        v.group_view(view_fp4, eidx, lens)
        self.assertEqual((v.stats["plan"], v.stats["reuse"]), (7, 1))
        self.assertIsNot(a[1], b[1])
        v.tag_layer(2, 2)
        v.group_view(view_fp4, eidx, lens)
        v.tag_layer(2, 2)                                               # the group start always recomputes
        v.group_view(view_fp4, eidx, lens)
        self.assertEqual(v.stats["plan"], 9)


# ------------------------------------------------------------------------------------------------ staged SGLang
def _sglang_root():
    for c in (os.environ.get("SGLANG_CANARY_PY"), ROOT / "runtime" / "sglang-canary" / "python",
              ROOT.parent / "ds41-release" / "runtime" / "sglang-canary" / "python", "/sgl-workspace/sglang/python"):
        if c and (Path(c) / "sglang" / "srt" / "layers" / "attention" / "deepseek_v4_backend.py").exists():
            return Path(c)
    return None


SG = _sglang_root()


def _src_of(path, qualname):
    text = path.read_text()
    node = P._node(ast.parse(text), qualname)
    return ast.get_source_segment(text, node)


@unittest.skipIf(SG is None, "no staged SGLang tree (SGLANG_CANARY_PY)")
class StagedSGLang(unittest.TestCase):
    """What the group reuse relies on, read from v2.6's staged SGLang (the wrapped method is also sha-pinned)."""

    def test_one_reader_call_per_tagged_attention(self):
        f = _src_of(SG / "sglang/srt/layers/attention/deepseek_v4_backend.py", "DeepseekV4AttnBackend._forward_attention")
        self.assertEqual(f.count("flash_mla_with_kvcache_sm120("), 1)
        self.assertIn("get_extra_key_buffer(layer_id)", f)
        self.assertIn("core_attn_metadata.sparse_page_indices(compress_ratio)", f)
        self.assertIn("layer_id = layer.layer_id", f)
        b = (SG / "sglang/srt/layers/attention/deepseek_v4_backend.py").read_text()
        self.assertIn("Fixed-shape top-k buffers the index_source layers fill in place", b)

    def test_only_sources_write(self):
        m = (SG / "sglang/srt/models/deepseek_v4.py").read_text()
        self.assertIn("if self.layer_id in config.kv_source_layer_ids:\n                self.compressor = DeepseekV41Compressor(", m)
        self.assertIn("if self.layer_id in config.index_source_layer_ids:\n                self.indexer = DeepseekV41Indexer(", m)
        pool = (SG / "sglang/srt/mem_cache/deepseek_v4_memory_pool.py").read_text()
        self.assertIn("the nearest preceding kv_source layer for ratios 1/2", pool)

    def test_pin_covers_the_wrapped_method(self):
        text = (SG / "sglang/srt/layers/attention/deepseek_v4_backend.py").read_text()
        self.assertEqual(P.source_pin(text, "DeepseekV4AttnBackend._forward_attention"),
                         P.PINS[P.BACKEND_MODULE]["DeepseekV4AttnBackend._forward_attention"])


# ------------------------------------------------------------------------------------------------ routing
@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Routing(unittest.TestCase):
    def setUp(self):
        self.rec = []
        fi = types.ModuleType("flashinfer.mla._sparse_mla_sm120")
        fi._sparse_mla_sm120_paged_attention = lambda *a, **k: self.rec.append(("stock", a, k))
        at = types.ModuleType("flashinfer.autotuner")
        at.AutoTuner = types.SimpleNamespace(get=lambda: types.SimpleNamespace(is_tuning_mode=False))
        self.saved_at = sys.modules.get("flashinfer.autotuner")
        sys.modules["flashinfer.autotuner"] = at
        for n in ("flashinfer", "flashinfer.mla"):
            m = types.ModuleType(n)
            m.__path__ = []
            sys.modules.setdefault(n, m)
        self.saved = sys.modules.get("flashinfer.mla._sparse_mla_sm120")
        sys.modules["flashinfer.mla._sparse_mla_sm120"] = fi
        v.reset_group()
        v._scratch.clear()
        self.saved_receipt = v.group_receipt
        v.group_receipt = lambda: self.rec.append(("receipt",)) or "receipt"

    def tearDown(self):
        v.group_receipt = self.saved_receipt
        if self.saved_at is not None:
            sys.modules["flashinfer.autotuner"] = self.saved_at
        else:
            sys.modules.pop("flashinfer.autotuner", None)
        if self.saved is not None:
            sys.modules["flashinfer.mla._sparse_mla_sm120"] = self.saved
        else:
            sys.modules.pop("flashinfer.mla._sparse_mla_sm120", None)
        _env(DSV41_V41C_READER=None)

    def test_decode_group_calls_stock(self):
        _env(DSV41_V41C_READER=None)
        view_fp4, _, _ = _fp4_pool(2, 256, 80)
        eidx, lens = _verify_indices(1, 512, 512, 81)
        q = torch.zeros(6, 64, 512, dtype=torch.bfloat16)
        kv, idx = torch.zeros(4, 64, 1, 584, dtype=torch.uint8), torch.zeros(6, 192, dtype=torch.int32)
        mid_o, mid_l = torch.zeros(6, 64, 11, 512, dtype=torch.bfloat16), torch.zeros(6, 64, 11)
        out, lse = torch.zeros(6, 64, 512, dtype=torch.bfloat16), torch.zeros(6, 64)
        v.decode(q, kv, idx, mid_o, mid_l, out, lse, 0.04, topk_length=None, attn_sink=None, extra_kv=view_fp4,
                 extra_indices=eidx, extra_topk_length=lens)
        self.assertEqual(self.rec[0], ("receipt",))
        (name, a, k), = self.rec[1:]
        self.assertEqual(name, "stock")
        self.assertIs(a[0], q)
        self.assertIs(a[1], kv)
        self.assertEqual(tuple(k["extra_kv_cache"].shape), (6 * 512 // 64, 64, 1, 584))
        self.assertEqual(k["extra_kv_cache"].stride(0), v.SUB_PAGE_BYTES)
        self.assertEqual(k["extra_indices"].dtype, torch.int32)
        self.assertIs(k["mid_out"], mid_o)
        self.assertIs(k["mid_lse"], mid_l)
        self.assertIs(k["extra_topk_length"], lens)
        self.assertEqual(k["d_v"], 512)

    def test_decode_io_keeps_runner(self):
        _env(DSV41_V41C_READER="io")
        orig = v._runner
        try:
            v._runner = lambda: (_ for _ in ()).throw(RuntimeError("io runner"))
            with self.assertRaisesRegex(RuntimeError, "io runner"):
                v.decode(None, None, torch.zeros(1, 128), None, None, None, None, 0.1, topk_length=None,
                         attn_sink=None, extra_kv=None, extra_indices=torch.zeros(1, 512), extra_topk_length=None)
        finally:
            v._runner = orig

    def test_prefill_routes_by_rows(self):
        _env(DSV41_V41C_READER=None)
        view_fp4, _, _ = _fp4_pool(2, 256, 90)
        calls = []
        orig_g, orig_p, orig_t = v.group_view, v.group_page_view, v.transcode_for_prefill
        try:
            v.group_view = lambda *a: calls.append("group") or orig_g(*a)
            v.group_page_view = lambda *a: calls.append("group-page") or orig_p(*a)
            v.transcode_for_prefill = lambda *a: calls.append("page") or orig_t(*a, transcode=v.transcode_tokens_torch)
            for rows in (72, 96, 128):
                e = torch.randint(0, 512, (rows, 64), dtype=torch.int32)
                v.prefill(torch.zeros(rows, 16, 512, dtype=torch.bfloat16), None, None, 0.04, topk_length=None,
                          attn_sink=None, extra_fp4=view_fp4, extra_idx=e, extra_topk_length=None, head_dim_v=512)
            self.assertEqual(calls, ["group", "group", "group-page"])
            _env(DSV41_V41C_READER="io")
            calls.clear()
            for rows in (72, 128):
                v.prefill(torch.zeros(rows, 16, 512, dtype=torch.bfloat16), None, None, 0.04, topk_length=None,
                          attn_sink=None, extra_fp4=view_fp4, extra_idx=torch.zeros(rows, 64, dtype=torch.int32),
                          extra_topk_length=None, head_dim_v=512)
            self.assertEqual(calls, ["page", "page"])
        finally:
            v.group_view, v.group_page_view, v.transcode_for_prefill = orig_g, orig_p, orig_t


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class PageMode(unittest.TestCase):
    """> 96 rows (eager prefill chunks): f092906's page mode on live entries, planned once per layer group."""

    def setUp(self):
        _env(DSV41_V41C_READER=None, DSV41_V41C_DEDUP_ROWS=None, DSV41_V41C_DEDUP_MIN_ROWS=None)
        v.reset_group()
        for k_ in v.stats:
            v.stats[k_] = 0

    def test_live_entries_read_their_rows(self):
        for pbs, pages in ((256, 4), (128, 6)):
            view_fp4, d4, s4 = _fp4_pool(pages, pbs, 120 + pbs)
            g = torch.Generator().manual_seed(pbs)
            eidx = torch.randint(0, pages * pbs, (200, 64), generator=g, dtype=torch.int32)
            eidx[torch.rand(200, 64, generator=g) < 0.1] = -1
            lens = torch.randint(1, 65, (200,), generator=g, dtype=torch.int32)
            view, new = v.group_page_view(view_fp4, eidx, lens)
            live = v._live(eidx, lens)
            self.assertTrue(torch.equal(new >= 0, live))
            for r_, j in list(zip(*torch.nonzero(live, as_tuple=True)))[::7]:
                d, s = _read_rows(view, int(new[r_, j]))
                t = int(eidx[r_, j])
                self.assertTrue(torch.equal(d, d4[t]) and torch.equal(s, s4[t]), (int(r_), int(j)))
            touched = (eidx[live] // 64).unique().numel()
            self.assertEqual(view.shape[0], touched)                 # only sub-pages a live entry touches

    def test_reused_across_group(self):
        view_fp4, _, _ = _fp4_pool(3, 256, 130)
        eidx = torch.randint(0, 768, (300, 64), dtype=torch.int32)
        v.tag_layer(20, 20)
        a = v.group_page_view(view_fp4, eidx)
        for L in (21, 22, 23):
            v.tag_layer(L, 20)
            b = v.group_page_view(view_fp4, eidx)
            self.assertIs(b[1], a[1])
        v.tag_layer(24, 24)                                            # index source: refilled top-k
        v.group_page_view(view_fp4, eidx)
        self.assertEqual((v.stats["plan"], v.stats["reuse"], v.stats["page"]), (2, 3, 2))


if __name__ == "__main__":
    unittest.main(verbosity=1)
