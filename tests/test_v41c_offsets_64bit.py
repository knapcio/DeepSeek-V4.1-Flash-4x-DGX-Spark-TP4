"""CPU checks: the v41c / v41n FP4 record addressing stays 64-bit past the 2 GiB byte boundary of a ratio-1 pool.

Why: the 2026-11-04 capacity window (run-20261104-203157-2655898, CAP boot, DSV41_KV_LAYOUT=v41c, reader native)
granted 9,408,512 tokens. Its ratio-1 FP4 pool is one buffer of 36,753 pages x 73,728 B = 2,709,725,184 B (the
DSV41_V41C_POOL receipt), so every record at a token slot >= 7,456,540 sits above byte 2^31. A byte offset computed
in 32-bit integers would wrap there. 8 concurrent 1M-token requests held 7,776,256 tokens, and 2 of 8 needles were
not retrieved. This file pins the arithmetic of that boundary and checks that every reader site multiplies the page
index by the page stride in 64 bits, so such a wrap cannot be introduced silently.

- geometry: the boundary slot of the CAP pool, the previous capacity run's pool (9,298,176 tokens, 8/8 retrieved, same
  exposure) and the stock V4 ratio-1 pool at the A arms' pinned 4,194,304 tokens (584 B records: boundary 3,670,...);
  what a 32-bit computation would return there (a negative offset, i.e. an address before the buffer);
- CUDA sources (runtime/flashinfer_v41c): every `<page index> * <page stride>` product has a size_t operand;
- Triton (v41c.py): the token id is cast to int64 before `(tok // PBS) * src_stride` in both transcode kernels, and the
  V4 scratch rows are addressed from an int64 slot.

usage: python3 tests/test_v41c_offsets_64bit.py      (stdlib only)
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V41C = ROOT / "runtime" / "flashinfer_v41c"

TWO_GIB = 1 << 31
FP4_DATA, FP4_SCALE = 256, 32                 # KVLayout.V41_FP4: 256 e2m1 bytes + 32 e4m3 scales
FP4_PAGE_256 = 73728                          # 256 x 288, already a multiple of the 256-byte page align
V4_DATA, V4_SCALE, V4_ALIGN = 576, 8, 576


def v4_page_bytes(page_size):
    return -(-(page_size * (V4_DATA + V4_SCALE)) // V4_ALIGN) * V4_ALIGN


def record_end(loc, page_size, page_bytes, data_bytes, scale_bytes):
    """Last byte (exclusive) a record touches: PagedKV::scale_offset(loc) + scale_bytes (the scale rows follow the
    data rows of the page, so the scale row is the record's highest byte)."""
    page, slot = divmod(loc, page_size)
    return page * page_bytes + page_size * data_bytes + slot * scale_bytes + scale_bytes


def first_slot_past(limit, page_size, page_bytes, data_bytes, scale_bytes):
    """Smallest token slot whose data or scale byte lies at or above `limit` (binary search on a monotone bound)."""
    def data_start(loc):
        page, slot = divmod(loc, page_size)
        return page * page_bytes + slot * data_bytes

    def touches(loc):
        return record_end(loc, page_size, page_bytes, data_bytes, scale_bytes) > limit or data_start(loc) >= limit
    lo, hi = 0, 1 << 40
    while lo < hi:
        mid = (lo + hi) // 2
        if touches(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def wrap32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= 1 << 31 else x


def pool_pages(tokens, page_size=256):
    return -(-tokens // page_size)


class Geometry(unittest.TestCase):
    def test_cap_pool_crosses_2gib(self):
        pages = 36753                                   # receipt: r1=v41_fp4:page=256:pages=36753:bytes=2709725184
        self.assertEqual(pages * FP4_PAGE_256, 2709725184)
        self.assertEqual(pool_pages(9408512), 36752)    # the full pool; the receipt counts page 0 (padded slot) too
        self.assertGreater(pages * FP4_PAGE_256, TWO_GIB)
        b = first_slot_past(TWO_GIB, 256, FP4_PAGE_256, FP4_DATA, FP4_SCALE)
        # page 29127 is the first page whose scale rows cross 2^31 (29127 x 73728 = 2,147,475,456; + 65,536 data bytes)
        self.assertEqual(b // 256, 29127)
        self.assertEqual(29127 * FP4_PAGE_256 + 256 * FP4_DATA > TWO_GIB, True)
        self.assertLess(b, 7_776_256)                   # the CAP run's peak #full token: the pool was used past it
        self.assertLess(b, 9_408_512)

    def test_previous_run_had_the_same_exposure(self):
        pages = pool_pages(9298176) + 1                 # run-20261103-035000-1206614: 8/8 retrieved
        self.assertGreater(pages * FP4_PAGE_256, TWO_GIB)
        b = first_slot_past(TWO_GIB, 256, FP4_PAGE_256, FP4_DATA, FP4_SCALE)
        self.assertLess(b, 7_778_304)                   # its peak #full token

    def test_stock_v4_pool_crosses_earlier(self):
        pb = v4_page_bytes(256)
        self.assertEqual(pb, 149760)
        pages = pool_pages(4194304) + 1                 # the A / X arms' pinned pool, stock 584 B ratio-1 records
        self.assertGreater(pages * pb, TWO_GIB)
        b = first_slot_past(TWO_GIB, 256, pb, V4_DATA, V4_SCALE)
        self.assertEqual(b // 256, TWO_GIB // pb)       # page 14339
        self.assertLess(b, 4194304)

    def test_int32_wrap_would_point_before_the_buffer(self):
        loc = 9_000_000                                 # a slot the CAP pool can hand out
        page, slot = divmod(loc, 256)
        off64 = page * FP4_PAGE_256 + slot * FP4_DATA
        self.assertGreater(off64, TWO_GIB)
        self.assertLess(wrap32(page * FP4_PAGE_256) + slot * FP4_DATA, 0)


# a product of a page / block index with a page stride; the index operand must be cast to size_t (or the stride be
# size_t and the index promoted by it: written as `(size_t)idx * stride` in every site today)
STRIDES = r"(?:stride_kv_block(?:_extra|_now)?|stride_extra_kv_block|page_stride|section_stride)"
PRODUCT = re.compile(r"([()\w:]+)\s*\*\s*" + STRIDES + r"\b")


def code_lines(text):
    for n, ln in enumerate(text.splitlines(), 1):
        s = ln.split("//", 1)[0]
        if s.strip():
            yield n, s


class CudaSources(unittest.TestCase):
    def test_every_page_stride_product_is_64bit(self):
        files = sorted(list((V41C / "csrc").glob("*.cu")) + list((V41C / "include").rglob("*.cuh")))
        self.assertTrue(files)
        seen, bad = 0, []
        for f in files:
            for n, s in code_lines(f.read_text()):
                for m in PRODUCT.finditer(s):
                    seen += 1
                    if not m.group(1).startswith("(size_t)"):
                        bad.append(f"{f.relative_to(ROOT)}:{n}: {s.strip()}")
        self.assertGreater(seen, 10)
        self.assertEqual(bad, [])

    def test_stride_arguments_are_size_t(self):
        bad = []
        for f in sorted(list((V41C / "csrc").glob("*.cu")) + list((V41C / "include").rglob("*.cuh"))):
            for n, s in code_lines(f.read_text()):
                for m in re.finditer(r"\b(int|int32_t|unsigned|uint32_t)\s+(" + STRIDES + r")\b", s):
                    bad.append(f"{f.relative_to(ROOT)}:{n}: {s.strip()}")
        self.assertEqual(bad, [])

    def test_generated_prefill_gather(self):
        text = (V41C / "gen_v41n_prefill.py").read_text()
        self.assertIn("kv_ptr + (size_t)page * stride_kv_block", text)


class TritonSources(unittest.TestCase):
    def kernel(self, name):
        text = (V41C / "v41c.py").read_text()
        m = re.search(r"\n    def " + name + r"\(.*?(?=\n    @triton\.jit|\n    def _v41n_page_plan_triton)", text, re.S)
        self.assertIsNotNone(m, name)
        return m.group(0)

    def test_rows_kernel_token_is_int64(self):
        k = self.kernel("_v41c_rows")
        self.assertIn("t64 = tl.where(live, tok, 0).to(tl.int64)", k)
        self.assertIn("page = src + (t64 // PBS) * src_stride", k)
        self.assertIn("s64 = slot.to(tl.int64)", k)
        self.assertIn("row = dst + (s64 // 64) * 37440", k)
        self.assertIn("sc_row = dst + (s64 // 64) * 37440", k)

    def test_token_transcode_kernel_is_int64(self):
        k = self.kernel("_fp4_to_v4_kernel")
        self.assertIn("tok = tl.load(tok_ptr + i).to(tl.int64)", k)
        self.assertIn("page_base = src + (tok // PBS) * src_stride", k)

    def test_page_and_list_kernels_go_through_rows(self):
        for name in ("_v41n_page_kernel", "_v41c_list_kernel", "_v41c_plan_kernel"):
            k = self.kernel(name)
            self.assertIn("_v41c_rows(src, src_stride, PBS,", k, name)
            self.assertNotIn("* src_stride", k, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
