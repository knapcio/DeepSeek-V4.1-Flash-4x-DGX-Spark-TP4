"""CPU checks of DSV41_SWA_NATIVE_DECODE ("swan", runtime/flashinfer_v41c/v41c.py swan_*, csrc/swan_decode_dsv4.cu, the
256-token dispatch of csrc/v41n_decode_dsv4.cu, the canary block in runtime/flash_mla_sm120.canary.py).

v2.6.x converts the SWA pool's 256-token footer pages into 64-token pages on every decode call (the canary's
_split_kv_pages_to_64: mask fill + page mark + page split) because FlashInfer's decode-dsv4 is instantiated for 64-token
main pages only. The kernel's main-section address math is generic in the page size: entry idx loads 448 + 128 bytes at
(idx / pbs) * stride + (idx % pbs) * 576 and its 8-byte footer word at (idx / pbs) * stride + pbs * 576 + (idx % pbs) * 8.
So the same kernel at PAGE_BLOCK_SIZE 256 on the pool itself loads, for every entry, the bytes the 64-token split copy
holds, and every shared-memory row (hence every output bit) is the stock call's. What must hold:

- addressing: for random pools, strides and indices (incl. -1, page edges, the last token), the kernel's load rule on
  the native 256-token view == the same rule on the canary's split output (the real Triton kernels under
  TRITON_INTERPRET=1 when triton is present, else a torch copy of the split);
- launcher text: swan_decode_dsv4.cu == FlashInfer's stock launcher (runtime/flashinfer_pce/base) with exactly the
  listed edits; the served (heads, top-k) grid is the same in the C++ dispatch (swan and v41n) and in SWAN_HEADS x
  SWAN_TOPK; the binding exports swan_decode_dsv4 and requires 256-token pages;
- routing: off by default (no FlashInfer import, nothing counted); stock route for a V4 or no extra cache, v41n route
  for an FP4 extra cache with the native reader; every other case falls back to the split (counted, logged once per
  reason): group / io readers, unserved heads / top-k, > max rows, AutoTuner tuning mode, a non-256 or misaligned pool;
- swan_call: the canary's buffers (output, lse, split-K scratch of the same shapes), the stock tactic (native_tactic, the
  stock hot-cache key), the runner gets the 256-token view; v41n route calls native_decode with the view;
- canary: flag off = the split + stock call as before; flag on and served = no split, no stock call; the block text and
  place are pinned (tests/test_prefill_native_extra.py and tests/test_v41c_reader.py strip it for their base check).

usage: python3 tests/test_swa_native_decode.py     (torch needed for most cases; triton for the real split kernels)
"""
import ast
import importlib.util
import math
import os
import re
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")

ROOT = Path(__file__).resolve().parent.parent
V41C = ROOT / "runtime" / "flashinfer_v41c"
CANARY = ROOT / "runtime" / "flash_mla_sm120.canary.py"
BASE_LAUNCHER = ROOT / "runtime" / "flashinfer_pce" / "base" / "csrc" / "sparse_mla_sm120_decode_dsv4.cu"
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

SWAN_BLOCK_RE = re.compile(r"    # DSV41_SWA_NATIVE_DECODE=1 \(opt-in, default off\):.*?return _swan_out\n\n", re.S)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v = _load("v41c_swan_under_test", V41C / "v41c.py")

ROW, FOOT, BPT = 576, 8, 584


def _page_bytes(pbs):
    return math.ceil(pbs * BPT / ROW) * ROW


# ------------------------------------------------------------------------------------------------ the load rule
def kernel_loads(pool_2d, pbs, stride, idx):
    """What decode-dsv4's IO warp loads for each main-section entry (decode_dsv4_kernel.cuh issue_gather): the 448-byte
    nope + 128-byte rope bulk copies from (idx / pbs) * stride + (idx % pbs) * 576 and the 8-byte footer word from
    (idx / pbs) * stride + pbs * 576 + (idx % pbs) * 8. Masked entries (idx < 0) read slot 0's row and a zero scale word
    (the scale load is skipped), exactly like the stock kernel. Returns [n, 584] uint8."""
    flat = pool_2d.reshape(-1)
    out = torch.zeros(idx.numel(), BPT, dtype=torch.uint8)
    for n, raw in enumerate(idx.reshape(-1).tolist()):
        i = max(raw, 0)
        page, loc = i // pbs, i % pbs
        d0 = page * stride + loc * ROW
        out[n, :ROW] = flat[d0:d0 + ROW]
        if raw >= 0:
            s0 = page * stride + pbs * ROW + loc * FOOT
            out[n, ROW:] = flat[s0:s0 + FOOT]
    return out


def torch_split(pool_2d, src_pbs, touched=None, dst=None):
    """A torch copy of the canary's split (_page_split_kernel): sub-page s of page p -> destination page 4p + s
    (data rows s*64 .. s*64+63, their footer words). touched: only pages holding a referenced token."""
    n = pool_2d.shape[0]
    r = src_pbs // 64
    pb64 = _page_bytes(64)
    if dst is None:
        dst = torch.full((n * r, pb64), 0xEE, dtype=torch.uint8)
    pages = range(n) if touched is None else sorted({t // src_pbs for t in touched.reshape(-1).tolist() if t >= 0})
    for p in pages:
        for s in range(r):
            dst[p * r + s, :64 * ROW] = pool_2d[p, s * 64 * ROW:(s + 1) * 64 * ROW]
            dst[p * r + s, 64 * ROW:64 * BPT] = pool_2d[p, src_pbs * ROW + s * 64 * FOOT:src_pbs * ROW + (s + 1) * 64 * FOOT]
    return dst


def real_split_fn():
    """The canary's own _split_kv_pages_to_64 + Triton kernels (TRITON_INTERPRET=1), cut out with ast."""
    src = CANARY.read_text()
    tree = ast.parse(src)
    import triton as _triton
    import triton.language as _tl
    from typing import Optional

    res = types.SimpleNamespace(buffers={})
    rc = types.ModuleType("sglang.srt.runtime_context")
    rc.get_resources = lambda: res
    for name in ("sglang", "sglang.srt"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules.setdefault(name, m)
    sys.modules["sglang.srt.runtime_context"] = rc
    ns = {"torch": torch, "triton": _triton, "tl": _tl, "math": math, "Optional": Optional}
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith("_") for t in node.targets):
            try:
                exec(compile(ast.Module([node], []), "canary", "exec"), ns)
            except Exception:  # noqa: BLE001
                pass
    for name in ("_page_split_kernel", "_page_mark_kernel", "_split_kv_pages_to_64"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        seg = ast.get_source_segment(src, fn)
        if fn.decorator_list:
            seg = "@triton.jit\n" + seg
        exec(compile(seg, "canary", "exec"), ns)
    return ns["_split_kv_pages_to_64"], res


def _pool(n_pages, stride, gen):
    raw = torch.randint(0, 256, (n_pages, stride), generator=gen, dtype=torch.uint8)
    return raw


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Addressing(unittest.TestCase):
    def _indices(self, n_tokens, rows, topk, gen):
        idx = torch.randint(0, n_tokens, (rows, topk), generator=gen, dtype=torch.int32)
        idx[torch.rand(rows, topk, generator=gen) < 0.1] = -1
        idx[0, :4] = torch.tensor([0, 255, 256, n_tokens - 1], dtype=torch.int32)   # page edges, last token
        return idx

    def test_native_256_equals_split_64_torch(self):
        gen = torch.Generator().manual_seed(11)
        for n_pages, stride in ((5, _page_bytes(256)), (3, _page_bytes(256) + 576)):
            pool = _pool(n_pages, stride, gen)
            idx = self._indices(n_pages * 256, 6, 128, gen)
            split = torch_split(pool, 256, touched=idx)
            a = kernel_loads(split, 64, _page_bytes(64), idx)
            b = kernel_loads(pool, 256, stride, idx)
            self.assertTrue(torch.equal(a, b), f"stride {stride}")

    def test_page_geometry(self):
        self.assertEqual(_page_bytes(256), 149760)
        self.assertEqual(_page_bytes(64), 37440)
        self.assertEqual(4 * _page_bytes(64), _page_bytes(256))
        self.assertEqual(v.SWAN_MIN_PAGE_BYTES, 256 * 584)
        # every bulk-copy source and footer word stays 16 / 8-byte aligned at 256-token pages
        for loc in range(256):
            self.assertEqual((loc * ROW) % 16, 0)
            self.assertEqual((loc * ROW + 448) % 16, 0)
            self.assertEqual((256 * ROW + loc * FOOT) % 8, 0)
        self.assertEqual(_page_bytes(256) % 16, 0)

    @unittest.skipUnless(HAVE_TRITON, "needs triton (TRITON_INTERPRET=1)")
    def test_native_256_equals_real_canary_split(self):
        split_fn, _ = real_split_fn()
        gen = torch.Generator().manual_seed(7)
        n_pages = 6
        stride = _page_bytes(256)
        raw = _pool(n_pages, stride, gen)
        view = torch.as_strided(raw, (n_pages, 256, 1, BPT), (stride, BPT, BPT, 1))
        for rnd in range(3):
            idx = self._indices(n_pages * 256, 24, 128, gen)
            out = split_fn(view, 256, touched_indices=idx)        # the stock kernels, the canary's call
            self.assertEqual(tuple(out.shape), (n_pages * 4, 64, 1, BPT))
            out_2d = torch.as_strided(out, (n_pages * 4, _page_bytes(64)), (_page_bytes(64), 1))
            a = kernel_loads(out_2d, 64, _page_bytes(64), idx)
            b = kernel_loads(raw, 256, stride, idx)
            self.assertTrue(torch.equal(a, b), f"round {rnd}")
            # the native view swan hands the binding: pbs 256, the pool's page stride, same storage
            sv = v.swan_view(view)
            self.assertEqual((tuple(sv.shape), sv.stride(0), sv.data_ptr()), ((n_pages, 256, 1, BPT), stride,
                                                                               view.data_ptr()))


# ------------------------------------------------------------------------------------------------ C++ text
def _dispatch_pairs(text, macro):
    return sorted({(int(h), int(k)) for h, k in re.findall(rf"^\s*{macro}\((\d+), (\d+)\)\s*$", text, re.M)})


class LauncherText(unittest.TestCase):
    def test_swan_launcher_is_the_stock_launcher(self):
        base = BASE_LAUNCHER.read_text()
        got = (V41C / "csrc" / "swan_decode_dsv4.cu").read_text()
        want = base
        for a, b in (
                ("#include <flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh>\n"
                 "#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>\n",
                 "#include <flashinfer_v41c/sparse_mla_sm120/decode_dsv4_kernel.cuh>\n"
                 "#include <flashinfer_v41c/sparse_mla_sm120/model/kv_cache_traits.cuh>\n"),
                ("static bool launch_decode_dsv4_impl(", "static bool launch_swan_decode_dsv4_impl("),
                ("auto kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;",
                 "auto kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, /*EXTRA_FP4=*/false,\n"
                 "                                              /*NATIVE_MODE=*/0>;"),
                ("bool launch_sparse_mla_decode_dsv4(", "bool launch_swan_decode_dsv4("),
                ("  if (mt != ModelType::DSV4 || page_block_size != 64) return false;",
                 "  if (mt != ModelType::DSV4 || page_block_size != 256) return false;  // swan: the SWA pool's own pages"),
                ("    return launch_decode_dsv4_impl<ModelType::DSV4, (H), (K), 64>(                          \\",
                 "    return launch_swan_decode_dsv4_impl<ModelType::DSV4, (H), (K), 256>(                    \\")):
            self.assertEqual(want.count(a), 1, a[:60])
            want = want.replace(a, b)
        # the only other differences: the ds41 comment block after the includes, and the dispatch grid cut to the
        # served shapes
        got_wo_comment = re.sub(r"\n// ds41 DSV41_SWA_NATIVE_DECODE.*?served shapes only.*?\n\n", "\n", got, flags=re.S)
        keep = {f"  DSV4_DISPATCH({h}, {k})\n" for h in v.SWAN_HEADS for k in v.SWAN_TOPK}
        want = "".join(ln for ln in want.splitlines(keepends=True)
                       if not re.match(r"^  DSV4_DISPATCH\(\d+, \d+\)\n$", ln) or ln in keep)
        self.assertEqual(got_wo_comment, want)

    def test_served_grid_is_the_same_everywhere(self):
        served = sorted((h, k) for h in v.SWAN_HEADS for k in v.SWAN_TOPK)
        self.assertEqual(_dispatch_pairs((V41C / "csrc" / "swan_decode_dsv4.cu").read_text(), "DSV4_DISPATCH"), served)
        nat = (V41C / "csrc" / "v41n_decode_dsv4.cu").read_text()
        self.assertEqual(_dispatch_pairs(nat, "SWAN_DISPATCH"), served)
        self.assertIn("launch_v41n_decode_dsv4_impl<ModelType::DSV4, (H), (K), 256, 1>(", nat)
        # the 64-token native grid and the diagnostics are unchanged
        self.assertEqual(len(_dispatch_pairs(nat, "DSV4_DISPATCH")), 25)
        self.assertIn("    if (page_block_size != 64) return false;\n", nat)

    def test_binding(self):
        b = (V41C / "csrc" / "v41c_binding.cu").read_text()
        self.assertIn("TVM_FFI_DLL_EXPORT_TYPED_FUNC(swan_decode_dsv4, flashinfer::sparse_mla_sm120::SwanDecodeDsv4);", b)
        fn = b[b.index("void SwanDecodeDsv4("):b.index("// ds41 v41n (DSV41_V41C_READER=native): V41cDecodeDsv4 with")]
        self.assertIn('parse_paged_kv_layout(ekv, BPT_DSV4, "extra_kv_cache")', fn)
        self.assertNotIn("288", fn)
        self.assertIn("TVM_FFI_ICHECK_EQ(page_block_size, 256)", fn)
        self.assertIn("launch_swan_decode_dsv4(", fn)

    def test_module_sources(self):
        self.assertIn("csrc/swan_decode_dsv4.cu", v.SOURCES)
        import json
        pinned = json.loads((V41C / "VENDORED.json").read_text())["files"]
        self.assertEqual(v.vendored(V41C), pinned)


# ------------------------------------------------------------------------------------------------ routing
class FakeTuner:
    tuning = False

    @classmethod
    def get(cls):
        return types.SimpleNamespace(is_tuning_mode=cls.tuning)


def _install_fake_flashinfer(calls):
    fi = types.ModuleType("flashinfer")
    fi.__path__ = []
    mla = types.ModuleType("flashinfer.mla")
    mla.__path__ = []
    S = types.ModuleType("flashinfer.mla._sparse_mla_sm120")
    S._DECODE_MAX_TOKENS = 64
    S._decode_dsv4_hot_cache = {}
    S._decode_dsv4_map_to_token_bucket = lambda n: ("bucket", n)

    class StockRunner:                      # flashinfer.autotuner.TunableRunner: __call__ -> forward
        def __call__(self, inputs, tactic=-1, **kw):
            return self.forward(inputs, tactic=tactic, **kw)

        def get_cache_key_extras(self, inputs):
            return ("extras", tuple(inputs[0].shape))

    S._get_sparse_mla_decode_dsv4_module = lambda: types.SimpleNamespace(runner_cls=StockRunner)
    S._sparse_mla_sm120_paged_attention = lambda *a, **k: calls.append(("stock", a, k))
    at = types.ModuleType("flashinfer.autotuner")
    at.AutoTuner = FakeTuner
    mla._sparse_mla_sm120 = S
    sys.modules.update({"flashinfer": fi, "flashinfer.mla": mla, "flashinfer.mla._sparse_mla_sm120": S,
                        "flashinfer.autotuner": at})
    return S


class _Fixture(unittest.TestCase):
    """Fake FlashInfer (tuner, hot cache, stock runner, stock paged attention), a fake module and tactic resolver;
    records every call in self.calls."""
    def setUp(self):
        self.calls = []
        self.saved = {k: sys.modules.get(k) for k in ("flashinfer", "flashinfer.mla", "flashinfer.mla._sparse_mla_sm120",
                                                      "flashinfer.autotuner")}
        self.S = _install_fake_flashinfer(self.calls)
        FakeTuner.tuning = False
        self.env = {k: os.environ.get(k) for k in (v.SWAN_ENV, v.ENV, v.READER_ENV)}
        for k in self.env:
            os.environ.pop(k, None)
        self._clear()
        for k in v.stats:
            v.stats[k] = 0
        v.native_tactics.clear()
        calls = self.calls

        class Mod:
            def swan_decode_dsv4(self, *a):
                calls.append(("swan", a))

        self.orig = (v._module, v.native_decode, v.native_tactic)
        v._module = lambda: Mod()
        v._swan_runner.cache_clear()
        v.swan_receipt.cache_clear()
        v.native_tactic = lambda S, inputs, kw, hot_key: calls.append(("tactic", hot_key)) or 3

        def fake_native(q, kv, idx, mid_out, mid_lse, output, out_lse, sm, **k):
            calls.append(("v41n", kv, idx, mid_out.shape, k))
            return output
        v.native_decode = fake_native

    def tearDown(self):
        v._module, v.native_decode, v.native_tactic = self.orig
        v._swan_runner.cache_clear()
        v.swan_receipt.cache_clear()
        for k, m in self.saved.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m
        for k, val in self.env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        self._clear()

    def _clear(self):
        v.swan_enabled.cache_clear()
        v.enabled.cache_clear()
        v.reader.cache_clear()

    def _on(self, **extra):
        os.environ[v.SWAN_ENV] = "1"
        for k, val in extra.items():
            os.environ[k] = val
        self._clear()

    def _args(self, rows=6, heads=16, topk=128, pbs=256, extra=584, stride=None, extra_topk=512):
        pb = stride or _page_bytes(pbs)
        raw = torch.zeros(4, pb, dtype=torch.uint8)
        k = torch.as_strided(raw, (4, pbs, 1, BPT), (pb, BPT, BPT, 1))
        q = torch.zeros(rows, 1, heads, 512, dtype=torch.bfloat16)
        idx = torch.zeros(rows, 1, topk, dtype=torch.int32)
        ex = None if extra is None else torch.zeros(3, 256, 1, extra, dtype=torch.uint8)
        eidx = None if extra is None else torch.zeros(rows, 1, extra_topk, dtype=torch.int32)
        elen = None if extra is None else torch.ones(rows, dtype=torch.int32)
        return q, k, idx, ex, eidx, elen

    def _call(self, **kw):
        q, k, idx, ex, eidx, elen = self._args(**kw)
        return v.swan_call(q, k, idx, None, None, 512, 0.04, ex, eidx, elen)


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Routing(_Fixture):
    def test_off_by_default(self):
        for k in ("flashinfer.mla._sparse_mla_sm120",):
            sys.modules.pop(k)
        self.assertIsNone(self._call())          # returns before importing FlashInfer
        self.assertEqual(self.calls, [])
        self.assertEqual(sum(v.stats.values()), 0)
        for val in ("0", "off", ""):
            os.environ[v.SWAN_ENV] = val
            v.swan_enabled.cache_clear()
            self.assertFalse(v.swan_enabled())
        os.environ[v.SWAN_ENV] = "maybe"
        v.swan_enabled.cache_clear()
        with self.assertRaises(ValueError):
            v.swan_enabled()

    def test_stock_route(self):
        self._on()
        for extra in (584, None):
            self.calls.clear()
            out, lse = self._call(extra=extra)
            self.assertIsNone(lse)
            self.assertEqual(tuple(out.shape), (6, 1, 16, 512))
            kinds = [c[0] for c in self.calls]
            self.assertEqual(kinds, ["tactic", "swan"])
            a = self.calls[1][1]
            q, kv, idx, mid_out, mid_lse, output, out_lse, num_splits, sm, tl, sink, ekv, eidx, elen, tactic = a
            self.assertEqual((tuple(kv.shape), kv.stride(0)), ((4, 256, 1, BPT), _page_bytes(256)))
            want_splits = 2 + (8 if extra else 0)               # ceil(128 / 64) + ceil(512 / 64), the canary's rule
            self.assertEqual(num_splits, want_splits)
            self.assertEqual(tuple(mid_out.shape), (6, 16, want_splits, 512))
            self.assertEqual(tuple(mid_lse.shape), (6, 16, want_splits))
            self.assertEqual((mid_out.dtype, mid_lse.dtype, output.dtype, out_lse.dtype),
                             (torch.bfloat16, torch.float32, torch.bfloat16, torch.float32))
            self.assertEqual(tuple(idx.shape), (6, 128))
            self.assertEqual(tactic, 3)
            hot = self.calls[0][1]
            self.assertEqual(hot[:5], (("bucket", 6), 16, 128, 512 if extra else 0, want_splits))
        self.assertEqual(v.stats["swan_decode"], 2)
        self.assertEqual(v.stats["swan_fallback"], 0)

    def test_v41n_route(self):
        self._on(**{v.ENV: "v41c", v.READER_ENV: "native"})
        out, _ = self._call(extra=288)
        self.assertEqual([c[0] for c in self.calls], ["v41n"])
        kv = self.calls[0][1]
        self.assertEqual((tuple(kv.shape), kv.stride(0)), ((4, 256, 1, BPT), _page_bytes(256)))
        self.assertEqual(tuple(self.calls[0][3]), (6, 16, 10, 512))
        self.assertEqual(v.stats["swan_v41n"], 1)

    def test_fallbacks(self):
        cases = [
            ({}, dict(extra=288)),                                    # FP4 extra without v41c: the canary's v41c path
            ({v.ENV: "v41c", v.READER_ENV: "group"}, dict(extra=288)),
            ({v.ENV: "v41c", v.READER_ENV: "io"}, dict(extra=288)),
            ({}, dict(heads=32)),
            ({}, dict(heads=8)),
            ({}, dict(topk=256)),
            ({}, dict(rows=65)),
            ({}, dict(pbs=64)),
            ({}, dict(pbs=128)),
            ({}, dict(stride=_page_bytes(256) + 8)),                 # not 16-byte aligned rows
        ]
        n = 0
        for env, kw in cases:
            for k in (v.ENV, v.READER_ENV):
                os.environ.pop(k, None)
            self._on(**env)
            self.assertIsNone(self._call(**kw), (env, kw))
            n += 1
            self.assertEqual(v.stats["swan_fallback"], n, (env, kw))
        self.assertEqual([c for c in self.calls if c[0] in ("swan", "v41n")], [])
        FakeTuner.tuning = True
        self._on()
        self.assertIsNone(self._call())
        self.assertEqual(v.stats["swan_fallback"], n + 1)

    def test_swan_view(self):
        pb = _page_bytes(256)
        raw = torch.zeros(5, pb, dtype=torch.uint8)
        k4 = torch.as_strided(raw, (5, 256, 1, BPT), (pb, BPT, BPT, 1))
        k3 = torch.as_strided(raw, (5, 256, BPT), (pb, BPT, 1))
        for k in (k4, k3):
            sv = v.swan_view(k)
            self.assertEqual((tuple(sv.shape), sv.stride(), sv.data_ptr()), ((5, 256, 1, BPT), (pb, BPT, BPT, 1),
                                                                             raw.data_ptr()))
        self.assertIsNone(v.swan_view(k4.view(torch.int8)))
        self.assertIsNone(v.swan_view(torch.zeros(5, 64, 1, BPT, dtype=torch.uint8)))
        small = torch.zeros(6 * 256 * BPT, dtype=torch.uint8)    # a page stride below 256 x 584 bytes
        self.assertIsNone(v.swan_view(torch.as_strided(small, (5, 256, 1, BPT), (256 * BPT - 16, BPT, BPT, 1))))
        odd = torch.zeros(6 * pb + 64, dtype=torch.uint8)            # a base that is not 16-byte aligned
        self.assertIsNone(v.swan_view(torch.as_strided(odd, (5, 256, 1, BPT), (pb, BPT, BPT, 1), 8)))


# ------------------------------------------------------------------------------------------------ canary
class CanaryText(unittest.TestCase):
    def test_block_is_pinned_and_placed(self):
        s = CANARY.read_text()
        blocks = SWAN_BLOCK_RE.findall(s)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], (
            "    # DSV41_SWA_NATIVE_DECODE=1 (opt-in, default off): the decode kernel reads the SWA pool's own 256-token"
            " pages, no\n"
            "    # per-call 64-token split (dsv41_v41c.swan_call); None = not served, the split below runs unchanged.\n"
            "    if src_pbs != _PBS_DST:\n"
            "        from .dsv41_v41c import swan_call as _swan_call\n\n"
            "        _swan_out = _swan_call(q, kv_u8, indices, topk_length, attn_sink, head_dim_v, softmax_scale,"
            " extra_k_cache,\n"
            "                               extra_indices, extra_topk_length)\n"
            "        if _swan_out is not None:\n"
            "            return _swan_out\n\n"))
        dec = s[s.index("def _flash_mla_flashinfer("):s.index("def _validate_flashinfer_sparse_mla_backend(")]
        pre = s[s.index("def _flash_mla_sm120_prefill("):s.index("def _flashinfer_dsv4_decode_capabilities(")]
        self.assertIn(blocks[0], dec)
        self.assertNotIn("swan", pre)
        self.assertLess(dec.index("src_pbs = "), dec.index(blocks[0]))
        self.assertLess(dec.index(blocks[0]), dec.index("kv_64 = ("))


def _canary_fn(recorder):
    src = CANARY.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_flash_mla_flashinfer")
    pkg = types.ModuleType("swanfake")
    pkg.__path__ = []
    sys.modules["swanfake"] = pkg
    sys.modules["swanfake.dsv41_v41c"] = v
    ns = {"__name__": "swanfake.flash_mla_sm120", "__package__": "swanfake", "torch": torch,
          "_split_kv_pages_to_64": lambda kv, pbs, touched_indices=None, buffer_tag="primary":
          recorder.append(("split", pbs)) or kv, "_PBS_DST": 64, "_PBS_SRC": 256}
    exec(compile(ast.get_source_segment(src, fn), str(CANARY), "exec"), ns)
    return ns["_flash_mla_flashinfer"]


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class CanaryRouting(_Fixture):
    def _run(self, **kw):
        f = _canary_fn(self.calls)
        q, k, idx, ex, eidx, elen = self._args(**kw)
        return f(q, k, idx, None, None, 512, 0.04, ex, eidx, elen)

    def test_canary_off_splits_and_calls_stock(self):
        self._run()
        self.assertEqual([c[0] for c in self.calls], ["split", "stock"])

    def test_canary_on_served_skips_split_and_stock(self):
        self._on()
        out, lse = self._run()
        self.assertEqual([c[0] for c in self.calls], ["tactic", "swan"])
        self.assertEqual(tuple(out.shape), (6, 1, 16, 512))

    def test_canary_on_unserved_is_unchanged(self):
        self._on()
        self._run(topk=256)
        self.assertEqual([c[0] for c in self.calls], ["split", "stock"])



# ------------------------------------------------------------------------------------------------ v2.6.4 profile
class Profile(unittest.TestCase):
    """v2.6.4: the production line of .env.tp4.example switches swan on; the module default stays off; the switch is
    volatile in adapter/autotune_keep.py (swan never tunes and runs the stock tactic, so toggling it must reuse the
    tuned cache instead of re-drawing tactics); both canary Dockerfiles build and verify the module."""

    def _production_line(self):
        lines = (ROOT / ".env.tp4.example").read_text().splitlines()
        i = next(n for n, line in enumerate(lines) if line.startswith("# Production settings, together with the line"))
        self.assertTrue(lines[i - 1].startswith('#EXTRA_CONTAINER_ENV="'), lines[i - 1][:60])
        return lines[i - 1]

    def test_production_line_enables_swan_once(self):
        items = self._production_line().split('"')[1].split()
        self.assertEqual([x for x in items if x.startswith("DSV41_SWA_NATIVE_DECODE")], ["DSV41_SWA_NATIVE_DECODE=1"])
        i = items.index("DSV41_SWA_NATIVE_DECODE=1")     # after v2.6.3's two keys (v2.6.4's decode yield follows)
        self.assertEqual(items[i - 2:i + 1], ["DSV41_BLOCK_VERIFY_FOLD=1", "DSV41_ROCE_IDLE_PARK=1",
                                              "DSV41_SWA_NATIVE_DECODE=1"])

    def test_only_the_production_line_enables_swan(self):
        text = (ROOT / ".env.tp4.example").read_text()
        on = [line for line in text.splitlines() if line.startswith("#EXTRA_CONTAINER_ENV=") and "SWA_NATIVE" in line]
        self.assertEqual(on, [self._production_line()])

    def test_volatile_in_autotune_keep(self):
        tree = ast.parse((ROOT / "adapter" / "autotune_keep.py").read_text())
        vol = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_VOLATILE")
        self.assertIn("DSV41_SWA_NATIVE_DECODE", vol)

    def test_default_off_in_module(self):
        src = (V41C / "v41c.py").read_text()
        self.assertIn('SWAN_ENV = "DSV41_SWA_NATIVE_DECODE"', src)
        self.assertIn('if raw in ("", "0", "off", "false", "no"):\n        return False', src)

    def test_dockerfiles_build_and_verify_the_module(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            d = (ROOT / name).read_text()
            self.assertIn("COPY runtime/flashinfer_v41c/v41c.py "
                          "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv41_v41c.py", d, name)
            self.assertIn("python3 -m sglang.kernels.ops.attention.dsv41_v41c build", d, name)
            self.assertIn("python3 -m sglang.kernels.ops.attention.dsv41_v41c verify", d, name)
            self.assertLess(d.index("dsv41_v41c verify"), d.index("COPY tests /opt/dsv41/tests") if
                            "COPY tests /opt/dsv41/tests" in d else len(d), name)


if __name__ == "__main__":
    unittest.main()
