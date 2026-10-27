"""CPU checks of DSV41_PREFILL_NATIVE_EXTRA (runtime/flashinfer_pce, runtime/flash_mla_sm120.canary.py).

- generator: on the vendored FlashInfer base the generated prefill dispatches the dual cache with extra pages of 128
  and 256 only (64 / 2 stay on the stock module), the single-cache dispatchers are gone, the two external symbols are
  renamed, the orchestrator refuses anything but main 64 + extra 128/256, and the manifest hashes what it wrote. Any
  anchor drift (each edit site mutated in turn), a missing input, or a span holding something unexpected stops it.
- loader: DSV41_PREFILL_NATIVE_EXTRA accepts 0 / 1 / unset only; routing needs the flag AND a 128/256 extra cache;
  generated sources that differ from the manifest, or an installed FlashInfer that differs from the generator's
  inputs, refuse.
- adapter, default OFF identical: the canary file minus the inserted block is byte-identical to f744d83's; with the
  flag off the stock op gets the split extra cache exactly as before; with it on, a 128/256 extra cache reaches the
  native module unsplit (the main source is still split) and no extra split buffer is requested; 64-page and absent
  extra caches keep the stock path. (Needs torch: the serving image or ds-b3-cutecheck:1; skipped elsewhere.)

usage: python3 tests/test_prefill_native_extra.py
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PCE = ROOT / "runtime" / "flashinfer_pce"
BASE = PCE / "base"
CANARY = ROOT / "runtime" / "flash_mla_sm120.canary.py"
# f744d83:runtime/flash_mla_sm120.canary.py (v2.6.1), the file the inserted block must be the only change to
CANARY_F744D83_SHA = "2fe25ba5e2ff91f6028b6a32c315ba8875d8780b9341bae9411af1cc927885bf"
CANARY_BLOCK = '''    # DSV41_PREFILL_NATIVE_EXTRA=1: the extra source's native 128/256-token
    # pages go straight to a module instantiated for them (no extra split
    # buffer). Default OFF: the stock path below, unchanged.
    from .dsv41_pce import routes as _pce_routes

    if _pce_routes(extra_kv_u8):
        from .dsv41_pce import paged_attention as _pce_paged_attention

        output = q2.new_empty((num_tokens, num_heads, head_dim_v), dtype=torch.bfloat16)
        out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=dev)
        _pce_paged_attention(
            q2, kv_64, idx, output, out_lse, softmax_scale,
            topk_length=topk_length, attn_sink=attn_sink, extra_kv=extra_kv_u8,
            extra_indices=extra_idx, extra_topk_length=extra_topk_length,
        )
        return (output.unsqueeze(1), None)
'''


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("gen_pce_sources", PCE / "gen_pce_sources.py")
pce = _load("dsv41_pce_under_test", PCE / "dsv41_pce.py")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class Generator(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "gen"
        self.m = gen.generate(BASE, self.out)
        self.pre = (self.out / "pce_prefill.cu").read_text()
        self.orc = (self.out / "pce_paged_attention.cu").read_text()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_dual_dispatch_is_128_256_only(self):
        self.assertEqual(re.findall(r"DISPATCH_FULLTILE_BY_NH_PBSX\((\d+)\);", self.pre), ["128", "256"])
        self.assertEqual(re.findall(r"DISPATCH_BY_NH_PBSX\((\d+)\);", self.pre), ["128", "256"])
        self.assertIn("(extra_page_block_size == 128 || extra_page_block_size == 256)) {", self.pre)
        self.assertNotIn("extra_page_block_size == 64", self.pre)
        self.assertNotIn("extra_page_block_size == 2)", self.pre)

    def test_single_cache_paths_removed(self):
        for gone in ("inline bool dispatch_v32(", "dispatch_v32<", "dispatch_dsv4_single(", "launch_prefill_sg<",
                     "switch (mt)"):
            self.assertNotIn(gone, self.pre)
        self.assertIn("return dispatch_dsv4_dual(", self.pre)
        self.assertIn("return false;  // dsv41 pce: dual-cache DSV4 only", self.pre)

    def test_symbols_renamed(self):
        self.assertEqual(len(re.findall(r"\bsparse_mla_prefill_dispatch\(", self.pre + self.orc)), 0)
        self.assertEqual(self.pre.count("bool pce_sparse_mla_prefill_dispatch("), 1)
        self.assertEqual(self.orc.count("pce_sparse_mla_prefill_dispatch("), 2)
        self.assertNotIn("void SparseMlaSm120PagedAttention(", self.orc)
        self.assertIn("void PceSparseMlaSm120PagedAttention(", self.orc)
        b = (self.out / "pce_binding.cu").read_text()
        self.assertIn("TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv41_pce_paged_attention,", b)
        self.assertNotIn("sparse_mla_sm120_paged_attention,", b)

    def test_orchestrator_guard(self):
        i = self.orc.index("page_block_size == 64 && extra_kv_ptr != nullptr")
        self.assertLess(i, self.orc.index("const bool ok = pce_sparse_mla_prefill_dispatch("))
        self.assertIn("(extra_page_block_size == 128 || extra_page_block_size == 256))", self.orc)

    def test_kernels_and_rest_unchanged(self):
        """Outside the edited sites the generated files are the base text (line multiset difference is the edit)."""
        base_pre = (BASE / "sparse_mla_sm120_prefill.cu").read_text()
        head = base_pre[:base_pre.index("template <ModelType MT>\ninline bool dispatch_v32(")]
        self.assertTrue(self.pre.split("\n", 1)[1].startswith(head))
        base_orc = (BASE / "sparse_mla_sm120.cu").read_text()
        norm = self.orc.split("\n", 1)[1]
        norm = norm.replace("pce_sparse_mla_prefill_dispatch(", "sparse_mla_prefill_dispatch(")
        norm = norm.replace("PceSparseMlaSm120PagedAttention(", "SparseMlaSm120PagedAttention(")
        norm = norm.replace("dsv41 pce: unsupported sparse-MLA", "Unsupported sparse-MLA")
        g0 = norm.index("  TVM_FFI_ICHECK(page_block_size == 64 && extra_kv_ptr")
        g1 = norm.index("  const bool ok = sparse_mla_prefill_dispatch(")
        self.assertEqual(norm[:g0] + norm[g1:], base_orc)

    def test_manifest(self):
        m = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(m, self.m)
        self.assertEqual(m["inputs"]["sparse_mla_sm120_prefill.cu"],
                         _sha((BASE / "sparse_mla_sm120_prefill.cu").read_bytes()))
        for name, h in m["outputs"].items():
            self.assertEqual(_sha((self.out / name).read_bytes()), h)
        self.assertEqual(m["native_extra_page_block_sizes"], [128, 256])
        again = gen.generate(BASE, self.tmp / "gen2")
        self.assertEqual(again["module_tag"], m["module_tag"])

    def _mutated(self, fname, old, new):
        d = self.tmp / "mut"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        for f in ("sparse_mla_sm120_prefill.cu", "sparse_mla_sm120.cu"):
            shutil.copy(BASE / f, d / f)
        t = (d / fname).read_text()
        self.assertIn(old, t)
        (d / fname).write_text(t.replace(old, new))
        return d

    def test_anchor_drift_refuses(self):
        cases = [
            ("sparse_mla_sm120_prefill.cu", "extra_page_block_size == 64 || extra_page_block_size == 2",
             "extra_page_block_size == 64 || extra_page_block_size == 4"),
            ("sparse_mla_sm120_prefill.cu", "      DISPATCH_FULLTILE_BY_NH_PBSX(2);", "      DISPATCH_FULLTILE_BY_NH_PBSX(4);"),
            ("sparse_mla_sm120_prefill.cu", "  } else if (extra_page_block_size == 2) {",
             "  } else if (extra_page_block_size == 4) {"),
            ("sparse_mla_sm120_prefill.cu", "inline bool dispatch_v32(", "inline bool dispatch_v3x("),
            ("sparse_mla_sm120_prefill.cu", "\nbool sparse_mla_prefill_dispatch(", "\nbool sparse_mla_prefill_dispatch2("),
            ("sparse_mla_sm120_prefill.cu", "    case ModelType::GLM_NSA:\n      return dispatch_v32",
             "    case ModelType::GLM_NSA:\n      return dispatch_dsv4_dual(0) || dispatch_v32"),
            ("sparse_mla_sm120.cu", "void SparseMlaSm120PagedAttention(", "void SparseMlaSm120PagedAttn("),
            ("sparse_mla_sm120.cu", "\"Unsupported sparse-MLA prefill configuration: \"", "\"Unsupported: \""),
            ("sparse_mla_sm120.cu", "  const bool ok = sparse_mla_prefill_dispatch(", "  bool ok = sparse_mla_prefill_dispatch("),
        ]
        for fname, old, new in cases:
            with self.subTest(old=old[:50]):
                d = self._mutated(fname, old, new)
                with self.assertRaises(gen.AnchorError):
                    gen.generate(d, self.tmp / "o")

    def test_cli(self):
        r = subprocess.run([sys.executable, str(PCE / "gen_pce_sources.py"), str(BASE), str(self.tmp / "cli")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PCE GEN OK module_tag=" + self.m["module_tag"], r.stdout)
        r = subprocess.run([sys.executable, str(PCE / "gen_pce_sources.py"), str(self.tmp / "nope"), str(self.tmp / "x")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("PCE GEN FAIL", r.stderr)
        d = self._mutated("sparse_mla_sm120_prefill.cu", "DISPATCH_BY_NH_PBSX(2);", "DISPATCH_BY_NH_PBSX(4);")
        r = subprocess.run([sys.executable, str(PCE / "gen_pce_sources.py"), str(d), str(self.tmp / "y")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("anchor", r.stderr)


@contextlib.contextmanager
def _env(value):
    old = os.environ.get(pce.ENV)
    if value is None:
        os.environ.pop(pce.ENV, None)
    else:
        os.environ[pce.ENV] = value
    pce.enabled.cache_clear()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(pce.ENV, None)
        else:
            os.environ[pce.ENV] = old
        pce.enabled.cache_clear()


class _Fake:
    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)


class Loader(unittest.TestCase):
    def test_enabled_values(self):
        for v, want in ((None, False), ("", False), ("0", False), (" 0 ", False), ("1", True)):
            with _env(v):
                self.assertIs(pce.enabled(), want, v)
        for v in ("true", "2", "on", "yes"):
            with _env(v):
                with self.assertRaises(ValueError):
                    pce.enabled()

    def test_routes(self):
        with _env("0"):
            self.assertFalse(pce.routes(_Fake(10, 256, 1, 584)))
        with _env("1"):
            self.assertTrue(pce.routes(_Fake(10, 256, 1, 584)))
            self.assertTrue(pce.routes(_Fake(10, 128, 1, 584)))
            self.assertTrue(pce.routes(_Fake(10, 128, 584)))
            self.assertFalse(pce.routes(_Fake(10, 64, 1, 584)))
            self.assertFalse(pce.routes(_Fake(10, 2, 1, 584)))
            self.assertFalse(pce.routes(_Fake(10, 149760)))
            self.assertFalse(pce.routes(None))
        with _env("bogus"):
            with self.assertRaises(ValueError):
                pce.routes(_Fake(10, 256, 1, 584))
            self.assertFalse(pce.routes(None))  # no extra cache: nothing to decide

    def test_check_sources(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            gen.generate(BASE, tmp / "g")
            m = pce.check_sources(tmp / "g", BASE)
            self.assertEqual(len(m["module_tag"]), 12)
            p = tmp / "g" / "pce_prefill.cu"
            p.write_text(p.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "does not match its manifest"):
                pce.check_sources(tmp / "g", BASE)
            gen.generate(BASE, tmp / "g")
            other = tmp / "fi"
            other.mkdir()
            for f in ("sparse_mla_sm120_prefill.cu", "sparse_mla_sm120.cu"):
                shutil.copy(BASE / f, other / f)
            (other / "sparse_mla_sm120.cu").write_text((other / "sparse_mla_sm120.cu").read_text() + "// x\n")
            with self.assertRaisesRegex(RuntimeError, "is not the source the module was generated"):
                pce.check_sources(tmp / "g", other)
        finally:
            shutil.rmtree(tmp)


class AdapterText(unittest.TestCase):
    def test_default_off_is_f744d83_plus_block(self):
        s = CANARY.read_text()
        self.assertEqual(s.count(CANARY_BLOCK), 1)
        self.assertEqual(_sha(s.replace(CANARY_BLOCK, "").encode()), CANARY_F744D83_SHA)

    def test_block_sits_before_the_extra_split(self):
        s = CANARY.read_text()
        fn = s[s.index("def _flash_mla_sm120_prefill("):s.index("@lru_cache(maxsize=1)\ndef _flashinfer_dsv4_decode")]
        self.assertLess(fn.index("kv_64 = ("), fn.index(CANARY_BLOCK))
        self.assertLess(fn.index(CANARY_BLOCK), fn.index('buffer_tag="extra"'))

    def test_dockerfiles_install_and_generate(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            t = (ROOT / name).read_text()
            i = t.index("COPY runtime/flash_mla_sm120.canary.py ")
            j = t.index("COPY runtime/flashinfer_pce/dsv41_pce.py "
                        "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv41_pce.py")
            k = t.index("gen_pce_sources.py \"$FI_CSRC\" /opt/dsv41/runtime/flashinfer_pce/gen")
            self.assertTrue(i < j < k, name)
            self.assertIn("p.check_sources(Path(p.DEFAULT_SRC)", t)
        self.assertEqual(pce.DEFAULT_SRC, "/opt/dsv41/runtime/flashinfer_pce/gen")


try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


def _prefill_fn(recorder):
    """_flash_mla_sm120_prefill from the canary file, executed against recording stand-ins."""
    import torch

    src = CANARY.read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_flash_mla_sm120_prefill")
    code = ast.get_source_segment(src, fn)

    pkg = types.ModuleType("pcefake")
    pkg.__path__ = []
    sys.modules["pcefake"] = pkg
    sub = _load("pcefake.dsv41_pce", PCE / "dsv41_pce.py")
    sys.modules["pcefake.dsv41_pce"] = sub

    def fake_pce(q, kv_64, indices, output, out_lse, sm_scale, **kw):
        recorder.append(("pce", kv_64, kw["extra_kv"], kw["extra_indices"]))
        output.fill_(1)
    sub.paged_attention = fake_pce

    fi = types.ModuleType("flashinfer.mla._sparse_mla_sm120")

    def stock(q2, kv, idx, output, out_lse, scale, **kw):
        recorder.append(("stock", kv, kw["extra_kv_cache"], kw["extra_indices"]))
        output.fill_(2)
    fi._sparse_mla_sm120_paged_attention = stock
    for n in ("flashinfer", "flashinfer.mla"):
        m = types.ModuleType(n)
        m.__path__ = []
        sys.modules.setdefault(n, m)
    sys.modules["flashinfer.mla._sparse_mla_sm120"] = fi

    def split(kv, src_pbs, touched_indices=None, buffer_tag="primary"):
        recorder.append(("split", buffer_tag, kv, src_pbs))
        return ("SPLIT", buffer_tag, id(kv))

    ns = {"__name__": "pcefake.flash_mla_sm120", "__package__": "pcefake", "torch": torch,
          "_split_kv_pages_to_64": split, "_PBS_DST": 64, "_PBS_SRC": 256}
    exec(compile(code, str(CANARY), "exec"), ns)
    return ns["_flash_mla_sm120_prefill"]


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class AdapterRouting(unittest.TestCase):
    def _run(self, flag, extra_pbs, main_pbs=256):
        import torch
        rec = []
        f = _prefill_fn(rec)
        q = torch.zeros(96, 16, 512, dtype=torch.bfloat16)
        k = torch.zeros(4, main_pbs, 1, 584, dtype=torch.uint8)
        idx = torch.zeros(96, 128, dtype=torch.int32)
        extra = None if extra_pbs is None else torch.zeros(3, extra_pbs, 1, 584, dtype=torch.uint8)
        eidx = None if extra is None else torch.zeros(96, 512, dtype=torch.int32)
        with _env(flag):
            out, _ = f(q, k, idx, None, None, 512, 0.04, extra, eidx, None)
        return rec, out, k, extra, eidx

    def test_off_is_stock_with_split(self):
        for flag in (None, "0"):
            for pbs in (128, 256):
                rec, out, k, extra, eidx = self._run(flag, pbs)
                self.assertEqual([r[0] for r in rec], ["split", "split", "stock"])
                self.assertEqual(rec[0][1], "primary")
                self.assertEqual((rec[1][1], rec[1][3]), ("extra", pbs))
                self.assertIs(rec[1][2], extra)
                self.assertEqual(rec[2][2], ("SPLIT", "extra", id(extra)))
                self.assertTrue(bool((out == 2).all()))

    def test_on_routes_native_pages(self):
        for pbs in (128, 256):
            rec, out, k, extra, eidx = self._run("1", pbs)
            self.assertEqual([r[0] for r in rec], ["split", "pce"])
            self.assertEqual(rec[0][1], "primary")  # SWA still split
            self.assertEqual(rec[1][1], ("SPLIT", "primary", id(k)))
            self.assertIs(rec[1][2], extra)  # extra unsplit, the pool tensor itself
            self.assertIs(rec[1][3], eidx)
            self.assertTrue(bool((out == 1).all()))
            self.assertEqual(tuple(out.shape), (96, 1, 16, 512))

    def test_on_keeps_stock_for_64_and_none(self):
        rec, out, *_ = self._run("1", 64)
        self.assertEqual([r[0] for r in rec], ["split", "stock"])
        rec, out, *_ = self._run("1", None)
        self.assertEqual([r[0] for r in rec], ["split", "stock"])
        self.assertIsNone(rec[1][2])

    def test_main_64_not_split(self):
        rec, *_ = self._run("1", 256, main_pbs=64)
        self.assertEqual([r[0] for r in rec], ["pce"])

    def test_bad_flag_refuses_before_the_attention_launch(self):
        import torch
        rec = []
        f = _prefill_fn(rec)
        extra = torch.zeros(3, 256, 1, 584, dtype=torch.uint8)
        with _env("yes"), self.assertRaises(ValueError):
            f(torch.zeros(96, 16, 512, dtype=torch.bfloat16), torch.zeros(4, 256, 1, 584, dtype=torch.uint8),
              torch.zeros(96, 128, dtype=torch.int32), None, None, 512, 0.04, extra,
              torch.zeros(96, 512, dtype=torch.int32), None)
        self.assertEqual([r[0] for r in rec], ["split"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
