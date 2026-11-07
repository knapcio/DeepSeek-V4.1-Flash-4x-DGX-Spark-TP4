"""CPU checks of the DSV41_KV_LAYOUT=v41c readers (runtime/flashinfer_v41c/v41c.py, runtime/flash_mla_sm120.canary.py).

- flag: unset / v4 off, v41c on, anything else raises; a 288-byte extra cache without the flag refuses.
- transcode: the module's torch reference turns an FP4 record into exactly the V4 row of tests/v41_fp4_ref.py (the CPU
  reference the v41w window checked on sm_121a against SGLang's writer and V4 store), on random latents, heavy tails,
  e2m1 value ties and e4m3 scale ties, and v4_rows(model rounding of x) == transcode(encode(x)).
- prefill plans: token mode (<= 96 rows: one slot per entry) and page mode (> 96 rows: touched 64-token sub-pages
  compacted) on 128- and 256-token FP4 pages: every valid entry, read through the rewritten index from the V4 page
  view, is the V4 row of the record the original index named; holes stay -1; entry order is unchanged; page mode
  holds only touched sub-pages.
- identity: vendored sources == VENDORED.json, every vendored header but the patched decode header == the approved
  FlashInfer closure (K8), installed-closure drift / a changed vendored header / a changed patched header refuse;
  the prebuilt module record and its .so sha256 refuse on mismatch.
- adapter: the canary minus the two v41c blocks and the K8 block is byte-identical to f744d83's; with a 584-byte extra
  cache nothing new runs; with a 288-byte one decode goes to the v41c decode and prefill to the v41c prefill, and
  without the flag that refuses. Dockerfiles build and verify the module.

usage: python3 tests/test_v41c_reader.py           (torch needed for most cases: ds-b3-cutecheck:1 / the image)
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V41C = ROOT / "runtime" / "flashinfer_v41c"
PCE = ROOT / "runtime" / "flashinfer_pce"
CANARY = ROOT / "runtime" / "flash_mla_sm120.canary.py"
CANARY_F744D83_SHA = "2fe25ba5e2ff91f6028b6a32c315ba8875d8780b9341bae9411af1cc927885bf"
BLOCK_RE = re.compile(r"    # DSV41_KV_LAYOUT=v41c:.*?return \(output\.unsqueeze\(1\), None\)\n(\n)?", re.S)
PCE_BLOCK_RE = re.compile(r"    # DSV41_PREFILL_NATIVE_EXTRA=1:.*?return \(output\.unsqueeze\(1\), None\)\n", re.S)
# DSV41_SWA_NATIVE_DECODE (tests/test_swa_native_decode.py pins its text and place)
SWAN_BLOCK_RE = re.compile(r"    # DSV41_SWA_NATIVE_DECODE=1 \(opt-in, default off\):.*?return _swan_out\n\n", re.S)

try:
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v = _load("v41c_under_test", V41C / "v41c.py")


class Flag(unittest.TestCase):
    def _env(self, val):
        if val is None:
            os.environ.pop(v.ENV, None)
        else:
            os.environ[v.ENV] = val
        v.enabled.cache_clear()

    def tearDown(self):
        self._env(None)

    def test_values(self):
        for val, want in ((None, False), ("", False), ("v4", False), ("v41c", True)):
            self._env(val)
            self.assertIs(v.enabled(), want)
        for val in ("v41", "V41C", "1"):
            self._env(val)
            with self.assertRaises(ValueError):
                v.enabled()

    def test_is_fp4(self):
        class F:
            def __init__(self, last):
                self.shape = (3, 256, 1, last)
        self._env(None)
        self.assertFalse(v.is_fp4(F(584)))
        self.assertFalse(v.is_fp4(None))
        with self.assertRaisesRegex(RuntimeError, "without DSV41_KV_LAYOUT=v41c"):
            v.is_fp4(F(288))
        self._env("v41c")
        self.assertTrue(v.is_fp4(F(288)))
        self.assertFalse(v.is_fp4(F(584)))


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class Transcode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = _load("v41_fp4_ref", ROOT / "tests" / "v41_fp4_ref.py")

    def _sets(self):
        r = self.ref
        return {"random": r.latents(512, 1), "tail": r.latents(512, 2, tail=True), "value_ties": r.value_ties(),
                "scale_ties": r.scale_ties()}

    def test_equals_reference(self):
        for name, x in self._sets().items():
            data, scales = self.ref.encode_rows(x)
            d_ref, s_ref = self.ref.transcode_fp4_to_v4(data, scales)
            d, s = v.v4_rows(v.decode_records(data, scales))
            with self.subTest(set=name):
                self.assertTrue(torch.equal(d, d_ref) and torch.equal(s, s_ref))
                # R1: the transcode is v2.6's V4 row of the model-rounded latent
                d2, s2 = self.ref.v4_rows(self.ref.decode_rows(data, scales).float())
                self.assertTrue(torch.equal(d, d2) and torch.equal(s, s2))

    def _fp4_pool(self, pages, pbs, seed):
        """A [pages, pbs, 1, 288] uint8 view over SGLang V41_FP4 pages filled with encoded latents."""
        r = self.ref
        pb = r.page_bytes(pbs)
        x = r.latents(pages * pbs, seed)
        flat = r.expected_pages(x, torch.arange(pages * pbs), pages, pbs)
        view = torch.as_strided(flat, (pages, pbs, 1, 288), (pb, 288, 288, 1))
        d4, s4 = r.transcode_fp4_to_v4(*r.encode_rows(x))
        return view, d4, s4

    def _read(self, pages_view, idx):
        """What a 64-token V4 page reader reads for index idx: (576 data bytes, 8 scale bytes)."""
        raw = torch.as_strided(pages_view, (pages_view.shape[0], v.SUB_PAGE_BYTES), (v.SUB_PAGE_BYTES, 1))
        p, o = idx // 64, idx % 64
        return raw[p, o * 576:(o + 1) * 576], raw[p, 64 * 576 + o * 8: 64 * 576 + (o + 1) * 8]

    def _check_plan(self, rows, k, pbs, pages, seed):
        view, d4, s4 = self._fp4_pool(pages, pbs, seed)
        g = torch.Generator().manual_seed(seed)
        eidx = torch.randint(0, pages * pbs, (rows, k), generator=g, dtype=torch.int32)
        eidx[torch.rand(rows, k, generator=g) < 0.1] = -1
        v._scratch.clear()
        v4, new = v.transcode_for_prefill(view, eidx, transcode=v.transcode_tokens_torch)
        self.assertEqual(new.shape, eidx.shape)
        self.assertTrue(torch.equal(new < 0, eidx < 0))
        for r_, j in zip(*torch.nonzero(eidx >= 0, as_tuple=True)):
            d, s = self._read(v4, int(new[r_, j]))
            t = int(eidx[r_, j])
            self.assertTrue(torch.equal(d, d4[t]) and torch.equal(s, s4[t]), (int(r_), int(j)))
        return v4, new, eidx

    def test_token_mode(self):
        for pbs in (128, 256):
            v4, new, eidx = self._check_plan(96, 128, pbs, 4, 3)
            self.assertEqual(v4.shape[0], 96 * 128 // 64)
            valid = new[new >= 0]
            self.assertEqual(valid.unique().numel(), valid.numel())

    def test_token_mode_is_capture_safe(self):
        # captured verify graphs: shapes must not depend on how many entries are holes (no mask indexing / sync)
        g = torch.Generator().manual_seed(7)
        for holes in (0.0, 0.3, 1.0):
            eidx = torch.randint(0, 1000, (16, 128), generator=g, dtype=torch.int32)
            eidx[torch.rand(16, 128, generator=g) < holes] = -1
            tokens, slots, new_idx, n_pages = v.plan_token_mode(eidx)
            self.assertEqual((tokens.numel(), slots.numel(), n_pages), (16 * 128, 16 * 128, 16 * 128 // 64))
            self.assertTrue(bool((tokens >= 0).all()))
            self.assertTrue(torch.equal(new_idx < 0, eidx < 0))
        v._scratch.clear()
        small = v._token_scratch(torch.device("cpu"), 4, 128)
        self.assertEqual(small.shape[0], v.TOKEN_MODE_MAX_ROWS * 128 // 64)     # sized for the largest call once
        self.assertIs(v._token_scratch(torch.device("cpu"), 96 * 128 // 64, 128), small)

    def test_page_mode(self):
        for pbs in (128, 256):
            v4, new, eidx = self._check_plan(200, 64, pbs, 6, 4)
            touched = (eidx[eidx >= 0] // 64).unique().numel()
            self.assertEqual(v4.shape[0], touched)
            self.assertLess(touched, 6 * pbs // 64 + 1)

    def test_page_mode_compacts(self):
        view, _, _ = self._fp4_pool(8, 256, 5)
        eidx = torch.full((120, 64), -1, dtype=torch.int32)
        eidx[:, :3] = torch.tensor([5, 700, 1900], dtype=torch.int32)
        v4, new = v.transcode_for_prefill(view, eidx, transcode=v.transcode_tokens_torch)
        self.assertEqual(v4.shape[0], 3)
        self.assertEqual(new[0, :3].tolist(), [5, 64 + 700 % 64, 128 + 1900 % 64])


class Identity(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_repo_is_consistent(self):
        ident = v.check_sources(V41C, PCE / "base" / "csrc", PCE / "base" / "include")
        self.assertEqual(len(ident["module_tag"]), 12)
        self.assertEqual(ident["vendored"], json.loads((V41C / "VENDORED.json").read_text())["files"])

    def _copy(self):
        d = self.tmp / "v41c"
        shutil.copytree(V41C, d)
        return d

    def test_vendored_drift_refuses(self):
        for rel in ("include/flashinfer_v41c/sparse_mla_sm120/common/kv_cache_io.cuh",
                    "include/flashinfer_v41c/sparse_mla_sm120/decode_dsv4_kernel.cuh", "csrc/v41c_binding.cu"):
            with self.subTest(rel=rel):
                d = self._copy()
                (d / rel).write_text((d / rel).read_text() + "// drift\n")
                with self.assertRaisesRegex(RuntimeError, "differ from VENDORED.json"):
                    v.check_sources(d, PCE / "base" / "csrc", PCE / "base" / "include")
                shutil.rmtree(d)

    def test_unapproved_vendored_header_refuses(self):
        """A vendored non-patched header re-pinned in VENDORED.json but different from the approved FlashInfer file."""
        d = self._copy()
        rel = "include/flashinfer_v41c/sparse_mla_sm120/common/kv_cache_io.cuh"
        (d / rel).write_text((d / rel).read_text() + "// drift\n")
        vj = json.loads((d / "VENDORED.json").read_text())
        vj["files"][rel] = hashlib.sha256((d / rel).read_bytes()).hexdigest()
        (d / "VENDORED.json").write_text(json.dumps(vj))
        with self.assertRaisesRegex(RuntimeError, "is not the approved FlashInfer"):
            v.check_sources(d, PCE / "base" / "csrc", PCE / "base" / "include")

    def test_installed_closure_drift_refuses(self):
        fi = self.tmp / "fi"
        shutil.copytree(PCE / "base", fi)
        h = fi / "include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh"
        h.write_text(h.read_text() + "// drift\n")
        with self.assertRaisesRegex(RuntimeError, "not the approved closure"):
            v.check_sources(V41C, fi / "csrc", fi / "include")

    def test_check_module(self):
        d = self._copy()
        ident = v.check_sources(d, PCE / "base" / "csrc", PCE / "base" / "include")
        with self.assertRaisesRegex(RuntimeError, "no prebuilt module record"):
            v.check_module(d, ident)
        (d / "lib").mkdir()
        name = v.module_name(ident["module_tag"])
        so = d / "lib" / f"{name}.so"
        so.write_bytes(b"\x7fELF v41c")
        mj = {"name": name, "module_tag": ident["module_tag"], "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest()}
        (d / "lib" / "module.json").write_text(json.dumps(mj))
        self.assertEqual(v.check_module(d, ident), so)
        so.write_bytes(b"\x7fELF other")
        with self.assertRaisesRegex(RuntimeError, "sha256"):
            v.check_module(d, ident)
        (d / "lib" / "module.json").write_text(json.dumps({**mj, "module_tag": "0" * 12}))
        with self.assertRaisesRegex(RuntimeError, "is not"):
            v.check_module(d, ident)


class AdapterText(unittest.TestCase):
    def test_default_off_text(self):
        s = CANARY.read_text()
        self.assertEqual(len(SWAN_BLOCK_RE.findall(s)), 1)
        s = SWAN_BLOCK_RE.sub("", s)
        self.assertEqual(len(BLOCK_RE.findall(s)), 2)
        base = PCE_BLOCK_RE.sub("", BLOCK_RE.sub("", s), count=1)
        self.assertEqual(hashlib.sha256(base.encode()).hexdigest(), CANARY_F744D83_SHA)

    def test_blocks_gate_on_288_byte_records(self):
        for b in (m.group(0) for m in BLOCK_RE.finditer(CANARY.read_text())):
            self.assertIn(".shape[-1] == 288", b.splitlines()[2])
            self.assertIn("_v41c_is_fp4(", b)

    def test_dockerfiles(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            t = (ROOT / name).read_text()
            self.assertIn("COPY runtime/flashinfer_v41c/v41c.py "
                          "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv41_v41c.py", t)
            b = t.index("FLASHINFER_CUDA_ARCH_LIST=12.1a python3 -m sglang.kernels.ops.attention.dsv41_v41c build")
            self.assertLess(b, t.index("python3 -m sglang.kernels.ops.attention.dsv41_v41c verify"))
            self.assertLess(t.index("dsv41_pce verify"), b)


def _fn(name, recorder):
    src = CANARY.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    pkg = types.ModuleType("v41cfake")
    pkg.__path__ = []
    sys.modules["v41cfake"] = pkg
    sub = _load("v41cfake.dsv41_v41c", V41C / "v41c.py")
    sub.decode = lambda *a, **k: recorder.append(("v41c-decode", k["extra_kv"]))

    def fake_prefill(q2, *a, **k):
        recorder.append(("v41c-prefill", k["extra_fp4"]))
        return torch.full((q2.shape[0], q2.shape[1], k["head_dim_v"]), 3, dtype=torch.bfloat16)
    sub.prefill = fake_prefill
    sys.modules["v41cfake.dsv41_v41c"] = sub
    pce = _load("v41cfake.dsv41_pce", PCE / "dsv41_pce.py")
    sys.modules["v41cfake.dsv41_pce"] = pce
    fi = types.ModuleType("flashinfer.mla._sparse_mla_sm120")
    fi._sparse_mla_sm120_paged_attention = lambda *a, **k: recorder.append(("stock", k["extra_kv_cache"]))
    fi._DECODE_MAX_TOKENS = 64
    for n in ("flashinfer", "flashinfer.mla"):
        m = types.ModuleType(n)
        m.__path__ = []
        sys.modules.setdefault(n, m)
    sys.modules["flashinfer.mla._sparse_mla_sm120"] = fi
    ns = {"__name__": "v41cfake.flash_mla_sm120", "__package__": "v41cfake", "torch": torch,
          "_split_kv_pages_to_64": lambda kv, pbs, touched_indices=None, buffer_tag="primary": kv,
          "_PBS_DST": 64, "_PBS_SRC": 256}
    exec(compile(ast.get_source_segment(src, fn), str(CANARY), "exec"), ns)
    return ns[name], sub


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class AdapterRouting(unittest.TestCase):
    def setUp(self):
        os.environ.pop(v.ENV, None)

    def tearDown(self):
        os.environ.pop(v.ENV, None)

    def _args(self, rows, last):
        q = torch.zeros(rows, 1, 16, 512, dtype=torch.bfloat16)
        k = torch.zeros(4, 64, 1, 584, dtype=torch.uint8)
        idx = torch.zeros(rows, 1, 128, dtype=torch.int32)
        extra = torch.zeros(3, 256, 1, last, dtype=torch.uint8)
        eidx = torch.zeros(rows, 1, 512, dtype=torch.int32)
        return q, k, idx, extra, eidx

    def _run(self, name, rows, last, flag):
        rec = []
        f, sub = _fn(name, rec)
        if flag:
            os.environ[v.ENV] = flag
        else:
            os.environ.pop(v.ENV, None)
        sub.enabled.cache_clear()
        q, k, idx, extra, eidx = self._args(rows, last)
        f(q, k, idx, None, None, 512, 0.04, extra, eidx, None)
        return [r[0] for r in rec]

    def test_decode(self):
        self.assertEqual(self._run("_flash_mla_flashinfer", 4, 584, None), ["stock"])
        self.assertEqual(self._run("_flash_mla_flashinfer", 4, 288, "v41c"), ["v41c-decode"])
        with self.assertRaisesRegex(RuntimeError, "without DSV41_KV_LAYOUT=v41c"):
            self._run("_flash_mla_flashinfer", 4, 288, None)

    def test_prefill(self):
        self.assertEqual(self._run("_flash_mla_sm120_prefill", 96, 584, None)[-1], "stock")
        self.assertEqual(self._run("_flash_mla_sm120_prefill", 96, 288, "v41c"), ["v41c-prefill"])
        self.assertEqual(self._run("_flash_mla_sm120_prefill", 300, 288, "v41c"), ["v41c-prefill"])
        with self.assertRaisesRegex(RuntimeError, "without DSV41_KV_LAYOUT=v41c"):
            self._run("_flash_mla_sm120_prefill", 96, 288, None)


if __name__ == "__main__":
    unittest.main(verbosity=1)
