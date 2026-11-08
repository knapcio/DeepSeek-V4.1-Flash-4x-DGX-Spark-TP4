#!/usr/bin/env python3
"""v2.7 (FlashInfer 0.7.x) CPU checks, stdlib only.

- boot.py refuses the FlashInfer-0.6.18-only knobs (v41c / v41n readers, swan, K8 native extra pages) once the
  installed FlashInfer is 0.7 or newer, and leaves 0.6.18 profiles alone;
- runtime/flash_mla_sm120.canary.py routes only 528-byte (KVLayout.V41) main caches to FlashInfer's DSv4.1 wrapper,
  before any V4 path, reads SGLANG_SM120_V41_PRECISION and refuses an unknown value (source-level checks: the module
  needs torch / triton / sglang to import);
- both canary recipes install FlashInfer 0.7.2 at its git commit and no longer build the 0.6.18-closure modules;
- the V4.1 route never calibrates sparse MLA in-process (boot.py v41_frozen_calibration): sparse_mla_sm120 joins
  --flashinfer-autotune-skip-ops (merged with the operator's ops), FLASHINFER_AUTOTUNE_DIR points at the shipped
  calibration unless set, a missing calibration refuses the boot, other layouts / FlashInfer 0.6.18 are untouched,
  serve() applies it after the dense-route args and hands the env to the engine; the shipped calibration is FlashInfer
  0.7.2's schema for GB10 with the four V4.1 TP4 request profiles; both recipes ship it.
usage: python3 tests/test_v27_fi07_guard.py
"""
import ast
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapter"))
sys.modules.setdefault("mxfp8_mode", types.SimpleNamespace(mxfp8_backend_disabled=lambda v: False))


def _load_boot():
    src = (ROOT / "boot.py").read_text()
    tree = ast.parse(src)
    names = {"FI0618_ONLY_KNOBS", "V41_CALIBRATION_DIR", "V41_CALIBRATION_FILE", "SKIP_OPS_FLAG"}
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))
            and (getattr(n, "name", "") in ("flashinfer_version", "retired_fi0618_knobs", "v41_frozen_calibration")
                 or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets)))]
    mod = types.ModuleType("boot_v27")
    mod.os = os
    mod.Path = Path
    import hashlib
    mod.hashlib = hashlib
    exec(compile(ast.Module(body=keep, type_ignores=[]), "boot.py", "exec"), mod.__dict__)
    return mod


class BootGuard(unittest.TestCase):
    def setUp(self):
        self.boot = _load_boot()

    def test_refuses_retired_knobs_on_07(self):
        env = {"DSV41_KV_LAYOUT": "v41c", "DSV41_SWA_NATIVE_DECODE": "1", "DSV41_PREFILL_NATIVE_EXTRA": "1"}
        for v in ("0.7.2", "0.7.3", "0.8.0"):
            self.assertEqual(self.boot.retired_fi0618_knobs(env, v), sorted(f"{k}={x}" for k, x in env.items()))

    def test_leaves_0618_and_clean_profiles(self):
        env = {"DSV41_KV_LAYOUT": "v41c", "DSV41_SWA_NATIVE_DECODE": "1"}
        self.assertEqual(self.boot.retired_fi0618_knobs(env, "0.6.18"), [])
        self.assertEqual(self.boot.retired_fi0618_knobs({"SGLANG_DSV4_KV_LAYOUT": "v41"}, "0.7.2"), [])
        self.assertEqual(self.boot.retired_fi0618_knobs({"DSV41_SWA_NATIVE_DECODE": "0"}, "0.7.2"), [])
        self.assertEqual(self.boot.retired_fi0618_knobs(env, "unknown"), [])

    def test_serve_calls_guard_first(self):
        src = (ROOT / "boot.py").read_text()
        body = src[src.index("def serve():"):]
        self.assertLess(body.index("retired_fi0618_knobs()"), body.index("_gpu_lines()"))


class CanaryV41Route(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "runtime" / "flash_mla_sm120.canary.py").read_text()
        self.tree = ast.parse(self.src)

    def test_v41_branch_precedes_every_backend_path(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "flash_mla_with_kvcache_sm120")
        body = ast.get_source_segment(self.src, fn)
        i_v41 = body.index("if k_cache.shape[-1] == _V41_MAIN_BYTES:")
        self.assertLess(i_v41, body.index('if _sm120_default_backend == "flashinfer":'))
        self.assertLess(i_v41, body.index('if _sm120_default_backend == "triton":'))

    def test_constants_and_wrapper_args(self):
        self.assertIn("_V41_MAIN_BYTES = 528", self.src)
        self.assertIn("_V41_FP4_BYTES = 288", self.src)
        for arg in ('kv_scale_format="ue8m0_g32"', "extra_kv_fp4=extra_fp4", "compute_precision=precision"):
            self.assertIn(arg, self.src)

    def test_precision_parser(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "_v41_precision")
        consts = next(n for n in self.tree.body if isinstance(n, ast.Assign)
                      and any(getattr(t, "id", "") == "_V41_PRECISIONS" for t in n.targets))
        ns = {"os": os}
        exec(compile(ast.Module(body=[consts, fn], type_ignores=[]), "canary", "exec"), ns)
        old = os.environ.pop("SGLANG_SM120_V41_PRECISION", None)
        try:
            self.assertEqual(ns["_v41_precision"](), "bf16")
            os.environ["SGLANG_SM120_V41_PRECISION"] = "Default"
            self.assertEqual(ns["_v41_precision"](), "default")
            os.environ["SGLANG_SM120_V41_PRECISION"] = "nvfp4"
            with self.assertRaises(ValueError):
                ns["_v41_precision"]()
        finally:
            os.environ.pop("SGLANG_SM120_V41_PRECISION", None)
            if old is not None:
                os.environ["SGLANG_SM120_V41_PRECISION"] = old


class Recipes(unittest.TestCase):
    @unittest.skipUnless((ROOT / "Dockerfile.canary").exists(), "no recipes here")
    def test_flashinfer_072_and_retired_modules(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            text = (ROOT / name).read_text()
            self.assertIn("ARG FLASHINFER_COMMIT=d84335d8c54e0c8f9f7e22faa15c7e34a58abf94", text, name)
            self.assertIn("--require-hashes", text, name)
            self.assertIn("-r /tmp/flashinfer-0.7.2.requirements", text, name)
            self.assertNotIn("dsv41_pce build", text, name)
            self.assertNotIn("dsv41_v41c build", text, name)
            # the gated Python fronts stay importable: the V4 SM120 bridge imports them on its 256-page paths
            self.assertIn("COPY runtime/flashinfer_pce/dsv41_pce.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv41_pce.py", text, name)
            self.assertIn("COPY runtime/flashinfer_v41c/v41c.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv41_v41c.py", text, name)
        self.assertIn("apply_fi7676.py --check", (ROOT / "Dockerfile.canary-roce").read_text())

    @unittest.skipUnless((ROOT / "files" / "flashinfer-0.7.2.requirements").exists(), "not a repo checkout (in an image)")
    def test_wheel_pins(self):
        want = {
            "flashinfer-python==0.7.2": "770550a0aa92b2f10e6c9a1f048dd43713f0f3bb6a12710186fcf8e4e9c99bd8",
            "flashinfer-cubin==0.7.2": "31704fa5b2562d459d34c67cee9e80e0abb96c82f89943d542276d5a92c88079",
            "flashinfer-jit-cache==0.7.2+cu130": "157e74a786c84df49c59cf1d7568a531a8833ec47a58e9fd52de52737705cc21",
            "flashinfer-jit-cache-sm121a==0.7.2+cu130": "1b827a99581a5166fe5a398610d1ef764055592498f5f92b4bd402e176ddaa8d",
        }
        lines = [l.split() for l in (ROOT / "files" / "flashinfer-0.7.2.requirements").read_text().splitlines()
                 if l and not l.startswith("#")]
        self.assertEqual({l[0]: l[1].split("sha256:")[1] for l in lines}, want)


class FrozenCalibration(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.boot = _load_boot()
        self.dir = tempfile.mkdtemp()
        (Path(self.dir) / "sparse_mla_sm120_cpb.json").write_text("{}")
        self.env = {"DSV41_KV_LAYOUT": "v41", "FLASHINFER_AUTOTUNE_DIR": self.dir}

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir)

    def run_(self, args, env=None, v="0.7.2"):
        return self.boot.v41_frozen_calibration(list(args), self.env if env is None else env, v)

    def test_adds_skip_op_and_keeps_args(self):
        args, add, receipt = self.run_(["--tp", "4", "--fp8-gemm-backend", "flashinfer_cutlass"])
        self.assertEqual(args, ["--tp", "4", "--fp8-gemm-backend", "flashinfer_cutlass",
                                "--flashinfer-autotune-skip-ops", "sparse_mla_sm120"])
        self.assertEqual(add, {})                      # the operator's dir stays
        self.assertIn("sparse-MLA calibration frozen: --flashinfer-autotune-skip-ops sparse_mla_sm120", receipt)
        self.assertIn(f"FLASHINFER_AUTOTUNE_DIR={self.dir}", receipt)
        self.assertIn("sha256=44136fa355b3678a", receipt)  # sha256 of "{}"

    def test_merges_operator_ops(self):
        args, _, _ = self.run_(["--flashinfer-autotune-skip-ops", "mxfp8_gemm", "bf16_gemm", "--tp", "4"])
        self.assertEqual(args, ["--tp", "4", "--flashinfer-autotune-skip-ops", "mxfp8_gemm", "bf16_gemm", "sparse_mla_sm120"])
        args, _, _ = self.run_(["--flashinfer-autotune-skip-ops=mxfp8_gemm,sparse_mla_sm120", "--x"])
        self.assertEqual(args, ["--x", "--flashinfer-autotune-skip-ops", "mxfp8_gemm", "sparse_mla_sm120"])
        args, _, _ = self.run_(["--flashinfer-autotune-skip-ops", "sparse_mla_sm120"])
        self.assertEqual(args.count("sparse_mla_sm120"), 1)

    def test_default_dir_and_refusal(self):
        env = {"DSV41_KV_LAYOUT": "v41"}
        self.boot.V41_CALIBRATION_DIR = self.dir
        _, add, _ = self.run_([], env)
        self.assertEqual(add, {"FLASHINFER_AUTOTUNE_DIR": self.dir})
        self.boot.V41_CALIBRATION_DIR = self.dir + "/nothing"
        with self.assertRaises(RuntimeError):
            self.run_([], env)
        with self.assertRaises(RuntimeError):
            self.run_([], {"DSV41_KV_LAYOUT": "v41", "FLASHINFER_AUTOTUNE_DIR": self.dir + "/nothing"})

    def test_untouched_elsewhere_and_escape(self):
        for env, v in (({"DSV41_KV_LAYOUT": "v41c"}, "0.7.2"), ({}, "0.7.2"), ({"DSV41_KV_LAYOUT": "v41"}, "0.6.18")):
            self.assertEqual(self.run_(["--a"], env, v), (["--a"], {}, None))
        args, add, receipt = self.run_(["--a"], {**self.env, "DSV41_V41_CALIBRATION": "inprocess"})
        self.assertEqual((args, add), (["--a"], {}))
        self.assertIn("IN-PROCESS", receipt)
        with self.assertRaises(ValueError):
            self.run_([], {**self.env, "DSV41_V41_CALIBRATION": "maybe"})

    def test_serve_wiring(self):
        src = (ROOT / "boot.py").read_text()
        body = src[src.index("def serve():"):]
        i = body.index("args = required_sglang_args(")
        j = body.index("args, calibration_env, calibration_receipt = v41_frozen_calibration(args)")
        self.assertLess(i, j)
        self.assertLess(j, body.index("save('launch.json'"))
        self.assertIn("env = dict(os.environ, DSV41_SOURCE=str(MODEL), **calibration_env)", body)
        self.assertIn("print(calibration_receipt, flush=True)", body)

    def test_shipped_calibration(self):
        import json
        p = ROOT / "files" / "fi-autotune" / "sparse_mla_sm120_cpb.json"
        if not p.exists():
            p = Path("/opt/dsv41/files/fi-autotune/sparse_mla_sm120_cpb.json")
        if not p.exists():
            self.skipTest("no shipped calibration here")
        d = json.loads(p.read_text())
        self.assertEqual((d["schema_version"], d["timing_protocol"]), (3, "cold_l2_per_call_graph_v1"))
        self.assertEqual(list(d["devices"]), ["0:NVIDIA GB10"])
        profiles = d["devices"]["0:NVIDIA GB10"]["profiles"]
        reqs = sorted((v["request"]["heads"], v["request"]["topk"], v["request"]["extra_topk"], v["request"]["extra_page_size"],
                       v["request"]["extra_kv_fp4"], v["request"]["compute_precision"], v["request"]["primary_page_size"])
                      for v in profiles.values())
        self.assertEqual(reqs, [(16, 128, 0, 0, False, "bf16", 256), (16, 128, 512, 128, True, "bf16", 256),
                                (16, 128, 512, 256, True, "bf16", 256), (16, 192, 0, 0, False, "bf16", 256)])
        for v in profiles.values():
            self.assertTrue({"1", "4", "8", "16", "24", "32", "48", "64"} <= set(v["buckets"]))

    @unittest.skipUnless((ROOT / "Dockerfile.canary").exists(), "no recipes here")
    def test_recipes_ship_it(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            self.assertIn("COPY files/fi-autotune /opt/dsv41/files/fi-autotune", (ROOT / name).read_text(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
