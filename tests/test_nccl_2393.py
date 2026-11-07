#!/usr/bin/env python3
"""v2.6.6: the canary-roce image's NCCL = 2.30.7 rebuilt with NVIDIA/nccl #2393 (scripts/nccl_receipt.py).

Repo checkout (CI, Mac): the nccl-builder stage of Dockerfile.canary-roce is pinned (source commit, patch sha256,
p2p.cc before / after, sm_121, the install path torch loads, the receipt check), scripts/fetch-nccl.sh stages the same
commit, the patch adds exactly the fence, the receipt's fence counter on synthetic disassembly. In the image
(/opt/dsv41, no .github, no Dockerfile): the library torch loads is the receipt's and is 2.30.7 with the fence.
"""
import hashlib
import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nccl_receipt", ROOT / "scripts" / "nccl_receipt.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

DOCKERFILE = ROOT / "Dockerfile.canary-roce"
PATCH = ROOT / "patches" / "nccl-2393-aarch64-cts-fence.patch"
IN_IMAGE = Path("/opt/dsv41/files/nccl-2393.json")

STOCK = """0000000000123400 <ncclIbIsend(void*, void*, unsigned long, int, void*, void*, void**)>:
  123400:	stp	x29, x30, [sp, #-96]!
  123404:	ldr	w28, [x26, #48]
  123408:	ret

0000000000123500 <ncclIbIrecv(void*, int, void**, unsigned long*, int*, void**, void**)>:
  123500:	dmb	ishld
"""
PATCHED = STOCK.replace("  123404:\tldr", "  123402:\tdmb\tishld\n  123404:\tldr")


class Receipt(unittest.TestCase):
    def test_fence_counter(self):
        self.assertEqual(R.count_isend_fences(STOCK), (1, 0))      # the Irecv fence is not ncclIbIsend's
        self.assertEqual(R.count_isend_fences(PATCHED), (1, 1))
        self.assertEqual(R.count_isend_fences(""), (0, 0))

    def test_problems(self):
        good = {"nccl_version_code": 23007, "banner": "NCCL version 2.30.7+cuda13.0", "isend_symbols": 1,
                "isend_dmb_ishld": 1}
        self.assertEqual(R.problems(good), [])
        for k, v in (("nccl_version_code", 23008), ("banner", ""), ("isend_dmb_ishld", 0), ("isend_symbols", 2)):
            self.assertTrue(R.problems(dict(good, **{k: v})), k)

    def test_banner_pattern(self):
        self.assertTrue(R.BANNER_RE.search(b"..NCCL version 2.30.7+cuda13.0\x00"))
        self.assertFalse(R.BANNER_RE.search(b"NCCL version 2.31.0+cuda13.0"))


@unittest.skipUnless(DOCKERFILE.exists() and PATCH.exists(), "not a repo checkout (in an image)")
class Recipe(unittest.TestCase):
    def test_patch_is_the_fence_only(self):
        self.assertEqual(hashlib.sha256(PATCH.read_bytes()).hexdigest(), R.PATCH_SHA256)
        added = [l for l in PATCH.read_text().splitlines() if l.startswith("+") and not l.startswith("+++")]
        self.assertEqual(len(added), 2)
        self.assertIn("std::atomic_thread_fence(std::memory_order_acquire);", added[1])
        # nothing removed (the "-- " line is format-patch's signature separator)
        self.assertFalse([l for l in PATCH.read_text().splitlines()
                          if l.startswith("-") and not l.startswith("---") and l != "-- "])

    def test_builder_stage_is_pinned(self):
        d = DOCKERFILE.read_text()
        stage = d[d.index("FROM ${BASE_IMAGE} AS nccl-builder"):]
        stage = stage[:stage.index("\nFROM ${BASE_IMAGE}\n")]
        for want in ("COPY runtime/nccl-src /work/nccl", R.REF, R.PATCH_SHA256,
                     "7379081678564a0065c60073a6ae3b7a7b259f632805ddc1a6dbac58e47df362",   # p2p.cc at 73cf112
                     "a0cc4e706d649d0582ff431bba252e4eb98a8d15e9f48c15a0e59e54ec73764a",   # p2p.cc + the patch
                     'NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"', "src.build",
                     "nccl_receipt.py write /opt/nccl/libnccl.so.2 /opt/nccl/nccl-2393.json"):
            self.assertIn(want, stage, want)
        # the patch is applied before the build, and the p2p.cc check sits between them
        self.assertLess(stage.index("patch -p1"), stage.index("a0cc4e706d64"))
        self.assertLess(stage.index("a0cc4e706d64"), stage.index("src.build"))

    def test_runtime_stage_installs_and_checks_the_library(self):
        d = DOCKERFILE.read_text()
        self.assertEqual(d.count("FROM ${BASE_IMAGE}"), 2)
        runtime = d[d.index("\nFROM ${BASE_IMAGE}\n"):]
        self.assertIn("COPY --from=nccl-builder /opt/nccl /tmp/nccl-2393", runtime)
        self.assertIn('install -m 0644 -o root -g root /tmp/nccl-2393/libnccl.so.2 "$target"', runtime)
        self.assertIn("target=" + R.PIP_LIB, runtime)
        check = "nccl_receipt.py check " + R.PIP_LIB + " /opt/dsv41/files/nccl-2393.json"
        self.assertIn(check, runtime)
        self.assertIn("tests/test_nccl_2393.py", runtime)
        # checked after the scripts are in the image, and nothing installs another NCCL afterwards
        self.assertLess(runtime.index("COPY scripts /opt/dsv41/scripts"), runtime.index(check))
        self.assertNotRegex(d, r"pip install[^\n]*nccl")

    def test_fetch_script_stages_the_same_commit(self):
        s = (ROOT / "scripts" / "fetch-nccl.sh").read_text()
        self.assertIn('REF="${NCCL_REF:-%s}"' % R.REF, s)
        self.assertIn("https://github.com/NVIDIA/nccl/archive/${REF}.tar.gz", s)
        self.assertRegex(s, r"TREE_SHA256=[0-9a-f]{64}")
        self.assertIn('echo "$REF" > "$DST.partial/REF"', s)
        self.assertTrue(os.access(ROOT / "scripts" / "fetch-nccl.sh", os.X_OK))
        if (ROOT / ".gitignore").exists():   # not in the image
            self.assertIn("runtime/nccl-src/", (ROOT / ".gitignore").read_text().splitlines())

    def test_only_canary_roce_carries_it(self):
        for name in ("Dockerfile", "Dockerfile.canary"):
            self.assertNotIn("nccl-builder", (ROOT / name).read_text(), name)


@unittest.skipUnless(IN_IMAGE.exists() and Path(R.PIP_LIB).exists(), "not in the canary-roce image")
class InImage(unittest.TestCase):
    def test_the_loaded_library_is_the_receipts(self):
        self.assertEqual(R.main(["nccl_receipt.py", "check", R.PIP_LIB, str(IN_IMAGE)]), 0)

    def test_torch_resolves_this_path(self):
        # torch's libtorch_cuda.so finds libnccl through DT_RPATH $ORIGIN/../../nvidia/nccl/lib: that directory
        torch_lib = Path("/opt/sglang/lib/python3.12/site-packages/torch/lib")
        if not torch_lib.exists():
            self.skipTest("no torch at the image path")
        self.assertTrue((torch_lib / ".." / ".." / "nvidia" / "nccl" / "lib" / "libnccl.so.2").resolve()
                        .samefile(R.PIP_LIB))


if __name__ == "__main__":
    unittest.main(verbosity=2)
