#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts" / "verify_sglang_layer20_tail.py"
SPEC = importlib.util.spec_from_file_location("verify_sglang_layer20_tail", VERIFY_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class Layer20BackportPackagingTests(unittest.TestCase):
    def test_patch_is_credited_and_pinned(self):
        patch = (ROOT / "patches" / "sglang-42901-layer20-tail.patch").read_text()
        self.assertIn("05ce0e980f00aa80864190d6f1cd5d9b685802f1", patch)
        self.assertIn("Yi (Vincent) Zhong", patch)
        self.assertIn("f80c91a4b97e52e17fbcff088aa6b47b6270f6b9", patch)

    def test_fetch_applies_and_verifies_both_backports(self):
        fetch = (ROOT / "scripts" / "fetch-sglang-canary.sh").read_text()
        self.assertIn("sglang-42901-layer20-tail.patch", fetch)
        self.assertIn("verify_sglang_layer20_tail.py", fetch)
        self.assertEqual(fetch.count("patch --dry-run --silent --fuzz=0 -p1"), 2)
        self.assertEqual(fetch.count("patch --silent --fuzz=0 -p1"), 2)
        self.assertIn("42901 05ce0e980f00aa80864190d6f1cd5d9b685802f1", fetch)

    def test_both_canary_images_verify_the_staged_source(self):
        verifier = (
            "RUN python3 /opt/dsv41/scripts/verify_sglang_layer20_tail.py "
            "/sgl-workspace/sglang/python"
        )
        test = "python3 /opt/dsv41/tests/test_sglang_layer20_tail_patch.py"
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn(verifier, source)
                self.assertIn(test, source)


class Layer20BackportVerifierTests(unittest.TestCase):
    BACKEND = """
class B:
    def __init__(self):
        self.tail_source_kv_layer_id: Optional[int] = None
    def enter_late_layer_tail(self):
        if self.tail_source_kv_layer_id is None:
            pass
    def exit_late_layer_tail(self):
        self.tail_source_kv_layer_id = None
    def write_tail_source_kv(self, layer):
        self.forward_low_ratio_sources(run_indexer=False)
        self.tail_source_kv_layer_id = layer.layer_id
    def forward_low_ratio_sources(self, layer, run_indexer=True):
        if layer.layer_id == self.tail_source_kv_layer_id:
            run_compressor = False
"""
    MODEL = """
class M:
    def __init__(self, config):
        self.tail_writes_source_kv = not _is_hip
        self.late_layer_start = last_source if self.tail_writes_source_kv else last_source + 1
        assert last_source in config.index_source_layer_ids
    def write_tail_source_kv(self):
        pass
    def _forward_layers_hc_pre_from_prev(self, attn_backend, i):
        self.layers[i].write_tail_source_kv(
        )
        attn_backend.enter_late_layer_tail(
        )
"""

    def make_tree(self, root: Path, *, reverse=False) -> None:
        files = {
            "sglang/srt/layers/attention/deepseek_v4_backend.py": self.BACKEND,
            "sglang/srt/models/deepseek_v4.py": self.MODEL,
        }
        if reverse:
            files["sglang/srt/models/deepseek_v4.py"] = self.MODEL.replace(
                "self.layers[i].write_tail_source_kv(\n        )\n        attn_backend.enter_late_layer_tail(\n        )",
                "attn_backend.enter_late_layer_tail(\n        )\n        self.layers[i].write_tail_source_kv(\n        )",
            )
        for relative, source in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            VERIFY.verify(root)

    def test_write_after_tail_switch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root, reverse=True)
            with self.assertRaises(SystemExit) as error:
                VERIFY.verify(root)
            self.assertIn("before entering the tail", str(error.exception))

    def test_marker_armed_before_write_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            path = root / "sglang/srt/layers/attention/deepseek_v4_backend.py"
            source = path.read_text().replace(
                "self.forward_low_ratio_sources(run_indexer=False)\n        self.tail_source_kv_layer_id = layer.layer_id",
                "self.tail_source_kv_layer_id = layer.layer_id\n        self.forward_low_ratio_sources(run_indexer=False)",
            )
            path.write_text(source)
            with self.assertRaises(SystemExit) as error:
                VERIFY.verify(root)
            self.assertIn("marker must arm only after", str(error.exception))


if __name__ == "__main__":
    unittest.main()
