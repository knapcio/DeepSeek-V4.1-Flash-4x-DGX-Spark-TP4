#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts" / "verify_sglang_bounded_replay.py"
SPEC = importlib.util.spec_from_file_location("verify_sglang_bounded_replay", VERIFY_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class BackportPackagingTests(unittest.TestCase):
    def test_patch_is_credited_and_pinned(self):
        patch = (ROOT / "patches" / "sglang-41303-bounded-replay-window.patch").read_text()
        self.assertIn("ed8c3139f1f643544b68102cf8c44504e5e55e3d", patch)
        self.assertIn("Vineel Bhat", patch)
        self.assertIn("Claude Fable 5.1", patch)
        self.assertIn("hnyls2002", patch)
        self.assertIn("f80c91a4b97e52e17fbcff088aa6b47b6270f6b9", patch)

    def test_fetch_is_exact_and_fail_closed(self):
        fetch = (ROOT / "scripts" / "fetch-sglang-canary.sh").read_text()
        self.assertIn("f80c91a4b97e52e17fbcff088aa6b47b6270f6b9", fetch)
        self.assertIn("patch --dry-run --silent --fuzz=0 -p1", fetch)
        self.assertIn("patch --silent --fuzz=0 -p1", fetch)
        self.assertIn("verify_sglang_bounded_replay.py", fetch)
        self.assertNotIn("--wildcards", fetch)
        self.assertNotIn(" -F", fetch)

    def test_both_canary_images_verify_the_staged_source(self):
        command = (
            "RUN python3 /opt/dsv41/scripts/verify_sglang_bounded_replay.py "
            "/sgl-workspace/sglang/python"
        )
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                self.assertIn(command, (ROOT / name).read_text())
                self.assertIn("test_swa_bounded_replay_backport.py", (ROOT / name).read_text())


class BackportVerifierTests(unittest.TestCase):
    SOURCES = {
        "sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py": """
def f(s, tail_len, seq_lens):
    a = max(0, s - tail_len)
    b = (seq_lens - tail_len).clamp_min(0)
""",
        "sglang/srt/mem_cache/cache_init_params.py": """
class P:
    decoder_swa_bounded_replay: bool = False
""",
        "sglang/srt/mem_cache/base_prefix_cache.py": """
class C:
    def swa_reprefill_tail_tokens(self, req=None) -> int:
        return 0
""",
        "sglang/srt/mem_cache/swa_radix_cache.py": """
class C:
    def swa_reprefill_tail_tokens(self, req=None) -> int:
        return 0
""",
        "sglang/srt/mem_cache/kv_cache_builder.py": """
def f(params, get_exec):
    params(decoder_swa_bounded_replay=get_exec().features.enable_decoder_swa_bounded_replay)
""",
        "sglang/srt/mem_cache/unified_radix_cache.py": """
class C:
    def swa_reprefill_tail_tokens(self, req=None) -> int:
        return 0
    def f(self, params, req):
        self.decoder_swa_bounded_replay = params.decoder_swa_bounded_replay
        return self.decoder_swa_bounded_replay and not self.session.will_lend_record(req)
""",
        "sglang/srt/session/streaming_session.py": """
from typing import Optional
class Req: pass
class C:
    def will_lend_record(self, req: Optional[Req]) -> bool:
        return False
""",
        "sglang/srt/managers/schedule_policy.py": """
def f(tree_cache, req):
    return tree_cache.swa_reprefill_tail_tokens(req)
""",
        "sglang/srt/managers/schedule_batch.py": """
class C:
    def f(self, tree_cache):
        a = tree_cache.swa_reprefill_tail_tokens(self)
        b = tree_cache.swa_reprefill_tail_tokens(self)
        return a + b
""",
    }

    def make_tree(self, root: Path) -> None:
        for relative, source in self.SOURCES.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            VERIFY.verify(root)

    def test_missing_semantic_hunk_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            path = root / "sglang/srt/mem_cache/unified_radix_cache.py"
            path.write_text("class C:\n    pass\n")
            with self.assertRaises(SystemExit) as error:
                VERIFY.verify(root)
            self.assertIn("will_lend_record", str(error.exception))

    def test_request_unaware_call_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            path = root / "sglang/srt/managers/schedule_policy.py"
            path.write_text(path.read_text() + "\ndef stale(tree_cache):\n    return tree_cache.swa_reprefill_tail_tokens()\n")
            with self.assertRaises(SystemExit) as error:
                VERIFY.verify(root)
            self.assertIn("request-unaware", str(error.exception))


if __name__ == "__main__":
    unittest.main()
