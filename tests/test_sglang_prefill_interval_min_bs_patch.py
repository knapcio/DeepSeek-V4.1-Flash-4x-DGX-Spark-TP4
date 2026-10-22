#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts" / "verify_sglang_prefill_interval_min_bs.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_sglang_prefill_interval_min_bs", VERIFY_PATH
)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class PrefillIntervalPackagingTests(unittest.TestCase):
    def test_patch_is_credited_pinned_and_default_off(self):
        patch = (ROOT / "patches" / "sglang-44001-prefill-interval-min-bs.patch").read_text()
        self.assertIn("9569a3656155a34dc763826b1608f247d3f66832", patch)
        self.assertIn("Shangming Cai", patch)
        self.assertIn("f80c91a4b97e52e17fbcff088aa6b47b6270f6b9", patch)
        self.assertIn("prefill_decode_interval_min_bs", patch)
        self.assertIn("] = 0", patch)

    def test_fetch_applies_and_verifies_scheduler_backport(self):
        fetch = (ROOT / "scripts" / "fetch-sglang-canary.sh").read_text()
        self.assertIn("sglang-44001-prefill-interval-min-bs.patch", fetch)
        self.assertIn("verify_sglang_prefill_interval_min_bs.py", fetch)
        self.assertGreaterEqual(fetch.count("patch --dry-run --silent --fuzz=0 -p1"), 3)
        self.assertGreaterEqual(fetch.count("patch --silent --fuzz=0 -p1"), 3)
        self.assertIn("44001 9569a3656155a34dc763826b1608f247d3f66832", fetch)

    def test_both_canary_images_verify_the_staged_source(self):
        verifier = (
            "RUN python3 /opt/dsv41/scripts/verify_sglang_prefill_interval_min_bs.py "
            "/sgl-workspace/sglang/python"
        )
        test = "python3 /opt/dsv41/tests/test_sglang_prefill_interval_min_bs_patch.py"
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn(verifier, source)
                self.assertIn(test, source)


class PrefillIntervalVerifierTests(unittest.TestCase):
    SCHEDULE = """
class Schedule:
    prefill_decode_interval_min_bs: A[
        int,
        "help",
    ] = 0
"""
    VALIDATION = """
def validate_prefill_decode_interval(server_args):
    cfg = resolving_view(server_args)
    if cfg.prefill_decode_interval_min_bs < 0:
        raise ValueError("--prefill-decode-interval-min-bs must be non-negative.")
"""
    SCHEDULER = """
class Scheduler:
    def init_chunked_prefill(self):
        self.prefill_decode_interval_min_bs = get_schedule().prefill_decode_interval_min_bs

    def _should_defer_prefill(self, running_batch: ScheduleBatch) -> bool:
        if self._prefill_decode_interval_remaining == 0:
            return False
        if running_batch.batch_size() < self.prefill_decode_interval_min_bs:
            self._prefill_decode_interval_remaining = 0
            return False
        self._prefill_decode_interval_remaining -= 1
        return True

    def get_next_batch_to_run(self, running_batch):
        return self._should_defer_prefill(running_batch)
"""

    def make_tree(self, root: Path, *, stale_call=False, wrong_order=False) -> None:
        scheduler = self.SCHEDULER
        if stale_call:
            scheduler += "\ndef stale(self):\n    return self._should_defer_prefill()\n"
        if wrong_order:
            scheduler = scheduler.replace(
                "        if running_batch.batch_size() < self.prefill_decode_interval_min_bs:\n",
                "        self._prefill_decode_interval_remaining -= 1\n"
                "        if running_batch.batch_size() < self.prefill_decode_interval_min_bs:\n",
            ).replace("        self._prefill_decode_interval_remaining -= 1\n        return True", "        return True")
        files = {
            "sglang/srt/arg_groups/fields/schedule.py": self.SCHEDULE,
            "sglang/srt/arg_groups/validation_hook.py": self.VALIDATION,
            "sglang/srt/managers/scheduler.py": scheduler,
        }
        for relative, source in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            VERIFY.verify(root)

    def test_stale_zero_argument_call_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root, stale_call=True)
            with self.assertRaises(SystemExit) as error:
                VERIFY.verify(root)
            self.assertIn("zero-argument", str(error.exception))

    def test_threshold_after_decrement_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root, wrong_order=True)
            with self.assertRaises(SystemExit) as error:
                VERIFY.verify(root)
            self.assertIn("remaining, batch threshold, decrement", str(error.exception))


class PrefillIntervalBehaviorTests(unittest.TestCase):
    @staticmethod
    def decide(remaining: int, batch_size: int, min_bs: int) -> tuple[bool, int]:
        if remaining == 0:
            return False, remaining
        if batch_size < min_bs:
            return False, 0
        return True, remaining - 1

    def test_default_zero_preserves_existing_interval(self):
        self.assertEqual(self.decide(2, 0, 0), (True, 1))
        self.assertEqual(self.decide(1, 0, 0), (True, 0))
        self.assertEqual(self.decide(0, 0, 0), (False, 0))

    def test_small_batch_clears_pending_interval(self):
        self.assertEqual(self.decide(4, 3, 4), (False, 0))

    def test_threshold_and_larger_batches_defer(self):
        self.assertEqual(self.decide(2, 4, 4), (True, 1))
        self.assertEqual(self.decide(2, 8, 4), (True, 1))


if __name__ == "__main__":
    unittest.main()
