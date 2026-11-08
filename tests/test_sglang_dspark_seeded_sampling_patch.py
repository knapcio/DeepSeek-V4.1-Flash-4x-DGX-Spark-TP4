#!/usr/bin/env python3
"""Packaging and tree checks for the local DSpark seeded-sampling patch.

Usage: test_sglang_dspark_seeded_sampling_patch.py [SGLANG_PYTHON_ROOT]

Without an argument the tree checks use runtime/sglang-canary/python when its
PATCHES file lists the patch, and are skipped otherwise. With an argument (the
fetch script and the images pass the staged tree) the tree checks always run.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "patches/sglang-local-dspark-seeded-sampling.patch"
PATCH_SHA256 = "baa01b0ff10f6cf7c64538cc8bb860b6b592dca17d2ad558da2a895b52a28450"

ACCEPT = "sglang/kernels/ops/speculative/dspark/dspark_accept.py"
SEEDED_DRAWS = "sglang/kernels/ops/speculative/dspark/seeded_draws.py"
DRAFT = "sglang/srt/speculative/dspark_components/dspark_draft.py"
SAMPLER = "sglang/srt/speculative/dspark_components/dspark_draft_sampler.py"
VERIFY = "sglang/srt/speculative/dspark_components/dspark_verify.py"
TOUCHED = (ACCEPT, SEEDED_DRAWS, DRAFT, SAMPLER, VERIFY)

# The unseeded draws on the f80c91a4 tree after the earlier patches.
UNSEEDED_VERIFY_COINS = (
    "    uniform_samples = torch.rand((bs, gamma), dtype=torch.float32, device=device)\n"
    "    uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)\n"
)
UNSEEDED_DRAFT_NOISE = "noise = self.exp_noise[:bs].exponential_()"

# A random draw in the changed files; each needs a `# seed-source:` comment
# within SEED_SOURCE_WINDOW lines above it.
DRAW_PATTERN = re.compile(r"torch\.rand\(|\.exponential_\(|torch\.multinomial\(|fill_seeded_exp_noise\(|_seeded_verify_coins\(")
SEED_SOURCE_WINDOW = 6

_TREE_ARG: Path | None = None


def _tree_root() -> Path | None:
    if _TREE_ARG is not None:
        return _TREE_ARG
    default = ROOT / "runtime/sglang-canary/python"
    patches = ROOT / "runtime/sglang-canary/PATCHES"
    if patches.is_file() and "local-dspark-seeded-sampling" in patches.read_text():
        return default
    return None


def _patch(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    with PATCH_PATH.open("rb") as stdin:
        return subprocess.run(
            ["patch", *args, "--silent", "--fuzz=0", "-p1"],
            cwd=cwd,
            stdin=stdin,
            capture_output=True,
            text=True,
        )


class DsparkSeededSamplingPackagingTests(unittest.TestCase):
    def test_patch_is_pinned_and_applied_with_zero_fuzz_after_earlier_patches(self):
        self.assertEqual(hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest(), PATCH_SHA256)
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn(f'DSPARK_SEEDED_SAMPLING_PATCH_SHA256="{PATCH_SHA256}"', fetch)
        self.assertIn(
            'PATCH_DSPARK_SEEDED_SAMPLING="$ROOT/patches/sglang-local-dspark-seeded-sampling.patch"',
            fetch,
        )
        dry = 'patch --dry-run --silent --fuzz=0 -p1 < "$PATCH_DSPARK_SEEDED_SAMPLING"'
        real = 'patch --silent --fuzz=0 -p1 < "$PATCH_DSPARK_SEEDED_SAMPLING"'
        self.assertEqual(fetch.count(dry), 1)
        self.assertEqual(fetch.count(real), 1)
        previous = fetch.index('patch --silent --fuzz=0 -p1 < "$PATCH_GREEDY_FLAGS_REFRESH"')
        self.assertLess(previous, fetch.index(dry))
        self.assertLess(fetch.index(dry), fetch.index(real))
        self.assertIn('tests/test_sglang_dspark_seeded_sampling_patch.py" "$DST/python"', fetch)
        self.assertIn("local-dspark-seeded-sampling patch-sha256=$DSPARK_SEEDED_SAMPLING_PATCH_SHA256", fetch)

    def test_patch_headers_name_exactly_the_touched_files(self):
        lines = PATCH_PATH.read_text().splitlines()
        new_files = {line[len("+++ b/python/"):].split("\t")[0] for line in lines if line.startswith("+++ b/python/")}
        self.assertEqual(new_files, set(TOUCHED))
        old_files = {line[len("--- a/python/"):].split("\t")[0] for line in lines if line.startswith("--- a/python/")}
        self.assertEqual(old_files, set(TOUCHED) - {SEEDED_DRAWS})
        self.assertIn("--- /dev/null", lines)

    def test_patch_adds_the_seeded_paths(self):
        added = "\n".join(line[1:] for line in PATCH_PATH.read_text().splitlines() if line.startswith("+"))
        for needle in (
            "_seeded_verify_coins(",
            "sampling_seed=sampling_info.sampling_seed[:bs]",
            "seq_lens=prefix_lens,",
            "fill_seeded_exp_noise(",
            "def stage_seeded_noise(",
            "draft_sampler.stage_seeded_noise(",
            "prefix_lens=batch.seq_lens,",
            "get_exec().deterministic.enable_deterministic_inference",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, added)
        self.assertGreaterEqual(added.count("# seed-source:"), 7)

    def test_images_run_this_test_on_the_staged_tree(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn(
                    "tests/test_sglang_dspark_seeded_sampling_patch.py /sgl-workspace/sglang/python", source
                )


class DsparkSeededSamplingTreeTests(unittest.TestCase):
    def setUp(self):
        self.tree = _tree_root()
        if self.tree is None:
            self.skipTest("no staged SGLang tree with the seeded-sampling patch")

    def test_tree_has_seeded_paths_and_seed_source_comments(self):
        accept = (self.tree / ACCEPT).read_text()
        sampler = (self.tree / SAMPLER).read_text()
        draft = (self.tree / DRAFT).read_text()
        verify = (self.tree / VERIFY).read_text()
        self.assertTrue((self.tree / SEEDED_DRAWS).is_file())
        self.assertIn("if sampling_info.sampling_seed is not None and seq_lens is not None:", accept)
        self.assertIn("_seeded_verify_coins(", accept)
        self.assertIn("seq_lens=prefix_lens,", verify)
        self.assertIn("fill_seeded_exp_noise(", sampler)
        self.assertIn("fill_seeded_exp_noise(", draft)
        self.assertIn("draft_sampler.stage_seeded_noise(", draft)
        # adapter/spec_sync_free.py requires exactly one unseeded draw in __call__.
        self.assertEqual(sampler.count("self.exp_noise[:bs].exponential_()"), 1)
        for rel, source in ((ACCEPT, accept), (SAMPLER, sampler), (DRAFT, draft)):
            lines = source.splitlines()
            for number, line in enumerate(lines):
                if line.lstrip().startswith("#") or not DRAW_PATTERN.search(line):
                    continue
                if line.lstrip().startswith(("def ", "from ", "import ")):
                    continue
                above = lines[max(0, number - SEED_SOURCE_WINDOW):number]
                with self.subTest(file=rel, line=number + 1):
                    self.assertTrue(
                        any("# seed-source:" in prior for prior in above),
                        f"{rel}:{number + 1} draws without a seed-source comment: {line.strip()}",
                    )

    def test_patch_applies_to_the_tree_before_it(self):
        if shutil.which("patch") is None:
            self.skipTest("the patch program is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for rel in TOUCHED:
                target = work / "python" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.tree / rel, target)
            reverse = _patch(["-R", "--dry-run"], work)
            self.assertEqual(reverse.returncode, 0, reverse.stdout + reverse.stderr)
            reverse = _patch(["-R"], work)
            self.assertEqual(reverse.returncode, 0, reverse.stdout + reverse.stderr)
            seeded_draws = work / "python" / SEEDED_DRAWS
            if seeded_draws.exists():
                self.assertEqual(seeded_draws.read_text(), "")
                seeded_draws.unlink()
            # The tree before the patch (f80c91a4 + earlier patches) has the unseeded draws.
            self.assertIn(UNSEEDED_VERIFY_COINS, (work / "python" / ACCEPT).read_text())
            self.assertIn(UNSEEDED_DRAFT_NOISE, (work / "python" / SAMPLER).read_text())
            forward = _patch(["--dry-run"], work)
            self.assertEqual(forward.returncode, 0, forward.stdout + forward.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        _TREE_ARG = Path(sys.argv.pop(1)).resolve()
        if not _TREE_ARG.is_dir():
            raise SystemExit(f"not a directory: {_TREE_ARG}")
    unittest.main(verbosity=2)
