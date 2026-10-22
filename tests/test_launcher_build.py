#!/usr/bin/env python3
"""Offline tests for the build step (`cmd_build`) of the launcher.

No Docker, SSH or network: the only thing driven here is the exclude list of the
rsync that stages the tree on the workers, because that one runs with `--delete`
against a copy of the repository and a wrong pattern removes files from it.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE_RSYNC = shutil.which("rsync") is not None


def build_source():
    """The body of cmd_build() from start.sh."""
    source = (ROOT / "start.sh").read_text()
    build = source[source.index("cmd_build()"):]
    return build[:build.index("\n}\n")]


def rsync_excludes():
    return re.findall(r"--exclude '([^']+)'", build_source())


def launcher_source():
    """start.sh from the variable defaults through the helper functions, no dispatch."""
    source = (ROOT / "start.sh").read_text()
    start = source.index("HEAD_IP=")
    end = source.index('case "$CMD" in')
    return "set -euo pipefail\n" + source[start:end]


def shell(code, directory, settings=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT)}
    for name in ("MODEL_DIR", "COMMON_MODEL", "SSH_IDENTITY", "NCCL_HOST_DIR",
                 "STATE_DIR", "LOG_DIR", "WORKER_DIR", "WORKER_ENGRAM_DIR", "ENGRAM_DIR", "HF_EXPORT_ROOT"):
        env[name] = str(directory / name)
    env.update(settings or {})
    result = subprocess.run(["bash", "-c", launcher_source() + "\n" + code],
                            env=env, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def check(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def rsync(source, destination):
    args = ["rsync", "-aH", "--delete"]
    for pattern in rsync_excludes():
        args += ["--exclude", pattern]
    args += [f"{source}/", f"{destination}/"]
    result = subprocess.run(args, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


class BuildTargetTests(unittest.TestCase):
    """`IMAGE` is whatever the profile names, so `build` must compile that recipe."""

    def test_build_compiles_the_configured_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "SSH_IDENTITY.pub").write_text("ssh-ed25519 AAAA test\n")
            log = Path(tmp, "docker.log")
            shell(r"""
              BUILD_DOCKERFILE=Dockerfile.canary
              BUILD_ARGS="--build-arg PIP_INDEX=https://mirror.invalid/simple"
              # Bash 3.2 treats expansion of an empty array as an unbound variable
              # under `set -u`. Keep one fake worker and stub the two remote effects;
              # this test is about the selected local Dockerfile/build arguments.
              WORKER_HOSTS=(worker)
              remote_on() { return 0; }
              rsync() { return 0; }
              docker() { printf '%s\n' "$*" >> "$LOG"
                         case "$1 $2" in "image inspect") echo arm64;; esac; }
              cmd_build
            """, Path(tmp), {"LOG": str(log)})
            calls = [line for line in log.read_text().splitlines() if line.startswith("build ")]
            self.assertEqual(len(calls), 1, log.read_text())
            self.assertIn(f"-f {ROOT}/Dockerfile.canary", calls[0])
            self.assertIn("--build-arg PIP_INDEX=https://mirror.invalid/simple", calls[0])

    def test_worker_builds_the_same_dockerfile(self):
        build = build_source()
        self.assertIn('docker build -f "$ROOT/$dockerfile" -t "$IMAGE"', build)
        workers = build[build.index("for h in \"${WORKER_HOSTS[@]}\""):]
        self.assertIn('docker build -f $(printf \'%q\' "$dockerfile")', workers)

    def test_a_dockerfile_outside_the_repository_is_rejected(self):
        """The workers only see what the rsync staged, so an absolute path cannot work."""
        build = build_source()
        self.assertIn("BUILD_DOCKERFILE must be inside the repository", build)


@unittest.skipUnless(HAVE_RSYNC, "rsync is not installed")
class RsyncExcludesTests(unittest.TestCase):
    """cmd_build must not exclude the staged SGLang tree from the workers."""

    def test_the_staged_tree_reaches_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            staged = "runtime/sglang-canary/python/sglang/srt/models"
            (src / staged).mkdir(parents=True)
            (src / staged / "deepseek_v4.py").write_text("branch")
            (dst / staged).mkdir(parents=True)
            (dst / staged / "deepseek_v4.py").write_text("stale")
            (src / "models").mkdir()
            (src / "models" / "shard.safetensors").write_text("weights")

            rsync(src, dst)

            self.assertEqual((dst / staged / "deepseek_v4.py").read_text(), "branch",
                             "the staged SGLang tree never reached the worker")
            self.assertFalse((dst / "models").exists(),
                             "the checkpoint directory at the repository root was copied")

    def test_every_exclude_is_anchored(self):
        """A bare name matches any path component; all five are root-level artifacts."""
        for pattern in rsync_excludes():
            if pattern in (".env", ".env.tp4"):
                continue  # exact filenames, meant to be protected everywhere
            self.assertTrue(pattern.startswith("/"), f"unanchored exclude: {pattern}")


class CanaryDockerfileTests(unittest.TestCase):
    """Dockerfile.canary installs two wheels from PyPI; an offline site needs a mirror."""

    def test_the_pip_index_is_overridable(self):
        text = (ROOT / "Dockerfile.canary").read_text()
        self.assertIn("ARG PIP_INDEX=https://pypi.org/simple", text)
        self.assertIn('-i "$PIP_INDEX"', text)


class DockerfilePrefillBypassTests(unittest.TestCase):
    def test_every_serving_image_checks_hook_against_its_final_scheduler(self):
        source_root = ('DSV41_PREFILL_BYPASS_SOURCE_ROOT='
                       '/sgl-workspace/sglang/python/sglang/srt/managers')
        for name in ('Dockerfile', 'Dockerfile.canary', 'Dockerfile.canary-roce'):
            with self.subTest(name=name):
                text = (ROOT / name).read_text()
                command = next(line for line in text.splitlines()
                               if 'test_prefill_bypass.py' in line)
                self.assertIn(source_root, command)
                self.assertIn('PYTHONPATH=/opt/dsv41/adapter', command)
                expected = 'unsupported' if name == 'Dockerfile' else 'supported'
                self.assertIn(f'DSV41_PREFILL_BYPASS_EXPECT={expected}', command)
