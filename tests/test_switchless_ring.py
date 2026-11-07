#!/usr/bin/env python3
"""Offline tests for the opt-in switchless ring (files/nccl.sh + the launcher hooks).

No Docker, SSH, sysfs or model paths are touched: `files/nccl.sh` is sourced
directly, and `start.sh` is sliced from its variable defaults to the command
dispatch so the real `docker_common_args` / `worker_env_lines` can be driven.

Default-off is the property that matters most here: with NCCL_SWITCHLESS_RING_ONLY
unset every generated argument list must be byte-identical to before this change.
"""
import os
import subprocess

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NCCL_SH = ROOT / "files" / "nccl.sh"


def launcher_source():
    """start.sh from the variable defaults through the helper functions, no dispatch."""
    source = (ROOT / "start.sh").read_text()
    start = source.index("HEAD_IP=")
    end = source.index('case "$CMD" in')
    return "set -euo pipefail\n" + source[start:end]


BASE_SETTINGS = {
    "NCCL_HOST_DIR": "/nonexistent",
    "NNODES": "4",
    "TP_SIZE": "4",
    "EP_SIZE": "2",
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "IB_HCA": "rocep1s0f0,rocep1s0f1",
}


def shell(code, directory, settings=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT)}
    for name in ("MODEL_DIR", "COMMON_MODEL", "SSH_IDENTITY", "NCCL_HOST_DIR",
                 "STATE_DIR", "LOG_DIR", "WORKER_DIR", "WORKER_ENGRAM_DIR", "ENGRAM_DIR"):
        env[name] = str(directory / name)
    Path(env["MODEL_DIR"]).mkdir(exist_ok=True)
    Path(env["MODEL_DIR"], "config.json").write_text("{}")
    # A four-node launcher needs three workers; the repo default is the 3-Spark fleet.
    env.update({
        "HEAD_IP": "10.0.0.1",
        "WORKER_IPS": "10.0.0.2 10.0.0.3 10.0.0.4",
        "WORKER_HOSTS": "spark2 spark3 spark4",
    })
    env.update(settings or {})
    return subprocess.run(
        ["bash", "-c", launcher_source() + "\n" + code],
        env=env, cwd=ROOT, text=True, capture_output=True, timeout=30,
    )


def nccl_shell(code, settings=None):
    """Run a snippet with files/nccl.sh sourced (no launcher)."""
    env = dict(os.environ)
    env.update(BASE_SETTINGS)
    env.update(settings or {})
    return subprocess.run(
        ["bash", "-c", 'set -uo pipefail\nsource "$0"\n' + code, str(NCCL_SH)],
        env=env, text=True, capture_output=True, timeout=30,
    )


def check(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def nfs_shell(code, directory, settings=None):
    """Run a snippet with files/nfs-share.sh sourced (sourcing it is side-effect free)."""
    env = {
        "PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT),
        "MODEL_DIR": str(directory), "EXPECTED_SHARDS": "2",
    }
    env.update(settings or {})
    return subprocess.run(
        ["bash", "-c", 'set -uo pipefail\nsource "$ROOT/files/nfs-share.sh"\n' + code],
        env=env, text=True, capture_output=True, timeout=30,
    )


def lines_after(out, tag):
    """Tokens printed one-per-line by `printf '<tag>%s\\n' "${array[@]}"`."""
    return [line[len(tag):] for line in out.splitlines() if line.startswith(tag)]


class SwitchOffByDefaultTests(unittest.TestCase):
    """With the switch unset nothing may change."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def test_enabled_predicate_defaults_to_false(self):
        self.assertIn("enabled=0", check(nccl_shell(
            'if switchless_ring_enabled; then echo enabled=1; else echo enabled=0; fi')))
        self.assertIn("enabled=0", check(nccl_shell(
            'NCCL_SWITCHLESS_RING_ONLY=0; if switchless_ring_enabled; then echo enabled=1; else echo enabled=0; fi')))

    def test_ring_args_and_env_string_are_empty(self):
        out = check(nccl_shell('''
          a=(); switchless_ring_args a; echo "count=${#a[@]}"
          echo "envstr=[$(switchless_ring_env_string)]"
        '''))
        self.assertIn("count=0", out)
        self.assertIn("envstr=[]", out)

    def test_validation_passes_without_the_switch(self):
        # NNODES=3 / a switched fabric is a perfectly good non-ring deployment.
        check(nccl_shell("nccl_validate_config", {"NNODES": "3", "TP_SIZE": "3"}))

    def test_worker_env_has_no_stray_whitespace(self):
        """Regression guard: the ring hook must not leave `  \\` behind when off.

        The first version appended the ring env to the heredoc as `$_extra_env
        $_ring_env`, which is byte-identical except for a trailing space whenever
        the switch is off. Harmless to bash, but "off changes nothing" has to mean
        the bytes too, so the separator now comes from the launcher.
        """
        out = check(shell('worker_env_lines "$HEAD_IP" 3 1', Path(self.tmp.name), BASE_SETTINGS))
        for line in out.splitlines():
            self.assertFalse(line.endswith(" "), repr(line))
            self.assertNotIn("  \\", line, repr(line))
        self.assertIn("-e DSV41_EXTRA_ENV=1 \\", out)

    def test_head_and_worker_arguments_are_unchanged(self):
        out = check(shell('''
          a=(); docker_common_args a "$HEAD_IP" 3
          printf '%s\\n' "${a[@]}"
        ''', Path(self.tmp.name), BASE_SETTINGS))
        self.assertNotIn("NCCL_SWITCHLESS_RING_ONLY", out)
        self.assertNotIn("NCCL_ALGO", out)
        self.assertNotIn("NCCL_PIP_SO", out)


class SwitchOnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.on = dict(BASE_SETTINGS)
        self.on.update({
            "NCCL_SWITCHLESS_RING_ONLY": "1",
            "NCCL_OVERLAY_PIP": "1",
            "NCCL_HOST_DIR": "/nonexistent",
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_ring_args_carry_the_ring_environment(self):
        out = check(nccl_shell('''
          a=(); switchless_ring_args a; printf '%s\\n' "${a[@]}"
        ''', self.on))
        for expected in ("NCCL_SWITCHLESS_RING_ONLY=1", "NCCL_ALGO=Ring",
                         "NCCL_SKIP_TREE_CONNECT=1", "NCCL_IB_SUBNET_PREFIX_LEN=24",
                         "NCCL_MIN_NCHANNELS=4", "NCCL_P2P_LEVEL=SYS"):
            self.assertIn(expected, out)

    def test_env_string_and_array_agree(self):
        """The worker heredoc and the head array must inject the same variables."""
        out = check(nccl_shell('''
          a=(); switchless_ring_args a
          printf 'ARGV %s\\n' "${a[@]}"
          printf 'STRV %s\\n' $(switchless_ring_env_string)
        ''', self.on))
        argv = sorted(lines_after(out, "ARGV "))
        strv = sorted(lines_after(out, "STRV "))
        self.assertEqual(argv, strv)
        self.assertIn("NCCL_SWITCHLESS_RING_ONLY=1", strv)

    def test_env_string_survives_quoted_values(self):
        out = check(nccl_shell('''
          eval "set -- $(switchless_ring_env_string)"
          printf '[%s]\\n' "$@"
        ''', self.on))
        self.assertIn("[NCCL_ALGO=Ring]", out)

    def test_worker_env_mirrors_the_ring_environment(self):
        out = check(shell('worker_env_lines "$HEAD_IP" 3 1', Path(self.tmp.name), self.on))
        for expected in ("NCCL_SWITCHLESS_RING_ONLY=1", "NCCL_ALGO=Ring",
                         "NCCL_SKIP_TREE_CONNECT=1", "NCCL_P2P_LEVEL=SYS"):
            self.assertIn(expected, out)

    def test_head_uses_the_overlay_mount_not_ld_library_path(self):
        nccl_dir = Path(self.tmp.name) / "nccl"
        nccl_dir.mkdir()
        (nccl_dir / "libnccl.so.2.30.7").write_text("SWITCHLESS_RING_ONLY\n")
        out = check(shell('''
          a=(); docker_common_args a "$HEAD_IP" 3
          printf '%s\\n' "${a[@]}"
        ''', Path(self.tmp.name), dict(self.on, NCCL_HOST_DIR=str(nccl_dir))))
        self.assertIn(":/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro", out)
        self.assertNotIn("LD_LIBRARY_PATH", out)

    def test_mount_refuses_a_missing_library_when_the_switch_is_on(self):
        result = nccl_shell("a=(); nccl_mount_args a", dict(self.on, NCCL_HOST_DIR="/nonexistent"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no readable library", result.stderr)



class OverlayWithoutTheRingTests(unittest.TestCase):
    """NCCL_OVERLAY_PIP=1 on a switched fabric (ring off): head and workers mount the same library.

    Until v2.6.5 docker_common_args honoured the overlay only with NCCL_SWITCHLESS_RING_ONLY=1 and put
    NCCL_HOST_DIR on LD_LIBRARY_PATH otherwise (inert for torch, which resolves libnccl through its RPATH),
    while every worker's nccl_mount_args mounted the library over the pip NCCL: the ranks ran two different
    NCCL builds. With the overlay off the head's arguments must stay byte-identical to v2.6.5's.
    """
    PIP = "/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.dir = Path(self.tmp.name)
        self.nccl = self.dir / "nccl"
        self.nccl.mkdir()
        (self.nccl / "libnccl.so.2.30.7").write_text("stock or patched, the launcher does not care\n")
        self.off = dict(BASE_SETTINGS, NCCL_HOST_DIR=str(self.nccl))

    def tearDown(self):
        self.tmp.cleanup()

    def head(self, settings):
        return shell("a=(); docker_common_args a \"$HEAD_IP\" 3; printf 'ARG %s\\n' \"${a[@]}\"",
                     self.dir, settings)

    def nccl_tokens(self, out):
        args = lines_after(out, "ARG ")
        keep = []
        for i, a in enumerate(args):
            if a == "-v" and "nccl" in args[i + 1]:
                keep += [a, args[i + 1]]
            if a == "-e" and args[i + 1].startswith(("LD_LIBRARY_PATH=", "NCCL_SWITCHLESS_RING_ONLY=")):
                keep += [a, args[i + 1]]
        return keep

    def worker_tokens(self, settings):
        # what the generated worker script runs: the launcher's nccl_worker_settings payload (variables + functions)
        # in a fresh shell, then nccl_mount_args, as in cmd_serve's worker heredoc
        out = check(shell("""payload="$(nccl_worker_settings)"
          env -i HOME="$HOME" PATH="$PATH" bash -c "set -e
          $payload
          nccl_args=(); nccl_mount_args nccl_args; printf 'ARG %s\\n' \\"\\${nccl_args[@]}\\"" """, self.dir, settings))
        return lines_after(out, "ARG ")

    def test_head_overlays_with_the_switch_off(self):
        on = dict(self.off, NCCL_OVERLAY_PIP="1")
        out = check(self.head(on))
        lib = str(self.nccl / "libnccl.so.2.30.7")
        self.assertEqual(self.nccl_tokens(out), ["-v", f"{lib}:{self.PIP}:ro"])
        self.assertNotIn("LD_LIBRARY_PATH", out)
        self.assertNotIn("NCCL_SWITCHLESS_RING_ONLY", out)

    def test_head_and_workers_mount_the_same_library(self):
        for overlay in ("0", "1"):
            settings = dict(self.off, NCCL_OVERLAY_PIP=overlay)
            head = self.nccl_tokens(check(self.head(settings)))
            self.assertEqual(head, self.worker_tokens(settings), overlay)

    def test_overlay_off_is_unchanged(self):
        # v2.6.5's head branch: NCCL_HOST_DIR on LD_LIBRARY_PATH when it holds a library, else nothing
        out = check(self.head(self.off))
        self.assertEqual(self.nccl_tokens(out),
                         ["-v", f"{self.nccl}:/nccl:ro", "-e", "LD_LIBRARY_PATH=/nccl"])
        empty = self.dir / "empty"
        empty.mkdir()
        out = check(self.head(dict(self.off, NCCL_HOST_DIR=str(empty))))
        self.assertEqual(self.nccl_tokens(out), [])

    def test_head_refuses_an_overlay_without_a_library(self):
        empty = self.dir / "empty"
        empty.mkdir()
        result = self.head(dict(self.off, NCCL_HOST_DIR=str(empty), NCCL_OVERLAY_PIP="1"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no readable library", result.stderr)
        self.assertIn("NCCL setup failed", result.stderr)

    def test_validation_checks_the_overlay_with_the_switch_off(self):
        check(nccl_shell("nccl_validate_config", {"NCCL_OVERLAY_PIP": "1", "NCCL_PIP_SO": self.PIP}))
        for bad in ({"NCCL_OVERLAY_PIP": "yes"},
                    {"NCCL_OVERLAY_PIP": "1", "NCCL_PIP_SO": "relative/libnccl.so.2"},
                    {"NCCL_OVERLAY_PIP": "1", "NCCL_PIP_SO": self.PIP, "NCCL_HOST_DIR": "/a:b"}):
            result = nccl_shell("nccl_validate_config", bad)
            self.assertNotEqual(result.returncode, 0, bad)

    def test_serve_validates_the_switches_with_the_ring_off(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        gate = serve.index("nccl_validate_config")
        self.assertLess(gate, serve.index("if switchless_ring_enabled"))
        self.assertLess(gate, serve.index("docker rm"))

    def test_default_follows_the_switch(self):
        out = check(shell('echo "overlay=$NCCL_OVERLAY_PIP"', self.dir, self.off))
        self.assertIn("overlay=0", out)
        out = check(shell('echo "overlay=$NCCL_OVERLAY_PIP"', self.dir, dict(self.off, NCCL_SWITCHLESS_RING_ONLY="1")))
        self.assertIn("overlay=1", out)


class ValidationTests(unittest.TestCase):
    def run_validation(self, settings):
        return nccl_shell("nccl_validate_config", dict(BASE_SETTINGS, **settings))

    def test_accepts_ep_from_one_to_tp(self):
        for ep in ("1", "2", "4"):
            with self.subTest(ep=ep):
                check(self.run_validation({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                           "NCCL_OVERLAY_PIP": "1", "EP_SIZE": ep}))

    def test_rejects_ep_above_tp_and_zero(self):
        for ep in ("5", "0", "x"):
            with self.subTest(ep=ep):
                result = self.run_validation({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                              "NCCL_OVERLAY_PIP": "1", "EP_SIZE": ep})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("1 <= EP_SIZE <= TP_SIZE", result.stderr)

    def test_rejects_topology_that_cannot_carry_the_ring(self):
        for settings in ({"NNODES": "3"}, {"TP_SIZE": "3"}):
            with self.subTest(**settings):
                result = self.run_validation(dict({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                                   "NCCL_OVERLAY_PIP": "1"}, **settings))
                self.assertNotEqual(result.returncode, 0)

    def test_rejects_transport_that_is_not_the_ring(self):
        for settings in ({"NCCL_ALGO": "Tree"}, {"NCCL_NET": "Socket"},
                         {"NCCL_IB_DISABLE": "1"}, {"NCCL_OVERLAY_PIP": "0"}):
            with self.subTest(**settings):
                result = self.run_validation(dict({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                                   "NCCL_OVERLAY_PIP": "1"}, **settings))
                self.assertNotEqual(result.returncode, 0)

    def test_rejects_channel_range_that_is_empty(self):
        result = self.run_validation({"NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_OVERLAY_PIP": "1",
                                      "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "4"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NCCL_MIN_NCHANNELS", result.stderr)

    def test_rejects_non_boolean_switches(self):
        for flag in ("NCCL_SWITCHLESS_RING_ONLY", "NCCL_OVERLAY_PIP"):
            with self.subTest(flag=flag):
                result = self.run_validation({flag: "yes"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be 0 or 1", result.stderr)


class GidIndexTests(unittest.TestCase):
    """ring_gid_index against a synthetic sysfs tree."""

    def build(self, directory, gids, types=None):
        """gids: {port_hca: {index: gid}}; ports are always ACTIVE."""
        for hca, table in gids.items():
            base = directory / hca / "ports" / "1"
            (base / "gid_attrs" / "types").mkdir(parents=True)
            (base / "gids").mkdir(parents=True)
            (base / "state").write_text("4: ACTIVE\n")
            for index, gid in table.items():
                (base / "gids" / index).write_text(gid + "\n")
                kind = (types or {}).get((hca, index), "RoCE v2")
                (base / "gid_attrs" / "types" / index).write_text(kind + "\n")
        return str(directory)

    IPV4 = "0000:0000:0000:0000:0000:ffff:c0a8:32a8"
    IPV6 = "fe80:0000:0000:0000:0000:0000:0000:0001"
    OTHER_IPV4 = "0000:0000:0000:0000:0000:ffff:c0a8:32a9"
    # the dotted address IPV4 maps, as the launcher would detect it
    ADDRESS = ".".join(str(b) for b in bytes.fromhex(IPV4[-9:].replace(":", "")))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_the_common_ipv4_index(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"0": self.IPV6, "3": self.IPV4},
                            "rocep1s0f1": {"0": self.IPV6, "3": self.IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs}'))
        self.assertEqual(out.strip(), "3")

    def test_rejects_ports_without_a_shared_index(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": self.IPV4},
                            "rocep1s0f1": {"5": self.IPV4}})
        result = nccl_shell(f'ring_gid_index {sysfs}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no common IPv4 RoCE v2 GID", result.stderr)

    def test_rejects_a_zero_gid(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": "0000:0000:0000:0000:0000:ffff:0000:0000"}})
        self.assertNotEqual(nccl_shell(f'ring_gid_index {sysfs}').returncode, 0)

    def test_honours_an_explicit_override(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": self.IPV4, "5": self.IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs}',
                               {"NCCL_IB_GID_INDEX": "5", "IB_HCA": "rocep1s0f0"}))
        self.assertEqual(out.strip(), "5")

    def test_auto_accepts_a_different_index_per_port(self):
        """NCCL_IB_GID_AUTO=1 and no detected index: nothing is pinned, NCCL picks one per device."""
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"0": self.IPV6, "3": self.IPV4},
                            "rocep1s0f1": {"3": self.IPV6, "5": self.OTHER_IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs}', {"NCCL_IB_GID_AUTO": "1"}))
        self.assertEqual(out.strip(), "auto")

    def test_auto_validates_a_detected_index_on_every_port(self):
        """resolved_gid pins a detected index even with AUTO=1, so the preflight checks that index."""
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"0": self.IPV6, "3": self.IPV4},
                            "rocep1s0f1": {"3": self.IPV6, "5": self.OTHER_IPV4}})
        result = nccl_shell(f'ring_gid_index {sysfs} {self.ADDRESS}', {"NCCL_IB_GID_AUTO": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no common IPv4 RoCE v2 GID", result.stderr)

    def test_auto_detection_skips_port_qualified_entries(self):
        """The launcher detects on port 1 of plain names only, so a:2,b:2 pins nothing."""
        directory = Path(self.tmp.name)
        for hca, table in {"a": {"3": self.IPV4}, "b": {"3": self.IPV6, "5": self.OTHER_IPV4}}.items():
            base = directory / hca / "ports" / "2"
            (base / "gid_attrs" / "types").mkdir(parents=True)
            (base / "gids").mkdir()
            (base / "state").write_text("4: ACTIVE\n")
            for index, gid in table.items():
                (base / "gids" / index).write_text(gid + "\n")
                (base / "gid_attrs" / "types" / index).write_text("RoCE v2\n")
        out = check(nccl_shell(f'ring_gid_index {directory} {self.ADDRESS}',
                               {"NCCL_IB_GID_AUTO": "1", "IB_HCA": "a:2,b:2"}))
        self.assertEqual(out.strip(), "auto")

    def test_auto_detection_reads_ib_hca_as_the_launcher_does(self):
        """The launcher does not strip a leading '=', so '=a,b' detects nothing on a."""
        sysfs = self.build(Path(self.tmp.name),
                           {"a": {"0": self.IPV6, "3": self.IPV4},
                            "b": {"3": self.IPV6, "5": self.OTHER_IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs} {self.ADDRESS}',
                               {"NCCL_IB_GID_AUTO": "1", "IB_HCA": "=a,b"}))
        self.assertEqual(out.strip(), "auto")
        result = nccl_shell(f'ring_gid_index {sysfs} {self.ADDRESS}',
                            {"NCCL_IB_GID_AUTO": "1", "IB_HCA": "a,b"})
        self.assertNotEqual(result.returncode, 0)

    def test_auto_reports_a_detected_index(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"0": self.IPV6, "3": self.IPV4},
                            "rocep1s0f1": {"0": self.IPV6, "3": self.OTHER_IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs} {self.ADDRESS}', {"NCCL_IB_GID_AUTO": "1"}))
        self.assertEqual(out.strip(), "3")

    def test_auto_still_needs_an_ipv4_gid_on_every_port(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": self.IPV4},
                            "rocep1s0f1": {"3": self.IPV6,
                                           "5": "0000:0000:0000:0000:0000:ffff:0000:0000"}})
        result = nccl_shell(f'ring_gid_index {sysfs}', {"NCCL_IB_GID_AUTO": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rocep1s0f1", result.stderr)

    def test_an_explicit_override_wins_over_auto(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": self.IPV4, "5": self.IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs}',
                               {"NCCL_IB_GID_AUTO": "1", "NCCL_IB_GID_INDEX": "5",
                                "IB_HCA": "rocep1s0f0"}))
        self.assertEqual(out.strip(), "5")

    def test_worker_payload_carries_auto(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": self.IPV4},
                            "rocep1s0f1": {"5": self.IPV4}})
        payload = check(nccl_shell("nccl_worker_settings", {"NCCL_IB_GID_AUTO": "1"}))
        result = subprocess.run(["bash", "-c", payload + f"\nring_gid_index {sysfs}"],
                                env={"PATH": os.environ["PATH"]},
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "auto")

    def test_doctor_accepts_a_worker_auto_answer(self):
        source = (ROOT / "start.sh").read_text()
        doctor = source[source.index("cmd_doctor()"):]
        doctor = doctor[:doctor.index("\n}\n")]
        self.assertIn('[[ "$wgid" =~ ^([0-9]+|auto)$ ]]', doctor)
        # the preflight learns each host's address, so it validates what serve will pin
        self.assertIn('nccl_preflight "$HEAD_IP"', doctor)
        self.assertIn('nccl_preflight $(printf \'%q\' "${WORKER_IPS[$wi]:-}")', doctor)

    def test_rejects_an_inactive_port(self):
        sysfs = self.build(Path(self.tmp.name), {"rocep1s0f0": {"3": self.IPV4}})
        Path(sysfs, "rocep1s0f0", "ports", "1", "state").write_text("1: DOWN\n")
        result = nccl_shell(f'ring_gid_index {sysfs}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not ACTIVE", result.stderr)

    def test_rejects_a_malformed_hca_name(self):
        result = nccl_shell(f'ring_gid_index {self.tmp.name}',
                            {"IB_HCA": "rocep1s0f0;rm -rf"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact HCA names", result.stderr)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def library(self, content):
        directory = Path(self.tmp.name)
        directory.mkdir(exist_ok=True)
        (directory / "libnccl.so.2.30.7").write_text(content)
        return str(directory)

    def test_rejects_a_stock_library(self):
        result = nccl_shell("nccl_preflight",
                            {"NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_OVERLAY_PIP": "0",
                             "NCCL_HOST_DIR": self.library("stock nccl")})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lacks SWITCHLESS_RING_ONLY", result.stderr)

    def test_rejects_a_missing_library(self):
        result = nccl_shell("nccl_preflight",
                            {"NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_OVERLAY_PIP": "0",
                             "NCCL_HOST_DIR": "/nonexistent"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no readable library", result.stderr)

    def test_is_free_when_the_switch_is_off(self):
        """A switched fabric may legitimately have no patched library at all."""
        check(nccl_shell("nccl_preflight",
                         {"NCCL_SWITCHLESS_RING_ONLY": "0", "NCCL_HOST_DIR": "/nonexistent"}))

    def test_worker_settings_maps_the_home_relative_path(self):
        out = check(nccl_shell('nccl_worker_settings | head -1',
                              {"NCCL_HOST_DIR": os.environ["HOME"] + "/nccl-2.30.7"}))
        self.assertIn('NCCL_HOST_DIR="$HOME"/nccl-2.30.7', out)

    def test_worker_settings_honours_an_absolute_override(self):
        out = check(nccl_shell('nccl_worker_settings | head -1',
                               {"NCCL_WORKER_DIR": "/opt/nccl"}))
        self.assertIn("NCCL_HOST_DIR=/opt/nccl", out)

    def test_worker_settings_exports_the_ring_functions(self):
        out = check(nccl_shell("nccl_worker_settings"))
        for function in ("switchless_ring_enabled", "nccl_library", "nccl_mount_args",
                         "ring_gid_index", "nccl_preflight"):
            self.assertIn(f"{function} ()", out)

    def test_worker_payload_does_not_swallow_a_missing_function(self):
        """Every call in the remote payload is guarded with `|| return 0`, so a
        function left out of it is a silent no-op rather than an error."""
        payload = check(nccl_shell("nccl_worker_settings",
                                   {"NCCL_SWITCHLESS_RING_ONLY": "1"}))
        result = subprocess.run(["bash", "-c", payload + "\ntype -t switchless_ring_enabled"],
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "function")


class SourceHygieneTests(unittest.TestCase):
    def test_nccl_sh_is_side_effect_free(self):
        """Sourcing it must not probe sysfs, docker or the filesystem."""
        result = subprocess.run(
            ["bash", "-c", 'set -euo pipefail\nsource "$0"\necho sourced-ok', str(NCCL_SH)],
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sourced-ok", result.stdout)

    def test_switch_variables_have_defaults_in_start_sh(self):
        source = (ROOT / "start.sh").read_text()
        for name in ("NCCL_SWITCHLESS_RING_ONLY", "NCCL_OVERLAY_PIP", "NCCL_PIP_SO"):
            self.assertIn(f'{name}="${{{name}:-', source, name)

    def test_every_ring_hook_is_behind_the_switch(self):
        """The ring must not run on a switched fabric by accident."""
        source = (ROOT / "start.sh").read_text()
        head = source[source.index("docker_common_args()"):]
        head = head[:head.index("\n}\n")]
        self.assertIn('if [[ "$NCCL_SWITCHLESS_RING_ONLY" == "1" ]]', head)
        self.assertIn("switchless_ring_args _a", head)
        self.assertIn("nccl_mount_args _a", head)

    def test_serve_gates_the_ring_before_replacing_containers(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        gate = serve.index("nccl_validate_config")
        self.assertIn("switchless_ring_enabled", serve)
        # The gate must precede the first container replacement in serve.
        for later in ("docker rm", "docker run -d", "start_worker"):
            if later in serve:
                self.assertLess(gate, serve.index(later), later)

    def test_worker_containers_mount_the_same_nccl_as_the_head(self):
        """The ring skips the tree connect on every rank, so a worker that loads a
        different NCCL than the head never completes ncclCommInitRank."""
        source = (ROOT / "start.sh").read_text()
        head = source[source.index("docker_common_args()"):]
        head = head[:head.index("\n}\n")]
        self.assertIn("nccl_mount_args _a", head)
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        workers = serve[serve.index("Starting workers"):serve.index("Starting head")]
        self.assertIn("$(nccl_worker_settings)", workers)
        self.assertIn("nccl_mount_args nccl_args", workers)
        self.assertIn("${nccl_args[@]}", workers)
        self.assertNotIn("LD_LIBRARY_PATH", workers)


class LocalWeightsTests(unittest.TestCase):
    """local_model_has_weights: the NFS_SHARE=0 guard before four containers go down."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.model = Path(self.tmp.name, "model")
        self.model.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def complete(self, shards=2):
        (self.model / "config.json").write_text('{"model_type": "deepseek_v4"}')
        for i in range(shards):
            (self.model / f"model-{i:05d}-of-{shards:05d}.safetensors").write_text("x" * 32)

    def sent_command(self):
        """The remote script local_model_volume_is_local really sends. Its own
        remote_on call redirects to /dev/null, so the stub writes to a file."""
        sent = Path(self.tmp.name) / "sent"
        check(nfs_shell(r'''
          remote_on() { local d; if [[ "${2:-}" == "--timeout" ]]; then d="$4"; else d="$2"; fi; printf '%s\n' "$d" > "$SENT"; }
          NFS_VOLUME=dsv41-weights
          local_model_volume_is_local fakehost
        ''', ROOT, {"SENT": str(sent)}))
        self.assertTrue(sent.exists(), "remote_on was never called")
        return sent.read_text()

    def test_accepts_a_complete_checkpoint(self):
        self.complete()
        check(nfs_shell("local_model_has_weights", self.model))

    def test_rejects_a_missing_config(self):
        self.complete()
        (self.model / "config.json").unlink()
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_rejects_an_empty_config(self):
        """A truncated copy leaves a zero-byte file, which the old check accepted."""
        self.complete()
        (self.model / "config.json").write_text("")
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_rejects_too_few_shards(self):
        self.complete(shards=1)
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_rejects_an_empty_shard(self):
        self.complete()
        (self.model / "model-00000-of-00002.safetensors").write_text("")
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_volume_probe_rejects_a_leftover_nfs_volume(self):
        """NFS_SHARE=0 must not silently keep reading over NFS.

        A leftover NFS volume has the same name and still has config.json, so the
        existence probe passes and the worker keeps reading over NFS — which fails
        confusingly the moment the exporter is gone. `Driver` cannot tell the two
        apart (both report "local"); the mount options can. This asserts the shape
        of the command only; the test below runs it.
        """
        out = self.sent_command()
        self.assertIn("{{json .Options}}", out)
        self.assertIn("dsv41-weights", out)
        self.assertIn('type":"nfs', out)
        self.assertIn("exit 1", out)

    def test_volume_probe_logic_distinguishes_real_volumes(self):
        """Run the command files/nfs-share.sh actually sends, against real docker.

        Not a copy of the logic: the fragment is captured from the shipped
        `local_model_volume_is_local`, so changing the shipped criterion fails this.
        """
        import shutil
        if not shutil.which("docker") or \
                subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            self.skipTest("docker daemon not reachable")
        fragment = self.sent_command()
        names = {"qa-probe-local": False, "qa-probe-nfs": True}
        for name, is_nfs in names.items():
            subprocess.run(["docker", "volume", "rm", name], capture_output=True)
            create = ["docker", "volume", "create"]
            if is_nfs:
                create += ["--driver", "local", "--opt", "type=nfs",
                           "--opt", "o=addr=10.0.0.9,nolock,soft",
                           "--opt", "device=:/var/tmp/x"]
            subprocess.run(create + [name], capture_output=True, check=True)
        try:
            for name, is_nfs in names.items():
                shipped = fragment.replace("dsv41-weights", name)
                accepted = subprocess.run(["bash", "-c", shipped],
                                          capture_output=True).returncode == 0
                self.assertEqual(accepted, not is_nfs,
                                 f"{name}: accepted={accepted}, expected local={not is_nfs}")
        finally:
            for name in names:
                subprocess.run(["docker", "volume", "rm", name], capture_output=True)

    def test_worker_probe_does_not_pull_or_create_a_volume(self):
        """The old probe pulled alpine and `-v` would create an empty volume."""
        source = (ROOT / "files" / "nfs-share.sh").read_text()
        probe = source[source.index("nfs_worker_has_model()"):]
        probe = probe[:probe.index("\n}\n")]
        self.assertIn("--pull=never", probe)
        self.assertIn("--network none", probe)
        self.assertIn("docker volume inspect", probe)
        self.assertNotIn("alpine", probe)
        self.assertNotIn("-v \'${NFS_VOLUME}", probe)


class NfsShareOffTests(unittest.TestCase):
    """NFS_SHARE=0 is the local-weights profile the ring needs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def test_share_is_a_noop(self):
        """serve calls cmd_share unconditionally; with NFS_SHARE=0 it must not touch NFS."""
        out = check(shell('''
          nfs_ensure_server() { echo "NFS-STARTED"; return 1; }
          nfs_publish_model()  { echo "NFS-PUBLISHED"; }
          nfs_ensure_worker_volume() { echo "NFS-VOLUME"; }
          cmd_share
        ''', Path(self.tmp.name), dict(BASE_SETTINGS, NFS_SHARE="0")))
        self.assertIn("keeping local worker volumes", out)
        for forbidden in ("NFS-STARTED", "NFS-PUBLISHED", "NFS-VOLUME"):
            self.assertNotIn(forbidden, out)

    def test_share_still_runs_when_enabled(self):
        out = check(shell('''
          nfs_ensure_server() { echo "NFS-STARTED"; }
          nfs_publish_model()  { echo "NFS-PUBLISHED"; }
          nfs_ensure_worker_volume() { :; }
          nfs_worker_has_model() { return 0; }
          ensure_ssh_keys() { :; }
          touch "$MODEL_DIR/config.json"
          cmd_share
        ''', Path(self.tmp.name), dict(BASE_SETTINGS, NFS_SHARE="1")))
        self.assertIn("NFS-STARTED", out)
        self.assertIn("NFS-PUBLISHED", out)

    def test_serve_refuses_a_leftover_nfs_volume(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        branch = serve[serve.index('if [[ "$NFS_SHARE" == "1" ]]'):]
        self.assertIn("local_model_volume_is_local", branch)
        self.assertIn("would still read over NFS", branch)

    def test_doctor_flags_it_too(self):
        source = (ROOT / "start.sh").read_text()
        doctor = source[source.index("cmd_doctor()"):]
        doctor = doctor[:doctor.index("\n}\n")]
        self.assertIn("local_model_volume_is_local", doctor)

    def test_serve_validates_local_weights_before_replacing_containers(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        branch = serve.index('if [[ "$NFS_SHARE" == "1" ]]')
        self.assertIn("local_model_has_weights", serve[branch:])
        self.assertIn("disables NFS setup", serve[branch:])
        for later in ("docker rm", "docker run -d", "start_worker"):
            if later in serve:
                self.assertLess(branch, serve.index(later), later)

    def test_status_reads_the_volume_not_the_head_symlink(self):
        source = (ROOT / "start.sh").read_text()
        status = source[source.index("cmd_status()"):]
        status = status[:status.index("\n}\n")]
        worker = status[status.index("for h in"):]
        self.assertNotIn("$COMMON_MODEL/config.json", worker)
        self.assertIn("nfs_worker_has_model", worker)

    def test_example_configures_local_weights_for_the_ring(self):
        example = (ROOT / ".env.tp4.example").read_text()
        self.assertIn("NFS_SHARE=0", example)

class DualPciDomainTests(unittest.TestCase):
    """The two-listener-GID cap that silently idles every device past the second."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def test_warns_when_ib_hca_exceeds_the_gid_cap(self):
        """The warning goes to stderr: it is advice, not a diagnostic result."""
        result = nccl_shell(
            'nccl_gid_publication_check',
            {"NCCL_SWITCHLESS_RING_ONLY": "1",
             "IB_HCA": "rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1",
             "NCCL_IB_EXTENDED_IPV4_GIDS": "0"})
        self.assertEqual(result.returncode, 0, "a capped list must not be fatal")
        self.assertIn("capped at 2", result.stderr)
        self.assertIn("rocep1s0f0", result.stderr)

    def test_silent_at_or_below_the_cap(self):
        for hca in ("rocep1s0f0", "rocep1s0f0,rocep1s0f1"):
            with self.subTest(hca=hca):
                result = nccl_shell("nccl_gid_publication_check",
                                    {"NCCL_SWITCHLESS_RING_ONLY": "1", "IB_HCA": hca,
                                     "NCCL_IB_EXTENDED_IPV4_GIDS": "0"})
                self.assertEqual(result.stderr.strip(), "")

    def test_silent_once_publication_is_extended(self):
        result = nccl_shell("nccl_gid_publication_check",
                            {"NCCL_SWITCHLESS_RING_ONLY": "1",
                             "IB_HCA": "rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1",
                             "NCCL_IB_EXTENDED_IPV4_GIDS": "1"})
        self.assertEqual(result.stderr.strip(), "")

    def test_silent_when_the_ring_is_off(self):
        result = nccl_shell("nccl_gid_publication_check",
                            {"NCCL_SWITCHLESS_RING_ONLY": "0",
                             "IB_HCA": "a,b,c,d", "NCCL_IB_EXTENDED_IPV4_GIDS": "0"})
        self.assertEqual(result.stderr.strip(), "")

    def test_extension_flags_reach_both_rank_paths(self):
        """Head and workers build their own environment; both must carry the flags."""
        on = dict(BASE_SETTINGS, NCCL_SWITCHLESS_RING_ONLY="1",
                  IB_HCA="rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1",
                  NCCL_IB_EXTENDED_IPV4_GIDS="1")
        argv = check(nccl_shell('a=(); switchless_ring_args a; printf "%s\\n" "${a[@]}"', on))
        envs = check(nccl_shell('switchless_ring_env_string', on))
        for name in ("NCCL_IB_EXTENDED_IPV4_GIDS=1", "NCCL_IB_PRESERVE_PCI_DOMAIN=1",
                     "NCCL_IB_ROUTE_DIAGNOSTICS=1", "NCCL_IB_QPS_PER_CONNECTION=1"):
            self.assertIn(name, argv, name)
            self.assertIn(name, envs, name)

    def test_head_and_worker_extension_parity(self):
        on = dict(BASE_SETTINGS, NCCL_SWITCHLESS_RING_ONLY="1",
                  NCCL_IB_EXTENDED_IPV4_GIDS="1")
        out = check(nccl_shell('''
          a=(); switchless_ring_args a
          printf 'ARGV %s\\n' "${a[@]}"
          printf 'STRV %s\\n' $(switchless_ring_env_string)
        ''', on))
        argv = sorted(lines_after(out, "ARGV "))
        strv = sorted(lines_after(out, "STRV "))
        self.assertEqual(argv, strv)

    def test_extension_is_off_by_default(self):
        """Byte-identical off: no extension flag may appear unless asked for."""
        out = check(nccl_shell('''
          a=(); switchless_ring_args a
          printf '%s\\n' "${a[@]}"
          switchless_ring_env_string
        ''', dict(BASE_SETTINGS, NCCL_SWITCHLESS_RING_ONLY="1")))
        self.assertNotIn("NCCL_IB_EXTENDED_IPV4_GIDS", out)
        self.assertNotIn("NCCL_IB_PRESERVE_PCI_DOMAIN", out)

    def test_doctor_reports_the_cap(self):
        source = (ROOT / "start.sh").read_text()
        doctor = source[source.index("cmd_doctor()"):]
        doctor = doctor[:doctor.index("\n}\n")]
        self.assertIn("nccl_gid_publication_check", doctor)

    def test_worker_settings_ship_the_new_helpers(self):
        out = check(nccl_shell("nccl_worker_settings"))
        for function in ("dual_pci_domain_args", "dual_pci_domain_env_string",
                         "nccl_gid_publication_check"):
            self.assertIn(f"{function} ()", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
