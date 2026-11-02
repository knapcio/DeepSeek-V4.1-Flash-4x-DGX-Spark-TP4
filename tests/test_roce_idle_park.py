#!/usr/bin/env python3
"""Checks for the RoCE proxy idle parking (DSV41_ROCE_IDLE_PARK=1, adapter/roce_idle_park.py).

- Gate off is v2.6: with every "#if ROCE_IDLE_PARK" block removed the proxy source is f744d83's
  byte for byte (pinned hash), the gate-off build of the current file has the same .text / .rodata /
  .data / .dynsym as that stripped source (Linux), _proxy.py builds and binds exactly as before
  (same command, same cache name, no roce_park_* lookups), and sitecustomize imports nothing unless
  the gate is set.
- Gate on: the build adds -DROCE_IDLE_PARK=1 under its own cache name keyed on the header; the
  bindings check the park ABI; the adapter refuses the roce / roce_ring transports, validates its
  env and configures every runtime it constructs.
- Scheduler hooks: idle mark once per busy->idle transition and BEFORE on_idle's own body (where
  --sleep-on-idle sleeps); wake once per idle->busy transition and BEFORE run_batch's body; a
  decode (run_batch after run_batch) makes no proxy call at all; process_input_requests wakes early
  only with requests; receipts (tid once, stats per new park).
- Against SGLang's scheduler.py when present (the image): on_idle / run_batch /
  process_input_requests exist, on_idle reaches maybe_sleep_on_idle, and both event loops call
  on_idle only in their no-batch branch.
- Native (gcc + libibverbs headers, Linux): the park build compiles with -Wall -Wextra -Werror and
  exports the API; tests/roce_idle_park/park_unit.c (state machine, lost-wakeup race) and
  park_stress.c (four real proxy loops over an in-process fake RDMA: baseline, every wake called,
  wakes skipped on one rank) pass.

No GPU, torch or network. usage: PYTHONPATH=adapter python3 tests/test_roce_idle_park.py
  PARK_STRESS_BURSTS=<n> (default 600) sets the stress length.
"""
import ast
import ctypes
import hashlib
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
B12X = ROOT / "runtime/b12x/b12x/comm"
if not (B12X / "roce_twoshot").is_dir():  # an overlay image: the package is installed in /opt/b12x only
    B12X = Path("/opt/b12x/b12x/comm")
TWO = B12X / "roce_twoshot"
ADAPTER = ROOT / "adapter"
CTESTS = Path(__file__).resolve().parent / "roce_idle_park"
SCHEDULER_PY = Path(os.environ.get(
    "SGLANG_SCHEDULER_PY", "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"))

# roce_twoshot/_roce_proxy.c and _proxy.py as of f744d83 (v2.6.1, production since 2026-10-27)
V26_PROXY_C_SHA256 = "69871c68c92e32201a9b0ca037d95de0406cf7e21356caff3c11486e0b73e095"
GATE_BLOCK = re.compile(r"^#if defined\(ROCE_IDLE_PARK\) && ROCE_IDLE_PARK\n.*?^#endif\n", re.S | re.M)
DOC_PARAGRAPH = re.compile(r"//\n// ds41 idle parking \(DSV41_ROCE_IDLE_PARK=1.*?the v2\.6 proxy's\.\n", re.S)

sys.path.insert(0, str(ADAPTER))
import roce_idle_park as park  # noqa: E402


def stripped_proxy_source():
    text = (TWO / "_roce_proxy.c").read_text()
    text = GATE_BLOCK.sub("", text)
    text = DOC_PARAGRAPH.sub("", text, count=1)
    return text.rstrip("\n") + "\n"


def load_proxy_module():
    """A fresh copy of roce_twoshot/_proxy.py (stdlib only), outside any package."""
    spec = importlib.util.spec_from_file_location("park_test_proxy", TWO / "_proxy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GateOffIsV26(unittest.TestCase):
    def test_stripped_source_is_v26(self):
        text = stripped_proxy_source()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), V26_PROXY_C_SHA256)

    def test_every_addition_is_gated(self):
        text = (TWO / "_roce_proxy.c").read_text()
        blocks = GATE_BLOCK.findall(text)
        self.assertEqual(len(blocks), 7)
        for b in blocks:
            self.assertNotIn("#if", b.split("\n", 1)[1])  # no nesting: the strip is exact
        rest = GATE_BLOCK.sub("", text)
        self.assertNotIn("park", DOC_PARAGRAPH.sub("", rest).lower())
        self.assertIn('#include "_roce_idle_park.h"', "".join(blocks))

    def _capture_build(self, mod, idle_park):
        mod.IDLE_PARK = idle_park
        seen = {}

        def fake_run(cmd, capture_output, text):
            seen["cmd"] = list(cmd)
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"so")
            return types.SimpleNamespace(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"B12X_ROCE_CACHE_DIR": tmp, "CC": "cc"}), \
                mock.patch.object(mod.subprocess, "run", fake_run), \
                mock.patch.object(mod.shutil, "which", lambda name: "/usr/bin/" + name):
            target = mod._build()
            name = target.name
            tmpname = Path(seen["cmd"][seen["cmd"].index("-o") + 1]).name
            seen["cmd"][seen["cmd"].index("-o") + 1] = "<tmp>"
        return name, tmpname, seen["cmd"]

    def test_gate_off_build_command_and_name(self):
        mod = load_proxy_module()
        name, tmpname, cmd = self._capture_build(mod, False)
        source = (TWO / "_roce_proxy.c").read_bytes()
        self.assertEqual(name, f"roce_twoshot_proxy-{hashlib.sha256(source).hexdigest()[:16]}.so")
        self.assertTrue(tmpname.startswith(".roce_twoshot_proxy-"))
        self.assertEqual(cmd, ["cc", "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", "<tmp>",
                               str(TWO / "_roce_proxy.c"), "-libverbs", "-lpthread"])

    def test_gate_on_build_command_and_name(self):
        mod = load_proxy_module()
        off, _, _ = self._capture_build(mod, False)
        name, tmpname, cmd = self._capture_build(mod, True)
        self.assertTrue(name.startswith("roce_twoshot_proxy-park-"))
        self.assertTrue(tmpname.startswith(".roce_twoshot_proxy-park-"))
        self.assertNotEqual(name.split("-")[-1], off.split("-")[-1])
        self.assertEqual(cmd[:2], ["cc", "-DROCE_IDLE_PARK=1"])
        self.assertEqual(cmd[2:], ["-O2", "-std=gnu11", "-shared", "-fPIC", "-o", "<tmp>",
                                   str(TWO / "_roce_proxy.c"), "-libverbs", "-lpthread"])
        # the header is part of the key: a header change is a new library
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "h.h"
            other.write_bytes((TWO / "_roce_idle_park.h").read_bytes() + b"\n// changed\n")
            with mock.patch.object(mod, "_PARK_HEADER", other):
                changed, _, _ = self._capture_build(mod, True)
        self.assertNotEqual(changed, name)

    def _fake_lib(self, park_abi=1):
        accessed = []

        class Fn:
            def __init__(self, name):
                self.name = name

            def __call__(self, *args):
                return {"roce_abi_version": 103, "roce_park_abi": park_abi}.get(self.name, 0)

        class Lib:
            def __getattr__(self, name):
                accessed.append(name)
                fn = Fn(name)
                object.__setattr__(self, name, fn)
                return fn

        return Lib(), accessed

    def test_gate_off_binds_no_park_symbol(self):
        mod = load_proxy_module()
        lib, accessed = self._fake_lib()
        with mock.patch.object(mod, "_build", lambda: Path("/x.so")), \
                mock.patch.object(mod.ctypes, "CDLL", lambda *a, **k: lib):
            mod.load()
        self.assertTrue(accessed)
        self.assertFalse([n for n in accessed if "park" in n])

    def test_gate_on_binds_and_checks_park_abi(self):
        mod = load_proxy_module()
        mod.IDLE_PARK = True
        lib, accessed = self._fake_lib()
        with mock.patch.object(mod, "_build", lambda: Path("/x.so")), \
                mock.patch.object(mod.ctypes, "CDLL", lambda *a, **k: lib):
            mod.load()
        self.assertEqual({n for n in accessed if "park" in n},
                         {"roce_park_abi", "roce_park_configure", "roce_park_allow", "roce_park_wake",
                          "roce_park_stat"})
        mod = load_proxy_module()
        mod.IDLE_PARK = True
        lib, _ = self._fake_lib(park_abi=2)
        with mock.patch.object(mod, "_build", lambda: Path("/x.so")), \
                mock.patch.object(mod.ctypes, "CDLL", lambda *a, **k: lib), \
                self.assertRaises(RuntimeError):
            mod.load()

    def test_stat_names_match_the_header(self):
        header = (TWO / "_roce_idle_park.h").read_text()
        idx = {int(v): k for k, v in re.findall(r"#define ROCE_PARK_STAT_(\w+) (\d+)", header)}
        mod = load_proxy_module()
        self.assertEqual(len(idx), len(mod.PARK_STATS))
        expect = {"PARKS": "parks", "WAKES": "wakes", "TIMEOUTS": "timeouts", "LATE": "late_doorbells",
                  "PARKED_NS": "parked_ns", "LAST_WAKE_NS": "last_wake_ns", "MAX_WAKE_NS": "max_wake_ns",
                  "LAST_LEAD_NS": "last_lead_ns", "TID": "tid", "PARKED": "parked", "ALLOW": "allow",
                  "AFTER_NS": "after_ns", "POLL_NS": "poll_ns"}
        for i, name in enumerate(mod.PARK_STATS):
            self.assertEqual(expect[idx[i]], name)

    def test_sitecustomize_gates_before_import(self):
        src = (ADAPTER / "sitecustomize.py").read_text()
        for fn in ("install_runtime as install_roce_idle_park",
                   "install_scheduler as install_roce_idle_park_scheduler"):
            i = src.index(f"from roce_idle_park import {fn}")
            gate = src.rindex("if os.environ.get('DSV41_ROCE_IDLE_PARK', '0').strip() == '1':", 0, i)
            self.assertLess(i - gate, 200)
        self.assertEqual(src.count("from roce_idle_park import"), 2)
        roce_branch = src[src.index("elif module.__name__ in ('b12x.comm.roce.roce_oneshot'"):]
        self.assertLess(roce_branch.index("install_roce_idle_park(module)"),
                        roce_branch.index("elif module.__name__ == 'sglang.srt.layers.logits_processor'"))
        sched = src[src.index("elif module.__name__ == 'sglang.srt.managers.scheduler':"):]
        self.assertLess(sched.index("install_roce_idle_park_scheduler(module)"),
                        sched.index("elif module.__name__ == 'sglang.srt.managers.schedule_batch':"))
        self.assertIn("'sglang.srt.managers.scheduler'", src[src.index("class EngramFinder"):])


class AdapterConfig(unittest.TestCase):
    def test_enabled(self):
        self.assertFalse(park.enabled({}))
        self.assertFalse(park.enabled({"DSV41_ROCE_IDLE_PARK": "0"}))
        self.assertTrue(park.enabled({"DSV41_ROCE_IDLE_PARK": "1"}))

    def test_defaults_and_values(self):
        self.assertEqual(park.config({}), (1000, 20))
        self.assertEqual(park.config({"DSV41_ROCE_IDLE_PARK_MS": "250", "DSV41_ROCE_IDLE_PARK_POLL_MS": "5"}),
                         (250, 5))
        self.assertEqual(park.config({"DSV41_ROCE_IDLE_PARK_MS": " "}), (1000, 20))

    def test_refused(self):
        for env in ({"DSV41_ROCE_IDLE_PARK_MS": "0"}, {"DSV41_ROCE_IDLE_PARK_MS": "x"},
                    {"DSV41_ROCE_IDLE_PARK_MS": "3600001"}, {"DSV41_ROCE_IDLE_PARK_POLL_MS": "0"},
                    {"DSV41_ROCE_IDLE_PARK_POLL_MS": "1001"}, {"DSV41_ROCE_IDLE_PARK_POLL_MS": "-3"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                park.config(env)


class FakeProxy:
    def __init__(self, tid=4242):
        self.calls = []
        self.stats = {n: 0 for n in ("parks", "wakes", "timeouts", "late_doorbells", "parked_ns",
                                     "last_wake_ns", "max_wake_ns", "last_lead_ns", "tid", "parked",
                                     "allow", "after_ns", "poll_ns")}
        self.stats["tid"] = tid

    def park_configure(self, after_ms, poll_ms):
        self.calls.append(("configure", after_ms, poll_ms))

    def park_allow(self):
        self.calls.append("allow")

    def park_wake(self):
        self.calls.append("wake")

    def park_stats(self):
        return dict(self.stats)


def fresh_adapter():
    park._ENTRIES.clear()
    park._STATE["allowed"] = False


class InstallRuntime(unittest.TestCase):
    def setUp(self):
        fresh_adapter()
        self.saved = {k: sys.modules.get(k) for k in ("b12x.comm.roce_twoshot._proxy",)}

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        fresh_adapter()

    def fake_module(self, name, lib_loaded=False):
        proxy_mod = types.ModuleType("b12x.comm.roce_twoshot._proxy")
        proxy_mod.IDLE_PARK = False
        proxy_mod._LIB = object() if lib_loaded else None
        sys.modules["b12x.comm.roce_twoshot._proxy"] = proxy_mod
        mod = types.ModuleType(name)

        class RoceOneshotAllReduce:
            def __init__(self, rank=0):
                self.rank = rank
                self._proxy = FakeProxy(tid=100 + rank)

        mod.RoceOneshotAllReduce = RoceOneshotAllReduce
        return mod, proxy_mod

    def test_installs_on_twoshot(self):
        mod, proxy_mod = self.fake_module(park.MODULE)
        with mock.patch.dict(os.environ, {"DSV41_ROCE_IDLE_PARK_MS": "1500", "DSV41_ROCE_IDLE_PARK_POLL_MS": "7"}), \
                self.assertLogs(park.logger, logging.WARNING) as logs:
            park.install_runtime(mod)
            park.install_runtime(mod)  # idempotent
        self.assertTrue(proxy_mod.IDLE_PARK)
        self.assertEqual(sum("armed after_ms=1500 poll_ms=7" in m for m in logs.output), 1)
        rt = mod.RoceOneshotAllReduce(rank=2)
        self.assertEqual(rt._proxy.calls, [("configure", 1500, 7)])
        self.assertEqual(len(park._ENTRIES), 1)

    def test_refuses_other_transports(self):
        for name in park.OTHER_MODULES:
            mod, _ = self.fake_module(name)
            with self.subTest(name=name), mock.patch.dict(os.environ, {"DSV41_ROCE_TWOSHOT": "0"}), \
                    self.assertRaises(RuntimeError):
                park.install_runtime(mod)

    def test_ignores_other_packages_when_twoshot_is_the_transport(self):
        mod, proxy_mod = self.fake_module(park.OTHER_MODULES[0])
        with mock.patch.dict(os.environ, {"DSV41_ROCE_TWOSHOT": "1"}):
            park.install_runtime(mod)
        self.assertFalse(proxy_mod.IDLE_PARK)
        self.assertFalse(getattr(mod.RoceOneshotAllReduce, "_dsv41_idle_park", False))

    def test_refuses_a_library_loaded_without_the_gate(self):
        mod, _ = self.fake_module(park.MODULE, lib_loaded=True)
        with self.assertRaises(RuntimeError):
            park.install_runtime(mod)


def make_scheduler_module(idle_name="on_idle", with_process=True):
    log = []
    mod = types.ModuleType("sglang.srt.managers.scheduler")

    def idle_body(self):
        log.append(("idle_body", park._STATE["allowed"]))  # the sleep happens in here

    def run_batch(self, batch):
        log.append(("run_batch_body", park._STATE["allowed"]))
        return batch

    ns = {idle_name: idle_body, "run_batch": run_batch}
    if with_process:
        def process_input_requests(self, recv_reqs):
            log.append(("process_body", len(recv_reqs)))
        ns["process_input_requests"] = process_input_requests
    mod.Scheduler = type("Scheduler", (), ns)
    return mod, log


class SchedulerHooks(unittest.TestCase):
    def setUp(self):
        fresh_adapter()
        self.proxies = [FakeProxy(tid=10 + r) for r in range(2)]
        self.runtimes = []
        for r, p in enumerate(self.proxies):
            rt = types.SimpleNamespace(rank=r, _proxy=p)
            self.runtimes.append(rt)
            park._ENTRIES.append(park._Entry.__new__(park._Entry))
            e = park._ENTRIES[-1]
            e.runtime = (lambda rt=rt: rt)
            e.allowed = False
            e.tid_logged = False
            e.parks_logged = 0
        self.mod, self.log = make_scheduler_module()
        park.install_scheduler(self.mod)
        self.s = self.mod.Scheduler()

    def tearDown(self):
        fresh_adapter()

    def calls(self):
        return [[c for c in p.calls if c in ("allow", "wake")] for p in self.proxies]

    def test_decode_makes_no_proxy_call(self):
        for step in range(500):  # one request's decode: a batch every iteration, never idle
            self.s.run_batch(step)
        self.assertEqual(self.calls(), [[], []])

    def test_idle_marks_once_and_before_the_body(self):
        for _ in range(5):
            self.s.on_idle()
        self.assertEqual(self.calls(), [["allow"], ["allow"]])
        self.assertEqual(self.log[0], ("idle_body", True))  # marked before on_idle's sleep

    def test_wake_once_before_run_batch(self):
        self.s.on_idle()
        self.s.run_batch(1)
        self.assertEqual(self.log[-1], ("run_batch_body", False))  # woken before the forward
        for step in range(50):
            self.s.run_batch(step)
        self.assertEqual(self.calls(), [["allow", "wake"], ["allow", "wake"]])
        self.s.on_idle()
        self.s.on_idle()
        self.s.run_batch(9)
        self.assertEqual(self.calls(), [["allow", "wake", "allow", "wake"]] * 2)

    def test_requests_wake_early(self):
        self.s.on_idle()
        self.s.process_input_requests([])
        self.assertEqual(self.calls(), [["allow"], ["allow"]])
        self.s.process_input_requests(["req"])
        self.assertEqual(self.calls(), [["allow", "wake"], ["allow", "wake"]])
        self.s.run_batch(1)
        self.assertEqual(self.calls(), [["allow", "wake"], ["allow", "wake"]])
        self.assertEqual(self.log[-2:], [("process_body", 1), ("run_batch_body", False)])

    def test_receipts(self):
        with self.assertLogs(park.logger, logging.WARNING) as logs:
            self.s.on_idle()
            self.s.run_batch(1)
            self.proxies[0].stats.update(parks=1, wakes=1, parked_ns=2_500_000_000, last_wake_ns=40_000,
                                         max_wake_ns=40_000, last_lead_ns=900_000)
            self.s.on_idle()
            self.s.run_batch(2)
            self.s.on_idle()
        out = "\n".join(logs.output)
        self.assertEqual(out.count("DSV41_ROCE_IDLE_PARK rank=0 proxy_tid=10"), 1)
        self.assertEqual(out.count("DSV41_ROCE_IDLE_PARK rank=1 proxy_tid=11"), 1)
        self.assertEqual(out.count("DSV41_ROCE_IDLE_PARK_STATS"), 1)
        self.assertIn("DSV41_ROCE_IDLE_PARK_STATS rank=0 parks=1 wakes=1 timeouts=0 late_doorbells=0 "
                      "parked_s=2.500 last_wake_us=40.0 max_wake_us=40.0 last_lead_us=900.0", out)

    def test_closed_runtime_is_skipped(self):
        self.runtimes[1]._proxy = None
        self.s.on_idle()
        self.s.run_batch(1)
        self.assertEqual(self.calls()[0], ["allow", "wake"])

    def test_fallback_and_refusal(self):
        mod, log = make_scheduler_module("self_check_during_idle", with_process=False)
        park.install_scheduler(mod)
        s = mod.Scheduler()
        s.self_check_during_idle()
        s.run_batch(1)
        self.assertEqual(self.calls(), [["allow", "wake"], ["allow", "wake"]])
        mod = types.ModuleType("sglang.srt.managers.scheduler")
        mod.Scheduler = type("Scheduler", (), {"run_batch": lambda self, b: b})
        with self.assertRaises(RuntimeError):
            park.install_scheduler(mod)

    def test_idempotent(self):
        before = self.mod.Scheduler.run_batch
        park.install_scheduler(self.mod)
        self.assertIs(self.mod.Scheduler.run_batch, before)


@unittest.skipUnless(SCHEDULER_PY.is_file(), "needs SGLang's scheduler.py (the image)")
class SGLangScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SCHEDULER_PY.read_text())
        cls.cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        cls.methods = {f.name: f for f in cls.cls.body if isinstance(f, ast.FunctionDef)}

    def calls_in(self, node):
        return [n.func.attr for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"]

    def test_hooked_methods_exist(self):
        for name in ("on_idle", "run_batch", "process_input_requests"):
            self.assertIn(name, self.methods)
        args = [a.arg for a in self.methods["process_input_requests"].args.args]
        self.assertEqual(args[:2], ["self", "recv_reqs"])

    def test_on_idle_contains_the_sleep(self):
        self.assertIn("maybe_sleep_on_idle", self.calls_in(self.methods["on_idle"]))

    @staticmethod
    def direct_on_idle(stmts):
        return any(isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                   and ast.unparse(st.value.func) == "self.on_idle" for st in stmts)

    def test_event_loops_idle_only_without_a_batch(self):
        for loop in ("event_loop_normal", "event_loop_overlap"):
            fn = self.methods[loop]
            self.assertIn("run_batch", self.calls_in(fn))
            sites = 0
            for node in ast.walk(fn):
                if not isinstance(node, ast.If):
                    continue
                test = ast.unparse(node.test)
                if self.direct_on_idle(node.body):  # `if <no batch>: self.on_idle()`
                    sites += 1
                    self.assertEqual(test, "batch is None", loop)
                if self.direct_on_idle(node.orelse):  # `if batch: ... else: self.on_idle()`
                    sites += 1
                    self.assertEqual(test, "batch", loop)
            self.assertEqual(sites, 1, loop)


def _verbs_available():
    cc = shutil.which("gcc") or shutil.which("cc")
    return sys.platform.startswith("linux") and cc is not None and any(
        Path(p, "infiniband/verbs.h").is_file() for p in ("/usr/include", "/usr/local/include"))


@unittest.skipUnless(_verbs_available(), "needs Linux, gcc and libibverbs headers")
class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cc = shutil.which("gcc") or "cc"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, out, sources, *flags, shared=True, libs=("-libverbs", "-lpthread")):
        cmd = [self.cc, *flags, "-O2", "-std=gnu11"] + (["-shared", "-fPIC"] if shared else []) + \
              ["-o", os.path.join(self.tmp, out), *map(str, sources), *libs]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, " ".join(cmd) + "\n" + proc.stderr)
        return os.path.join(self.tmp, out)

    def section(self, so, name):
        out = os.path.join(self.tmp, f"{os.path.basename(so)}{name}.bin")
        subprocess.run(["objcopy", "-O", "binary", f"--only-section={name}", so, out], check=True)
        return Path(out).read_bytes()

    @unittest.skipUnless(shutil.which("objcopy"), "needs objcopy")
    def test_gate_off_object_code_is_v26(self):
        ref = Path(self.tmp, "v26_proxy.c")
        ref.write_text(stripped_proxy_source())
        a = self.build("v26.so", [ref])
        b = self.build("off.so", [TWO / "_roce_proxy.c"])
        for name in (".text", ".rodata", ".data", ".dynsym", ".dynstr"):
            with self.subTest(section=name):
                self.assertEqual(self.section(a, name), self.section(b, name))

    def test_park_build_is_warning_clean_and_exports_the_api(self):
        so = self.build("on.so", [TWO / "_roce_proxy.c"], "-DROCE_IDLE_PARK=1", "-Wall", "-Wextra", "-Werror")
        lib = ctypes.CDLL(so)
        self.assertEqual(lib.roce_abi_version(), 103)
        self.assertEqual(lib.roce_park_abi(), 1)
        for sym in ("roce_park_configure", "roce_park_allow", "roce_park_wake", "roce_park_stat"):
            self.assertTrue(hasattr(lib, sym))
        off = ctypes.CDLL(self.build("off_w.so", [TWO / "_roce_proxy.c"], "-Wall", "-Wextra", "-Werror"))
        self.assertFalse(hasattr(off, "roce_park_abi"))

    def test_state_machine_unit(self):
        exe = self.build("park_unit", [CTESTS / "park_unit.c"], "-Wall", "-Wextra", "-Werror", "-pthread",
                         f"-I{TWO}", shared=False, libs=())
        proc = subprocess.run([exe, "20000"], capture_output=True, text=True, timeout=300)
        sys.stderr.write(proc.stdout)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertRegex(proc.stdout, r"UNIT PASS 14 tests")

    def test_multi_rank_stress(self):
        bursts = os.environ.get("PARK_STRESS_BURSTS", "600")
        base = self.build("stress_base", [CTESTS / "park_stress.c"], "-Wall", "-Werror", "-pthread", f"-I{TWO}",
                          shared=False, libs=("-libverbs", "-lm"))
        on = self.build("stress_park", [CTESTS / "park_stress.c"], "-Wall", "-Werror", "-pthread",
                        "-DROCE_IDLE_PARK=1", f"-I{TWO}", shared=False, libs=("-libverbs", "-lm"))
        for exe, mode in ((base, "baseline"), (on, "nofault"), (on, "fault")):
            with self.subTest(mode=mode):
                proc = subprocess.run([exe, mode, bursts], capture_output=True, text=True, timeout=600)
                sys.stderr.write(proc.stdout)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertRegex(proc.stdout, r"^STRESS .* PASS$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
