"""CPU tests for prefix_scan.py: a fake OpenAI-compatible server, no model or GPU.

  python3 -B tests/test_prefix_scan_cpu.py
"""
import http.server
import json
import os
import sys
import random
import re
import tempfile
import threading
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import prefix_scan as P  # noqa: E402


class Fake(http.server.BaseHTTPRequestHandler):
    mode = "clean"     # clean | degenerate | stale | nohit | phantom | worse | loopabort | loopabort500
    report = True      # usage.prompt_tokens_details.cached_tokens (SGLang --enable-cache-report)
    seen = set()
    hits = 0.0
    lock = threading.Lock()
    pfx = None

    def log_message(self, *a):
        pass

    def _send(self, obj, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(("sglang:cached_tokens_total{model_name=\"m\"} %f\n" % Fake.hits).encode(), "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            text = body["messages"][0]["content"]
            # ~33 tokens per record + the question: enough structure for the sizing loop
            return self._send({"count": 11 + 33 * text.count("\n") + len(text.split("\n\n")[-1]) // 4})
        content = body["messages"][0]["content"]
        q = content.split("\n\n")[-1]
        lines = content.split("\n\n")[0].split("\n")
        pfx = P.Prefix(len(lines), 7)
        salt = body.get("cache_salt")
        with Fake.lock:
            # round salts are shared by the 8 concurrent cold requests: "warm" only after that batch
            n = sum(1 for x in Fake.seen if x[0] == salt)
            warm = n >= (8 if re.search(r"-r\d+$", salt or "") else 1)
            Fake.seen.add((salt, n))
            if warm and Fake.mode != "nohit":
                Fake.hits += 90000
        if Fake.mode == "loopabort500" and warm:
            # The named loop exit (DSV41_LOOP_NAMED_EXIT=1): SGLang's non-stream error body, HTTP 500.
            b = json.dumps({"object": "error", "message": "loop_abort: DSV41 repetition (art=ngram, period=32, "
                            "output_tokens=128, kept=32)", "type": "500", "param": None, "code": 500}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b)
            return
        if q.startswith("Give the ref code"):
            a, b = (int(x) for x in re.findall(r"Record (\d{5})", q))
            ca = "ZZZZZ" if (warm and Fake.mode == "worse") else pfx.codes[a]
            ans = f"Record {a:05d}: {ca}\nRecord {b:05d}: {pfx.codes[b]}"
        elif q.startswith("Quote verbatim the last"):
            ans = "\n".join(pfx.lines[-3:])
        elif q.startswith("Quote verbatim"):
            want = [int(x) for x in re.findall(r"Record (\d{5})", q)]
            if warm and Fake.mode == "phantom":
                want = want[:-1] + [want[-1] - 1]
            ans = "\n".join(pfx.lines[k] for k in want)
        else:
            ans = f"Record {len(lines) - 9:05d}" if (warm and Fake.mode == "phantom") else "NONE"
        ids = list(range(40)) + ([7] * 16 if Fake.mode == "degenerate" else [])
        rnd = random.Random(hash((salt, warm)) & 0xffff)
        lps = [-0.5 + rnd.uniform(-0.03, 0.03) + (0.3 if (warm and Fake.mode == "stale") else 0.0) for _ in ids]
        ch = {"finish_reason": "stop", "response_token_ids": ids, "message": {"content": ans, "reasoning_content": ""}}
        if Fake.mode == "loopabort" and warm:
            ch["matched_stop"] = "repetition"
        if body.get("logprobs"):
            ch["logprobs"] = {"content": [{"logprob": v} for v in lps]}
        usage = {"prompt_tokens": 1000}
        if Fake.report:
            usage["prompt_tokens_details"] = {"cached_tokens": 90000 if (warm and Fake.mode != "nohit") else 0}
        self._send({"choices": [ch], "usage": usage})


class PrefixScanCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:%d" % cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def go(self, mode, lines=400, strict=False, report=True, drift_task="quote"):
        Fake.mode = mode
        Fake.report = report
        Fake.seen = set()  # salts carry a per-second stamp: two runs in one second must not share cache state
        d = tempfile.mkdtemp()
        a = SimpleNamespace(label=mode, base=self.base, model="m", conc=8, rounds=1, temp=0.8, top_p=0.95,
                            max_tokens=64, solo_max_tokens=64, thinking="default", lines=lines, seed=7, logprobs=False,
                            strict=strict, block=2304, drift_reps=3, drift_tokens=16, drift_kinds="full,partial",
                            drift_step=7, drift_task=drift_task, partial_frac=0.6, out=os.path.join(d, "r.json"))
        rc = P.run(a)
        with open(a.out) as f:
            return rc, json.load(f)

    def test_scoring_primitives(self):
        pfx = P.Prefix(400, 7)
        self.assertEqual(P.longest_run([1, 2, 2, 2, 3]), 3)
        self.assertGreaterEqual(P.longest_loop([4, 5] * 30), 48)
        self.assertEqual(P.longest_loop(list(range(100))), 0)
        self.assertEqual(P.parse_scan("NONE"), "NONE")
        self.assertEqual(P.parse_scan("Record 00012\nRecord 00013"), [12, 13])
        self.assertEqual(P.parse_lookup("Record 00042: ABCDE"), {42: "ABCDE"})
        sc = P.score(pfx, "quote", (5,), {"content": pfx.lines[5].replace("ref=", "ref=Q"), "token_ids": [1]})
        self.assertTrue(sc["misquotes"])
        self.assertEqual(P.drift({"token_ids": [1, 2, 3], "lps": [-1.0, -2.0, -3.0]},
                                 {"token_ids": [1, 2, 9], "lps": [-1.5, -2.5, -9.0]}), [2, 0.5])

    def test_clean_passes(self):
        rc, res = self.go("clean")
        self.assertEqual(rc, 0, res["summary"])
        self.assertEqual(res["summary"]["degenerate"], 0)
        self.assertEqual(len(res["rounds"]), 2)  # cold + warm
        for kind in ("full", "partial"):
            self.assertLessEqual(res["summary"]["drift"][kind]["cold_warm"], res["summary"]["drift"][kind]["limit"])
            self.assertEqual(res["summary"]["drift"][kind]["reps"], 3)

    def test_clean_passes_from_metrics(self):
        rc, res = self.go("clean", report=False)
        self.assertEqual(rc, 0, res["summary"])

    def test_loop_abort_is_degenerate(self):
        rc, res = self.go("loopabort")
        self.assertEqual(rc, 1)
        self.assertGreater(res["summary"]["degenerate"], 0)

    def test_named_loop_exit_is_degenerate_not_error(self):
        rc, res = self.go("loopabort500")
        self.assertEqual(rc, 1)
        self.assertGreater(res["summary"]["degenerate"], 0)
        self.assertFalse(any("request error" in r for r in res["summary"]["reasons"]), res["summary"])

    def test_degenerate_fails(self):
        rc, res = self.go("degenerate")
        self.assertEqual(rc, 1)
        self.assertEqual(res["summary"]["degenerate"], 16)

    def test_stale_cache_state_fails_drift(self):
        rc, res = self.go("stale")
        self.assertEqual(rc, 1)
        self.assertTrue(any("drift" in r for r in res["summary"]["reasons"]), res["summary"])

    def test_warm_only_anomaly_fails(self):
        for task in ("scan", "quote"):
            rc, res = self.go("phantom", drift_task=task)
            self.assertEqual(rc, 1, task)
            self.assertTrue(res["summary"]["drift"]["full"]["warm_only_anomalies"], res["summary"])

    def test_warm_only_anomaly_fails_scan(self):
        rc, res = self.go("phantom", drift_task="scan")
        self.assertEqual(rc, 1)
        self.assertTrue(res["summary"]["drift"]["full"]["warm_only_anomalies"], res["summary"])
        self.assertTrue(any("warm-only" in r for r in res["summary"]["reasons"]))

    def test_warm_worse_than_cold_fails(self):
        rc, res = self.go("worse")
        self.assertEqual(rc, 1)
        self.assertTrue(any(r.startswith("warm incorrect") or r.startswith("warm misquotes") for r in res["summary"]["reasons"]),
                        res["summary"])

    def test_model_errors_symmetric_pass(self):
        # the same wrong answer cold and warm is a model property, not a cache fault
        res = {"rounds": [{"phase": ph, "responses": [{"kind": "lookup", "score": {"correct": False, "misquotes": ["x"]}}] * 5}
                          for ph in ("cold", "warm")],
               "drift": [{"cc": [5, 0.05], "cw": [5, 0.05], "warm_hits": 9e4,
                          "texts": {"coldA": "Record 00997", "coldB": "Record 00997", "warmA": "Record 00997"}}] * 3}
        self.assertEqual(P.summarize(res)["verdict"], "PASS")

    def test_issue2_release_data_fails(self):
        # 2026-09-29 release without the fix (fastprobe, issue prompt): drift 0.25-0.36 vs 0.065, warm-only 36559
        res = {"rounds": [], "drift": [{"cc": [4, 0.065], "cw": [2, v], "warm_hits": 99072.0,
                                         "texts": {"coldA": "NONE", "coldB": "NONE", "warmA": "Record 36559"}}
                                        for v in (0.2658, 0.2497, 0.3629)]}
        sm = P.summarize(res)
        self.assertEqual(sm["verdict"], "FAIL")
        self.assertTrue(any("drift" in r for r in sm["reasons"]) and any("warm-only" in r for r in sm["reasons"]))
        fixed = {"rounds": [], "drift": [{"cc": [4, 0.0439], "cw": [n, v], "warm_hits": 99072.0,
                                           "texts": {"coldA": "NONE", "coldB": "NONE", "warmA": "NONE"}}
                                          for n, v in ((54, 0.0218), (64, 0.0233), (4, 0.0413))]}
        self.assertEqual(P.summarize(fixed)["verdict"], "PASS")

    def test_no_cache_hit_fails(self):
        rc, res = self.go("nohit")
        self.assertEqual(rc, 1)
        self.assertTrue(any("did not hit" in r for r in res["summary"]["reasons"]), res["summary"])

    def test_auto_sizing_hits_block_phase(self):
        rc, res = self.go("clean", lines=0)
        self.assertEqual(rc, 0, res["summary"])
        note = res["prefix"]["sizing"]
        lo, hi = (int(x) for x in re.search(r"P mod 2304 (\d+)-(\d+)", note).groups())
        self.assertTrue(P.TARGET_LO <= lo <= hi <= P.TARGET_HI, note)


if __name__ == "__main__":
    unittest.main()
