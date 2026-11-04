"""CPU tests for ds_gate.py: a fake SGLang server (/tokenize, /generate, /v1/chat/completions), no model or GPU.

  python3 -B tests/test_ds_gate_cpu.py
"""
import contextlib
import http.server
import io
import json
import math
import os
import sys
import random
import tempfile
import threading
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import ds_gate as P  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
KIND = {p: k for k, _, p in P.PROMPTS}


def dist(prev, k, mode):
    """Fake next-token top-k: token ids (prev*31 + j) % 5000, probabilities 0.5^(j+1) (hot mode flattens them)."""
    lps = [(j + 1) * math.log(0.5) for j in range(k)]
    if mode == "hot":
        lps = [x * 0.6 for x in lps]
        z = math.log(sum(math.exp(x) for x in lps) / 0.97)
        lps = [x - z for x in lps]
    return [((prev * 31 + j) % 5000, lp) for j, lp in enumerate(lps)]


def ent(tid, lp, shape):
    return {"logprob": lp, "token_id": tid, "token": None} if shape == "dict" else [lp, tid, None]


class Fake(http.server.BaseHTTPRequestHandler):
    mode = "clean"      # clean | hot | cached
    shape = "list"      # list | dict (meta_info entry shape)
    chat = {}           # prompt kind -> clean | salad | loop | repetition | badjson
    calls = []
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with Fake.lock:
            Fake.calls.append((self.path, body))
        if self.path == "/tokenize":
            return self._send({"tokens": [ord(c) for c in body["prompt"]], "count": len(body["prompt"]),
                               "max_model_len": 1048576})
        if self.path == "/generate":
            return self._send(self.generate(body))
        if Fake.chat.get(KIND[body["messages"][0]["content"]]) == "namedexit":
            # The named loop exit (DSV41_LOOP_NAMED_EXIT=1): SGLang's non-stream error body, HTTP 500.
            b = json.dumps({"object": "error", "message": "loop_abort: DSV41 repetition (art=line, period=0, "
                            "output_tokens=900, kept=900)", "type": "500", "param": None, "code": 500}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b)
            return
        return self._send(self.chat_reply(body))

    def generate(self, body):
        ids, sp, k, start = body["input_ids"], body["sampling_params"], body["top_logprobs_num"], body.get("logprob_start_len", len(body["input_ids"]))
        sh = Fake.shape
        in_tok, in_top = [], []
        for i in range(start, len(ids)):
            if i == 0:
                in_tok.append(ent(ids[0], None, sh))
                in_top.append(None)
                continue
            d = dist(ids[i - 1], k, Fake.mode)
            in_tok.append(ent(ids[i], dict(d).get(ids[i], -9.0), sh))
            in_top.append([ent(t, lp, sh) for t, lp in d])
        out, out_tok, out_top, prev = [], [], [], ids[-1]
        for _ in range(sp["max_new_tokens"]):
            d = dist(prev, k, Fake.mode)
            out.append(d[0][0])
            out_tok.append(ent(d[0][0], d[0][1], sh))
            out_top.append([ent(t, lp, sh) for t, lp in d])
            prev = d[0][0]
        cached = 5 if (Fake.mode == "cached" and start == 0) else 0
        return {"text": "x", "output_ids": out, "meta_info": {
            "prompt_tokens": len(ids), "cached_tokens": cached, "finish_reason": {"type": "length"},
            "input_token_logprobs": in_tok, "input_top_logprobs": in_top,
            "output_token_logprobs": out_tok, "output_top_logprobs": out_top}}

    def chat_reply(self, body):
        prompt = body["messages"][0]["content"]
        kind = KIND[prompt]
        mode = Fake.chat.get(kind, "clean")
        rnd = random.Random(len(prompt) + (body.get("seed") or 0))
        words = "river bridge ferry council budget risk schedule traffic data cache token model plan".split()
        text = " ".join(rnd.choice(words) + str(rnd.randint(0, 99)) for _ in range(90))
        if kind in ("json", "schema"):
            text = json.dumps({"items": [{"name": "tent", "qty": 1}]})
        finish, matched = "stop", None
        if mode == "salad":
            text += " 这是乱码文字混入"
        elif mode == "loop":
            text += " and then the light went out." * 20
            finish = "length"
        elif mode == "repetition":
            finish, matched = "stop", "repetition"
        elif mode == "badjson":
            text = '{"items": [{"name": "tent", "qty": 1}'
        return {"choices": [{"index": 0, "finish_reason": finish, "matched_stop": matched,
                             "message": {"role": "assistant", "content": text, "reasoning_content": "thinking " * 5}}],
                "usage": {"completion_tokens": 123}}


class DsGateCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        P.BASE = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Fake.mode, Fake.shape, Fake.chat, Fake.calls = "clean", "list", {}, []

    def run_quiet(self, fn, a):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fn(a)
        return rc, buf.getvalue()

    def collect(self, name, panel="long", texts=None, k=4, gen=6, tf=True):
        a = SimpleNamespace(panel=panel, out=os.path.join(self.tmp, name + ".json"), k=k, gen=gen, tf=tf,
                            texts=texts or os.path.join(HERE, "kld_long_texts.json"))
        rc, out = self.run_quiet(P.kld_collect, a)
        self.assertEqual(rc, 0)
        with open(a.out) as f:
            return a.out, json.load(f), out

    def compare(self, ref, cand, max_kl=None):
        return self.run_quiet(P.kld_compare, SimpleNamespace(ref=ref, cand=cand, max_kl=max_kl))

    def small_texts(self):
        path = os.path.join(self.tmp, "small.json")
        with open(path, "w") as f:
            json.dump([{"id": "a", "kind": "prose", "tokens": 50, "text": "The quick brown fox jumps. " * 3},
                       {"id": "b", "kind": "code", "tokens": 77, "text": "def f(x):\n    return x * 2\n" * 2}], f)
        return path

    # ---------------------------------------------------------------- math
    def test_kl_hand_example(self):
        p = {1: math.log(0.5), 2: math.log(0.3)}
        q = {1: math.log(0.4), 2: math.log(0.4)}
        self.assertAlmostEqual(P.kl_top(p, q), 0.5 * math.log(0.5 / 0.4) + 0.3 * math.log(0.3 / 0.4), places=12)
        self.assertEqual(P.kl_top(p, p), 0.0)
        # token 2 missing from q's top-k gets q's floor (0.2); tail buckets 0.1 vs 0.3
        q2 = {1: math.log(0.5), 3: math.log(0.2)}
        want = 0.5 * math.log(0.5 / 0.5) + 0.3 * math.log(0.3 / 0.2) + 0.2 * math.log(0.2 / 0.3)
        self.assertAlmostEqual(P.kl_top(p, q2), want, places=12)
        p3 = {1: math.log(0.6), 2: math.log(0.3)}
        want3 = 0.6 * math.log(0.6 / 0.5) + 0.3 * math.log(0.3 / 0.2) + 0.1 * math.log(0.1 / 0.3)
        self.assertAlmostEqual(P.kl_top(p3, q2), want3, places=12)
        self.assertIsNone(P.kl_top(None, q))
        self.assertIsNone(P.kl_top(p, {}))

    def test_top_dict_shapes(self):
        want = {7: -0.1, 9: -2.5}
        self.assertEqual(P.top_dict([[-0.1, 7, "a"], [-2.5, 9, None]]), want)
        self.assertEqual(P.top_dict([{"logprob": -0.1, "token_id": 7, "token": "a"}, {"logprob": -2.5, "token_id": 9}]), want)
        self.assertEqual(P.top_dict({"7": -0.1, "9": {"logprob": -2.5}}), want)
        self.assertIsNone(P.top_dict(None))
        self.assertIsNone(P.top_dict([]))
        self.assertIsNone(P.top_dict([[None, 7, None]]))
        self.assertEqual(P.entry([None, 3, None]), (None, 3))

    # ---------------------------------------------------------------- compare thresholds
    def write_panel(self, name, pn, p, q=None):
        rec = {"id": "t", "kind": "x", "n_prompt": 3, "prompt_lp": [None, p, q or p], "prompt_tok_lp": [None, -1.0, -2.0],
               "gen_tokens": [1, 2], "gen_top": [p, p], "tf_cached": 0}
        path = os.path.join(self.tmp, name + ".json")
        with open(path, "w") as f:
            json.dump({"panel": pn, "items": [rec]}, f)
        return path

    def test_compare_thresholds(self):
        p = {"1": math.log(0.5), "2": math.log(0.3)}
        q = {"1": math.log(0.4), "2": math.log(0.4)}
        kl = P.kl_top(p, q)
        ref = self.write_panel("r", "short", p)
        cand = self.write_panel("c", "short", q, q)  # position 1 and 2: KL(p||q) and KL(p||q)... p fixed in ref
        rc, out = self.compare(ref, cand, max_kl=kl + 1e-6)
        self.assertEqual(rc, 0, out)
        self.assertIn(f"GATE kld_short {kl:.5f} PASS 2", out)
        self.assertIn("GATE kld_short_agree", out)
        self.assertIn("GATE kld_short_cont", out)
        rc, out = self.compare(ref, cand, max_kl=kl - 1e-6)
        self.assertEqual(rc, 1)
        self.assertIn(f"GATE kld_short {kl:.5f} FAIL 2", out)
        # default threshold 0.035: this KL (~0.025) passes, a flatter candidate fails
        rc, _ = self.compare(ref, cand)
        self.assertEqual((rc, kl < 0.035), (0, True))
        flat = {"1": math.log(0.2), "2": math.log(0.2)}
        rc, out = self.compare(ref, self.write_panel("f", "short", flat, flat))
        self.assertEqual(rc, 1, out)
        # top-1 disagreement leaves the agree-only subset empty
        swap = {"1": math.log(0.3), "2": math.log(0.5)}
        rc, out = self.compare(ref, self.write_panel("s", "short", swap, swap), max_kl=10)
        self.assertIn("GATE kld_short_agree nan info 0", out)
        self.assertIn("top-1 0.00 %", out)
        # panel mismatch always fails
        rc, out = self.compare(ref, self.write_panel("l", "long", p))
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCHED PANEL", out)
        self.assertIn("GATE kld_short nan FAIL 0", out)

    # ---------------------------------------------------------------- collect via fake /generate
    def test_collect_list_and_dict_shapes(self):
        texts = self.small_texts()
        ra, A, out = self.collect("shape-list", texts=texts)
        self.assertIn("WARN b: only 54 of 77 tokens available", out)
        a, b = A["items"]
        self.assertEqual((a["n_prompt"], b["n_prompt"]), (50, 54))
        gens = [body for path, body in Fake.calls if path == "/generate"]
        self.assertEqual(len(gens), 4)
        want = [ord(c) for c in "The quick brown fox jumps. " * 3][:50]
        self.assertEqual(gens[0]["input_ids"], want)
        self.assertEqual(gens[1]["input_ids"], want)
        self.assertEqual([g.get("logprob_start_len") for g in gens], [None, 0, None, 0])
        self.assertTrue(gens[0]["cache_salt"] and gens[0]["cache_salt"] != gens[2]["cache_salt"])
        self.assertEqual([g["sampling_params"]["max_new_tokens"] for g in gens], [6, 1, 6, 1])
        self.assertTrue(all(g["return_logprob"] and g["top_logprobs_num"] == 4 and g["sampling_params"]["temperature"] == 0
                            and "model" not in g for g in gens))
        tok = [(p, b) for p, b in Fake.calls if p == "/tokenize"]
        self.assertTrue(all(b["add_special_tokens"] is False and b["model"] == P.MODEL for _, b in tok))
        self.assertEqual(len(a["prompt_lp"]), 50)
        self.assertIsNone(a["prompt_lp"][0])
        self.assertEqual(len(a["prompt_lp"][1]), 4)
        self.assertEqual(len(a["prompt_tok_lp"]), 50)
        self.assertEqual((len(a["gen_tokens"]), len(a["gen_top"])), (6, 6))
        self.assertEqual(a["tf_cached"], 0)

        Fake.shape = "dict"
        rd, D, _ = self.collect("shape-dict", texts=texts)
        self.assertEqual(json.dumps(A["items"]), json.dumps(D["items"]))
        rc, out = self.compare(ra, rd)
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE kld_long 0.00000 PASS 102", out)  # 49 + 53 scored positions
        self.assertIn("identical 6/6 tokens (full)", out)

        Fake.shape, Fake.mode = "list", "hot"
        rh, H, _ = self.collect("hot", texts=texts)
        rc, out = self.compare(ra, rh)
        self.assertEqual(rc, 1, out)
        self.assertRegex(out, r"GATE kld_long 0\.\d+ FAIL 102")
        self.assertIn("top-1 100.00 %", out)
        rc, _ = self.compare(ra, rh, max_kl=1.0)
        self.assertEqual(rc, 0)

        Fake.mode = "cached"
        rcached, Cd, out = self.collect("cached", texts=texts)
        self.assertEqual(Cd["items"][0]["tf_cached"], 5)
        self.assertIn("teacher-forced request reported cached_tokens 5", out)
        self.assertTrue(any("cached_tokens 5" in w for w in Cd["warnings"]))
        _, out = self.compare(ra, rcached)
        self.assertIn("WARN a: cand teacher-forced request had cached_tokens 5", out)

    def test_tokenize_cut_exact_counts(self):
        with open(os.path.join(HERE, "kld_long_texts.json")) as f:
            texts = {t["id"]: t["text"] for t in json.load(f)}
        _, L, out = self.collect("long", k=2, gen=2)
        self.assertEqual([i["n_prompt"] for i in L["items"]], [16500, 18000, 24003, 40001])
        self.assertEqual([len(i["prompt_lp"]) for i in L["items"]], [16500, 18000, 24003, 40001])
        self.assertNotIn("WARN", out)
        gens = [b for p, b in Fake.calls if p == "/generate"]
        self.assertEqual(gens[-1]["input_ids"], [ord(c) for c in texts["mixed-long"][:40001]])

        Fake.calls = []
        _, S, out = self.collect("short", panel="short", k=2, gen=2)
        self.assertEqual(len(S["items"]), 12)
        self.assertEqual(len([1 for p, _ in Fake.calls if p == "/tokenize"]), 6)  # one per distinct source text
        gens = [b for p, b in Fake.calls if p == "/generate"][1::2]
        src = dict(texts, polish=P.POLISH, json=P.json_doc())
        for (i, kind, s, off, n), rec, g in zip(P.SHORT, S["items"], gens):
            n = n or len(src[s])
            self.assertEqual((rec["id"], rec["n_prompt"], rec["offset"]), (i, n, off))
            self.assertEqual(g["input_ids"], [ord(c) for c in src[s][off:off + n]])
        self.assertNotIn("WARN", out)
        self.assertEqual(P.json_doc(), P.json_doc())  # deterministic across arms
        json.loads(P.json_doc())

    # ---------------------------------------------------------------- tscan
    def test_flag_row(self):
        clean = "Plain prose about bridges and ferries, " + " ".join(f"word{i}" for i in range(80))
        row = lambda **kw: dict({"kind": "prose", "finish": "stop", "matched_stop": None, "content": clean,
                                 "reasoning": ""}, **kw)
        r = row()
        self.assertEqual(P.flag_row(r), (False, False))
        self.assertEqual(r["flags"], [])
        r = row(content=clean + "这是乱码文字")
        self.assertEqual(P.flag_row(r), (True, False))
        self.assertEqual(r["flags"], ["SALAD"])
        r = row(content=clean + "���")
        self.assertIn("SALAD", (P.flag_row(r), r["flags"])[1])
        r = row(content=clean + " the end." * 20, finish="length")
        self.assertEqual(P.flag_row(r), (True, False))
        self.assertEqual(r["flags"], ["LOOP"])
        r = row(content="short", matched_stop="repetition")
        self.assertEqual(P.flag_row(r), (True, False))
        self.assertEqual(r["flags"], ["LOOP"])
        r = row(kind="json", content='{"a": 1')
        self.assertEqual(P.flag_row(r), (False, True))
        self.assertEqual(r["flags"], ["BADJSON"])
        r = row(kind="json", content='```json\n{"a": 1}\n```')
        self.assertEqual(P.flag_row(r), (False, False))
        r = row(kind="json", content='{"a": 1', finish="length")  # truncated: not judged
        self.assertEqual(P.flag_row(r), (False, False))

    def tscan(self, label, thinking="on"):
        a = SimpleNamespace(label=label, max_tokens=64, thinking=thinking, out_dir=os.path.join(self.tmp, label))
        rc, out = self.run_quiet(P.tscan, a)
        with open(os.path.join(a.out_dir, "tscan.jsonl")) as f:
            return rc, [json.loads(x) for x in f], out

    def test_tscan_clean_and_bodies(self):
        rc, rows, out = self.tscan("clean")
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(rows), 35)
        self.assertTrue(all(r["flags"] == [] for r in rows), [r["flags"] for r in rows])
        self.assertIn("TSCAN: 0 salad of 35 outputs (errors 0, bad json 0)", out)
        chats = [b for p, b in Fake.calls if p == "/v1/chat/completions"]
        self.assertEqual(len(chats), 35)
        self.assertTrue(all(b["model"] == P.MODEL and b["chat_template_kwargs"] == {"thinking": True} for b in chats))
        self.assertEqual(chats[2]["reasoning_effort"], "high")
        self.assertEqual(chats[0]["reasoning_effort"], "low")
        self.assertEqual([b.get("seed") for b in chats[:11]], [20260929 + i for i in range(11)])
        self.assertTrue(all("seed" not in b for b in chats[11:22]))
        self.assertIn("response_format", chats[10])
        self.assertEqual(sum(b["temperature"] == 0.0 for b in chats[22:]), 2)
        self.assertEqual(sum(r["phase"] == "c4" for r in rows), 13)
        Fake.calls = []
        self.tscan("off", thinking="off")
        chats = [b for p, b in Fake.calls if p == "/v1/chat/completions"]
        self.assertTrue(all(b["chat_template_kwargs"] == {"thinking": False} and "reasoning_effort" not in b for b in chats))

    def test_tscan_flags(self):
        Fake.chat = {"prose": "salad", "story": "loop", "think": "repetition", "json": "badjson"}
        rc, rows, out = self.tscan("bad")
        self.assertEqual(rc, 1)
        by = lambda kind: {tuple(r["flags"]) for r in rows if r["kind"] == kind}
        self.assertEqual(by("prose"), {("SALAD",)})
        self.assertEqual(by("story"), {("LOOP",)})
        self.assertEqual(by("think"), {("LOOP",)})
        self.assertEqual(by("json"), {("BADJSON",)})
        self.assertEqual(by("code"), {()})
        n_prose, n_story, n_think, n_json = (sum(r["kind"] == k for r in rows) for k in ("prose", "story", "think", "json"))
        self.assertIn(f"TSCAN: {n_prose + n_story + n_think} salad of 35 outputs (errors 0, bad json {n_json})", out)
        self.assertIn("(repetition) LOOP", out)
        Fake.chat = {"think": "namedexit"}
        rc, rows, out = self.tscan("named-exit")
        self.assertEqual(rc, 1)
        self.assertEqual({tuple(r["flags"]) for r in rows if r["kind"] == "think"}, {("LOOP",)})
        self.assertIn(f"TSCAN: {n_think} salad of 35 outputs (errors 0, bad json 0)", out)
        self.assertIn("(loop_abort) LOOP", out)
        Fake.chat = {"json": "badjson"}
        rc, _, out = self.tscan("badjson-only")
        self.assertEqual(rc, 0, out)  # BADJSON is a warning only


def _test_cont_only_gate(self):
    texts = os.path.join(self.tmp, "t2.json")
    with open(texts, "w") as f:
        json.dump([{"id": "a", "kind": "prose", "text": "x" * 3000, "tokens": 2000}], f)
    _, A, _ = self.collect("c1", texts=texts, tf=False)
    _, B, _ = self.collect("c2", texts=texts, tf=False)
    self.assertEqual(A["items"][0]["prompt_lp"], [])
    a = SimpleNamespace(ref=os.path.join(self.tmp, "c1.json"), cand=os.path.join(self.tmp, "c2.json"), max_kl=None)
    rc, out = self.run_quiet(P.kld_compare, a)
    self.assertEqual(rc, 0, out)
    self.assertIn("_cont", out.split("GATE")[-1])
    gens = [body for path, body in Fake.calls if path == "/generate"]
    self.assertTrue(gens and all("logprob_start_len" not in g for g in gens))



DsGateCPU.test_cont_only_gate = _test_cont_only_gate

if __name__ == "__main__":
    unittest.main()
