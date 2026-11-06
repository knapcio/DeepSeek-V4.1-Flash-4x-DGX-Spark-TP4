#!/usr/bin/env python3
"""Shared-long-prefix scan: concurrency x prefix-cache correctness gate (stdlib only).

DeepSeek-V4.1 / SGLang port (2026-09-29) of the GLM-5.3 gate. DS differences: SGLang's UnifiedRadixCache
(full + sliding-window components) with compressed attention and Engram; cache hits are read from
usage.prompt_tokens_details.cached_tokens (--enable-cache-report) or the sglang:cached_tokens_total counter;
cache_salt namespaces the radix tree; loop-aborted requests (an HTTP error named "loop_abort: ...", or matched_stop
"repetition" with DSV41_LOOP_NAMED_EXIT=0) count as degenerate.
There is no recurrent-state block, so instead of one block phase the drift test spreads its repetitions over
prompt lengths (P mod 256 differs per repetition), and a second drift kind checks a PARTIAL hit: the warm request
shares only the long prefix with an earlier request (next question on the same document), so the hit ends
mid-prompt at an arbitrary position.

Why: quality gates at c1/c4 with fresh prompts never read state back out of the prefix cache under concurrency.
This bench sends one long synthetic prefix (default 3050 records, ~99k tokens) to N concurrent requests with distinct short
tasks, first on a cold prefix (a fresh ``cache_salt``: the first request computes it, the others read it from the
prefix cache while or after it is written) and then warm (same salt: every request hits), and verifies every answer
against ground truth.

Prefix: LINES records "Record NNNNN host=node-HH state=S retries=R at t=Ts ref=CODE; lease held then released.",
where CODE is a random 5-letter code (seeded). The codes cannot be derived from a pattern, so a correct quote proves
the model read that record.

Tasks (8): ref-code lookups spread over the prefix (including its last records), verbatim quotes of records, and
an anomaly scan whose only correct answer is NONE (the prefix has no malformed records).

Checks per response:
  degenerate   longest run of one identical consecutive token >= RUN_FLAG (8) or a periodic loop (period 2..16,
               >= 4 repeats) spanning >= LOOP_FLAG (48) tokens; non-finite logprobs when requested
  quotes       every quoted full record must exist verbatim; every "Record N ... ref=CODE" / "Record N: CODE" must
               carry N's code
  anomaly      the scan task must answer NONE; any listed record is a false claim
Drift test (T=0, thinking off, c1, logprobs), two kinds, each repetition on its own prompt length:
  full     a cold request (salt A), a second cold request (salt B) and a warm request (salt A): the warm request
           repeats the whole prompt, so the hit ends near its end (page-aligned by the cache);
  partial  salt A is first seeded with the first --partial-frac (0.6) of the records plus a DIFFERENT question
           (1 output token); the warm request (whole prefix + drift question, salt A) then hits only up to where the
           two prompts diverge, mid-document, and computes the remaining ~40 % on top of the cached state;
           cold A' and cold B use fresh salts.
The mean |logprob difference| over the common token prefix, cold-vs-warm against cold-vs-cold, measures whether a
cache hit restores the state the cold path had. The hit size is read per request (cached_tokens).
GLM issue #2 (KDA checkpoint 1152 tokens stale): cold-vs-warm 0.25-0.36 against a 0.04-0.09 floor; fixed: 0.02-0.04.
The drift prompt (--drift-task quote, default) asks for five verbatim records spread over the prefix, so the greedy
answer is ~170 tokens that depend on the cached state; GLM's scan prompt answers "NONE" (3 tokens on DS).
Repetition j adds j * --drift-step records (default 7, ~230 tokens), so the prompt length and the hit position
move through different residues mod 128 / 256 (SWA window, compression ratio, page).

Gate (exit 0 = PASS). It tests the cache, not the model: GLM misreads near-duplicate records and is not
reproducible at T=0 within a boot, cold as well as warm, so absolute correctness and cold/warm identity are reported
but not required.
  (a) zero degenerate outputs;
  (b) warm incorrect / misquoting / false-anomaly counts each <= cold + 2 (24 responses per phase by default);
  (c) per drift kind: median cold-vs-warm drift <= max(1.5 x median cold-vs-cold, 0.06), every warm drift request
      hitting the cache (cached_tokens > 0), >= 3 repetitions; also the worst single repetition <= 2 x that limit;
  (d) no warm-only scan anomalies: a record the warm scan flags that neither cold scan flagged.
GLM issue #2 without the fix: drift 0.25-0.36 vs 0.065 and warm-only anomalies 3/3 (fail); with the fix: 0.02-0.04
vs 0.044, none (pass).

  python3 scripts/prefix_scan.py run LABEL [--base http://127.0.0.1:8888 --conc 8 --rounds 3 --temp 0.8 --top-p 0.95
      --max-tokens 1200 --lines 3050] --out prefix-scan-LABEL.json
  python3 scripts/prefix_scan.py score FILE      re-score a saved result

Origin: GLM-5.3-Flash-4x-DGX-Spark-TP4 issue #2 (KDA checkpoint 1152 tokens stale under prefix caching).
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

RUN_FLAG = 8
LOOP_FLAG = 48
TARGET_LO, TARGET_HI = 300, 900   # P mod block for every prompt (see the drift-test note)
TOLERANCE = 2                      # (b) warm error counts may exceed cold by at most this (24 responses per phase)
DRIFT_RATIO, DRIFT_FLOOR = 1.5, 0.06   # (c) cold-vs-warm drift limit = max(1.5 x cold-vs-cold, 0.06)
ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"
STATES = ("idle", "dead", "drained", "leased", "stale")
FMT = "Record %05d host=node-%02d state=%s retries=%d at t=%ds ref=%s; lease held then released."


# --------------------------------------------------------------------------------------------------- prefix + tasks
class Prefix:
    def __init__(self, lines=3050, seed=7):
        rnd = random.Random(seed)
        self.n = lines
        self.codes = ["".join(rnd.choice(ALPHA) for _ in range(5)) for _ in range(lines)]
        self.lines = [FMT % (i, i % 29, STATES[rnd.randrange(len(STATES))], rnd.randrange(6), 7 * i, self.codes[i])
                      for i in range(lines)]
        self.lineset = set(self.lines)
        self.text = "\n".join(self.lines)

    def tasks(self):
        n = self.n
        pick = lambda f: max(0, min(n - 1, int(f * n)))  # noqa: E731
        look = [(pick(0.10), pick(0.35)), (pick(0.60), pick(0.85)), (n - 60, n - 25), (n - 40, n - 8)]
        t = []
        for a, b in look:
            t.append(("lookup", (a, b), "Give the ref code of Record %05d and of Record %05d, one per line exactly as "
                      "'Record N: CODE'. Output only those two lines." % (a, b)))
        t.append(("quote", (pick(0.5), pick(0.5) + 1), "Quote verbatim, exactly as they appear, the lines for Record %05d "
                  "and Record %05d, one per line." % (pick(0.5), pick(0.5) + 1)))
        t.append(("quote", (n - 3, n - 2, n - 1), "Quote verbatim the last three lines of the log, one per line."))
        scan = ("Scan every line of the log for anything malformed, truncated, merged with another line, out of order "
                "or missing. If every record is well-formed and none is missing, answer exactly: NONE. Otherwise answer "
                "with one line per affected record, exactly as 'Record N'.")
        t.append(("scan", (), scan))
        t.append(("scan", (), "Before answering, check the record numbers are consecutive from 00000 to %05d. " % (n - 1)
                  + scan))
        return t


# ------------------------------------------------------------------------------------------------------------ http
class Client:
    def __init__(self, base, model):
        self.base, self.model = base.rstrip("/"), model

    def post(self, path, body, timeout=3600):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def metrics(self):
        try:
            with urllib.request.urlopen(self.base + "/metrics", timeout=30) as r:
                txt = r.read().decode()
        except Exception:  # noqa: BLE001
            return {}
        out = {}
        for ln in txt.splitlines():
            m = re.match(r"(sglang:(?:cached_tokens|prompt_tokens|generation_tokens))(?:_total)?"
                         r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", ln)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
        return out

    def chat(self, content, a, salt, temp=None, thinking=None, max_tokens=None, logprobs=False):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens or a.max_tokens, "temperature": a.temp if temp is None else temp,
                "top_p": a.top_p if (temp is None or temp > 0) else 1.0, "return_token_ids": True, "cache_salt": salt}
        if (thinking or a.thinking) == "off":
            body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
        if a.logprobs or logprobs:
            body["logprobs"] = True
        t0 = time.time()
        try:
            r = self.post("/v1/chat/completions", body)
        except urllib.error.HTTPError as exc:
            text = exc.read()[:300]
            if b"loop_abort:" in text:
                # The server's loop abort: a degenerate output, not a request error.
                return {"finish": "loop_abort", "matched_stop": None, "usage": {}, "s": round(time.time() - t0, 2),
                        "reasoning": "", "content": "", "token_ids": [], "cached": None,
                        "loop_abort": text.decode(errors="replace")}
            return {"error": f"HTTP {exc.code}: {text!r}", "s": round(time.time() - t0, 2)}
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)[:300], "s": round(time.time() - t0, 2)}
        ch = r["choices"][0]
        msg = ch["message"]
        usage = r.get("usage") or {}
        rec = {"finish": ch.get("finish_reason"), "matched_stop": ch.get("matched_stop"), "usage": usage,
               "s": round(time.time() - t0, 2),
               "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "", "content": msg.get("content") or "",
               "token_ids": ch.get("token_ids") or ch.get("response_token_ids"),
               "cached": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))}
        if not rec["token_ids"]:
            try:
                rec["token_ids"] = self.post("/tokenize", {"model": self.model, "prompt": rec["reasoning"] + rec["content"],
                                                           "add_special_tokens": False})["tokens"]
                if isinstance(rec["token_ids"], list) and rec["token_ids"] and isinstance(rec["token_ids"][0], list):
                    rec["token_ids"] = rec["token_ids"][0]
                rec["ids_from"] = "tokenize"
            except Exception:  # noqa: BLE001
                rec["token_ids"] = []
        if ch.get("logprobs"):
            lps = [c.get("logprob") for c in (ch["logprobs"].get("content") or [])]
            rec["lps"] = lps
            rec["lp_tokens"] = [c.get("token") for c in (ch["logprobs"].get("content") or [])]
            rec["lp_bad"] = sum(1 for v in lps if not isinstance(v, (int, float)) or v != v or abs(v) == float("inf"))
        return rec


# ---------------------------------------------------------------------------------------------------------- scoring
def longest_run(ids):
    best = cur = 0
    prev = object()
    for t in ids:
        cur = cur + 1 if t == prev else 1
        prev = t
        best = max(best, cur)
    return best


def longest_loop(ids, pmin=2, pmax=16, min_reps=4):
    best = 0
    for p in range(pmin, pmax + 1):
        run = 0
        for i in range(p, len(ids)):
            run = run + 1 if ids[i] == ids[i - p] else 0
            if run + p >= p * min_reps:
                best = max(best, run + p)
    return best


def drift(x, y):
    """(common-prefix tokens, mean |dlogprob| over it, at least the first token)."""
    if x.get("error") or y.get("error") or not x.get("lps") or not y.get("lps"):
        return None
    a, b = x.get("token_ids") or x.get("lp_tokens") or [], y.get("token_ids") or y.get("lp_tokens") or []
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    k = max(n, 1)
    d = [abs(p - q) for p, q in zip(x["lps"][:k], y["lps"][:k]) if isinstance(p, (int, float)) and isinstance(q, (int, float))]
    return [n, round(sum(d) / len(d), 4) if d else None]


def parse_lookup(text):
    return {int(n): c for n, c in re.findall(r"Record\s*(\d{1,5})\s*[:=\-]?\s*(?:ref=)?\s*([A-Z]{5})\b", text)}


def parse_scan(text):
    body = text.strip()
    if re.fullmatch(r"\**NONE\.?\**", body, re.I):
        return "NONE"
    recs = sorted({int(n) for n in re.findall(r"Record\s*(\d{1,5})", body)})
    return recs if recs else "UNPARSED:" + body[:80]


def score(pfx, kind, target, rec):
    if "error" in rec:
        return {"error": rec["error"], "degenerate": False, "misquotes": [], "false_anomaly": None}
    ids = rec.get("token_ids") or []
    run, loop = longest_run(ids), longest_loop(ids)
    content = rec.get("content", "")
    text = rec.get("reasoning", "") + "\n" + content
    mis = []
    for m in re.finditer(r"Record \d{5} host=[^\n]*?released\.", content):
        if m.group(0) not in pfx.lineset:
            mis.append(m.group(0)[:200])
    for n, code in parse_lookup(content).items():
        if n < pfx.n and code != pfx.codes[n]:
            mis.append(f"Record {n:05d}: {code} (truth {pfx.codes[n]})")
    aborted = rec.get("matched_stop") == "repetition" or rec.get("finish") == "loop_abort"
    out = {"ntok": len(ids), "run": run, "loop": loop, "loop_abort": aborted,
           "degenerate": run >= RUN_FLAG or loop >= LOOP_FLAG or bool(rec.get("lp_bad")) or aborted,
           "lp_bad": rec.get("lp_bad"), "finish": rec.get("finish"), "misquotes": mis[:8], "false_anomaly": None}
    if kind == "lookup":
        got = parse_lookup(content)
        out["answer"] = {str(k): got.get(k) for k in target}
        out["correct"] = all(got.get(k) == pfx.codes[k] for k in target)
    elif kind == "quote":
        out["correct"] = all(pfx.lines[k] in content for k in target)
    elif kind == "scan":
        ans = parse_scan(content)
        out["answer"] = ans
        out["correct"] = ans == "NONE"
        out["false_anomaly"] = ans if isinstance(ans, list) else None
    return out


# -------------------------------------------------------------------------------------------------------------- run
def prompt_tokens(cl, text, thinking):
    body = {"model": cl.model, "messages": [{"role": "user", "content": text}], "add_generation_prompt": True}
    if thinking == "off":
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    c = cl.post("/tokenize", body)["count"]
    return c[0] if isinstance(c, list) else c


def size_prefix(cl, a):
    """Choose the record count so every prompt (all tasks, both thinking modes) sits at P mod block in
    [TARGET_LO, TARGET_HI]. Returns (Prefix, sizing note)."""
    if a.lines:
        return Prefix(a.lines, a.seed), "fixed --lines"
    lines = 3050
    for _ in range(16):
        pfx = Prefix(lines, a.seed)
        try:
            ps = [prompt_tokens(cl, pfx.text + "\n\n" + q, th) for _, _, q in pfx.tasks() for th in ("off", a.thinking)]
        except Exception as exc:  # noqa: BLE001
            return Prefix(3050, a.seed), f"/tokenize failed ({exc!r:.80}); 3050 records, block phase unchecked"
        rs = [p % a.block for p in ps]
        if TARGET_LO <= min(rs) and max(rs) <= TARGET_HI:
            return pfx, f"{lines} records, P {min(ps)}-{max(ps)}, P mod {a.block} {min(rs)}-{max(rs)}"
        mid = (TARGET_LO + TARGET_HI) // 2
        shift = (mid - (min(rs) + max(rs)) // 2) % a.block
        if shift > a.block // 2:
            shift -= a.block
        lines += max(1, round(abs(shift) / 32.5)) * (1 if shift > 0 else -1)
    return pfx, f"{lines} records, block phase not reached ({min(rs)}-{max(rs)})"


def run(a):
    cl = Client(a.base, a.model)
    pfx, sizing = size_prefix(cl, a)
    print(f"[{a.label}] prefix: {sizing}", flush=True)
    tasks = pfx.tasks()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    res = {"label": a.label, "started": stamp, "args": vars(a), "rounds": [], "solo": [], "drift": [],
           "prefix": {"lines": pfx.n, "sizing": sizing}}

    def fire(salt, phase, rnd):
        m0 = cl.metrics()
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.conc) as ex:
            futs = [ex.submit(cl.chat, pfx.text + "\n\n" + tasks[i % len(tasks)][2], a, salt) for i in range(a.conc)]
            recs = [f.result() for f in futs]
        m1 = cl.metrics()
        out = []
        for i, rec in enumerate(recs):
            kind, target, _ = tasks[i % len(tasks)]
            sc = score(pfx, kind, target, rec)
            out.append({"i": i, "kind": kind, "score": sc, "content": rec.get("content", "")[:4000],
                        "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens"), "cached": rec.get("cached")})
            print(f"[{a.label} r{rnd} {phase:4s} #{i} {kind:6s}] ntok={sc.get('ntok')} run={sc.get('run')} loop={sc.get('loop')} "
                  f"correct={sc.get('correct')} misquotes={len(sc.get('misquotes') or [])} cached={rec.get('cached')} "
                  f"{'DEGENERATE' if sc.get('degenerate') else ''}{' ERROR ' + sc['error'] if sc.get('error') else ''}", flush=True)
        dm = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        return {"round": rnd, "phase": phase, "salt": salt, "wall_s": round(time.time() - t0, 1), "metrics_delta": dm,
                "responses": out}

    for r in range(a.rounds):
        salt = f"prefix-scan-{a.label}-{stamp}-r{r}"
        res["rounds"].append(fire(salt, "cold", r))
        res["rounds"].append(fire(salt, "warm", r))

    # Drift test (full + partial hits), each repetition on its own prompt length
    def hits(rec, m0):
        if rec.get("cached") is not None:
            return float(rec["cached"])
        return cl.metrics().get("sglang:cached_tokens", 0) - m0.get("sglang:cached_tokens", 0)

    for kind in a.drift_kinds.split(","):
        for j in range(a.drift_reps):
            dp = Prefix(pfx.n + a.drift_step * (j + 1), a.seed)
            dt = dp.tasks()
            if a.drift_task == "scan":
                scan_q = [t for t in dt if t[0] == "scan"][0][2]
            else:  # a long faithful answer: five verbatim records spread over the prefix (~170 tokens)
                recs5 = [max(0, min(dp.n - 1, int(f * dp.n))) for f in (0.05, 0.3, 0.55, 0.8)] + [dp.n - 2]
                scan_q = ("Quote verbatim, exactly as they appear, the lines for Record " +
                          ", ".join("%05d" % r for r in recs5[:-1]) + " and Record %05d, one per line." % recs5[-1])
            other_q = [t for t in dt if t[0] == "lookup"][0][2]
            base = f"prefix-scan-{a.label}-{stamp}-{kind}{j}"
            recs = {}
            if kind == "partial":
                cut = "\n".join(dp.lines[:int(a.partial_frac * dp.n)])  # hit ends mid-document
                seed = cl.chat(cut + "\n\n" + other_q, a, base + "-a", temp=0.0, thinking="off", max_tokens=1)
                recs["seed"] = seed
            for name, salt in (("coldA", base + ("-c" if kind == "partial" else "-a")), ("coldB", base + "-b"),
                               ("warmA", base + "-a")):
                m0 = cl.metrics()
                recs[name] = cl.chat(dp.text + "\n\n" + scan_q, a, salt, temp=0.0, thinking="off",
                                     max_tokens=a.drift_tokens, logprobs=True)
                recs[name]["hits"] = hits(recs[name], m0)
            cc, cw = drift(recs["coldA"], recs["coldB"]), drift(recs["coldA"], recs["warmA"])
            ptok = (recs["warmA"].get("usage") or {}).get("prompt_tokens")
            res["drift"].append({"kind": kind, "rep": j, "lines": dp.n, "cc": cc, "cw": cw, "warm_hits": recs["warmA"]["hits"],
                                 "cold_hits": [recs["coldA"]["hits"], recs["coldB"]["hits"]], "prompt_tokens": ptok,
                                 "texts": {k: v.get("content", "")[:300] for k, v in recs.items() if k != "seed"}})
            print(f"[{a.label} drift {kind} {j}] P={ptok} cold-cold {cc} cold-warm {cw} warm hit {recs['warmA']['hits']:.0f} "
                  f"cold hits {recs['coldA']['hits']:.0f}/{recs['coldB']['hits']:.0f}", flush=True)
    for j, (kind, target, q) in enumerate([t for t in tasks if t[0] in ("lookup", "scan")]):
        salt = f"prefix-scan-{a.label}-{stamp}-solo{j}"
        pair = {}
        for phase in ("cold", "warm"):
            rec = cl.chat(pfx.text + "\n\n" + q, a, salt, temp=0.0, thinking="off", max_tokens=a.solo_max_tokens)
            sc = score(pfx, kind, target, rec)
            pair[phase] = {"score": sc, "content": rec.get("content", "")[:2000], "error": rec.get("error")}
        res["solo"].append({"kind": kind, "target": list(target), **pair})
        print(f"[{a.label} solo {kind:6s}] cold={pair['cold']['score'].get('answer')} ({pair['cold']['score'].get('correct')}) "
              f"warm={pair['warm']['score'].get('answer')} ({pair['warm']['score'].get('correct')})", flush=True)

    res["summary"] = summarize(res, a.strict)
    print(json.dumps(res["summary"], indent=1), flush=True)
    with open(a.out, "x") as fo:
        json.dump(res, fo, indent=1)
    print(f"wrote {a.out}", flush=True)
    return 0 if res["summary"]["verdict"] == "PASS" else 1


def summarize(res, strict=False):
    rs = [x for rnd in res["rounds"] for x in rnd["responses"]]
    by_phase = {}
    for rnd in res["rounds"]:
        d = by_phase.setdefault(rnd["phase"], {"responses": 0, "degenerate": 0, "errors": 0, "misquotes": 0,
                                               "false_anomaly": 0, "scan": 0, "incorrect": 0})
        for x in rnd["responses"]:
            sc = x["score"]
            d["responses"] += 1
            d["degenerate"] += bool(sc.get("degenerate"))
            d["errors"] += bool(sc.get("error"))
            d["misquotes"] += bool(sc.get("misquotes"))
            d["incorrect"] += sc.get("correct") is False
            if x["kind"] == "scan":
                d["scan"] += 1
                d["false_anomaly"] += sc.get("false_anomaly") is not None
    degenerate = sum(bool(x["score"].get("degenerate")) for x in rs)
    errors = sum(bool(x["score"].get("error")) for x in rs)
    reasons = []
    if degenerate:
        reasons.append(f"{degenerate} degenerate output(s)")
    if errors:
        reasons.append(f"{errors} request error(s)")
    # (b) the cache must not make answers worse: warm counts within cold + TOLERANCE
    cold, warm = by_phase.get("cold", {}), by_phase.get("warm", {})
    for key in ("incorrect", "misquotes", "false_anomaly"):
        if warm.get(key, 0) > cold.get(key, 0) + TOLERANCE:
            reasons.append(f"warm {key} {warm.get(key, 0)} > cold {cold.get(key, 0)} + {TOLERANCE}")
    # (c) T=0 cold-vs-warm logprob drift within the cold-vs-cold floor, per kind; (d) no warm-only scan anomalies
    drift_summary = {}
    allr = [d for d in res.get("drift", []) if d.get("cc") and d.get("cw") and d["cc"][1] is not None and d["cw"][1] is not None]
    kinds = sorted({d.get("kind", "full") for d in res.get("drift", [])}) or ["full"]
    for kind in kinds:
        dr = [d for d in allr if d.get("kind", "full") == kind]
        if len(dr) < 3:
            reasons.append(f"drift test {kind} incomplete ({len(dr)} of >= 3 repetitions)")
        if not dr:
            continue
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        mcc, mcw = med([d["cc"][1] for d in dr]), med([d["cw"][1] for d in dr])
        worst = max(d["cw"][1] for d in dr)
        limit = max(DRIFT_RATIO * mcc, DRIFT_FLOOR)
        missed = sum(1 for d in dr if not d.get("warm_hits"))
        phantoms = []
        for d in dr:
            ans = {k: parse_scan(v) for k, v in d.get("texts", {}).items()}
            cold_recs = {n for k in ("coldA", "coldB") if isinstance(ans.get(k), list) for n in ans[k]}
            if isinstance(ans.get("warmA"), list) and set(ans["warmA"]) - cold_recs:
                phantoms.append(sorted(set(ans["warmA"]) - cold_recs))
        drift_summary[kind] = {"cold_cold": mcc, "cold_warm": mcw, "worst_cold_warm": worst, "limit": round(limit, 4),
                               "reps": len(dr), "warm_without_hit": missed, "warm_only_anomalies": phantoms,
                               "warm_hits": [d.get("warm_hits") for d in dr], "prompt_tokens": [d.get("prompt_tokens") for d in dr],
                               "identical_prefix_tokens": [d["cw"][0] for d in dr]}
        if missed:
            reasons.append(f"{kind}: {missed} warm drift request(s) did not hit the prefix cache")
        if mcw > limit:
            reasons.append(f"{kind}: cold-vs-warm drift {mcw} > {limit:.3f} (cold-vs-cold {mcc})")
        if worst > 2 * limit:
            reasons.append(f"{kind}: worst cold-vs-warm drift {worst} > {2 * limit:.3f}")
        if phantoms:
            reasons.append(f"{kind}: warm-only scan anomalies: {phantoms}")
    if strict:
        for ph, d in by_phase.items():
            if d["misquotes"] or d["false_anomaly"]:
                reasons.append(f"{ph}: {d['misquotes']} misquoting, {d['false_anomaly']} false-anomaly response(s)")
    examples = [x["score"]["misquotes"][0] for x in rs if x["score"].get("misquotes")][:5]
    examples += [f"scan -> {x['score']['false_anomaly']}" for x in rs if x["score"].get("false_anomaly")][:5]
    solo = [{"kind": x["kind"], "target": x["target"], "cold": x["cold"]["score"].get("correct"),
             "warm": x["warm"]["score"].get("correct")} for x in res.get("solo", [])]
    return {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons, "drift": drift_summary, "by_phase": by_phase,
            "degenerate": degenerate, "prefix": res.get("prefix"), "solo_correct": solo, "examples": examples}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("--base", default="http://127.0.0.1:8888")
    r.add_argument("--model", default="deepseek-v4.1-flash")
    r.add_argument("--conc", type=int, default=8)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--temp", type=float, default=0.8)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--max-tokens", type=int, default=1200)
    r.add_argument("--solo-max-tokens", type=int, default=200)
    r.add_argument("--thinking", default="default", choices=["default", "off"])
    r.add_argument("--lines", type=int, default=3050, help="records in the prefix (~32.5 tokens each, 3050 = ~99k tokens); "
                   "0 = size via /tokenize to P mod --block in [300, 900]")
    r.add_argument("--block", type=int, default=2304, help="state block for --lines 0 sizing (GLM KDA; unused on DS)")
    r.add_argument("--drift-reps", type=int, default=3)
    r.add_argument("--drift-kinds", default="full,partial")
    r.add_argument("--partial-frac", type=float, default=0.6)
    r.add_argument("--drift-step", type=int, default=7, help="records added per drift repetition")
    r.add_argument("--drift-tokens", type=int, default=128)
    r.add_argument("--drift-task", default="quote", choices=["quote", "scan"],
                   help="drift prompt: quote = five verbatim records (long exact answer, DS default); scan = GLM's NONE scan")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--logprobs", action="store_true", help="also flag non-finite token logprobs")
    r.add_argument("--strict", action="store_true", help="misquotes / false anomaly claims in sampled rounds also fail")
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("file")
    s.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "score":
        with open(a.file) as f:
            res = json.load(f)
        print(json.dumps(summarize(res, a.strict), indent=1))
        return 0
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists: choose a new output")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
