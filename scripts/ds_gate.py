#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash gate helpers for SGLang (port of the GLM final_bench kldlong / tscan). Stdlib only.
Run on the head (Spark_01) against the SGLang server (default --url http://127.0.0.1:8888, --model deepseek-v4.1-flash).

  kld collect --panel long|short --out F.json [--texts kld_long_texts.json] [--k 10] [--gen 256]
      Fidelity panel scored on identical input_ids in every arm (no chat template, no special tokens).
      long:  the four texts of kld_long_texts.json cut to exact token counts (16500 / 18000 / 24003 / 40001: several
             full prefill chunks, a short final chunk, row counts that are not a multiple of TP = 4).
      short: 12 texts of ~1-3k tokens: 10 token slices of the long texts at offsets > 0 (prose, code, mixed docs)
             plus an embedded Polish text (whole, ~1k tokens) and a seeded generated JSON document.
      Per text, via SGLang native POST /generate:
        1. continuation: greedy, --gen tokens, top-k output logprobs only, a fresh cache_salt (cold prefill);
        2. with --tf only, teacher-forced: max_new_tokens 1, logprob_start_len 0, top_logprobs_num k. The ds41 launch
           runs decoder SWA bounded replay, which cannot return prompt logprobs (adapter/replay_guard.py answers 400),
           so without --tf compare gates on the continuation positions up to and including the first divergence.
      Cold prefill assumption: SGLang caps a request's radix prefix match at its logprob_start_len when
      return_logprob is set, so request 2 recomputes every prompt position even though request 1 just cached them.
      meta_info.cached_tokens is recorded per request (tf_cached / cont_cached); collect and compare print a WARN when
      the teacher-forced request reports cached tokens > 0, which would mean the assumption does not hold on this build.
  kld compare REF.json CAND.json [--max-kl X]
      KL(ref || cand) over ref's top-k support plus one tail bucket, per text and pooled; top-1 agreement; the same
      numbers on the positions where ref and cand agree on top-1; mean NLL of the actual prompt tokens; last-2048
      positions (long panel); continuation KL up to and including the first divergence. Prints
        GATE kld_<panel> <mean> <PASS|FAIL> <positions>
        GATE kld_<panel>_agree <mean> info <positions>
        GATE kld_<panel>_cont <mean> info <positions>
      --max-kl defaults to 0.035 (both panels). Exit code 0 on PASS.
  tscan LABEL [--max-tokens 768] [--thinking on|off] [--out-dir DIR]
      T > 0 correctness scan on /v1/chat/completions: 11 prompts (prose / think / code / json / Polish / story /
      json_schema) at (T 1.0, top-p 0.95, seeded) and (T 0.6, top-p 0.95, unseeded) at c=1, then one c=4 pass mixing
      T=1 and T=0 requests. Thinking via chat_template_kwargs {"thinking": bool}; with thinking on, the GLM effort of
      each prompt maps to the top-level reasoning_effort (low -> low, high/max -> high). Flags: SALAD (>= 5 CJK
      characters or >= 3 U+FFFD), LOOP (tail is one unit of <= 200 chars repeated >= 6 times, or the server loop
      abort: an HTTP error whose message starts "loop_abort:", or with DSV41_LOOP_NAMED_EXIT=0 finish "stop" with
      matched_stop "repetition"), BADJSON (json tasks, warning only).
      Prints "TSCAN: <n> salad of <m> outputs ..."; rows go to DIR/tscan.jsonl (default ~/ds-gate/LABEL/).
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "deepseek-v4.1-flash"
HERE = os.path.dirname(os.path.abspath(__file__))
MAX_KL = {"long": 0.035, "short": 0.035}


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(text):
    err = None
    for body in ({"model": MODEL, "prompt": text, "add_special_tokens": False}, {"model": MODEL, "prompt": text}):
        try:
            r = post("/tokenize", body, timeout=300)
            if r.get("tokens"):
                return [int(x) for x in r["tokens"]]
        except Exception as exc:  # noqa: BLE001
            err = exc
    raise RuntimeError(f"/tokenize failed: {err!r}")


def generate(ids, max_new, k, start, salt=None):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
            "return_logprob": True, "top_logprobs_num": k}
    if start is not None:
        body["logprob_start_len"] = start
    if salt:
        body["cache_salt"] = salt
    r = post("/generate", body)
    return r[0] if isinstance(r, list) else r


# ------------------------------------------------------------------------------------------------ logprob parsing
def entry(e):
    """(logprob, token_id) from one SGLang logprob entry: [logprob, token_id, text] or {"logprob", "token_id"}."""
    if isinstance(e, dict):
        return e.get("logprob"), e.get("token_id", e.get("id"))
    if isinstance(e, (list, tuple)) and len(e) >= 2:
        return e[0], e[1]
    return None, None


def top_dict(x):
    """{token_id: logprob} for one position (list of entries or a {token_id: logprob} map); None when empty."""
    if not x:
        return None
    if isinstance(x, dict):
        pairs = ((k, v["logprob"] if isinstance(v, dict) else v) for k, v in x.items())
    else:
        pairs = ((tid, lp) for lp, tid in map(entry, x))
    d = {int(t): float(lp) for t, lp in pairs if t is not None and lp is not None}
    return d or None


# ------------------------------------------------------------------------------------------------ panels
SHORT = [  # id, kind, source, token offset, tokens (None: the whole text)
    ("docs-a", "prose", "prose-docs", 1500, 1531), ("docs-b", "prose", "prose-docs", 9000, 2049),
    ("docs-c", "prose", "prose-docs", 14000, 1203), ("runner-a", "code", "code-runner", 2500, 2555),
    ("runner-b", "code", "code-runner", 12000, 1777), ("sched-a", "code", "code-sched", 5000, 3001),
    ("sched-b", "code", "code-sched", 18000, 1409), ("mixed-a", "mixed", "mixed-long", 8000, 2311),
    ("mixed-b", "prose", "mixed-long", 31000, 1027), ("mixed-c", "mixed", "mixed-long", 37000, 2803),
    ("polish", "polish", "polish", 0, None), ("json", "json", "json", 0, 2500),
]

POLISH = """Stare miasteczko nad rzeką przez wieki żyło z handlu drewnem. Tratwy spływały wiosną do Gdańska, a flisacy wracali pieszo, niosąc sól, sukno i wiadomości ze świata. Kiedy w dziewiętnastym wieku poprowadzono kolej inną doliną, rynek opustoszał w ciągu jednego pokolenia. Zostały kamienice z podcieniami, kościół z drewnianą dzwonnicą i przekonanie mieszkańców, że historia kiedyś o nich przypomni.

Przypomniała sobie w latach dziewięćdziesiątych, gdy młody burmistrz postanowił odnowić nabrzeże. Nie miał pieniędzy na wielkie inwestycje, więc zaczął od drobiazgów: ławek, latarni, ścieżki rowerowej wzdłuż wału. Po trzech latach do miasteczka zaczęli przyjeżdżać kajakarze, a za nimi właściciele pensjonatów. Dziś w sezonie trudno znaleźć wolny pokój, a zimą rynek znowu cichnie, jakby odpoczywał po letnim zgiełku.

Dobry żurek zaczyna się od zakwasu. Mąkę żytnią razową zalewa się przegotowaną, letnią wodą, dodaje ząbek czosnku, skórkę chleba i liść laurowy, po czym odstawia w ciepłe miejsce na cztery, pięć dni. Zakwas powinien pachnieć kwaśno, ale czysto; jeśli pojawi się pleśń, trzeba zacząć od nowa. Wywar gotuje się na wędzonce z warzywami, a zakwas wlewa na końcu, mieszając, żeby nie powstały grudki. Majeranek dodaje się dopiero przed podaniem.

Zespół, który planuje migrację bazy danych, powinien najpierw spisać wszystkie zależności: usługi, raporty, zadania wsadowe i ręczne skrypty, o których nikt już nie pamięta. Następnie warto przygotować środowisko testowe z kopią danych produkcyjnych i sprawdzić, ile trwa pełne przeniesienie. Plan wycofania nie jest formalnością. Jeśli nie da się go przećwiczyć, lepiej przesunąć termin, niż liczyć na to, że wszystko pójdzie gładko.

Jesień w górach potrafi zaskoczyć. Rano słońce i dwanaście stopni, po południu mgła tak gęsta, że szlak znika po kilku metrach, a wieczorem pierwszy śnieg na graniach. Doświadczeni turyści zabierają czołówkę nawet na krótką wycieczkę, sprawdzają prognozę w dwóch źródłach i mówią komuś w schronisku, dokąd idą. Najczęstszym błędem nie jest brak sprzętu, tylko upór: niechęć do zawrócenia, gdy warunki się pogarszają.

Babcia twierdziła, że pogodę najlepiej przewidują jaskółki i jej lewe kolano. Dziadek wolał barometr, który wisiał w sieni od czasów przedwojennych i którego nikt poza nim nie umiał odczytać. Spierali się o to przez pięćdziesiąt lat, a my, wnuki, prowadziliśmy tabelę trafień w zeszycie w kratkę. Wynik był remisowy, co obie strony uznały za dowód własnej racji.

Nauka programowania przypomina naukę języka obcego: najpierw zapamiętuje się słówka, potem zdania, a dopiero po długim czasie zaczyna się myśleć w nowym języku. Początkujący często przepisują kod z poradników bez zrozumienia, co jest normalnym etapem, o ile w końcu zaczną zadawać pytania: dlaczego ta pętla kończy się tutaj, co się stanie, gdy lista będzie pusta, skąd program wie, jaki jest typ tej zmiennej. Takie pytania są ważniejsze niż kolejny kurs.
"""


def json_doc(n=60, seed=20260929):
    rnd = random.Random(seed)
    first = ["Anna", "Piotr", "Maria", "Jan", "Olivia", "Liam", "Katarzyna", "Tomasz", "Sofia", "Mateo", "Zofia", "Noah"]
    last = ["Nowak", "Kowalski", "Smith", "Garcia", "Wiśniewska", "Müller", "Rossi", "Dubois", "Lewandowski", "Kim"]
    cities = ["Warszawa", "Kraków", "Gdańsk", "Berlin", "Lyon", "Porto", "Toronto", "Osaka", "Wrocław", "Austin"]
    status = ["pending", "paid", "shipped", "delivered", "refunded", "cancelled"]
    orders = []
    for i in range(n):
        items = [{"sku": f"SKU-{rnd.randint(1000, 9999)}", "qty": rnd.randint(1, 5),
                  "unit_price": round(rnd.uniform(1.5, 499.0), 2), "gift": rnd.random() < 0.1}
                 for _ in range(rnd.randint(1, 4))]
        orders.append({"order_id": f"ORD-{2026_0000 + i * 37}", "customer": {
            "name": f"{rnd.choice(first)} {rnd.choice(last)}", "email": f"user{rnd.randint(10, 99999)}@example.com",
            "address": {"city": rnd.choice(cities), "postal_code": f"{rnd.randint(10, 99)}-{rnd.randint(100, 999)}"}},
            "created_at": f"2026-{rnd.randint(1, 9):02d}-{rnd.randint(1, 28):02d}T{rnd.randint(0, 23):02d}:"
                          f"{rnd.randint(0, 59):02d}:00Z",
            "status": rnd.choice(status), "items": items,
            "total": round(sum(x["qty"] * x["unit_price"] for x in items), 2),
            "note": rnd.choice([None, "leave at the door", "call before delivery", "fragile", "zadzwonić przed dostawą"])})
    return json.dumps({"orders": orders}, indent=2, ensure_ascii=False)


def panel(name, texts_path):
    with open(texts_path, encoding="utf-8") as f:
        texts = json.load(f)
    if name == "long":
        return [{"id": t["id"], "kind": t.get("kind", ""), "text": t["text"], "offset": 0, "tokens": int(t["tokens"])}
                for t in texts]
    src = {t["id"]: t["text"] for t in texts}
    src.update(polish=POLISH, json=json_doc())
    return [{"id": i, "kind": k, "text": src[s], "offset": o, "tokens": n} for i, k, s, o, n in SHORT]


# ------------------------------------------------------------------------------------------------ KLD collect
def kld_collect(a):
    out = {"model": MODEL, "url": BASE, "panel": a.panel, "k": a.k, "gen": a.gen, "tf": a.tf, "items": []}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    toks, warns = {}, []
    for t in panel(a.panel, a.texts):
        if t["text"] not in toks:
            toks[t["text"]] = tokenize(t["text"])
        full = toks[t["text"]]
        want = t["tokens"] or len(full)
        ids = full[t["offset"]:t["offset"] + want]
        n = len(ids)
        rec = {"id": t["id"], "kind": t["kind"], "offset": t["offset"], "target": want, "n_prompt": n}
        t0 = time.time()
        r = generate(ids, a.gen, a.k, None, salt=f"kld-{stamp}-{t['id']}")  # output logprobs only, cold prefill
        rec["cont_s"] = round(time.time() - t0, 2)
        mi = r.get("meta_info") or {}
        out_lp = mi.get("output_token_logprobs") or []
        rec["gen_tokens"] = [int(x) for x in (r.get("output_ids") or [entry(e)[1] for e in out_lp])]
        rec["gen_top"] = [top_dict(x) for x in mi.get("output_top_logprobs") or []]
        rec["finish"] = mi.get("finish_reason")
        rec["cont_cached"] = mi.get("cached_tokens")
        if not a.tf:
            rec["prompt_lp"], rec["prompt_tok_lp"] = [], []
            out["items"].append(rec)
            print(f"  {t['id']:12s} prompt {n} tok (offset {t['offset']}) continuation {rec['cont_s']} s "
                  f"({len(rec['gen_tokens'])} tok, cached {rec['cont_cached']})", flush=True)
            continue
        t0 = time.time()
        r = generate(ids, 1, a.k, 0)
        rec["tf_s"] = round(time.time() - t0, 2)
        mi = r.get("meta_info") or {}
        rec["prompt_lp"] = [top_dict(x) for x in mi.get("input_top_logprobs") or []]
        rec["prompt_tok_lp"] = [entry(e)[0] for e in mi.get("input_token_logprobs") or []]
        rec["tf_cached"] = mi.get("cached_tokens")
        rec["tf_prompt_tokens"] = mi.get("prompt_tokens")
        if n < want:
            warns.append(f"{t['id']}: only {n} of {want} tokens available")
        if rec["tf_cached"]:
            warns.append(f"{t['id']}: teacher-forced request reported cached_tokens {rec['tf_cached']} (not a cold prefill)")
        if len(rec["prompt_lp"]) != n:
            warns.append(f"{t['id']}: {len(rec['prompt_lp'])} teacher-forced positions for {n} prompt tokens")
        out["items"].append(rec)
        print(f"  {t['id']:12s} prompt {n} tok (target {want}, offset {t['offset']}) continuation {rec['cont_s']} s "
              f"({len(rec['gen_tokens'])} tok, cached {rec['cont_cached']}) teacher-forced {rec['tf_s']} s "
              f"({len(rec['prompt_lp'])} positions, cached {rec['tf_cached']})", flush=True)
    out["warnings"] = warns
    with open(a.out, "w") as f:
        json.dump(out, f)
    for w in warns:
        print("WARN " + w)
    print(f"KLD-COLLECT {a.panel} {a.out}: {len(out['items'])} texts, prompts {[i['n_prompt'] for i in out['items']]}, "
          f"{sum(i['n_prompt'] for i in out['items'])} tokens, {len(warns)} warning(s)")
    return 0


# ------------------------------------------------------------------------------------------------ KLD compare
def kl_top(p, q):
    """KL(p || q) over p's top-k support plus one tail bucket; q entries missing from its top-k get q's floor."""
    if not p or not q:
        return None
    floor = min(q.values())
    kl, pm, qm = 0.0, 0.0, 0.0
    for tok, lp in p.items():
        lq = q.get(tok, floor)
        pp = math.exp(lp)
        kl += pp * (lp - lq)
        pm += pp
        qm += math.exp(lq)
    pt, qt = max(1e-12, 1 - pm), max(1e-12, 1 - min(qm, 1 - 1e-12))
    return max(0.0, kl + pt * math.log(pt / qt))


def stats(xs):
    if not xs:
        return {"n": 0, "mean": float("nan"), "p99": float("nan"), "max": float("nan")}
    s = sorted(xs)
    return {"n": len(s), "mean": sum(s) / len(s), "p99": s[max(0, int(0.99 * len(s)) - 1)], "max": s[-1]}


def nll(rec):
    v = [-x for x in rec.get("prompt_tok_lp") or [] if x is not None]
    return sum(v) / len(v) if v else float("nan")


def argmax(d):
    return max(d, key=d.get)


def kld_compare(a):
    with open(a.ref) as f, open(a.cand) as g:
        R, C = json.load(f), json.load(g)
    pn = R.get("panel", "long")
    tag = f"kld_{pn}"
    max_kl = a.max_kl if a.max_kl is not None else MAX_KL.get(pn, 0.035)
    same_panel = C.get("panel", "long") == pn and len(R["items"]) == len(C["items"]) and all(
        r["id"] == c["id"] and r["n_prompt"] == c["n_prompt"] for r, c in zip(R["items"], C["items"]))
    if not same_panel:
        print(f"MISMATCHED PANEL: {pn} {[(i['id'], i['n_prompt']) for i in R['items']]} vs "
              f"{C.get('panel', 'long')} {[(i['id'], i['n_prompt']) for i in C['items']]}")
        print(f"GATE {tag} nan FAIL 0")
        return 1
    pooled, agree_all, tail_all, cont_all, top1 = [], [], [], [], 0
    for r, c in zip(R["items"], C["items"]):
        kls, agree = [], []
        for p, q in zip(r["prompt_lp"], c["prompt_lp"]):
            k = kl_top(p, q)
            if k is None:
                continue
            kls.append(k)
            if argmax(p) == argmax(q):
                agree.append(k)
        tail = kls[-2048:] if pn == "long" else []
        gr, gc = r["gen_tokens"], c["gen_tokens"]
        div = next((i for i, (u, v) in enumerate(zip(gr, gc)) if u != v), min(len(gr), len(gc)))
        ck = [k for k in (kl_top(p, q) for p, q in zip(r["gen_top"][:div + 1], c["gen_top"][:div + 1])) if k is not None]
        pooled += kls
        agree_all += agree
        tail_all += tail
        cont_all += ck
        top1 += len(agree)
        s, sa = stats(kls), stats(agree)
        print(f"{r['id']:12s} prompt {r['n_prompt']:6d}  tf KL mean {s['mean']:.5f} p99 {s['p99']:.4f} "
              f"top1 {100 * len(agree) / max(1, len(kls)):.2f} %  agree-only KL {sa['mean']:.5f}"
              + (f"  last-2048 KL {stats(tail)['mean']:.5f}" if tail else "")
              + f"  NLL {nll(r):.4f} -> {nll(c):.4f}  continuation: identical {div}/{len(gr)} tokens"
              f"{' (full)' if gr == gc else ''}, KL to divergence {stats(ck)['mean']:.5f} ({len(ck)} pos)")
        for side, x in (("ref", r), ("cand", c)):
            if x.get("tf_cached"):
                print(f"WARN {r['id']}: {side} teacher-forced request had cached_tokens {x['tf_cached']}")
    s, sa, sc = stats(pooled), stats(agree_all), stats(cont_all)
    if s["n"] == 0:  # no teacher-forced positions (bounded-replay server): gate on the continuation
        ok = sc["n"] > 0 and sc["mean"] <= max_kl
        print(f"continuation-only panel: {sc['n']} positions (up to and including the first divergence), mean KL "
              f"{sc['mean']:.5f}, p99 {sc['p99']:.4f}, max {sc['max']:.4f}")
        print(f"GATE {tag}_cont {sc['mean']:.5f} {'PASS' if ok else 'FAIL'} {sc['n']}")
        return 0 if ok else 1
    ok = s["n"] > 0 and s["mean"] <= max_kl
    n_ref = [nll(r) for r in R["items"]]
    n_cand = [nll(c) for c in C["items"]]
    print(f"pooled: {s['n']} teacher-forced positions, mean KL {s['mean']:.5f}, p99 {s['p99']:.4f}, max {s['max']:.4f}, "
          f"top-1 {100 * top1 / max(1, s['n']):.2f} %; agree-only {sa['n']} positions, mean KL {sa['mean']:.5f}, "
          f"p99 {sa['p99']:.4f}"
          + (f"; last-2048 mean {stats(tail_all)['mean']:.5f}" if tail_all else "")
          + f"; mean NLL {sum(n_ref) / len(n_ref):.4f} -> {sum(n_cand) / len(n_cand):.4f}"
          f"; continuation mean {sc['mean']:.5f} ({sc['n']} pos)")
    print(f"GATE {tag} {s['mean']:.5f} {'PASS' if ok else 'FAIL'} {s['n']}")
    print(f"GATE {tag}_agree {sa['mean']:.5f} info {sa['n']}")
    print(f"GATE {tag}_cont {sc['mean']:.5f} info {sc['n']}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ T > 0 scan
PROMPTS = [
    ("prose", "low", "Explain how to decide whether a medium-sized software project is ready for a database migration. "
     "Cover dependencies, schema compatibility, tests, rollout and rollback in several paragraphs."),
    ("prose", "low", "Describe how a city could reduce traffic congestion over ten years without building new roads. "
     "Discuss pricing, public transport, zoning and the politics of each option."),
    ("think", "high", "A small town has one bridge that is closing for two years of repairs. The council must choose "
     "between a free ferry, a temporary pontoon bridge, and expanded bus service over a longer road. Reason carefully "
     "through the costs and risks of each option, then recommend one."),
    ("think", "high", "Three friends split a restaurant bill unevenly because one arrived late and ordered less, another "
     "paid the tip, and the third covered a previous taxi. Work out a fair way to settle up step by step."),
    ("code", "low", "Write a Python 3 module with a thread-safe bounded LRU cache with per-item TTL: get, set, delete, "
     "clear, __len__, injectable monotonic clock, plus five unittest tests. Return only code."),
    ("code", "low", "Write a Python function that parses an INI-like config format with sections, comments, multi-line "
     "values and typed getters, with docstrings and doctests. Return only code."),
    ("json", "low", "Return only a JSON object describing three fictional employees: name, role, start_date (ISO 8601), "
     "skills (array of strings), manager (name or null). No prose, no code fences."),
    ("json", "low", "Return only a JSON array of five cities with fields name, country, population (integer), "
     "coordinates {lat, lon} and a one-sentence note. No prose, no code fences."),
    ("polish", "low", "Napisz po polsku kilka akapitów o tym, jak przygotować się do pierwszego maratonu: plan "
     "treningowy, odżywianie, sprzęt i regeneracja."),
    ("story", "low", "Write a short story (about 500 words) about a lighthouse keeper who finds a message in a bottle "
     "written in her own handwriting."),
    ("schema", "low", "Give me a packing list for a three-day hiking trip as JSON."),
]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "packing", "schema": {
    "type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "qty": {"type": "integer"}}, "required": ["name", "qty"]}}},
    "required": ["items"]}}}
EFFORT = {"low": "low", "medium": "medium", "high": "high", "max": "high"}


def loop_tail(s, min_rep=6, max_unit=200):
    t = s[-1500:]
    if len(t) < 300:
        return False
    for u in range(1, max_unit + 1):
        unit = t[-u:]
        reps, i = 1, len(t) - 2 * u
        while i >= 0 and t[i:i + u] == unit:
            reps += 1
            i -= u
        if reps >= min_rep and reps * u >= 120:
            return True
    return False


def scan_text(text):
    cjk = sum(0x4E00 <= ord(ch) <= 0x9FFF or 0x3040 <= ord(ch) <= 0x30FF or 0xAC00 <= ord(ch) <= 0xD7AF for ch in text)
    bad = text.count("�")
    flags = []
    if cjk >= 5 or bad >= 3:
        flags.append("SALAD")
    if loop_tail(text):
        flags.append("LOOP")
    return flags, cjk, bad


def json_ok(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        json.loads(t)
        return True
    except Exception:  # noqa: BLE001
        return False


def one_chat(kind, effort, prompt, temp, top_p, seed, max_tokens, thinking=True):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temp, "top_p": top_p, "chat_template_kwargs": {"thinking": thinking}}
    if thinking:
        body["reasoning_effort"] = EFFORT.get(effort, "high")
    if seed is not None:
        body["seed"] = seed
    if kind == "schema":
        body["response_format"] = SCHEMA
    t0 = time.time()
    try:
        r = post("/v1/chat/completions", body, timeout=900)
    except urllib.error.HTTPError as exc:
        # The server's loop abort ends the request as an error named "loop_abort: ..." (HTTP 500).
        text = exc.read().decode(errors="replace")
        if "loop_abort:" not in text:
            raise
        return {"kind": kind, "temp": temp, "top_p": top_p, "seed": seed, "finish": "loop_abort",
                "matched_stop": None, "tokens": None, "wall_s": round(time.time() - t0, 2), "content": "",
                "reasoning": "", "loop_abort": text[:300]}
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    usage = r.get("usage") or {}
    return {"kind": kind, "temp": temp, "top_p": top_p, "seed": seed, "finish": ch.get("finish_reason"),
            "matched_stop": ch.get("matched_stop"), "tokens": usage.get("completion_tokens"),
            "wall_s": round(time.time() - t0, 2), "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or ""}


def flag_row(r):
    """Adds flags / cjk / fffd to a result row; returns (salad_or_loop, badjson)."""
    flags, cjk, bad = scan_text((r.get("reasoning") or "") + "\n" + (r.get("content") or ""))
    if (r.get("matched_stop") == "repetition" or r.get("finish") == "loop_abort") and "LOOP" not in flags:
        flags.append("LOOP")
    badjson = r["kind"] in ("json", "schema") and r.get("finish") == "stop" and r.get("matched_stop") != "repetition" \
        and not json_ok(r.get("content") or "")
    if badjson:
        flags.append("BADJSON")
    r.update(flags=flags, cjk=cjk, fffd=bad)
    return any(f in ("SALAD", "LOOP") for f in flags), badjson


def tscan(a):
    out_dir = os.path.expanduser(a.out_dir or os.path.join("~/ds-gate", a.label))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tscan.jsonl")
    think = a.thinking == "on"
    rows = []
    for temp, top_p, seeded in ((1.0, 0.95, True), (0.6, 0.95, False)):
        for i, (kind, effort, prompt) in enumerate(PROMPTS):
            seed = (20260929 + i) if seeded else None
            try:
                rows.append(dict(one_chat(kind, effort, prompt, temp, top_p, seed, a.max_tokens, think), phase="c1"))
            except Exception as exc:  # noqa: BLE001
                rows.append({"phase": "c1", "kind": kind, "temp": temp, "seed": seed, "error": repr(exc)[:300]})
    # c = 4: every prompt at T = 1 plus two greedy requests, four at a time (mixed sampled / greedy batches)
    mix = [(k, e, p, 1.0, 0.95, 777 + i) for i, (k, e, p) in enumerate(PROMPTS)]
    mix.insert(3, ("prose", "low", PROMPTS[0][2], 0.0, 1.0, None))
    mix.insert(8, ("code", "low", PROMPTS[4][2], 0.0, 1.0, None))
    with cf.ThreadPoolExecutor(4) as ex:
        futs = [(m, ex.submit(one_chat, *m, a.max_tokens, think)) for m in mix]
        for m, f in futs:
            try:
                rows.append(dict(f.result(), phase="c4"))
            except Exception as exc:  # noqa: BLE001
                rows.append({"phase": "c4", "kind": m[0], "temp": m[3], "seed": m[5], "error": repr(exc)[:300]})
    salad = errors = badjson = 0
    with open(path, "w") as fo:
        for r in rows:
            if "error" in r:
                errors += 1
                r["flags"] = ["ERROR"]
            else:
                s, b = flag_row(r)
                salad += s
                badjson += b
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{r.get('phase', '?'):3s} {r.get('kind', '?'):7s} T={r.get('temp')} seed={r.get('seed')} "
                  f"tokens {r.get('tokens')} finish {r.get('finish')}"
                  f"{' (' + str(r['matched_stop']) + ')' if r.get('matched_stop') == 'repetition' else ''}"
                  f"{' (loop_abort)' if r.get('finish') == 'loop_abort' else ''} "
                  f"{' '.join(r.get('flags', []))}", flush=True)
    print(f"TSCAN: {salad} salad of {len(rows)} outputs (errors {errors}, bad json {badjson}) -> {path}")
    return 0 if salad == 0 and errors == 0 else 1


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    global BASE, MODEL
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=BASE)
    common.add_argument("--model", default=MODEL)
    ap = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash SGLang gate helpers")
    sp = ap.add_subparsers(dest="cmd", required=True)
    k = sp.add_parser("kld")
    ksp = k.add_subparsers(dest="sub", required=True)
    c = ksp.add_parser("collect", parents=[common])
    c.add_argument("--panel", choices=("long", "short"), default="long")
    c.add_argument("--out", required=True)
    c.add_argument("--texts", default=os.path.join(HERE, "kld_long_texts.json"))
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--gen", type=int, default=256)
    c.add_argument("--tf", action="store_true", help="also teacher-forced prompt logprobs (needs a server without "
                   "decoder SWA bounded replay; the ds41 launch rejects them with HTTP 400)")
    m = ksp.add_parser("compare")
    m.add_argument("ref")
    m.add_argument("cand")
    m.add_argument("--max-kl", type=float, default=None, help="default 0.035 (long and short)")
    t = sp.add_parser("tscan", parents=[common])
    t.add_argument("label")
    t.add_argument("--max-tokens", type=int, default=768)
    t.add_argument("--thinking", choices=("on", "off"), default="on")
    t.add_argument("--out-dir", default=None, help="default ~/ds-gate/LABEL")
    a = ap.parse_args(argv)
    BASE, MODEL = getattr(a, "url", BASE).rstrip("/"), getattr(a, "model", MODEL)
    if a.cmd == "kld":
        return kld_collect(a) if a.sub == "collect" else kld_compare(a)
    return tscan(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
