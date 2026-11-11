#!/usr/bin/env python3
"""Single-GB10 unit probe of DSV41_MOE_EVICT_FIRST (b12x_next evict-first patch, adapter/moe_b12x_next.py).

One idle GB10, no network, no engine, no checkpoint: synthetic W4A8 experts in the production TP4 geometry (E=384
routed experts, K=5120 hidden, N=576 per rank, top-6, SwiGLU limit 10; MXFP4 bytes + E8M0 scales repacked in place
by b12x_next exactly as the adapter does), ~1.8 GB of device memory, no pinned host memory. Runs the adapter's own
_Geometry (production profile: DSV41_MOE_B12X_NEXT_DETERMINISTIC=1, the default plan table, DSV41_MOE_LAUNCH_PREPARED
=plan,direct) with the gate armed. Nothing here reads or sets clocks, fans or power limits. Designed to run as a child
process under an external TERM / KILL deadline: every record is one flushed JSON line, and SIGTERM ends with an
INVALID verdict line.

Cells
  bitwise   M in 1 6 12 24 48 96: the same prepared layer, same inputs, flag off vs on, eager (ef=False / ef=True)
            and under CUDA graph (off: captured with ef=False; on: captured through the adapter's own per-capture
            decision, gate on, M <= 128). torch.equal of all four outputs (eager off / on, graph off / on); the off
            run is repeated to prove determinism. Actually on: every launch is traced to the compiled kernel it ran
            (b12x_next's _get_dynamic_kernel and the prepared launch's plan): eager on and the on capture must run
            only flag-on kernels (cache key + ("evict_first_weights",)), eager off and the off capture none.
  moe_time  M in 1 6 12 24 48 96, flag off / on: flush L2 (64 MiB read), replay the layer's graph, CUDA events
            around the replay. Interleaved sample by sample with a rotating order; n = --reps (>= 200).
  protect   M in 6 96, prefetch kind in l2pf (adapter l2_prefetch's cp.async.bulk.prefetch.L2 kernel, the shared
            FC2 stand-in: 6 MiB in 8 segments) and l2pf_el (its evict_last MoE-window variant): flush, prefetch, 50 us
            spin, replay the layer (timed), then a consumer read of the 6 MiB (timed). Flag off / on. References per
            prefetch kind: WARM (prefetch, no MoE) and COLD (no prefetch). Interleaved with the moe_time cells.
            Without nvcc the prefetch is a plain read (kind "read"; recorded). Per sample the MoE replay and the
            consumer read are timed back to back; total_us = the median of their per-sample sum.
  done      after the timing: a device synchronize; the record proves the run ended without an asynchronous fault.

Verdict (pre-registered, on medians; astra's efu contract, 2026-11-11):
  KILL     FIRST, before any completeness rule: a credible difference = a bitwise cell whose on launches ran flag-on
           kernels and off launches stock ones, whose off output is finite and identical to itself (graph: to eager
           off too), with on != off or graph on != eager on. Outputs are compared bit for bit (int16 views).
  INVALID  any harness error or device fault (an error record; no "done" record) / missing or duplicated planned cell
           / a timed cell with n, n_moe or n_total < 200 (--reps below 200 is a smoke run) / a time that is not finite
           and positive / a prefetch kind other than the planned l2pf / l2pf_el (no nvcc: "read") / the off output not
           bit-identical run to run, not finite, or eager off != graph off (the stock path itself) / not compared as
           int16 / not actually on: no flag-on kernel compiled at load, an on launch or the on capture that ran a
           stock kernel (or none), an off launch that ran a flag-on kernel, or the on graph's capture not on
  KILL     the MoE layer with the flag on > 1.02x the flag-off time at any M; or the MoE + consumer net balance at
           M=6 with the l2pf prefetch (total_off - total_on, us) <= 0 (a faster consumer bought with a slower MoE)
  HOLD     the M=96 l2pf net balance <= 0: not admitted (the lever covers M <= 128; requalify with
           DSV41_MOE_EVICT_FIRST_MAX_ROWS below 96)
  PASS     otherwise; reports per M and prefetch kind the consumer warmth gain (consumer_off - consumer_on) and the
           net balance (total_off - total_on); GREY is appended if the M=6 l2pf net balance is < 1 us
Final line: ``EFU VERDICT: PASS|HOLD|KILL|INVALID [GREY] <details>``.

usage
  python3 tests/gpu_test_moe_evict_first.py plan                      cell plan + runtime estimate (no GPU)
  python3 tests/gpu_test_moe_evict_first.py run --out efu.jsonl [--reps 300] [--cache DIR]
  python3 tests/gpu_test_moe_evict_first.py --selftest                decider on synthetic records (CPU)
Runtime on one GB10: setup 1-5 min (synthetic weights + repack + plan preparation + the flag-off / flag-on kernel
compiles; ~1 min with a warm --cache), bitwise ~10 s, timing ~10-20 s at --reps 300: ~2-6 min in total.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = "efu-20261110"
E, K, N, TOPK, LIMIT = 384, 5120, 576, 6, 10.0
MS = (1, 6, 12, 24, 48, 96)
PROTECT_MS = (6, 96)
PF_KINDS = ("l2pf", "l2pf_el")
PRIMARY_PF = "l2pf"
FLAGS = ("off", "on")
N_MIN = 200
TIME_RATIO_MAX = 1.02
GREY_GAIN_US = 1.0                 # on the net balance (MoE + consumer) at M=6, l2pf
FLUSH_MB, PF_MB, PF_SEGS, SPIN_CYCLES = 64, 6, 8, 110_000      # 110k cycles ~ 50 us at the 2.2 GHz cap
MIB = 1 << 20


# ------------------------------------------------------------------------------------------------- plan / decider
def planned_cells():
    cells = [("bitwise", m, mode) for m in MS for mode in ("eager", "graph")]
    cells += [("moe_time", m, f) for m in MS for f in FLAGS]
    cells += [("protect", m, pf, f) for m in PROTECT_MS for pf in PF_KINDS for f in FLAGS]
    cells += [("ref", pf, c) for pf in PF_KINDS for c in ("WARM", "COLD")]
    return cells


def cell_id(rec):
    t = rec.get("type")
    if t == "bitwise":
        return ("bitwise", rec["M"], rec["mode"])
    if t == "moe_time":
        return ("moe_time", rec["M"], rec["flag"])
    if t == "protect":
        return ("protect", rec["M"], rec["pf_planned"], rec["flag"])
    if t == "ref":
        return ("ref", rec["pf_planned"], rec["cell"])
    return None


def _pos(x):
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def decide(records):
    """(verdict, flags, detail) from the run's records, by the pre-registered rules in the module docstring."""
    bitrecs = [r for r in records if r.get("type") == "bitwise"]
    # 1. a CREDIBLE difference decides first, before any completeness rule (astra batch 4): the cell's on launches ran
    #    flag-on kernels and its off launches stock ones, the off output is finite and identical to itself (and, under
    #    graph, to eager off)
    cred = [r for r in bitrecs if r.get("on_kernels") == "on" and r.get("off_kernels") == "off"
            and r.get("finite") is True and r.get("off_repeat_equal") is True
            and (r.get("mode") == "eager" or (r.get("capture") == "on" and r.get("graph_equals_eager") is True))]
    diff = sorted({(r.get("M"), r.get("mode")) for r in cred if r.get("equal") is False
                   or (r.get("mode") == "graph" and r.get("on_graph_equals_eager") is False)}, key=str)
    if diff:
        return "KILL", [], f"flag-on outputs differ bitwise (on vs off, or graph on vs eager on) at {diff}"
    errors = [r for r in records if r.get("type") == "error"]
    if errors:
        return "INVALID", [], f"harness error or device fault: {str(errors[0].get('error', ''))[:200]}"
    have = {}
    for r in records:
        try:
            c = cell_id(r)
        except KeyError:
            return "INVALID", [], f"a malformed {r.get('type')} record"
        if c is not None:
            if c in have:
                return "INVALID", [], f"cell {c} recorded twice"
            have[c] = r
    missing = [c for c in planned_cells() if c not in have]
    if missing:
        return "INVALID", [], f"missing cells {missing[:6]}{' ...' if len(missing) > 6 else ''} ({len(missing)})"
    if not any(r.get("type") == "done" for r in records):
        return "INVALID", [], "no 'done' record (the run did not reach its final device synchronize)"
    for c, r in have.items():
        if c[0] == "bitwise":
            continue
        ns = [r.get("n")] + ([r.get("n_moe"), r.get("n_total")] if c[0] == "protect" else [])
        if not all(isinstance(n, int) and n >= N_MIN for n in ns):
            return "INVALID", [], f"cell {c}: n {ns} < {N_MIN} (smoke run?)"
        ts = ([r.get("median_us")] if c[0] == "moe_time" else [r.get("consumer_us")]
              + ([r.get("moe_us"), r.get("total_us")] if c[0] == "protect" else []))
        ts += [r[k] for k in ("p10_us", "p90_us", "consumer_p10_us", "consumer_p90_us") if k in r]
        if not all(_pos(t) for t in ts):
            return "INVALID", [], f"cell {c}: a time that is not finite and positive ({ts})"
        if c[0] in ("protect", "ref") and r.get("pf") != r.get("pf_planned"):
            return "INVALID", [], f"cell {c}: prefetch kind {r.get('pf')!r}, not the planned {r.get('pf_planned')!r}"
    setup = next((r for r in records if r.get("type") == "setup"), {})
    if not setup.get("flag_on_kernels") or not setup.get("ef_caps"):
        return "INVALID", [], f"no flag-on kernel compiled at load ({setup.get('flag_on_kernels')}, caps {setup.get('ef_caps')})"
    bit = [have[("bitwise", m, mode)] for m in MS for mode in ("eager", "graph")]
    for key, what in (("finite", "synthetic outputs not finite"),
                      ("off_repeat_equal", "the flag-off output is not bit-identical run to run")):
        bad = [(r["M"], r["mode"]) for r in bit if r.get(key) is not True]
        if bad:
            return "INVALID", [], f"{what} at {bad}"
    bad = [r["M"] for r in bit if r["mode"] == "graph" and r.get("graph_equals_eager") is not True]
    if bad:
        return "INVALID", [], f"the stock path itself: graph off != eager off at M={bad}"
    bad = [(r["M"], r["mode"]) for r in bit if r.get("cmp") != "int16"]
    if bad:
        return "INVALID", [], f"not compared bit for bit (int16 views) at {bad}"
    unv = [(r["M"], r["mode"], r.get("on_kernels"), r.get("off_kernels"), r.get("capture")) for r in bit if r not in cred]
    if unv:
        return "INVALID", [], f"not actually on / off (M, mode, on kernels, off kernels, capture): {unv[:4]}"
    if any(r.get("equal") is not True or (r["mode"] == "graph" and r.get("on_graph_equals_eager") is not True) for r in bit):
        return "INVALID", [], "a bitwise cell without a definite equal / on_graph_equals_eager"
    ratios = {m: have[("moe_time", m, "on")]["median_us"] / have[("moe_time", m, "off")]["median_us"] for m in MS}
    slow = {m: round(v, 4) for m, v in ratios.items() if v > TIME_RATIO_MAX}
    if slow:
        return "KILL", [], f"MoE layer on/off > {TIME_RATIO_MAX} at {slow}"
    gains, net = {}, {}
    for m in PROTECT_MS:
        for pf in PF_KINDS:
            off, on = have[("protect", m, pf, "off")], have[("protect", m, pf, "on")]
            gains[f"M{m}/{pf}"] = round(off["consumer_us"] - on["consumer_us"], 2)
            net[f"M{m}/{pf}"] = round(off["total_us"] - on["total_us"], 2)
    detail = ("net us " + json.dumps(net, sort_keys=True) + "; consumer gain us " + json.dumps(gains, sort_keys=True)
              + "; moe on/off " + json.dumps({f"M{m}": round(v, 4) for m, v in ratios.items()}, sort_keys=True))
    if net[f"M6/{PRIMARY_PF}"] <= 0:
        return "KILL", [], f"MoE + consumer net balance at M6/{PRIMARY_PF} <= 0; " + detail
    if net[f"M96/{PRIMARY_PF}"] <= 0:
        return "HOLD", [], (f"MoE + consumer net balance at M96/{PRIMARY_PF} <= 0: not admitted (requalify with "
                            "DSV41_MOE_EVICT_FIRST_MAX_ROWS below 96); " + detail)
    flags = ["GREY"] if net[f"M6/{PRIMARY_PF}"] < GREY_GAIN_US else []
    return "PASS", flags, detail


def verdict_line(records):
    v, flags, detail = decide(records)
    return f"EFU VERDICT: {v}{' ' + ' '.join(flags) if flags else ''} {detail}"


def selftest():
    def good(gain=3.0, ratio=1.0, n=300, moe_cost=0.0):
        recs = [{"type": "setup", "flag_on_kernels": 2, "ef_caps": [5, 6]}]
        recs += [{"type": "bitwise", "M": m, "mode": mode, "equal": True, "off_repeat_equal": True, "finite": True,
                  "graph_equals_eager": True, "on_graph_equals_eager": True, "on_kernels": "on", "off_kernels": "off",
                  "cmp": "int16", "capture": "on" if mode == "graph" else None} for m in MS for mode in ("eager", "graph")]
        recs += [{"type": "moe_time", "M": m, "flag": f, "median_us": 300.0 * (ratio if f == "on" else 1.0), "n": n}
                 for m in MS for f in FLAGS]
        recs += [{"type": "protect", "M": m, "pf_planned": pf, "pf": pf, "flag": f, "n": n, "n_moe": n, "n_total": n,
                  "consumer_us": 40.0 - (gain if f == "on" else 0.0), "moe_us": 300.0 + (moe_cost if f == "on" else 0.0),
                  "total_us": 340.0 - (gain if f == "on" else 0.0) + (moe_cost if f == "on" else 0.0)}
                 for m in PROTECT_MS for pf in PF_KINDS for f in FLAGS]
        recs += [{"type": "ref", "pf_planned": pf, "pf": pf, "cell": c, "n": n, "consumer_us": 30.0}
                 for pf in PF_KINDS for c in ("WARM", "COLD")]
        recs.append({"type": "done"})
        return recs

    def mutate(recs, pred, **kw):
        for r in recs:
            if pred(r):
                r.update(kw)
        return recs

    cases = [
        ("pass", good(), ("PASS", [])),
        ("grey", good(gain=0.5), ("PASS", ["GREY"])),
        ("kill bitwise", mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 48 and r["mode"] == "graph",
                                equal=False), ("KILL", [])),
        ("kill time", good(ratio=1.03), ("KILL", [])),
        ("pass at 1.02", good(ratio=1.019), ("PASS", [])),
        ("invalid error", good() + [{"type": "error", "error": "boom"}], ("INVALID", [])),
        ("invalid missing", [r for r in good() if not (r.get("type") == "moe_time" and r["M"] == 96)], ("INVALID", [])),
        ("invalid n", good(n=150), ("INVALID", [])),
        ("invalid nondeterministic off", mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 6,
                                                off_repeat_equal=False), ("INVALID", [])),
        ("invalid non-finite", mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 96, finite=False,
                                      equal=False), ("INVALID", [])),
        ("invalid no flag-on kernel", mutate(good(), lambda r: r.get("type") == "setup", flag_on_kernels=0),
         ("INVALID", [])),
        ("invalid capture not on", mutate(good(), lambda r: r.get("type") == "bitwise" and r["mode"] == "graph"
                                          and r["M"] == 1, capture="off"), ("INVALID", [])),
        # a bitwise difference outranks a GREY pass; an error outranks everything
        ("kill beats grey", mutate(good(gain=0.1), lambda r: r.get("type") == "bitwise" and r["M"] == 1, equal=False),
         ("KILL", [])),
        # astra's efu contract (2026-11-11)
        ("kill: graph on != eager on", mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 24
                                              and r["mode"] == "graph", on_graph_equals_eager=False), ("KILL", [])),
        ("invalid: stock graph off != eager off", mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 12
                                                         and r["mode"] == "graph", graph_equals_eager=False),
         ("INVALID", [])),
        ("invalid: an on launch ran a stock kernel", mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 96
                                                            and r["mode"] == "eager", on_kernels="off"), ("INVALID", [])),
        ("invalid: mixed on capture", mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 6
                                             and r["mode"] == "graph", on_kernels="mixed"), ("INVALID", [])),
        ("invalid: an off launch ran a flag-on kernel", mutate(good(), lambda r: r.get("type") == "bitwise"
                                                               and r["M"] == 1, off_kernels="on"), ("INVALID", [])),
        ("kill beats an unverified cell", mutate(mutate(good(), lambda r: r.get("type") == "bitwise" and r["M"] == 1,
                                                        on_kernels="none"),
                                                 lambda r: r.get("type") == "bitwise" and r["M"] == 48, equal=False),
         ("KILL", [])),
        ("invalid: no done record (async fault)", [r for r in good() if r.get("type") != "done"], ("INVALID", [])),
        ("invalid: n_moe < 200", mutate(good(), lambda r: r.get("type") == "protect" and r["M"] == 96, n_moe=199),
         ("INVALID", [])),
        ("kill: faster consumer, slower MoE (net <= 0)", good(gain=3.0, moe_cost=3.5), ("KILL", [])),
        ("grey: net < 1 us", good(gain=3.0, moe_cost=2.5), ("PASS", ["GREY"])),
        ("hold: M96 net <= 0 (not admitted)", mutate(good(), lambda r: r.get("type") == "protect" and r["M"] == 96
                                                    and r["flag"] == "on", total_us=341.0), ("HOLD", [])),
        # astra batch 4
        ("kill: a credible difference beats a later error", mutate(good(), lambda r: r.get("type") == "bitwise"
                                                                   and r["M"] == 12, equal=False)
         + [{"type": "error", "error": "boom in timing"}], ("KILL", [])),
        ("kill: a credible difference beats missing cells", [r for r in mutate(good(), lambda r: r.get("type") == "bitwise"
                                                                               and r["M"] == 6, equal=False)
                                                             if r.get("type") not in ("moe_time", "done")], ("KILL", [])),
        ("invalid: compared numerically (no int16)", mutate(good(), lambda r: r.get("type") == "bitwise", cmp=None),
         ("INVALID", [])),
        ("invalid: NaN time", mutate(good(), lambda r: r.get("type") == "moe_time" and r["M"] == 6, median_us=float("nan")),
         ("INVALID", [])),
        ("invalid: zero consumer time", mutate(good(), lambda r: r.get("type") == "ref", consumer_us=0.0), ("INVALID", [])),
        ("invalid: prefetch was a plain read", mutate(good(), lambda r: r.get("type") in ("protect", "ref"), pf="read"),
         ("INVALID", [])),
        ("invalid: n_total 1", mutate(good(), lambda r: r.get("type") == "protect" and r["M"] == 6, n_total=1),
         ("INVALID", [])),
        ("invalid: a duplicated cell", good() + [good()[1]], ("INVALID", [])),
        ("invalid: no total_us", mutate(good(), lambda r: r.get("type") == "protect", total_us=None), ("INVALID", [])),
    ]
    bad = []
    for name, recs, want in cases:
        v, flags, detail = decide(recs)
        ok = (v, flags) == want
        print(f"[{'OK' if ok else 'FAIL'}] selftest {name}: {v} {flags} {detail[:120]}")
        if not ok:
            bad.append(name)
    line = verdict_line(good())
    ok = line.startswith("EFU VERDICT: PASS net us")
    print(f"[{'OK' if ok else 'FAIL'}] selftest verdict line: {line[:100]}")
    if not ok:
        bad.append("line")
    print("SELFTEST " + ("PASS" if not bad else f"FAIL {bad}"))
    return 0 if not bad else 1


def plan_text(reps):
    cells = planned_cells()
    timed = [c for c in cells if c[0] in ("moe_time", "protect", "ref")]
    lines = [f"{KIT}: E={E} K={K} N={N} top-{TOPK} synthetic W4A8 experts (~1.8 GB), production adapter profile",
             f"bitwise cells ({len([c for c in cells if c[0] == 'bitwise'])}): M {MS} x eager / graph, off vs on",
             f"timed cells ({len(timed)}), n={reps} each, interleaved sample by sample (rotating order):"]
    lines += [f"  {c}" for c in timed]
    lines.append(f"rules: INVALID (error or fault / missing / no done / n<{N_MIN} / off nondeterministic / stock eager != "
                 f"graph / not actually on or off), KILL (any on output differs, on/off > {TIME_RATIO_MAX} at any M, "
                 f"MoE + consumer net at M6/{PRIMARY_PF} <= 0), PASS (+GREY if that net < {GREY_GAIN_US} us or the "
                 f"M96/{PRIMARY_PF} net <= 0)")
    est = len(timed) * reps * 1.1e-3
    lines.append(f"estimate on one GB10: setup 1-5 min (cold compile cache), bitwise ~10 s, timing ~{est:.0f} s")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- GPU run
class Terminated(Exception):
    pass


def _on_term(signum, frame):
    raise Terminated(f"signal {signum}")


def run(args):
    out = open(args.out, "a" if args.append else "w")
    records = []

    def emit(rec):
        rec = dict(rec, kit=KIT)
        records.append(rec)
        line = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        out.write(line + "\n")
        out.flush()
        print(line, flush=True)

    signal.signal(signal.SIGTERM, _on_term)
    t_start = time.time()
    try:
        _run(args, emit)
    except (Terminated, KeyboardInterrupt) as exc:
        emit({"type": "error", "error": f"terminated: {exc}"})
    except Exception as exc:  # noqa: BLE001
        import traceback
        emit({"type": "error", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]})
    line = verdict_line(records) + f" | {time.time() - t_start:.0f} s"
    out.write(json.dumps({"type": "verdict", "kit": KIT, "line": line}) + "\n")
    out.close()
    print(line, flush=True)
    return 0 if line.startswith("EFU VERDICT: PASS") else 1


def _setup_env(args):
    tmp = tempfile.mkdtemp(prefix="efu-")
    env = {"DSV41_MOE_B12X_NEXT": "1", "DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1", "DSV41_MOE_B12X_NEXT_TUNE": "0",
           "DSV41_MOE_B12X_NEXT_WARM": "1", "DSV41_MOE_B12X_NEXT_GRAPH_BS": "1,2,4,8,16",
           "DSV41_MOE_B12X_NEXT_LADDER": "128", "CUDA_GRAPH_MAX_BS_DECODE": "16", "DSPARK_BLOCK_SIZE": "5",
           "DSV41_MOE_EVICT_FIRST": "1", "DSV41_MOE_EVICT_FIRST_MAX_ROWS": "128",
           "DSV41_MOE_LAUNCH_PREPARED": args.prepared}
    for k, v in env.items():
        os.environ[k] = v
    for k in ("CHUNKED_PREFILL_SIZE", "DSV41_AB_VARIANTS", "DSV41_PREFILL_SP", "DSV41_PREFILL_SP_FP8_MOE"):
        os.environ.pop(k, None)
    os.environ["B12X_NEXT_COMPILE_CACHE_DIR"] = args.cache or os.path.join(tmp, "b12x-next-compile")
    os.environ["B12X_COMPILE_CACHE_DIR"] = os.path.join(tmp, "b12x-compile")      # l2pf .so cache (nvcc)
    os.environ.setdefault("STATE_PATH", tmp)
    os.makedirs(os.environ["B12X_NEXT_COMPILE_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["B12X_COMPILE_CACHE_DIR"], exist_ok=True)
    try:
        import moe_b12x_next  # noqa: F401
    except ImportError:
        for p in (os.path.join(ROOT, "adapter"), os.path.join(ROOT, "runtime", "b12x_next")):
            sys.path.insert(0, p)
    return env


def _run(args, emit):
    env = _setup_env(args)
    import torch
    import moe_b12x_next as ad
    dev = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(dev)
    emit({"type": "env", "device": props.name, "capability": list(torch.cuda.get_device_capability(dev)),
          "sms": props.multi_processor_count, "l2_bytes": getattr(props, "L2_cache_size", None),
          "torch": torch.__version__, "reps": args.reps, "env": env, "prepared": args.prepared,
          "cache": os.environ["B12X_NEXT_COMPILE_CACHE_DIR"], "pid": os.getpid()})
    torch.manual_seed(0)
    t0 = time.time()
    B = ad._b12x()
    fm = B["fm"]
    from b12x_next.moe.fused_moe import _impl as I
    if not ad._EF["armed"]:
        raise RuntimeError("the adapter did not arm DSV41_MOE_EVICT_FIRST")
    key = (E, K, N, TOPK, 0, 1, LIMIT, dev.index)
    geom = ad._Geometry(key, E, K, N, TOPK, 0, LIMIT, dev, ep_size=1)
    # synthetic checkpoint-layout experts: MXFP4 bytes, E8M0 scales 2^-9 .. 2^-5 (bounded, no inf / nan)
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev, generator=g)
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=dev, generator=g)
    s13 = torch.randint(118, 123, (E, 2 * N, K // 32), dtype=torch.uint8, device=dev, generator=g)
    s2 = torch.randint(118, 123, (E, K, N // 32), dtype=torch.uint8, device=dev, generator=g)
    ones = torch.ones(E, dtype=torch.float32, device=dev)
    experts = fm.prepare_weights(plan=geom.weight_plan, weights=fm.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2, w13_global_scales=ones, w2_global_scales=ones))
    torch.cuda.synchronize(dev)
    t_prep = time.time() - t0
    geom.build(experts)
    impl = experts._impl
    ef_keys = [k for k in I._DYNAMIC_KERNEL_CACHE if isinstance(k, tuple) and k and k[-1] == "evict_first_weights"]
    emit({"type": "setup", "prepare_weights_s": round(t_prep, 1), "build_s": round(time.time() - t0 - t_prep, 1),
          "caps": geom.caps, "ef_caps": geom.ef_caps, "ef_refused": sorted(geom.ef_refused),
          "flag_on_kernels": len(ad._EF["kernels"]), "flag_on_keys_in_cache": len(ef_keys),
          "evict_first_stats": dict(getattr(I, "EVICT_FIRST_STATS", {})), "report": geom.report,
          "mem_allocated_gib": round(torch.cuda.memory_allocated(dev) / 2**30, 2)})

    # inputs: random top-6 distinct experts per row, softmax weights, bf16 activations
    mmax = max(MS)
    x = torch.randn(mmax, K, device=dev, generator=g).to(torch.bfloat16)
    ids = torch.rand(mmax, E, device=dev, generator=g).argsort(dim=1)[:, :TOPK].to(torch.int32).contiguous()
    wts = torch.softmax(torch.randn(mmax, TOPK, device=dev, generator=g), dim=1).float().contiguous()
    outs = {}

    def launch(m, ef, out):
        """forward() for one chunk of m rows, with the evict-first decision forced (ef True / False) or taken by the
        adapter (None: the per-capture decision)."""
        if geom.direct:
            i, w = ids[:m], wts[:m]
        else:
            i, w = geom.ids_buf[:m], geom.w_buf[:m]
            ad.ep_remap(ids[:m], wts[:m], i, w, geom.offset, geom.E)
        geom._run(geom.cap_for(m), x[:m], i, w, out, impl, ef=ef)
        return out

    decisions = []
    adapter_decide = ad.ef_decide

    # actually on: trace every launch to the compiled kernel it ran. A stock launch (and every launch under capture)
    # looks the kernel up through _impl._get_dynamic_kernel; an eager launch through a prepared plan runs p.compiled
    # (moe_launch_prepared._launch). A flag-on kernel is a cache value whose key ends in "evict_first_weights".
    ran = []

    def ef_compiled(c):
        return any(v is c for k, v in list(I._DYNAMIC_KERNEL_CACHE.items())
                   if isinstance(k, tuple) and k and k[-1] == "evict_first_weights")

    orig_get = I._get_dynamic_kernel

    def traced_get(*a, **kw):
        res = orig_get(*a, **kw)
        ran.append(ef_compiled(res[0] if isinstance(res, tuple) else res))
        return res

    I._get_dynamic_kernel = traced_get
    try:
        import moe_launch_prepared as mlp
        orig_launch = mlp._launch

        def traced_launch(p, kw):
            ran.append(ef_compiled(p.compiled))
            return orig_launch(p, kw)

        mlp._launch = traced_launch
    except ImportError:
        mlp = None

    def kernels(fn):
        """'on' / 'off' / 'mixed' / 'none': the flag-on state of every kernel fn() launched."""
        ran.clear()
        fn()
        return "none" if not ran else "on" if all(ran) else "off" if not any(ran) else "mixed"

    def spy(m, refused=False):          # what the adapter's per-capture decision returned
        v = adapter_decide(m, refused=refused)
        decisions.append(v)
        return v

    ad.ef_decide = spy
    def beq(x, y):
        """Bit-for-bit equality: int16 views of the bf16 outputs (torch.equal on bf16 is numeric: +0 == -0)."""
        return bool(x.dtype == y.dtype == torch.bfloat16 and x.shape == y.shape
                    and torch.equal(x.view(torch.int16), y.view(torch.int16)))

    graphs = {}
    for m in MS:
        o = {k: torch.empty(m, K, dtype=torch.bfloat16, device=dev) for k in ("e_off", "e_off2", "e_on", "g_off", "g_on")}
        ek = {"off": kernels(lambda: launch(m, False, o["e_off"]))}
        launch(m, False, o["e_off2"])
        ek["on"] = kernels(lambda: launch(m, True, o["e_on"]))
        torch.cuda.synchronize(dev)
        captured, gk = {}, {}
        for flag, ef in (("off", False), ("on", None)):
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                launch(m, False if flag == "off" else True, o["g_" + flag])      # warm-up outside capture
            torch.cuda.current_stream(dev).wait_stream(s)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            decisions.clear()
            with torch.cuda.graph(graph):
                gk[flag] = kernels(lambda: launch(m, ef, o["g_" + flag]))
            captured[flag] = "on" if (ef is None and decisions and all(decisions)) else "off"
            graphs[(m, flag)] = graph
            o["g_" + flag].zero_()
            graph.replay()
        torch.cuda.synchronize(dev)
        for mode in ("eager", "graph"):
            a, b = (o["e_off"], o["e_on"]) if mode == "eager" else (o["g_off"], o["g_on"])
            emit({"type": "bitwise", "M": m, "mode": mode, "cap": geom.cap_for(m), "cmp": "int16",
                  "equal": beq(a, b), "off_repeat_equal": beq(o["e_off"], o["e_off2"]),
                  "graph_equals_eager": beq(o["g_off"], o["e_off"]), "on_graph_equals_eager": beq(o["g_on"], o["e_on"]),
                  "on_kernels": (ek if mode == "eager" else gk)["on"], "off_kernels": (ek if mode == "eager" else gk)["off"],
                  "finite": bool(torch.isfinite(a.float()).all()),
                  "capture": captured["on"] if mode == "graph" else None,
                  "max_abs_diff": float((a.float() - b.float()).abs().max())})
        outs[m] = o
    del outs
    I._get_dynamic_kernel = orig_get
    if mlp is not None:
        mlp._launch = orig_launch
    timing(args, emit, ad, graphs, dev)
    torch.cuda.synchronize(dev)
    emit({"type": "done", "elapsed_s": round(time.time() - t0, 1)})


def timing(args, emit, ad, graphs, dev):
    import torch
    flush = torch.empty(FLUSH_MB * MIB // 4, dtype=torch.int32, device=dev).fill_(1)
    buf = torch.empty(PF_MB * MIB // 4, dtype=torch.int32, device=dev).fill_(3)
    seg = PF_MB * MIB // PF_SEGS
    segs = torch.tensor([[buf.data_ptr() + i * seg, seg] for i in range(PF_SEGS)], dtype=torch.int64, device=dev)
    kinds = {}
    try:
        import l2_prefetch
        lib = l2_prefetch._lib()
        chunk = l2_prefetch._CHUNK

        def pf_normal(stream):
            rc = lib.dsv41_l2pf(segs.data_ptr(), PF_SEGS, chunk, stream.cuda_stream)
            if rc:
                raise RuntimeError(f"l2pf launch failed: cuda error {rc}")

        def pf_el(stream):
            rc = lib.dsv41_l2pf_el(segs.data_ptr(), PF_SEGS, segs.data_ptr(), 0, chunk, stream.cuda_stream)
            if rc:
                raise RuntimeError(f"l2pf_el launch failed: cuda error {rc}")

        kinds = {"l2pf": ("l2pf", pf_normal), "l2pf_el": ("l2pf_el", pf_el)}
    except Exception as exc:  # noqa: BLE001  (no nvcc in the image: plain read, recorded)
        emit({"type": "note", "note": f"l2_prefetch kernel unavailable ({type(exc).__name__}: {exc}); "
                                      "prefetch = plain read"})

        def pf_read(stream):
            buf.max()

        kinds = {pf: ("read", pf_read) for pf in PF_KINDS}
    stream = torch.cuda.current_stream(dev)

    def consumer():
        return buf.max()

    cells = [("moe_time", m, f) for m in MS for f in FLAGS]
    cells += [("protect", m, pf, f) for m in PROTECT_MS for pf in PF_KINDS for f in FLAGS]
    cells += [("ref", pf, c) for pf in PF_KINDS for c in ("WARM", "COLD")]
    samples = {c: {"moe": [], "cons": [], "total": []} for c in cells}

    def sample(c):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        flush.max()
        if c[0] == "moe_time":
            ev[0].record()
            graphs[(c[1], c[2])].replay()
            ev[1].record()
            return ev[:2], None
        pf = c[2] if c[0] == "protect" else c[1]
        if not (c[0] == "ref" and c[2] == "COLD"):
            kinds[pf][1](stream)
            torch.cuda._sleep(SPIN_CYCLES)
        if c[0] == "protect":
            ev[0].record()
            graphs[(c[1], c[3])].replay()
            ev[1].record()
        ev[2].record()
        consumer()
        ev[3].record()
        return (ev[:2] if c[0] == "protect" else None), ev[2:]

    # warm every path once, then interleave
    for c in cells:
        sample(c)
    torch.cuda.synchronize(dev)
    pending = []
    for r in range(args.reps):
        k = r % len(cells)
        for c in cells[k:] + cells[:k]:
            pending.append((c, *sample(c)))
        if len(pending) >= 25 * len(cells) or r == args.reps - 1:
            torch.cuda.synchronize(dev)
            for c, moe_ev, cons_ev in pending:
                if moe_ev is not None:
                    samples[c]["moe"].append(moe_ev[0].elapsed_time(moe_ev[1]) * 1000.0)
                if cons_ev is not None:
                    samples[c]["cons"].append(cons_ev[0].elapsed_time(cons_ev[1]) * 1000.0)
                if moe_ev is not None and cons_ev is not None:
                    samples[c]["total"].append(samples[c]["moe"][-1] + samples[c]["cons"][-1])
            pending.clear()

    def med(v):
        return round(statistics.median(v), 3) if v else None

    def pct(v, q):
        return round(sorted(v)[min(len(v) - 1, int(len(v) * q))], 3) if v else None

    for c in cells:
        s = samples[c]
        if c[0] == "moe_time":
            emit({"type": "moe_time", "M": c[1], "flag": c[2], "n": len(s["moe"]), "median_us": med(s["moe"]),
                  "p10_us": pct(s["moe"], 0.1), "p90_us": pct(s["moe"], 0.9)})
        elif c[0] == "protect":
            emit({"type": "protect", "M": c[1], "pf_planned": c[2], "pf": kinds[c[2]][0], "flag": c[3],
                  "n": len(s["cons"]), "n_moe": len(s["moe"]), "consumer_us": med(s["cons"]),
                  "consumer_p10_us": pct(s["cons"], 0.1), "consumer_p90_us": pct(s["cons"], 0.9),
                  "moe_us": med(s["moe"]), "total_us": med(s["total"]), "n_total": len(s["total"])})
        else:
            emit({"type": "ref", "pf_planned": c[1], "pf": kinds[c[1]][0], "cell": c[2], "n": len(s["cons"]),
                  "consumer_us": med(s["cons"]), "consumer_p10_us": pct(s["cons"], 0.1),
                  "consumer_p90_us": pct(s["cons"], 0.9)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", choices=("plan", "run"), default=None)
    ap.add_argument("--out", default="efu.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--cache", default=None, help="B12X_NEXT_COMPILE_CACHE_DIR (default: a fresh temp dir)")
    ap.add_argument("--prepared", default="plan,direct", help="DSV41_MOE_LAUNCH_PREPARED (production: plan,direct)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.mode == "plan":
        print(plan_text(a.reps))
        return 0
    if a.mode == "run":
        return run(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
