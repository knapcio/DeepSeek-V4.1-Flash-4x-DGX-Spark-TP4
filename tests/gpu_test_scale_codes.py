"""GPU bit-identity gate for DSV41_MOE_SCALE_CODES (one idle GB10; the first gate of the scale-codes window).

Two processes, because the gate is read at import: `run --codes 0` and `run --codes 1`, both with production's
DSV41_MOE_B12X_NEXT_DETERMINISTIC=1 and production's plan settings (the adapter defaults), drive the engine's own MoE
method through tests/test_moe_b12x_next.py's harness (real FusedMoE loader, real 1017 checkpoint slices from
MODEL_DIR, CKPT_EXPERTS experts per layer). `compare` judges against a manifest FROZEN HERE, not against what either
process happened to produce (REVIEW_sol_scalecodes R3/R4):

  layers x TP slices   layers.3 and mtp.2 on all four moe_tp ranks; layers.20 (rank 0), layers.39 (rank 1), mtp.0
                       (rank 2), mtp.1 (rank 3). mtp.1 / mtp.2 hold the widest exponent spans of the checkpoint.
  plans                every capacity the adapter prepares (EXPECTED_CAPS: the CUDA-graph verify/draft row counts and
                       the prefill ladder); each geometry must prepare exactly that set (else INVALID: the plan
                       manifest changed). Every plan's config string and implementation are recorded.
  rows                 every capacity, capacity - 1, and 2049 / 4095 / 4097 / 5000 (ladder and chunk boundaries),
                       each with three routings: uniform, concentrated (every token on the same experts) and
                       last-expert (expert E-1 in every row); one CUDA-graph padding row (id -1) in each.
  graphs               capture + replay with changed inputs at every decode capacity <= 96.
  checks               every expected output present, bf16 [M, 5120], finite, nonzero on live rows, zero on the
                       padding row; codes off == codes on byte for byte; each process identical to itself on a rerun
                       (else INVALID); the coded process converted every layer build, each receipt freed >= 90 % of
                       native - codes (measured after the release), coded layer bytes < native layer bytes; the native
                       process converted nothing. Missing or extra keys, wrong mode metadata or shapes: FAIL.
  prepared launch      with DSV41_MOE_LAUNCH_PREPARED=plan in the environment (v2.6 production), each run records
                       adapter/moe_launch_prepared.py's counters: both runs must have built and hit prepared plans
                       (else INVALID: the production launch path was not exercised); the coded run's plans must hold
                       the coded kernel and the native run's none (else FAIL).
  micro                the adapter pins <= 8-row plans to `dynamic`; a separate geometry with that pin lifted
                       (plan table "1-8=heur") is run in both processes and its plan implementations recorded; if b12x
                       picks `micro` there, its outputs join the comparison, otherwise the report says the compact
                       micro coded path was not exercised (production does not select it at default settings).

  sanitize --tool memcheck|racecheck|synccheck: one layer (layers.3, rank 3), codes on, eager, M 1, 6, 96, 2049, for
  running under compute-sanitizer with the strict release receipt (the kit requires "ERROR SUMMARY: 0 errors", exit 0
  and "SANITIZE-RECEIPT <tool>: released" for each tool).

  layers --codes 0|1 --out F / compare-layers A B (2026-10-30): every MoE layer (layers.0-39, mtp.0-2), one TP rank per
  layer (index % 4), CKPT_EXPERTS experts from a window that rotates through the layer's experts, M 6 / 96 / 2049,
  uniform and last-expert routing: codes off == codes on byte for byte, 43 receipts released.

  v2.7.3 production (2026-11-10): unless set, every mode runs with the production profile's DSV41_MOE_LAUNCH_PREPARED=
  plan,direct and CHUNKED_PREFILL_SIZE=8192 (SC_CHUNKED_PREFILL_SIZE, else CHUNKED_PREFILL_SIZE, sets the chunk the
  manifest is built for; SC_CHUNKED_PREFILL_SIZE=0 = the adapter's default ladder): the plans include the 8192-row
  capacity, and rows 8191 / 8192 / 8193 join the manifest.

  retain [--out DIR] / retain-run --out F / compare-retain F (2026-11-10, the in-boot A/B arm X_sc): ONE process with
  the A/B armed (DSV41_AB_VARIANTS=3; v0 codes off, v1 codes on with DSV41_MOE_SCALE_CODES_MIN_ROWS=1, v2 codes on at
  the default floor 2), i.e. moe_b12x_next's retain mode: both expert sets per layer, plans native. For layers.3 (rank
  3) and mtp.2 (rank 0), M in RETAIN_MS (1 / 6 / 12 / 24 / 48 / 96), uniform and last-expert routing with a padding
  row: eager native (no capture) vs eager coded (inside ab_variant.capturing(1), the capture warm-up path) vs CUDA graphs
  captured under variants 0 and 1 and replayed on new inputs, each against the eager native output of the replayed
  inputs, byte for byte. The adapter's own decisions are judged too: v0 native everywhere, v1 coded everywhere,
  v2 native at M=1 and coded above; the native scale storage still holds its bytes; the process-wide
  B12X_NEXT_W4A8_SCALE_CODES is unset; with the prepared launch requested, coded and native prepared plans both exist.
  sanitize --retain: the same one-layer sanitizer workload with retain mode, native and coded calls in one process.

  plan: prints the manifest and the environment the modes would use, without torch or a GPU (dry check).

usage (in the overlay image, one idle node):
  PYTHONPATH=/opt/dsv41/adapter:/opt/b12x:/opt/b12x_next MODEL_DIR=/model python3 gpu_test_scale_codes.py [--out DIR]
exit 0 PASS, 1 FAIL, 75 INVALID. The window treats anything but PASS as a hard stop.
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
K_HID = 5120
GRAPH_BS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)          # the adapter's default DSV41_MOE_B12X_NEXT_GRAPH_BS
BLOCK = 5                                                     # DSPARK_BLOCK_SIZE
LADDER = (128, 256, 512, 1024, 2048, 4096)                    # DSV41_MOE_B12X_NEXT_LADDER
# v2.7.3 production: CHUNKED_PREFILL_SIZE=8192 adds the 8192-row capacity to the adapter's ladder
CHUNK = int(os.environ.get("SC_CHUNKED_PREFILL_SIZE") or os.environ.get("CHUNKED_PREFILL_SIZE") or 8192)
PROD_ENV = {"DSV41_MOE_LAUNCH_PREPARED": "plan,direct", "CHUNKED_PREFILL_SIZE": str(CHUNK) if CHUNK else ""}
EXPECTED_CAPS = tuple(sorted({r * b for r in (BLOCK, BLOCK + 1) for b in GRAPH_BS} | set(LADDER)
                             | ({CHUNK} if CHUNK > max(LADDER) else set())))
EXTRA_MS = (2049, 4095, 4097, 5000) + ((CHUNK + 1,) if CHUNK > max(LADDER) else ())
MS = tuple(sorted(set(EXPECTED_CAPS) | {c - 1 for c in EXPECTED_CAPS if c > 1} | set(EXTRA_MS)))
GRAPH_MS = tuple(c for c in EXPECTED_CAPS if c <= 96)
ROUTINGS = ("uniform", "concentrated", "last")
BUILDS = tuple([("layers.3", 6, r) for r in range(4)] + [("layers.20", 6, 0), ("layers.39", 6, 1), ("mtp.0", 3, 2),
                                                          ("mtp.1", 3, 3)] + [("mtp.2", 3, r) for r in range(4)])


def build_key(prefix, rank):
    return f"{prefix}@tp{rank}"


def expected_keys():
    keys = set()
    for prefix, _, rank in BUILDS:
        b = build_key(prefix, rank)
        keys |= {f"{b}/{rt}/M{m}" for m in MS for rt in ROUTINGS}
        keys |= {f"{b}/graph/M{m}" for m in GRAPH_MS}
    return keys


def routed(h, torch, m, n_global, topk, seed, kind):
    x, ids, w = h.routing(m, n_global, topk, seed)
    if kind == "concentrated":
        ids[:-1] = torch.arange(topk, dtype=ids.dtype, device=ids.device)
    elif kind == "last":
        ids[:-1, 0] = n_global - 1
        ids[:-1, 1:] = torch.arange(topk - 1, dtype=ids.dtype, device=ids.device)
    if m > 1:
        ids[-1] = -1                                          # CUDA-graph padding row
    return x, ids.contiguous(), w


def production_env():
    """The v2.7.3 production profile's MoE launch settings: the prepared launch unless the caller set it (an empty
    value unsets), the chunk the manifest was built for (CHUNK; 0 = the adapter's default ladder)."""
    os.environ.setdefault("DSV41_MOE_LAUNCH_PREPARED", PROD_ENV["DSV41_MOE_LAUNCH_PREPARED"])
    if not os.environ["DSV41_MOE_LAUNCH_PREPARED"]:
        del os.environ["DSV41_MOE_LAUNCH_PREPARED"]
    if CHUNK:
        os.environ["CHUNKED_PREFILL_SIZE"] = str(CHUNK)
    else:
        os.environ.pop("CHUNKED_PREFILL_SIZE", None)
    os.environ["DSV41_MOE_B12X_NEXT_DETERMINISTIC"] = "1"
    # the harness caps the process at TEST_MAX_GB (2.0 by default): the 8192-row plan's scratch and rows need more
    os.environ.setdefault("TEST_MAX_GB", "6")


def setup(codes):
    # the codes-only (release) path: since v2.7.5 DSV41_MOE_SCALE_CODES=1 alone is retain mode (--retain checks that one)
    os.environ["DSV41_MOE_SCALE_CODES"] = "release" if codes else "0"
    production_env()
    sys.argv = [sys.argv[0], "--quick"]
    sys.path.insert(0, HERE)
    import torch
    import test_moe_b12x_next as h                     # engine stubs + adapter install + checkpoint access
    assert h.ad.SCALE_CODES == bool(codes), (h.ad.SCALE_CODES, codes)
    assert (os.environ.get("B12X_NEXT_W4A8_SCALE_CODES") == "1") == bool(codes)
    return torch, h


def prepared_state():
    """adapter/moe_launch_prepared.py's counters (None when the prepared launch is not installed)."""
    P = sys.modules.get("moe_launch_prepared")
    if P is None or not P._STATE.get("installed"):
        return None
    s = {k: P._STATE.get(k) for k in ("mode", "calls", "hits", "builds", "stock")}
    s["plans"] = len(P._PLANS)
    s["coded_plans"] = sum(1 for p in P._PLANS.values() if p.kernel_key and "w4a8_coded_scales" in p.kernel_key)
    return s


def make(h, prefix, topk, rank):
    geo = dict(topk=topk, ep_size=1, ep_rank=0, moe_tp_size=4, moe_tp_rank=rank)
    ly, me, _, _ = h.make_layer(prefix, n_global=h.NE, **geo)
    return ly, me


def run(codes, out):
    torch, h = setup(codes)
    ad = h.ad
    outs, rerun, layer_mem, plans, caps_seen, prep_seen = {}, {}, {}, {}, {}, {}
    for prefix, topk, rank in BUILDS:
        b = build_key(prefix, rank)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        ly, me = make(h, prefix, topk, rank)
        torch.cuda.synchronize()
        layer_mem[b] = torch.cuda.memory_allocated() - before
        g = ly._dsv41_b12x_next.geom
        caps_seen[b] = sorted(int(c) for c in g.plans)
        plans[b] = {str(c): str(getattr(p, "config", ""))[:240] for c, p in g.plans.items()}
        for m in MS:
            for i, kind in enumerate(ROUTINGS):
                x, ids, w = routed(h, torch, m, h.NE, topk, 1000 + 7 * m + i, kind)
                o = h.apply(ly, me, x, ids, w)
                o2 = h.apply(ly, me, x, ids, w)
                k = f"{b}/{kind}/M{m}"
                rerun[k] = bool(torch.equal(o, o2))
                outs[k] = o.detach().cpu()
        for m in GRAPH_MS:
            x, ids, w = routed(h, torch, m, h.NE, topk, 2000 + m, "uniform")
            sx, si, sw = x.clone(), ids.clone(), w.clone()
            h.apply(ly, me, sx, si, sw)                  # warm outside capture
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                so = h.apply(ly, me, sx, si, sw)
            x2, ids2, w2 = routed(h, torch, m, h.NE, topk, 3000 + m, "concentrated")
            sx.copy_(x2)
            si.copy_(ids2)
            sw.copy_(w2)
            gr.replay()
            torch.cuda.synchronize()
            k = f"{b}/graph/M{m}"
            outs[k] = so.detach().cpu().clone()
            rerun[k] = bool(torch.equal(h.apply(ly, me, x2, ids2, w2).cpu(), outs[k]))
            del gr
        ps = prepared_state()
        if ps is not None:                               # plans may be dropped on an env change: keep the maxima
            for key in ("plans", "coded_plans"):
                prep_seen[key] = max(prep_seen.get(key, 0), ps[key])
        del ly, me, g
        torch.cuda.empty_cache()
    # micro probe: the <= 8 row pin lifted (separate geometry)
    micro = {"plans": {}, "outputs": 0}
    saved = dict(ad._GEOMS), ad.PLAN_TABLE
    ad._GEOMS.clear()
    ad.PLAN_TABLE = ad.parse_plan_table("1-8=heur", False)
    try:
        ly, me = make(h, "layers.3", 6, 3)
        g = ly._dsv41_b12x_next.geom
        micro["plans"] = {str(c): str(getattr(p, "config", ""))[:240] for c, p in g.plans.items() if c <= 8}
        if any("micro" in v for v in micro["plans"].values()):
            for m in (1, 3, 5, 6, 8):
                x, ids, w = routed(h, torch, m, h.NE, 6, 4000 + m, "uniform")
                o = h.apply(ly, me, x, ids, w)
                k = f"micro/layers.3@tp3/M{m}"
                outs[k] = o.detach().cpu()
                rerun[k] = bool(torch.equal(o, h.apply(ly, me, x, ids, w)))
                micro["outputs"] += 1
        del ly, me
    finally:
        ad._GEOMS.clear()
        ad._GEOMS.update(saved[0])
        ad.PLAN_TABLE = saved[1]
    meta = {"codes": bool(codes), "scale_codes_flag": bool(ad.SCALE_CODES), "rerun_same": rerun,
            "layer_bytes": layer_mem, "plans": plans, "caps_seen": caps_seen, "micro": micro,
            "codes_stats": {k: v for k, v in ad._CODES.items()}, "experts": h.NE, "model": h.MODEL,
            "prepared_requested": os.environ.get("DSV41_MOE_LAUNCH_PREPARED", "0").strip().lower()
            not in ("", "0", "off", "false", "no"),
            "prepared": None if prepared_state() is None else {**prepared_state(), **{
                f"max_{k}": v for k, v in prep_seen.items()}},
            "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    torch.save({"outs": outs, "meta": meta}, out)
    print(json.dumps({"saved": out, "codes": bool(codes), "outputs": len(outs),
                      "rerun_identical": sum(rerun.values()), "of": len(rerun)}))


# racecheck is slow in the stock kernels' prepare at large M (10-27: cut at 3,300 s at M 2049): the kit bounds it
SANITIZE_MS = tuple(int(x) for x in os.environ.get("SC_SANITIZE_MS", "1,6,96,2049").split(","))


def sanitize_retain(tool):
    """The sanitizer workload in A/B retain mode: one process, the native and the coded experts of one layer, each M
    run native (eager) and coded (inside ab_variant.capturing(1)). Nothing is released, so there is no receipt; the
    racecheck rule stays differential against `sanitize --codes 0`."""
    torch, h, ab = setup_retain()
    ly, me = make(h, "layers.3", 6, 3)
    for m in SANITIZE_MS:
        x, ids, w = routed(h, torch, m, h.NE, 6, 5000 + m, "uniform")
        o = h.apply(ly, me, x, ids, w)
        with ab.capturing(1):
            oc = h.apply(ly, me, x, ids, w)
        torch.cuda.synchronize()
        assert torch.isfinite(o).all() and torch.isfinite(oc).all()
    print(f"sanitize {tool}: ran M {SANITIZE_MS} in retain mode (native + coded); decisions "
          f"{sorted((k[1], v) for k, v in h.ad._RETAIN['decisions'].items())}", flush=True)


def sanitize(tool, codes=1):
    """Memory-safety workload for compute-sanitizer (codes on), with the STRICT receipt (2026-10-30). Until 10-27 the
    receipt inferred the release from references and refused under every tool (views of the native scales stayed
    referenced only under the sanitizer), so this leg downgraded it to information. With single-storage ownership the
    native storage is emptied, the receipt checks the storages, the coded tree and the allocator, and a refusal here
    fails the leg (non-zero exit). DSV41_MOE_SCALE_CODES_HOLDERS=1 names whatever still holds an (empty) alias: the
    10-27 holders' identity, printed as SANITIZE-HOLDERS."""
    os.environ["DSV41_MOE_SCALE_CODES_HOLDERS"] = "1"
    torch, h = setup(codes)
    ad = h.ad
    ly, me = make(h, "layers.3", 6, 3)
    if not codes:                                  # the racecheck baseline: the same workload on native scales
        for m in SANITIZE_MS:
            x, ids, w = routed(h, torch, m, h.NE, 6, 5000 + m, "uniform")
            o = h.apply(ly, me, x, ids, w)
            torch.cuda.synchronize()
            assert torch.isfinite(o).all()
        print(f"sanitize {tool}: ran M {SANITIZE_MS} with codes off (baseline)")
        return
    r = (ad._CODES.get("per_layer") or [{}])[-1]
    ok = r.get("released") == 2 and r.get("freed", -1) >= 0.9 * (r.get("native", 0) - r.get("coded", 0)) > 0
    print(f"SANITIZE-RECEIPT {tool}: {'released' if ok else 'NOT RELEASED'} (strict receipt passed under the sanitizer)"
          f" native {r.get('native')} coded {r.get('coded')} freed {r.get('freed')} empty aliases {r.get('aliases')}",
          flush=True)
    print(f"SANITIZE-HOLDERS {tool}: {r.get('holders', 'none')}", flush=True)
    if not ok:
        raise SystemExit(f"sanitize {tool}: receipt record incomplete: {r}")
    for m in SANITIZE_MS:
        x, ids, w = routed(h, torch, m, h.NE, 6, 5000 + m, "uniform")
        o = h.apply(ly, me, x, ids, w)
        torch.cuda.synchronize()
        assert torch.isfinite(o).all()
    print(f"sanitize {tool}: ran M {SANITIZE_MS} with codes on")


# ---- A/B retain mode (2026-11-10): native and coded experts in one process, coded only in captures of variant 1/2
RETAIN_MS = (1, 6, 12, 24, 48, 96)
RETAIN_BUILDS = (("layers.3", 6, 3), ("mtp.2", 3, 0))
RETAIN_ROUTINGS = ("uniform", "last")
RETAIN_MODES = ("eager-native", "eager-coded", "graph-native", "graph-coded", "graph-ref")
RETAIN_VARIANTS = {"DSV41_AB_V0": "DSV41_MOE_SCALE_CODES=0",
                   "DSV41_AB_V1": "DSV41_MOE_SCALE_CODES=1;DSV41_MOE_SCALE_CODES_MIN_ROWS=1",
                   "DSV41_AB_V2": "DSV41_MOE_SCALE_CODES=1"}


def retain_want(variant, m):
    """The adapter's expected choice: v0 native, v1 coded (floor 1), v2 coded from 2 rows (the default floor)."""
    return variant == 1 or (variant == 2 and m >= 2)


def retain_keys():
    return {f"{build_key(p, r)}/{mode}/{rt}/M{m}" for p, _, r in RETAIN_BUILDS for mode in RETAIN_MODES
            for rt in RETAIN_ROUTINGS for m in RETAIN_MS}


def setup_retain():
    """Arm the in-boot A/B as sitecustomize does (configure() before the adapter import), then the harness."""
    os.environ.pop("DSV41_MOE_SCALE_CODES", None)
    os.environ.update(DSV41_AB_VARIANTS="3", **RETAIN_VARIANTS)
    os.environ.setdefault("DSV41_AB_FILE", "/tmp/sc-retain-ab_variant")
    production_env()
    sys.argv = [sys.argv[0], "--quick"]
    sys.path.insert(0, HERE)
    import ab_variant as ab
    assert ab.configure(), "the in-boot A/B did not arm"
    assert os.environ.get("DSV41_MOE_SCALE_CODES") == "1", "the union did not switch the scale-codes gate on"
    import torch
    import test_moe_b12x_next as h
    assert h.ad.RETAIN and h.ad._AB is ab, "moe_b12x_next is not in A/B retain mode"
    assert "B12X_NEXT_W4A8_SCALE_CODES" not in os.environ, "retain mode set the process-wide switch"
    return torch, h, ab


def run_retain(out):
    torch, h, ab = setup_retain()
    ad = h.ad
    outs, decisions, storages, why = {}, {}, {}, []
    for prefix, topk, rank in RETAIN_BUILDS:
        b = build_key(prefix, rank)
        ly, me = make(h, prefix, topk, rank)
        st = ly._dsv41_b12x_next
        if st.coded_impl is None or st.coded_impl is st.impl:
            why.append(f"{b}: the layer does not hold two expert sets")
        storages[b] = [int(t.untyped_storage().nbytes()) for t in (ly.w13_weight_scale_inv, ly.w2_weight_scale_inv)]
        storages[b + "/coded_caps"] = sorted(int(c) for c in st.geom.coded_caps)
        storages[b + "/cap_of"] = {str(m): int(st.geom.cap_for(m)) for m in RETAIN_MS}
        for m in RETAIN_MS:
            for i, kind in enumerate(RETAIN_ROUTINGS):
                x, ids, w = routed(h, torch, m, h.NE, topk, 9000 + 7 * m + i, kind)
                key = f"{b}/{{}}/{kind}/M{m}"
                outs[key.format("eager-native")] = h.apply(ly, me, x, ids, w).detach().cpu()
                with ab.capturing(1):
                    outs[key.format("eager-coded")] = h.apply(ly, me, x, ids, w).detach().cpu()
                x2, ids2, w2 = routed(h, torch, m, h.NE, topk, 9500 + 7 * m + i, "concentrated" if m > 1 else kind)
                outs[key.format("graph-ref")] = h.apply(ly, me, x2, ids2, w2).detach().cpu()
                for v, mode in ((0, "graph-native"), (1, "graph-coded")):
                    sx, si, sw = x.clone(), ids.clone(), w.clone()
                    with ab.capturing(v):
                        h.apply(ly, me, sx, si, sw)                 # the capture warm-up (eager, same variant)
                        torch.cuda.synchronize()
                        gr = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(gr):
                            so = h.apply(ly, me, sx, si, sw)
                    sx.copy_(x2)
                    si.copy_(ids2)
                    sw.copy_(w2)
                    gr.replay()
                    torch.cuda.synchronize()
                    outs[key.format(mode)] = so.detach().cpu().clone()
                    del gr
            with ab.capturing(2):                                   # the default floor: decisions only
                x, ids, w = routed(h, torch, m, h.NE, topk, 9900 + m, "uniform")
                h.apply(ly, me, x, ids, w)
        for (v, m, k), coded in ad._RETAIN["decisions"].items():
            if k == topk:
                decisions[f"{b}/v{v}/M{m}"] = bool(coded)
        del ly, me, st
        torch.cuda.empty_cache()
    meta = {"retain": True, "decisions": decisions, "storages": storages, "why": why,
            "retain_stats": {k: v for k, v in ad._RETAIN.items() if k in ("layers", "native", "coded", "max_span")},
            "switch": os.environ.get("B12X_NEXT_W4A8_SCALE_CODES"), "experts": h.NE, "model": h.MODEL,
            "prepared_requested": os.environ.get("DSV41_MOE_LAUNCH_PREPARED", "0").strip().lower()
            not in ("", "0", "off", "false", "no"), "prepared": prepared_state(),
            "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    torch.save({"outs": outs, "meta": meta}, out)
    print(json.dumps({"saved": out, "retain": True, "outputs": len(outs), "decisions": len(decisions)}))


def compare_retain(path):
    """Judge one retain-run file against the frozen retain manifest. Returns (verdict, exit code, report)."""
    import torch
    try:
        doc = torch.load(path)
        A, meta = doc["outs"], doc["meta"]
    except Exception as exc:  # noqa: BLE001
        return "FAIL", 1, {"verdict": "FAIL", "why": [f"unreadable run file: {exc!r}"[:200]]}
    why, invalid = list(meta.get("why") or []), []
    want = retain_keys()
    if set(A) != want:
        why.append(f"manifest: expected {len(want)}, missing {len(want - set(A))}, unexpected {len(set(A) - want)}")
    if meta.get("retain") is not True or meta.get("switch") is not None:
        why.append(f"not a retain run, or the process-wide switch was set ({meta.get('switch')!r})")
    differ = []
    for p, _, r in RETAIN_BUILDS:
        b = build_key(p, r)
        st = (meta.get("storages") or {}).get(b)
        if not st or min(st) <= 0:
            why.append(f"{b}: native scale storage released in retain mode ({st})")
        caps = (meta.get("storages") or {}).get(b + "/coded_caps") or []
        cap_of = (meta.get("storages") or {}).get(b + "/cap_of") or {}
        for m in RETAIN_MS:
            for v in (0, 1, 2):
                got = (meta.get("decisions") or {}).get(f"{b}/v{v}/M{m}")
                if got is None:
                    why.append(f"{b}: no decision recorded for variant {v} M={m}")
                elif got != retain_want(v, m):
                    # a capacity whose plan is compact micro is not prepared for codes: not exercised, not wrong
                    unprepared = retain_want(v, m) and cap_of.get(str(m)) not in caps
                    (invalid if unprepared else why).append(
                        f"{b}: variant {v} M={m} ran {'coded' if got else 'native'} (capacity {cap_of.get(str(m))}, "
                        f"coded capacities {caps})")
            for rt in RETAIN_ROUTINGS:
                k = f"{b}/{{}}/{rt}/M{m}"
                for mode in RETAIN_MODES:
                    t = A.get(k.format(mode))
                    if t is None:
                        continue
                    if t.dtype != torch.bfloat16 or tuple(t.shape) != (m, K_HID) or not torch.isfinite(t.float()).all():
                        why.append(f"{k.format(mode)}: dtype {t.dtype} shape {tuple(t.shape)} or non-finite")
                        continue
                    live = t[:-1] if m > 1 else t
                    if float(live.float().abs().sum()) == 0.0 or (m > 1 and float(t[-1].float().abs().max()) != 0.0):
                        why.append(f"{k.format(mode)}: zero live rows or a nonzero padding row")
                for x_mode, y_mode in (("eager-native", "eager-coded"), ("graph-ref", "graph-coded"),
                                       ("graph-ref", "graph-native")):
                    x, y = A.get(k.format(x_mode)), A.get(k.format(y_mode))
                    if x is not None and y is not None and not (x.shape == y.shape and x.dtype == y.dtype
                                                                and torch.equal(_bits(x, torch), _bits(y, torch))):
                        differ.append(f"{k.format(y_mode)} != {x_mode}")
    rs = meta.get("retain_stats") or {}
    if rs.get("layers") != len(RETAIN_BUILDS) or not 0 < rs.get("coded", 0) < rs.get("native", 0) \
            or not 0 < rs.get("max_span", 0) <= 16:
        why.append(f"retain stats {rs}")
    if meta.get("prepared_requested"):
        ps = meta.get("prepared") or {}
        if ps.get("mode") not in ("plan", "plan,direct") or not (ps.get("builds") or 0) > 0:
            invalid.append(f"the prepared launch was requested but did not build plans ({ps})")
        elif not (ps.get("coded_plans") or 0) > 0 or not (ps.get("plans", 0) - ps.get("coded_plans", 0)) > 0:
            why.append(f"retain mode should hold coded and native prepared plans ({ps})")
    rep = {"expected_outputs": len(want), "n_differ": len(differ), "differ": differ[:20], "why": why[:30],
           "invalid": invalid, "retain_stats": rs, "prepared": meta.get("prepared")}
    verdict, code = ("FAIL", 1) if differ or why else ("INVALID", 75) if invalid else ("PASS", 0)
    rep["verdict"] = verdict
    return verdict, code, rep


def plan():
    """Dry: the manifests and the environment, no torch, no GPU."""
    env = {k: os.environ.get(k, v) for k, v in PROD_ENV.items()}
    return {"gate": {"builds": [list(b) for b in BUILDS], "expected_caps": list(EXPECTED_CAPS), "rows": list(MS),
                     "graph_rows": list(GRAPH_MS), "routings": list(ROUTINGS), "outputs": len(expected_keys())},
            "layers": {"builds": len(ALL_LAYERS), "rows": list(LAYER_MS)},
            "retain": {"builds": [list(b) for b in RETAIN_BUILDS], "rows": list(RETAIN_MS), "modes": list(RETAIN_MODES),
                       "variants": RETAIN_VARIANTS, "outputs": len(retain_keys()),
                       "decisions": {f"v{v}/M{m}": "coded" if retain_want(v, m) else "native"
                                     for v in (0, 1, 2) for m in RETAIN_MS}},
            "sanitize_rows": list(SANITIZE_MS), "env": env, "deterministic": "1"}


# ---- every MoE layer (2026-10-30): codes off vs on on each of the 43 layers' real scales, one TP rank per layer
ALL_LAYERS = tuple([(f"layers.{i}", 6) for i in range(40)] + [(f"mtp.{i}", 3) for i in range(3)])
LAYER_MS = (6, 96, 2049)
LAYER_ROUTINGS = ("uniform", "last")


def layer_builds(ne):
    """(prefix, topk, rank, first expert): rank = layer index % 4; the expert window rotates through the layer's
    experts (384 routed per main layer, 128 per DSpark stage) so the sweep covers different experts per layer."""
    out = []
    for i, (prefix, topk) in enumerate(ALL_LAYERS):
        n_exp = 384 if prefix.startswith("layers.") else 128
        out.append((prefix, topk, i % 4, (i * ne) % (n_exp - ne + 1)))
    return out


def layer_keys(ne):
    return {f"{p}@tp{r}+e{b}/{rt}/M{m}" for p, _, r, b in layer_builds(ne) for m in LAYER_MS for rt in LAYER_ROUTINGS}


def run_layers(codes, out):
    torch, h = setup(codes)
    ad = h.ad
    outs, rerun = {}, {}
    for prefix, topk, rank, base in layer_builds(h.NE):
        os.environ["CKPT_EXPERT_BASE"] = str(base)
        ly, me = make(h, prefix, topk, rank)
        b = f"{prefix}@tp{rank}+e{base}"
        for m in LAYER_MS:
            for i, kind in enumerate(LAYER_ROUTINGS):
                x, ids, w = routed(h, torch, m, h.NE, topk, 7000 + 7 * m + i, kind)
                o = h.apply(ly, me, x, ids, w)
                k = f"{b}/{kind}/M{m}"
                rerun[k] = bool(torch.equal(o, h.apply(ly, me, x, ids, w)))
                outs[k] = o.detach().cpu()
        del ly, me
        torch.cuda.empty_cache()
        print(f"LAYER {b} done", flush=True)
    os.environ.pop("CKPT_EXPERT_BASE", None)
    meta = {"codes": bool(codes), "scale_codes_flag": bool(ad.SCALE_CODES), "rerun_same": rerun, "experts": h.NE,
            "codes_stats": {k: v for k, v in ad._CODES.items()}, "model": h.MODEL}
    torch.save({"outs": outs, "meta": meta}, out)
    print(json.dumps({"saved": out, "codes": bool(codes), "outputs": len(outs)}))


def compare_layers(a_path, b_path):
    import torch
    try:
        a, b = torch.load(a_path), torch.load(b_path)
        A, B, ma, mb = a["outs"], b["outs"], a["meta"], b["meta"]
    except Exception as exc:  # noqa: BLE001
        return "FAIL", 1, {"verdict": "FAIL", "why": [f"unreadable run files: {exc!r}"[:200]]}
    why, invalid = [], []
    ne = ma.get("experts")
    want = layer_keys(ne) if isinstance(ne, int) and ne == mb.get("experts") else set()
    if not want or set(A) != want or set(B) != want:
        why.append(f"manifest: expected {len(want)}, native {len(A)}, coded {len(B)}")
    if ma.get("codes") is not False or mb.get("codes") is not True:
        why.append("the files are not a codes-off and a codes-on run")
    for mm, name in ((ma, "native"), (mb, "coded")):
        if not all((mm.get("rerun_same") or {0: False}).values()):
            invalid.append(f"{name}: not deterministic on a rerun")
    differ = []
    for k in sorted(want & set(A) & set(B)):
        x, y = A[k], B[k]
        m = int(k.rsplit("/M", 1)[1])
        if x.dtype != torch.bfloat16 or tuple(x.shape) != (m, K_HID) or not torch.isfinite(x.float()).all() \
                or float(x[:-1].float().abs().sum()) == 0.0:
            why.append(f"native {k}: dtype / shape / non-finite / zero live rows")
        if not (x.shape == y.shape and x.dtype == y.dtype and torch.equal(_bits(x, torch), _bits(y, torch))):
            differ.append(k)
    per = (mb.get("codes_stats") or {}).get("per_layer") or []
    if len(per) != len(ALL_LAYERS):
        why.append(f"coded: {len(per)} receipts, {len(ALL_LAYERS)} expected")
    for i, r in enumerate(per):
        if not (r.get("released") == 2 and r.get("freed", -1) >= 0.9 * (r.get("native", 0) - r.get("coded", 0)) > 0
                and 0 < r.get("span", 99) <= 16):
            why.append(f"coded receipt {i}: {r}")
    if ((ma.get("codes_stats") or {}).get("layers", -1)) != 0:
        why.append("native: a layer was converted")
    rep = {"expected_outputs": len(want), "n_differ": len(differ), "differ": differ[:20], "why": why[:30],
           "invalid": invalid, "max_span": max([r.get("span", 0) for r in per] or [0])}
    verdict, code = ("FAIL", 1) if differ or why else ("INVALID", 75) if invalid else ("PASS", 0)
    rep["verdict"] = verdict
    return verdict, code, rep


def _bits(t, torch):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def compare(a_path, b_path):
    """Judge two run files against the frozen manifest. Returns (verdict, exit code, report)."""
    import torch
    try:
        a, b = torch.load(a_path), torch.load(b_path)
        A, B, ma, mb = a["outs"], b["outs"], a["meta"], b["meta"]
    except Exception as exc:  # noqa: BLE001 - unreadable evidence
        rep = {"verdict": "FAIL", "why": [f"unreadable run files: {exc!r}"[:200]]}
        return "FAIL", 1, rep
    why, invalid = [], []
    want = expected_keys()
    extra_micro = {k for k in set(A) | set(B) if k.startswith("micro/")}
    if set(A) - extra_micro != want or set(B) - extra_micro != want:
        why.append(f"output manifest: native missing {len(want - set(A))}, coded missing {len(want - set(B))}, "
                   f"unexpected {len((set(A) | set(B)) - want - extra_micro)} (expected {len(want)})")
    if {k for k in A if k.startswith("micro/")} != {k for k in B if k.startswith("micro/")}:
        why.append("the micro probe ran in one process only")
    if ma.get("codes") is not False or ma.get("scale_codes_flag") is not False:
        why.append("the first file is not a codes-off run")
    if mb.get("codes") is not True or mb.get("scale_codes_flag") is not True:
        why.append("the second file is not a codes-on run")
    for mm, name in ((ma, "native"), (mb, "coded")):
        builds = {build_key(p, r) for p, _, r in BUILDS}
        if set(mm.get("caps_seen", {})) != builds or any(v != list(EXPECTED_CAPS) for v in mm["caps_seen"].values()):
            invalid.append(f"{name}: the adapter prepared other plan capacities than production's")
        if set(mm.get("rerun_same", {})) != set(A if name == "native" else B):
            why.append(f"{name}: rerun records do not cover every output")
        elif not all(mm["rerun_same"].values()):
            invalid.append(f"{name}: not deterministic on a rerun ({sum(not v for v in mm['rerun_same'].values())})")
        if set(mm.get("layer_bytes", {})) != builds or any(int(v) <= 0 for v in mm["layer_bytes"].values()):
            why.append(f"{name}: per-build memory records missing or non-positive")
    # v2.6 production launches eager dynamic MoE through the prepared launch (DSV41_MOE_LAUNCH_PREPARED=plan)
    if ma.get("prepared_requested") != mb.get("prepared_requested"):
        why.append("the prepared launch was requested in one run only")
    elif mb.get("prepared_requested"):
        for mm, name, coded in ((ma, "native", False), (mb, "coded", True)):
            ps = mm.get("prepared") or {}
            if ps.get("mode") not in ("plan", "plan,direct") or not (ps.get("builds") or 0) > 0 \
                    or not (ps.get("hits") or 0) > 0:
                invalid.append(f"{name}: the prepared launch was requested but did not build and hit plans ({ps})")
            elif coded and not (ps.get("max_coded_plans") or 0) > 0:
                why.append(f"coded: no prepared plan holds the coded kernel ({ps})")
            elif not coded and (ps.get("max_coded_plans") or 0) != 0:
                why.append(f"native: a prepared plan holds the coded kernel ({ps})")
    differ = []
    for k in sorted(want & set(A) & set(B)) + sorted(extra_micro & set(A) & set(B)):
        x, y = A[k], B[k]
        m = int(k.rsplit("/M", 1)[1])
        for t, nm in ((x, "native"), (y, "coded")):
            if t.dtype != torch.bfloat16 or tuple(t.shape) != (m, K_HID) or not torch.isfinite(t.float()).all():
                why.append(f"{nm} {k}: dtype {t.dtype} shape {tuple(t.shape)} or non-finite")
                break
            live = t[:-1] if m > 1 else t
            if float(live.float().abs().sum()) == 0.0 or (m > 1 and float(t[-1].float().abs().max()) != 0.0):
                why.append(f"{nm} {k}: zero live rows or a nonzero padding row")
                break
        if x.shape == y.shape and x.dtype == y.dtype and not torch.equal(_bits(x, torch), _bits(y, torch)):
            differ.append(k)
    st = mb.get("codes_stats", {})
    per = st.get("per_layer") or []
    if len(per) != len(BUILDS) + 1:                     # every build + the micro probe geometry
        why.append(f"coded: {len(per)} conversion receipts, {len(BUILDS) + 1} expected")
    for i, r in enumerate(per):
        if not (r.get("freed", -1) >= 0.9 * (r.get("native", 0) - r.get("coded", 0)) > 0 and 0 < r.get("span", 99) <= 16
                and r.get("released") == 2):
            why.append(f"coded receipt {i}: {r}")
    if (ma.get("codes_stats") or {}).get("layers", -1) != 0:
        why.append("native: a layer was converted")
    for k in ma.get("layer_bytes", {}):
        if not mb.get("layer_bytes", {}).get(k, 1 << 62) < ma["layer_bytes"][k]:
            why.append(f"{k}: coded layer bytes not below native")
    rep = {"expected_outputs": len(want), "micro_outputs": len(extra_micro & set(A)), "differ": differ[:20],
           "n_differ": len(differ), "why": why[:30], "invalid": invalid, "micro_plans": (mb.get("micro") or {}).get("plans"),
           "plans_example": next(iter((mb.get("plans") or {}).values()), None),
           "prepared": {"native": ma.get("prepared"), "coded": mb.get("prepared")}, "coded_stats": {
               k: v for k, v in st.items() if k != "per_layer"}}
    if differ or why:
        verdict, code = "FAIL", 1
    elif invalid:
        verdict, code = "INVALID", 75
    else:
        verdict, code = "PASS", 0
    rep["verdict"] = verdict
    return verdict, code, rep


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(int(sys.argv[sys.argv.index("--codes") + 1]), sys.argv[sys.argv.index("--out") + 1])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "sanitize":
        if "--retain" in sys.argv:
            sanitize_retain(sys.argv[sys.argv.index("--tool") + 1])
            return 0
        sanitize(sys.argv[sys.argv.index("--tool") + 1],
                 int(sys.argv[sys.argv.index("--codes") + 1]) if "--codes" in sys.argv else 1)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "plan":
        print(json.dumps(plan(), indent=1))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "retain-run":
        run_retain(sys.argv[sys.argv.index("--out") + 1])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "compare-retain":
        verdict, code, rep = compare_retain(sys.argv[2])
        print(json.dumps(rep, indent=1, default=str))
        print(f"SCALE CODES RETAIN GATE: {verdict} ({rep.get('expected_outputs', 0)} expected outputs, "
              f"{rep.get('n_differ', 0)} differ)")
        return code
    if len(sys.argv) > 1 and sys.argv[1] == "retain":
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/tmp/scale-codes-retain"
        os.makedirs(out, exist_ok=True)
        p = os.path.join(out, "outs-retain.pt")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "retain-run", "--out", p])
        if r.returncode != 0:
            print(f"SCALE CODES RETAIN GATE: FAIL (retain-run exited {r.returncode})")
            return 1
        return subprocess.run([sys.executable, os.path.abspath(__file__), "compare-retain", p]).returncode
    if len(sys.argv) > 1 and sys.argv[1] == "layers":
        run_layers(int(sys.argv[sys.argv.index("--codes") + 1]), sys.argv[sys.argv.index("--out") + 1])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "compare-layers":
        verdict, code, rep = compare_layers(sys.argv[2], sys.argv[3])
        print(json.dumps(rep, indent=1, default=str))
        print(f"SCALE CODES LAYER SWEEP: {verdict} ({rep.get('expected_outputs', 0)} expected outputs, "
              f"{rep.get('n_differ', 0)} differ)")
        return code
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        verdict, code, rep = compare(sys.argv[2], sys.argv[3])
        print(json.dumps(rep, indent=1, default=str))
        print(f"SCALE CODES GPU GATE: {verdict} ({rep.get('expected_outputs', 0)} expected outputs, "
              f"{rep.get('n_differ', 0)} differ)")
        return code
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/tmp/scale-codes-gate"
    os.makedirs(out, exist_ok=True)
    paths = []
    for codes in (0, 1):
        p = os.path.join(out, f"outs-codes{codes}.pt")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "run", "--codes", str(codes), "--out", p])
        if r.returncode != 0:
            print(f"SCALE CODES GPU GATE: FAIL (run --codes {codes} exited {r.returncode})")
            return 1
        paths.append(p)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "compare", *paths])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
