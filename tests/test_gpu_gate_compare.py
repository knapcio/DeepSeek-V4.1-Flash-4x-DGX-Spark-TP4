"""CPU checks of tests/gpu_test_scale_codes.py's comparator (REVIEW_sol_scalecodes R3/R4), on shrunken constants.

The comparator judges two run files against the manifest frozen in the gate, never against what the runs happened
to contain. Fixtures: a complete PASS; Sol's reproduction (empty maps + plausible counters: used to PASS with 0
outputs); both files equally truncated; wrong mode metadata; a wrong shape; one differing byte; a nonzero padding
row; a nondeterministic rerun (INVALID); other plan capacities (INVALID); a receipt below 90 % (FAIL); v2.6's prepared
launch not engaged (INVALID), coded plans missing from the coded run or present in the native run (FAIL); v2.7.3's
plan,direct accepted. The A/B retain mode's comparator (compare_retain) and the dry `plan` mode.

usage: python3 tests/test_gpu_gate_compare.py      (torch; no GPU)
"""
import copy
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_test_scale_codes as gt  # noqa: E402

FAILS = []
gt.K_HID = 8
gt.EXPECTED_CAPS = (5, 6)
gt.MS = (4, 5, 6)
gt.GRAPH_MS = (5,)
gt.BUILDS = (("layers.3", 6, 0), ("mtp.2", 3, 1))


def check(c, m):
    if not c:
        FAILS.append(m)


def out_tensor(m, seed):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(m, gt.K_HID, generator=g) + 0.1).to(torch.bfloat16)
    if m > 1:
        t[-1] = 0
    return t


def runs():
    outs = {k: out_tensor(int(k.rsplit("/M", 1)[1]), i) for i, k in enumerate(sorted(gt.expected_keys()))}
    builds = {gt.build_key(p, r) for p, _, r in gt.BUILDS}
    base = {"rerun_same": {k: True for k in outs}, "caps_seen": {b: list(gt.EXPECTED_CAPS) for b in builds},
            "plans": {}, "micro": {"plans": {}, "outputs": 0}}
    a = {"outs": outs, "meta": dict(base, codes=False, scale_codes_flag=False, layer_bytes={b: 1000 for b in builds},
                                    codes_stats={"layers": 0, "per_layer": []})}
    per = [{"native": 400, "coded": 200, "freed": 200, "span": 9, "released": 2}] * (len(gt.BUILDS) + 1)
    b = {"outs": {k: v.clone() for k, v in outs.items()},
         "meta": dict(copy.deepcopy(base), codes=True, scale_codes_flag=True, layer_bytes={x: 800 for x in builds},
                      codes_stats={"layers": len(per), "per_layer": per, "max_span": 9})}
    return a, b


def layer_sweep():
    """2026-10-30: the every-layer sweep's comparator (compare_layers): PASS, one differing byte, a missing layer, a
    receipt without both storages released, a nondeterministic rerun (INVALID), a converted native run."""
    saved = gt.LAYER_MS
    gt.LAYER_MS = (4,)
    try:
        def lruns(ne=2):
            keys = sorted(gt.layer_keys(ne))
            outs = {k: out_tensor(4, i) for i, k in enumerate(keys)}
            per = [{"native": 400, "coded": 200, "freed": 200, "span": 9, "released": 2}] * len(gt.ALL_LAYERS)
            a = {"outs": outs, "meta": {"codes": False, "experts": ne, "rerun_same": {k: True for k in keys},
                                        "codes_stats": {"layers": 0, "per_layer": []}}}
            b = {"outs": {k: v.clone() for k, v in outs.items()},
                 "meta": {"codes": True, "experts": ne, "rerun_same": {k: True for k in keys},
                          "codes_stats": {"layers": len(per), "per_layer": per}}}
            return a, b

        def lv(a, b):
            d = tempfile.mkdtemp()
            pa, pb = os.path.join(d, "a.pt"), os.path.join(d, "b.pt")
            torch.save(a, pa)
            torch.save(b, pb)
            return gt.compare_layers(pa, pb)[0]
        check(len(gt.layer_builds(16)) == 43 and all(b + 16 <= (384 if p.startswith("layers.") else 128)
                                                     for p, _, _, b in gt.layer_builds(16)), "layer builds")
        a, b = lruns()
        check(lv(a, b) == "PASS", "layer sweep: identical runs did not PASS")
        a, b = lruns()
        k = sorted(b["outs"])[7]
        b["outs"][k] = b["outs"][k].clone()
        b["outs"][k].view(torch.int16)[0, 0] ^= 1
        check(lv(a, b) == "FAIL", "layer sweep: one differing bit passed")
        a, b = lruns()
        for x in (a, b):
            x["outs"].pop(sorted(x["outs"])[0])
        check(lv(a, b) == "FAIL", "layer sweep: an equally missing output passed")
        a, b = lruns()
        b["meta"]["codes_stats"]["per_layer"] = [dict(r, released=0) for r in b["meta"]["codes_stats"]["per_layer"]]
        check(lv(a, b) == "FAIL", "layer sweep: an unreleased receipt passed")
        a, b = lruns()
        b["meta"]["rerun_same"][sorted(b["outs"])[0]] = False
        check(lv(a, b) == "INVALID", "layer sweep: a nondeterministic rerun was not INVALID")
        a, b = lruns()
        a["meta"]["codes_stats"]["layers"] = 1
        check(lv(a, b) == "FAIL", "layer sweep: a converted native run passed")
    finally:
        gt.LAYER_MS = saved


def retain():
    """2026-11-10, A/B retain mode (compare_retain, one run file): PASS; a differing byte (eager and graph); a wrong
    decision (FAIL); a decision missed only because its capacity is compact micro (INVALID); a released native storage;
    the process-wide switch set; a missing output; a nonzero padding row; prepared plans without coded ones (FAIL)."""
    saved = gt.RETAIN_MS, gt.RETAIN_BUILDS
    gt.RETAIN_MS, gt.RETAIN_BUILDS = (1, 6), (("layers.3", 6, 3), ("mtp.2", 3, 0))
    try:
        def rrun():
            outs = {}
            for i, k in enumerate(sorted(gt.retain_keys())):
                m = int(k.rsplit("/M", 1)[1])
                group = k.split("/")[0] + k.split("/", 2)[2]       # the same tensor for every mode of one call
                outs[k] = out_tensor(m, abs(hash(group)) % 1000)
            dec, st = {}, {}
            for p, _, r in gt.RETAIN_BUILDS:
                b = gt.build_key(p, r)
                st[b] = [100, 50]
                st[b + "/coded_caps"] = [1, 6]
                st[b + "/cap_of"] = {"1": 1, "6": 6}
                for m in gt.RETAIN_MS:
                    for v in (0, 1, 2):
                        dec[f"{b}/v{v}/M{m}"] = gt.retain_want(v, m)
            meta = {"retain": True, "decisions": dec, "storages": st, "why": [], "switch": None,
                    "retain_stats": {"layers": 2, "native": 400, "coded": 200, "max_span": 12},
                    "prepared_requested": True,
                    "prepared": {"mode": "plan,direct", "builds": 8, "hits": 20, "plans": 6, "coded_plans": 3}}
            return {"outs": outs, "meta": meta}

        def rv(doc):
            d = tempfile.mkdtemp()
            pth = os.path.join(d, "r.pt")
            torch.save(doc, pth)
            return gt.compare_retain(pth)[0]

        check(rv(rrun()) == "PASS", "retain: complete run")
        for mode in ("eager-coded", "graph-coded", "graph-native"):
            d = rrun()
            k = next(k for k in sorted(d["outs"]) if f"/{mode}/" in k and k.endswith("/M6"))
            d["outs"][k] = d["outs"][k].clone()
            d["outs"][k].view(torch.int16)[0, 0] ^= 1
            check(rv(d) == "FAIL", f"retain: a differing byte in {mode} passed")
        d = rrun()
        d["meta"]["decisions"]["layers.3@tp3/v2/M1"] = True
        check(rv(d) == "FAIL", "retain: M=1 coded under the default floor passed")
        d = rrun()
        d["meta"]["decisions"]["layers.3@tp3/v0/M6"] = True
        check(rv(d) == "FAIL", "retain: the control variant ran codes and passed")
        d = rrun()
        d["meta"]["decisions"]["mtp.2@tp0/v1/M6"] = False
        check(rv(d) == "FAIL", "retain: a prepared capacity ran native and passed")
        d = rrun()
        d["meta"]["decisions"]["mtp.2@tp0/v1/M6"] = False
        d["meta"]["storages"]["mtp.2@tp0/coded_caps"] = [1]
        check(rv(d) == "INVALID", "retain: an unprepared (compact micro) capacity was not INVALID")
        d = rrun()
        d["meta"]["storages"]["layers.3@tp3"] = [0, 50]
        check(rv(d) == "FAIL", "retain: a released native storage passed")
        d = rrun()
        d["meta"]["switch"] = "1"
        check(rv(d) == "FAIL", "retain: the process-wide switch set passed")
        d = rrun()
        del d["outs"][sorted(d["outs"])[0]]
        check(rv(d) == "FAIL", "retain: a missing output passed")
        d = rrun()
        k = next(k for k in sorted(d["outs"]) if k.endswith("/M6"))
        d["outs"][k] = d["outs"][k].clone()
        d["outs"][k][-1] = 1
        check(rv(d) == "FAIL", "retain: a nonzero padding row passed")
        d = rrun()
        d["meta"]["prepared"]["coded_plans"] = 0
        check(rv(d) == "FAIL", "retain: no coded prepared plan passed")
        d = rrun()
        d["meta"]["retain_stats"]["layers"] = 1
        check(rv(d) == "FAIL", "retain: a missing layer conversion passed")
    finally:
        gt.RETAIN_MS, gt.RETAIN_BUILDS = saved


def dry_plan():
    """The dry `plan` mode needs neither torch nor a GPU; the production capacities include the 8192-row chunk."""
    import json
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("SC_CHUNKED_PREFILL_SIZE", "CHUNKED_PREFILL_SIZE",
                                                           "DSV41_MOE_LAUNCH_PREPARED")}
    r = subprocess.run([sys.executable, "-S", "-c",
                        "import sys, runpy; sys.modules['torch'] = None; sys.argv = ['g', 'plan']; "
                        f"runpy.run_path({os.path.abspath(gt.__file__)!r}, run_name='__main__')"],
                       capture_output=True, text=True, env=env)
    try:
        doc = json.loads(r.stdout)
    except ValueError:
        check(False, f"plan: no JSON (rc {r.returncode}): {r.stderr[-300:]}")
        return
    check(r.returncode == 0 and 8192 in doc["gate"]["expected_caps"] and 8193 in doc["gate"]["rows"]
          and doc["env"]["DSV41_MOE_LAUNCH_PREPARED"] == "plan,direct", f"plan: production defaults {doc['env']}")
    check(doc["retain"]["rows"] == [1, 6, 12, 24, 48, 96] and doc["retain"]["decisions"]["v2/M1"] == "native"
          and doc["retain"]["decisions"]["v2/M6"] == "coded" and doc["retain"]["decisions"]["v0/M96"] == "native",
          f"plan: retain manifest {doc['retain']}")


def verdict(a, b):
    d = tempfile.mkdtemp()
    pa, pb = os.path.join(d, "a.pt"), os.path.join(d, "b.pt")
    torch.save(a, pa)
    torch.save(b, pb)
    return gt.compare(pa, pb)[0]


def main():
    a, b = runs()
    check(verdict(a, b) == "PASS", "complete runs")
    # Sol's reproduction: empty maps with plausible aggregate counters
    e = {"outs": {}, "meta": {"codes": False, "rerun_same": {}, "layer_bytes": {}, "codes_stats": {"layers": 0}}}
    f = {"outs": {}, "meta": {"codes": True, "rerun_same": {}, "layer_bytes": {},
                              "codes_stats": {"layers": 4, "max_span": 13, "native": 100, "coded": 50, "freed": 50}}}
    check(verdict(e, f) == "FAIL", "empty maps passed")
    a, b = runs()
    drop = sorted(a["outs"])[:3]
    for x in (a, b):
        for k in drop:
            del x["outs"][k]
            del x["meta"]["rerun_same"][k]
    check(verdict(a, b) == "FAIL", "equally truncated runs passed")
    a, b = runs()
    b["meta"]["codes"] = False
    check(verdict(a, b) == "FAIL", "wrong mode passed")
    a, b = runs()
    k = sorted(b["outs"])[0]
    b["outs"][k] = b["outs"][k][:, :4].contiguous()
    check(verdict(a, b) == "FAIL", "wrong shape passed")
    a, b = runs()
    k = sorted(b["outs"])[1]
    b["outs"][k].view(torch.int16)[0, 0] ^= 1
    check(verdict(a, b) == "FAIL", "a differing byte passed")
    a, b = runs()
    k = next(k for k in sorted(a["outs"]) if k.endswith("/M5"))
    for x in (a, b):
        x["outs"][k][-1] = 1
    check(verdict(a, b) == "FAIL", "a nonzero padding row passed")
    a, b = runs()
    b["meta"]["rerun_same"][sorted(b["outs"])[0]] = False
    check(verdict(a, b) == "INVALID", "a nondeterministic rerun was not INVALID")
    a, b = runs()
    for x in (a, b):
        x["meta"]["caps_seen"] = {k: [5] for k in x["meta"]["caps_seen"]}
    check(verdict(a, b) == "INVALID", "other plan capacities were not INVALID")
    a, b = runs()
    b["meta"]["codes_stats"]["per_layer"] = [{"native": 400, "coded": 200, "freed": -200, "span": 9}] * 3
    check(verdict(a, b) == "FAIL", "a receipt below 90 % passed")
    a, b = runs()
    per = [dict(r, released=1) for r in b["meta"]["codes_stats"]["per_layer"]]
    b["meta"]["codes_stats"]["per_layer"] = per
    check(verdict(a, b) == "FAIL", "a receipt without both native storages released passed")
    # v2.6: the prepared launch (DSV41_MOE_LAUNCH_PREPARED=plan) must be exercised in both runs, coded plans in
    # the coded run only
    def prep(a, b, na=None, nb=None):
        for x, st in ((a, na), (b, nb)):
            x["meta"]["prepared_requested"] = True
            x["meta"]["prepared"] = st
        return verdict(a, b)
    on = {"mode": "plan", "builds": 4, "hits": 40, "max_coded_plans": 0}
    a, b = runs()
    check(prep(a, b, dict(on), dict(on, max_coded_plans=4)) == "PASS", "prepared: engaged, coded plans in the coded run")
    a, b = runs()
    check(prep(a, b, dict(on), dict(on, hits=0, max_coded_plans=4)) == "INVALID", "prepared: no hits was not INVALID")
    a, b = runs()
    check(prep(a, b, None, dict(on, max_coded_plans=4)) == "INVALID", "prepared: not installed was not INVALID")
    a, b = runs()
    check(prep(a, b, dict(on), dict(on)) == "FAIL", "prepared: coded run without coded plans passed")
    a, b = runs()
    check(prep(a, b, dict(on, max_coded_plans=1), dict(on, max_coded_plans=4)) == "FAIL",
          "prepared: native run with a coded plan passed")
    a, b = runs()
    a["meta"]["prepared_requested"] = True
    check(verdict(a, b) == "FAIL", "prepared: requested in one run only passed")
    a, b = runs()          # v2.7.3 production runs plan,direct
    check(prep(a, b, dict(on, mode="plan,direct"), dict(on, mode="plan,direct", max_coded_plans=4)) == "PASS",
          "prepared: plan,direct (v2.7.3) not accepted")
    layer_sweep()
    retain()
    dry_plan()
    for x in FAILS:
        print("FAIL " + x)
    print("test_gpu_gate_compare: " + ("ok" if not FAILS else f"{len(FAILS)} failure(s)"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
