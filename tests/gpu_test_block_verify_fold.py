"""GPU test of adapter/block_verify_fold.py (one GB10, serving stopped, ~5 min). Contract: bit-identical to
production block_verify.block_accept_probs.

  docker run --rm --gpus all -v $PWD:/w -w /w -e PYTHONPATH=/w/adapter:/w/tests <image> \
      python3 tests/gpu_test_block_verify_fold.py [--rows 12000] [--out result.json] [--run-id ID]

1. cases: every case of tests/bv_fold_cases.py on the GPU (the interpreter's edge cases: eta at / one ulp around h,
   ties, NaN / inf keys, zero / near-zero mass, dead positions, top-p rows, V = 129280): tau, token and the
   intermediates S, h, mass, residual, keys bit-identical to production's.
2. random: production-shaped rows (V = 129280, g = 5, bs 1..8, cutoff / none, top-p 1 / 0.95 / 0.8, ties) up to
   --rows: tau and token mismatches against production (contract 0), plus intermediate bit differences.
3. graph: the fold captured in a CUDA graph on static inputs, replayed 3 times: outputs == eager production.
4. speed (informational): torch path vs fold, eager and graph-captured, bs 1 / 4 / 8, bracketed torch | fold | torch
   with medians of 200; a bracket whose two torch medians differ by > 3 % is marked drifted.
Writes one JSON (verdict PASS iff 1-3 have zero differences and the schema is complete).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import block_verify as bv  # noqa: E402
import block_verify_fold as bvf  # noqa: E402
import bv_fold_cases as cm  # noqa: E402

G = cm.G
V = 129280


def bits_diff(a, b):
    if a is None or b is None:
        return 0 if (a is None and b is None) else 1
    if a.shape != b.shape:
        return a.numel() or 1
    return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum())


def compare(c):
    live_p = None if c["live"] is None else c["live"].to(torch.int64) - c["live_sub"]
    t_p, k_p = bv.block_accept_probs(c["X"].long(), c["tp"], c["dp"], eta=c["eta"], gumbel=c["gum"], live=live_p)
    t_m, k_m, dm = cm.prod_debug(c["X"].long(), c["tp"], c["dp"], c["eta"], c["gum"], live=live_p)
    df = {}
    t_f, k_f = bvf.fold(c["X"], c["tp"], c["dp"], c["eta"], c["gum"], live=c["live"], live_sub=c["live_sub"], debug=df)
    return {"rows": int(t_p.numel()),
            "mirror": int((t_p != t_m).sum() + (k_p != k_m).sum()),
            "tau": int((t_f != t_p).sum()), "tok": int((k_f != k_p).sum()),
            "inter": sum(bits_diff(dm[k], df[k]) for k in ("s", "h", "mass", "res", "keys"))}


def median_us(fn, n=200):
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--out", default="")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args()
    import triton
    out = {"test": "gpu_test_block_verify_fold", "run_id": a.run_id, "started": time.time(),
           "device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
           "prod_source_sha": cm.prod_source_sha(), "mirrored_sha": cm.MIRRORED_SHA}
    # 1
    tot = {"cases": 0, "rows": 0, "mirror": 0, "tau": 0, "tok": 0, "inter": 0}
    bad = []
    for name, c in cm.cases("cuda"):
        r = compare(c)
        tot["cases"] += 1
        for k in ("rows", "mirror", "tau", "tok", "inter"):
            tot[k] += r[k]
        if r["mirror"] or r["tau"] or r["tok"] or r["inter"]:
            bad.append({"case": name, **r})
    out["cases"] = {**tot, "failing": bad[:20]}
    # 2
    rnd = {"rows": 0, "tau": 0, "tok": 0, "inter": 0, "mirror": 0, "batches": 0}
    seed = 0
    while rnd["rows"] < a.rows:
        bs = (1, 2, 4, 8)[seed % 4]
        c = cm.random_case(bs, V, 50_000 + seed, "cuda", sharp_p=1.0 + seed % 4, topp=(None, 0.95, 0.8)[seed % 3],
                           live_mode=("cut", "none")[seed % 2], tie=seed % 17 == 0, q_mix=(seed % 5) / 5)
        r = compare(c)
        for k in ("rows", "tau", "tok", "inter", "mirror"):
            rnd[k] += r[k]
        rnd["batches"] += 1
        seed += 1
    out["random"] = rnd
    # 3 graph replay on static inputs
    c = cm.random_case(4, V, 777, "cuda", sharp_p=3.0, topp=0.95, live_mode="cut")
    live_p = c["live"] - 1
    ref = bv.block_accept_probs(c["X"].long(), c["tp"], c["dp"], eta=c["eta"], gumbel=c["gum"], live=live_p)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        bvf.fold(c["X"], c["tp"], c["dp"], c["eta"], c["gum"], live=c["live"], live_sub=1)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            got = bvf.fold(c["X"], c["tp"], c["dp"], c["eta"], c["gum"], live=c["live"], live_sub=1)
    torch.cuda.current_stream().wait_stream(st)
    gdiff = 0
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        gdiff += int((got[0] != ref[0]).sum() + (got[1] != ref[1]).sum())
    out["graph"] = {"replays": 3, "differences": gdiff}
    # 4 speed (informational)
    speed = {}
    for bs in (1, 4, 8):
        c = cm.random_case(bs, V, 999 + bs, "cuda", live_mode="cut")
        live_p = c["live"] - 1

        def torch_path():
            e = torch.rand(bs, G, device="cuda")
            gm = torch.empty(bs, V, device="cuda").exponential_()
            return bv.block_accept_probs(c["X"].long(), c["tp"], c["dp"], eta=e, gumbel=gm, live=live_p)

        def fold_path():
            e = torch.rand(bs, G, device="cuda")
            gm = torch.empty(bs, V, device="cuda").exponential_()
            return bvf.fold(c["X"], c["tp"], c["dp"], e, gm, live=c["live"], live_sub=1)

        for fn in (torch_path, fold_path):
            for _ in range(5):
                fn()
        torch.cuda.synchronize()
        t1, f, t2 = median_us(torch_path), median_us(fold_path), median_us(torch_path)
        speed[f"bs{bs}_eager_us"] = {"torch_a": round(t1, 1), "fold": round(f, 1), "torch_b": round(t2, 1),
                                     "drifted": abs(t1 - t2) > 0.03 * min(t1, t2)}
    out["speed"] = speed
    complete = tot["cases"] >= 60 and rnd["rows"] >= a.rows and out["graph"]["replays"] == 3
    zero = (tot["mirror"] + tot["tau"] + tot["tok"] + tot["inter"] + rnd["tau"] + rnd["tok"] + rnd["inter"]
            + rnd["mirror"] + gdiff) == 0
    sha_ok = out["prod_source_sha"] == out["mirrored_sha"]
    out["verdict"] = "PASS" if (complete and zero and sha_ok) else ("INVALID" if not (complete and sha_ok) else "FAIL")
    out["finished"] = time.time()
    line = json.dumps(out)
    print(line)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(line + "\n")
    sys.exit(0 if out["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
