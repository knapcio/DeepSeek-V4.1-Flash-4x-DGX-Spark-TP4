"""Single-GPU microbench of the DSpark draft LM head: the MXFP4 copy (M <= 8: the GEMV) vs the fp8 twin (production).

One rank's vocab shard (32320 x 5120 at TP4). Method (2026-11-12):
- direct timing markers: CUDA events right around each kernel call, eager; every call follows a READ-ONLY flush
  (a 256 MiB reduction that writes one scalar: no dirty lines, so no write-back lands inside the next timing) that keeps
  the GPU busy while the call is enqueued, so the head streams from DRAM as it does between the draft's other kernels;
- ABBA pairing: one sample = fp8, fp4, fp4, fp8 back to back; the paired gain = mean(fp8) - mean(fp4) of that quad;
- the sweep (SWEEP, 36 launches) screens each correct launch on SCREEN_QUADS quads; the winner is re-timed on an
  INDEPENDENT repeat (fresh activations, REPEAT_QUADS quads) and the verdict reads that repeat: mean paired gain with a
  95 % Student-t interval.
Correctness: each Triton path against a torch reference of its own weights (fp32 matmul of the dequantized copy): max
|diff| / max |ref| and row argmax (this checks the kernels, not the quantization). Fidelity (report only): both heads
against the bf16 head on the same activations (relative error, argmax agreement, MXFP4 vs fp8 argmax agreement); the
draft-side acceptance cost is the offline screen's and the A/B's to measure, never this bench's.

Bench verdict (pre-registered, diagnostics/drafthead-fp4-20261112; a cheap pre-screen for the engine A/B):
  repeat mean gain >= 150 us                     -> BENCH PASS      (the A/B runs; its step rule decides)
  90 us <= repeat mean gain < 150 us             -> BENCH CONTINUE  (the A/B runs and is decided on net tok/s at c1,
                                                                    acceptance and correctness: dh4_summary.py)
  gain < 90 us, or a kernel error >= 1e-2         -> BENCH KILL (exit 3)
The winner is printed as DSV41_DRAFT_HEAD_FP4_TILE=BN,BK,WARPS,STAGES (BK = the GEMV's K tile, STAGES 1 = no prefetch).

Never touches clocks, fans, or anything outside this process's tensors (no pinned host memory).

usage: PYTHONPATH=/opt/dsv41/adapter python3 bench_draft_head_fp4.py [--model-dir DIR --rank 1] [--json out.json]
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("DSV41_DRAFT_HEAD_FP8", "1")
os.environ.setdefault("DSV41_DRAFT_HEAD_FP4", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import draft_head_fp8 as dh  # noqa: E402

N_SHARD, K = 32320, 5120
PASS_US = 150.0
CONTINUE_US = 90.0
FLUSH_MIB = 256
SCREEN_QUADS = 6
REPEAT_QUADS = 30
ROW_QUADS = 8
# GEMV launches (BN, BK, WARPS, STAGES): rows per thread = BN * BK / (1024 * WARPS) in {1, 2, 4}, prefetch on (3) / off (1)
SHAPES = [(32, 256, 4), (16, 256, 4), (32, 256, 8), (16, 256, 2), (64, 256, 8), (32, 256, 2), (64, 256, 4), (128, 256, 8),
          (32, 128, 4), (64, 128, 8), (32, 128, 2), (64, 128, 4), (128, 128, 8), (64, 128, 2), (128, 128, 4),
          (16, 512, 8), (16, 512, 4), (32, 512, 8)]
SWEEP = [(bn, bk, w, s) for (bn, bk, w) in SHAPES for s in (3, 1)]
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13,
        20: 2.09, 25: 2.06, 29: 2.05, 30: 2.04, 40: 2.02, 60: 2.00}


def t95(df):
    return _T95.get(df) or _T95[max(k for k in _T95 if k <= df)] if df >= 1 else float("nan")


def mean_ci(xs):
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, None, None
    h = t95(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def load_shard(model_dir, rank, tp=4):
    from safetensors import safe_open
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, name), "pt") as f:
            if "head.weight" in f.keys():
                sl = f.get_slice("head.weight")
                n = sl.get_shape()[0] // tp
                return sl[rank * n:(rank + 1) * n].to(torch.bfloat16)
    raise SystemExit(f"no head.weight under {model_dir}")


class Timer:
    """Direct markers: flush (read-only), event, the call, event; one sync per group of calls."""

    def __init__(self, dev):
        self.buf = torch.zeros(FLUSH_MIB << 18, dtype=torch.float32, device=dev)   # FLUSH_MIB MiB

    def flush(self):
        return self.buf.sum()

    def run(self, fns):
        ev = []
        for fn in fns:
            self.flush()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            ev.append((a, b))
        torch.cuda.synchronize()
        return [a.elapsed_time(b) * 1e3 for a, b in ev]

    def flush_us(self, n=5):
        out = []
        for _ in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            self.flush()
            b.record()
            torch.cuda.synchronize()
            out.append(a.elapsed_time(b) * 1e3)
        return statistics.median(out)

    def abba(self, f8, f4, quads):
        """Paired quads fp8, fp4, fp4, fp8: (gains, fp8 times, fp4 times) in us."""
        g, t8, t4 = [], [], []
        for _ in range(quads):
            a, b, c, d = self.run([f8, f4, f4, f8])
            g.append((a + d) / 2 - (b + c) / 2)
            t8 += [a, d]
            t4 += [b, c]
        return g, t8, t4


def rel(y, r):
    return float((y - r).abs().max() / r.abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--ms", default="5,10,20,40,64")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    t0 = time.time()
    w = (load_shard(args.model_dir, args.rank) if args.model_dir else torch.randn(N_SHARD, K) * 0.02).to(torch.bfloat16)
    w = w.to(dev)
    src = "checkpoint rank %d" % args.rank if args.model_dir else "synthetic randn x 0.02"
    twin8 = dh.make_twin(w)
    twin4 = dh.make_fp4_twin(w)
    torch.cuda.empty_cache()
    ref8 = (twin8[0].float().view(N_SHARD, K // 32, 32)
            * torch.exp2(twin8[1].float() - 127).repeat_interleave(32, 0)[:N_SHARD].unsqueeze(-1)).view(N_SHARD, K)
    ref8 = ref8.to(torch.bfloat16)
    ref4 = dh.dequant_fp4(twin4)
    tm = Timer(dev)
    dev_name = torch.cuda.get_device_name()
    fl = tm.flush_us()
    print(f"[bench] {dev_name} head shard {N_SHARD} x {K} ({src}); fp8 {(twin8[0].numel() + twin8[1].numel()) / 1e6:.1f} MB, "
          f"mxfp4 {(twin4[0].numel() + twin4[1].numel()) / 1e6:.1f} MB; built in {time.time() - t0:.1f}s; read-only flush "
          f"{FLUSH_MIB} MiB = {fl:.0f} us", flush=True)
    mb4 = (twin4[0].numel() + twin4[1].numel()) / 1e6
    mb8 = (twin8[0].numel() + twin8[1].numel()) / 1e6
    res = {"device": dev_name, "source": src, "flush_us": fl, "rows": {}, "sweep_m5": {}, "fidelity": {},
           "method": {"timing": "cuda events around each eager call after a read-only flush", "flush_mib": FLUSH_MIB,
                      "pairing": "ABBA quads fp8 fp4 fp4 fp8", "screen_quads": SCREEN_QUADS,
                      "repeat_quads": REPEAT_QUADS, "row_quads": ROW_QUADS, "pass_us": PASS_US,
                      "continue_us": CONTINUE_US, "ci": "95 % Student-t of the repeat's paired gains"}}
    ms = [int(m) for m in args.ms.split(",")]
    for m in ms:
        x = torch.randn(m, K, device=dev).to(torch.bfloat16)
        r8 = x.float() @ ref8.float().t()
        r4 = x.float() @ ref4.float().t()
        rb = x.float() @ w.float().t()
        y8 = dh.head_fp8(x, twin8)
        y4 = dh.head_fp4(x, twin4)
        row = {"fp8_err": rel(y8, r8), "fp4_err": rel(y4, r4),
               "fp8_argmax_ok": float((y8.argmax(-1) == r8.argmax(-1)).float().mean()),
               "fp4_argmax_ok": float((y4.argmax(-1) == r4.argmax(-1)).float().mean()),
               "fp4_vs_fp8_argmax": float((y4.argmax(-1) == y8.argmax(-1)).float().mean())}
        res["fidelity"][m] = {"fp8_vs_bf16_err": rel(y8, rb), "fp4_vs_bf16_err": rel(y4, rb),
                              "fp8_vs_bf16_argmax": float((y8.argmax(-1) == rb.argmax(-1)).float().mean()),
                              "fp4_vs_bf16_argmax": float((y4.argmax(-1) == rb.argmax(-1)).float().mean()),
                              "fp4_vs_fp8_argmax": row["fp4_vs_fp8_argmax"]}
        f8, f4, fb = (lambda: dh.head_fp8(x, twin8)), (lambda: dh.head_fp4(x, twin4)), (lambda: F.linear(x, w))
        tm.run([f8, f4, fb])                                   # warm (compiles)
        g, t8, t4 = tm.abba(f8, f4, ROW_QUADS)
        row["bf16_us"] = statistics.median(tm.run([fb] * ROW_QUADS))
        row["fp8_us"], row["fp4_us"] = statistics.median(t8), statistics.median(t4)
        row["gain_us"], row["gain_lo"], row["gain_hi"] = mean_ci(g)
        row["fp4_gbs"] = mb4 / row["fp4_us"] * 1e3
        row["fp8_gbs"] = mb8 / row["fp8_us"] * 1e3
        res["rows"][m] = row
        fd = res["fidelity"][m]
        print(f"[bench] M={m:3d} bf16 {row['bf16_us']:7.1f} us | fp8 {row['fp8_us']:7.1f} us ({row['fp8_gbs']:.0f} GB/s) | "
              f"mxfp4 {row['fp4_us']:7.1f} us ({row['fp4_gbs']:.0f} GB/s) | paired gain {row['gain_us']:+7.1f} us "
              f"[{row['gain_lo']:+.1f}, {row['gain_hi']:+.1f}] | kernel err fp8 {row['fp8_err']:.1e} fp4 {row['fp4_err']:.1e} | "
              f"vs bf16: err fp8 {fd['fp8_vs_bf16_err']:.1e} fp4 {fd['fp4_vs_bf16_err']:.1e}, argmax fp8 "
              f"{fd['fp8_vs_bf16_argmax']:.3f} fp4 {fd['fp4_vs_bf16_argmax']:.3f}, fp4=fp8 {fd['fp4_vs_fp8_argmax']:.3f}",
              flush=True)
    ok_err = all(r["fp4_err"] < 1e-2 and r["fp8_err"] < 1e-2 for r in res["rows"].values())
    # --- screen: every correct M = 5 launch, ABBA against the fp8 twin
    x = torch.randn(5, K, device=dev).to(torch.bfloat16)
    r4 = x.float() @ ref4.float().t()
    f8 = lambda: dh.head_fp8(x, twin8)                         # noqa: E731
    best = None
    for tile in ([] if args.no_sweep else SWEEP) or [dh.FP4_DEFAULT_TILE]:
        f4 = lambda tile=tile: dh.head_fp4(x, twin4, tile=tile)   # noqa: E731
        try:
            err = rel(f4(), r4)
            tm.run([f4, f4])
            g, t8, t4 = tm.abba(f8, f4, SCREEN_QUADS)
        except Exception as exc:  # noqa: BLE001 - a launch that does not compile is skipped
            print(f"[bench] tile {tile}: skipped ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
            continue
        key = ",".join(map(str, tile))
        res["sweep_m5"][key] = {"err": err, "gain_median_us": statistics.median(g), "fp4_us": statistics.median(t4),
                                "fp8_us": statistics.median(t8)}
        print(f"[bench] screen M=5 tile BN,BK,W,S={tile}: mxfp4 {statistics.median(t4):7.1f} us, fp8 "
              f"{statistics.median(t8):7.1f} us, paired gain median {statistics.median(g):+7.1f} us, err {err:.1e}", flush=True)
        if err < 1e-2 and (best is None or statistics.median(g) > best[1]):
            best = (tile, statistics.median(g))
    if best is None:
        print("BENCH KILL: no MXFP4 launch compiled and passed the error check", flush=True)
        res["verdict"] = "KILL"
        if args.json:
            with open(args.json, "w") as f:
                json.dump(res, f, indent=1)
        sys.exit(3)
    # --- the independent repeat of the winner: fresh activations, its own quads
    tile = best[0]
    torch.manual_seed(1)
    x = torch.randn(5, K, device=dev).to(torch.bfloat16)
    r4 = x.float() @ ref4.float().t()
    f8 = lambda: dh.head_fp8(x, twin8)                         # noqa: E731
    f4 = lambda: dh.head_fp4(x, twin4, tile=tile)              # noqa: E731
    err_rep = rel(f4(), r4)
    tm.run([f8, f4])
    g, t8, t4 = tm.abba(f8, f4, REPEAT_QUADS)
    gm, glo, ghi = mean_ci(g)
    res["repeat"] = {"tile": ",".join(map(str, tile)), "quads": REPEAT_QUADS, "gain_mean_us": gm, "gain_ci95_us": [glo, ghi],
                     "gain_median_us": statistics.median(g), "fp8_us": statistics.median(t8), "fp4_us": statistics.median(t4),
                     "fp4_gbs": mb4 / statistics.median(t4) * 1e3, "fp8_gbs": mb8 / statistics.median(t8) * 1e3,
                     "err": err_rep, "gains_us": g}
    ok_err = ok_err and err_rep < 1e-2
    res["best_tile_m5"] = ",".join(map(str, tile))
    res["best_fp4_us_m5"] = statistics.median(t4)
    res["fp8_us_m5"] = statistics.median(t8)
    res["gain_us_m5"] = gm
    res["gain_ci95_us_m5"] = [glo, ghi]
    verdict = "KILL" if not ok_err else "PASS" if gm >= PASS_US else "CONTINUE" if gm >= CONTINUE_US else "KILL"
    res["verdict"] = verdict
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    print(f"[bench] repeat M=5 tile {res['best_tile_m5']}: {REPEAT_QUADS} ABBA quads, fp8 {res['fp8_us_m5']:.1f} us "
          f"({res['repeat']['fp8_gbs']:.0f} GB/s), mxfp4 {res['best_fp4_us_m5']:.1f} us ({res['repeat']['fp4_gbs']:.0f} GB/s)",
          flush=True)
    print(f"DSV41_DRAFT_HEAD_FP4_TILE={res['best_tile_m5']}", flush=True)
    print(f"BENCH {verdict}: M=5 fp8 {res['fp8_us_m5']:.1f} us, best mxfp4 {res['best_fp4_us_m5']:.1f} us, paired gain "
          f"{gm:+.1f} us [95 % CI {glo:+.1f}, {ghi:+.1f}] on the independent repeat (rule: >= {PASS_US:.0f} PASS, "
          f">= {CONTINUE_US:.0f} CONTINUE to the engine A/B on net tok/s, else KILL; kernel errors < 1e-2: "
          f"{'ok' if ok_err else 'ERR'})", flush=True)
    sys.exit(0 if verdict in ("PASS", "CONTINUE") else 3)


if __name__ == "__main__":
    main()
