"""GB10 gate of DSV41_PREFILL_NATIVE_EXTRA: one idle GPU, the overlay image, no engine.

The production adapter's own prefill entry (sglang.kernels.ops.attention.flash_mla_sm120._flash_mla_sm120_prefill)
runs flag OFF (stock: SWA split + extra split to 64-token pages, stock module) and flag ON (SWA split, extra read in its
native 128/256-token pages by the prebuilt pce module) on the same inputs. The adapter returns the attention OUTPUT
only (its LSE result is None and serving discards LSE), so the comparisons are on the output, bit for bit.

1. eager identity, N_EAGER cases: extra pages 128 / 256 x 16 / 64 heads (MG_N_HG 1 and 2) x 72 / 96 (wide verify) /
   1,031 / 4,096 rows x full-tile (no topk_length, extra top-k 512) / general (topk_length + extra_topk_length, ragged)
   dispatch, -1 holes, attention sink on the 16-head cases;
2. edge cases, N_EDGE (REVIEW_sol K8): per page size, rows 96, general dispatch: every row with zero extra entries,
   every extra entry a hole, extra indices on page first / last tokens up to the pool's last token, a non-finite
   (fp8 NaN) record in extra slot 0 (the slot a hole reads speculatively before masking), and minimum main length 1;
3. CUDA-graph capture and replay of the wide-verify route, N_REPLAY checks (REVIEW_sol K8 P2-2): per page size and
   rows 65 / 66 / 72 / 84 / 96 (64 goes to the decode kernel; 66..96 are the bs 11..16 verify graphs), one graph
   captured with the flag OFF and one with it ON on the same static inputs; then 3 times: every input (q, both index
   tensors, both length tensors, the extra pool's bytes) is rewritten in place, both graphs replay, and both outputs
   must equal a fresh eager OFF result on the new inputs bit for bit (also proves the replays read the new inputs);
4. route evidence: DSV41_PCE_HIT for both page sizes; an instrumented `_split_kv_pages_to_64` records every call: under
   ON no call may carry buffer_tag "extra" (every case, not only the first), and the extra split buffer must not exist
   after the first ON-only pass; under OFF the 128/256 cases must make that call (control);
5. module identity: the tag and the .so sha256 of the module that ran (dsv41_pce.check_sources + check_module);
6. memory (informational): the stock extra split buffer at a production-sized pool (--pool-pages, default 19,912 pages
   of 256 tokens = the v2.6 head's 5,097,472-token pool), by formula, and measured with --alloc-prod;
7. time (informational): per-call adapter time OFF vs ON at 4,096 rows, CUDA events, warm, medians of 20.

Exit 0 = PASS, 1 = FAIL (any identity or replay break, missing route evidence, an extra split under ON, a dead
control), 75 = INVALID (no GPU, the module did not verify or load before any case ran).
usage: python3 gpu_test_prefill_native_extra.py [--out F.json] [--alloc-prod]
"""
import argparse
import json
import logging
import math
import os
import statistics
import sys

import torch

D_NOPE, D_ROPE, ROW, FOOT = 448, 64, 576, 8
BPT = ROW + FOOT  # 584
SPLIT_PAGE_BYTES = math.ceil(64 * BPT / ROW) * ROW  # 37,440
PBSX = (128, 256)
N_EAGER = 32          # 2 page sizes x 2 head counts x 4 row counts x 2 dispatches
N_EDGE = 10           # 2 page sizes x 5 edge cases
N_REPLAY = 30         # 2 page sizes x 5 row counts x 3 input rewrites
REPLAY_ROWS = (65, 66, 72, 84, 96)
MAIN_PAGES = 8


def page_bytes(pbs):
    return math.ceil(pbs * BPT / ROW) * ROW


def fill_pool(view, gen):
    """Fill a V4 footer-format pool view [pages, pbs, 1, 584] (padded stride) with finite fp8 / bf16 / ue8m0 content."""
    pages, pbs = view.shape[0], view.shape[1]
    pb = view.stride(0)
    raw = torch.as_strided(view, (pages, pb), (pb, 1))
    data = raw[:, : pbs * ROW].reshape(pages, pbs, ROW)
    nope = (torch.randn(pages, pbs, D_NOPE, generator=gen, device=view.device) * 40).clamp(-448, 448)
    data[:, :, :D_NOPE] = nope.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = torch.randn(pages, pbs, D_ROPE, generator=gen, device=view.device).to(torch.bfloat16)
    data[:, :, D_NOPE:] = rope.view(torch.uint8).reshape(pages, pbs, 2 * D_ROPE)
    raw[:, : pbs * ROW] = data.reshape(pages, pbs * ROW)
    foot = raw[:, pbs * ROW: pbs * ROW + pbs * FOOT].reshape(pages, pbs, FOOT).clone()
    foot[:, :, :7] = torch.randint(118, 128, (pages, pbs, 7), generator=gen, device=view.device,
                                   dtype=torch.int32).to(torch.uint8)
    raw[:, pbs * ROW: pbs * ROW + pbs * FOOT] = foot.reshape(pages, pbs * FOOT)


def make_pool(pages, pbs, gen, dev):
    pb = page_bytes(pbs)
    raw = torch.zeros(pages, pb, dtype=torch.uint8, device=dev)
    view = torch.as_strided(raw, (pages, pbs, 1, BPT), (pb, BPT, BPT, 1))
    fill_pool(view, gen)
    return view


def make_idx(rows, k, pool_tokens, gen, dev, holes=0.1):
    idx = torch.randint(0, pool_tokens, (rows, k), generator=gen, device=dev, dtype=torch.int32)
    mask = torch.rand(rows, k, generator=gen, device=dev) < holes
    return idx.masked_fill(mask, -1)


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


class Hits(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--alloc-prod", action="store_true")
    ap.add_argument("--pool-pages", type=int, default=19912)
    a = ap.parse_args()
    res = {"cases": [], "replays": [], "verdict": "INVALID"}

    def done(code):
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "replays")}))
        sys.exit(code)

    if not torch.cuda.is_available():
        res["why"] = "no GPU"
        done(75)
    dev = torch.device("cuda", 0)
    from pathlib import Path

    from sglang.kernels.ops.attention import dsv41_pce
    from sglang.kernels.ops.attention import flash_mla_sm120 as A
    from sglang.srt.runtime_context import get_resources

    try:   # 5. identity of the module that will run (before any case: a refusal here is INVALID, not a result)
        src = Path(os.environ.get(dsv41_pce.SRC_ENV, dsv41_pce.DEFAULT_SRC))
        manifest = dsv41_pce.check_sources(src, *dsv41_pce._flashinfer_dirs())
        so = dsv41_pce.check_module(src, manifest)
        res["module"] = {"tag": manifest["module_tag"], "so_sha256": dsv41_pce._sha(so)}
        dsv41_pce._module()
    except Exception as e:  # noqa: BLE001
        res["why"] = f"module identity / load: {type(e).__name__}: {e}"[:2000]
        done(75)

    hits = Hits()
    logging.getLogger(dsv41_pce.__name__).addHandler(hits)
    logging.getLogger(dsv41_pce.__name__).setLevel(logging.WARNING)
    buffers = get_resources().buffers
    ekey = f"flash_mla_sm120_split:{dev}:extra"
    state = {"flag": None, "extra_split_on": 0, "extra_split_off": 0}
    real_split = A._split_kv_pages_to_64

    def split_recorder(kv, src_pbs, touched_indices=None, buffer_tag="primary"):
        if buffer_tag == "extra":
            state["extra_split_on" if state["flag"] == "1" else "extra_split_off"] += 1
        return real_split(kv, src_pbs, touched_indices=touched_indices, buffer_tag=buffer_tag)
    A._split_kv_pages_to_64 = split_recorder

    def setflag(flag):
        os.environ[dsv41_pce.ENV] = flag
        dsv41_pce.enabled.cache_clear()
        state["flag"] = flag

    def call(q, k_main, idx, tl, sink, extra, eidx, etl):
        out, _ = A._flash_mla_sm120_prefill(q, k_main, idx, tl, sink, 512, 512 ** -0.5, extra, eidx, etl)
        return out

    def run(flag, *args):
        setflag(flag)
        out = call(*args)
        torch.cuda.synchronize()
        return out.clone()

    gen = torch.Generator(device=dev)
    gen.manual_seed(20261027)
    k_main = make_pool(MAIN_PAGES, 256, gen, dev)
    fail = False

    def record(case, on, off):
        nonlocal fail
        ok = bits_equal(on, off)
        case["identical"] = ok
        if not ok:
            d = (on.float() - off.float()).abs()
            case["max_abs_diff"] = float(torch.nan_to_num(d, nan=float("inf")).max())
            case["n_diff"] = int((on.view(torch.int16) != off.view(torch.int16)).sum())
            fail = True
        return case

    try:
        pools = {pbs: make_pool(64 if pbs == 256 else 128, pbs, gen, dev) for pbs in PBSX}
        # ON-only first pass: the extra split buffer must never come into existence
        for pbs, extra in pools.items():
            q = (torch.randn(96, 16, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            idx = make_idx(96, 128, MAIN_PAGES * 256, gen, dev)
            eidx = make_idx(96, 512, extra.shape[0] * pbs, gen, dev)
            run("1", q, k_main, idx, None, None, extra, eidx, None)
        res["extra_buffer_after_on_only_pass"] = ekey in buffers
        if ekey in buffers:
            fail = True
        # 1. eager identity
        order = [("1", "on"), ("0", "off")]
        for pbs, extra in pools.items():
            ptok = extra.shape[0] * pbs
            for nh in (16, 64):
                for rows in (72, 96, 1031, 4096):
                    for mode, ek in (("fulltile", 512), ("general", 448)):
                        q = (torch.randn(rows, nh, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
                        idx = make_idx(rows, 128, MAIN_PAGES * 256, gen, dev)
                        eidx = make_idx(rows, ek, ptok, gen, dev)
                        sink = torch.randn(nh, generator=gen, device=dev) if nh == 16 else None
                        tl = etl = None
                        if mode == "general":
                            tl = torch.randint(1, 129, (rows,), generator=gen, device=dev, dtype=torch.int32)
                            etl = torch.randint(0, ek + 1, (rows,), generator=gen, device=dev, dtype=torch.int32)
                        outs = {name: run(flag, q, k_main, idx, tl, sink, extra, eidx, etl) for flag, name in order}
                        res["cases"].append(record({"kind": "eager", "pbs": pbs, "heads": nh, "rows": rows,
                                                    "mode": mode, "extra_topk": ek, "sink": sink is not None},
                                                   outs["on"], outs["off"]))
                        order.reverse()
        # 2. edge cases
        for pbs, extra in pools.items():
            ptok, rows, ek = extra.shape[0] * pbs, 96, 448
            for kind in ("zero_extra", "all_holes", "page_edges", "nan_slot0", "main_len_1"):
                q = (torch.randn(rows, 16, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
                idx = make_idx(rows, 128, MAIN_PAGES * 256, gen, dev)
                eidx = make_idx(rows, ek, ptok, gen, dev)
                tl = torch.randint(1, 129, (rows,), generator=gen, device=dev, dtype=torch.int32)
                etl = torch.full((rows,), ek, device=dev, dtype=torch.int32)
                ex = extra
                if kind == "zero_extra":
                    etl.zero_()
                elif kind == "all_holes":
                    eidx.fill_(-1)
                elif kind == "page_edges":
                    edges = torch.tensor([p * pbs + o for p in range(extra.shape[0]) for o in (0, pbs - 1)],
                                         device=dev, dtype=torch.int32)
                    eidx = edges[torch.randint(0, edges.numel(), (rows, ek), generator=gen, device=dev)]
                    eidx[:, 0] = ptok - 1   # the pool's last token in every row
                elif kind == "nan_slot0":
                    ex = make_pool(extra.shape[0], pbs, gen, dev)
                    pb = ex.stride(0)
                    torch.as_strided(ex, (pb,), (1,))[:D_NOPE] = 0x7F       # fp8 e4m3 NaN in slot 0's nope bytes
                    eidx = torch.where(eidx == 0, torch.ones_like(eidx), eidx)   # slot 0 only through holes
                elif kind == "main_len_1":
                    tl.fill_(1)
                outs = {name: run(flag, q, k_main, idx, tl, None, ex, eidx, etl) for flag, name in order}
                res["cases"].append(record({"kind": kind, "pbs": pbs, "heads": 16, "rows": rows, "mode": "general",
                                            "extra_topk": ek}, outs["on"], outs["off"]))
                order.reverse()
        # 3. CUDA-graph capture and replay of the wide-verify route
        for pbs, extra in pools.items():
            ptok, ek = extra.shape[0] * pbs, 512
            for rows in REPLAY_ROWS:
                q = (torch.randn(rows, 16, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
                idx = make_idx(rows, 128, MAIN_PAGES * 256, gen, dev)
                eidx = make_idx(rows, ek, ptok, gen, dev)
                tl = torch.randint(1, 129, (rows,), generator=gen, device=dev, dtype=torch.int32)
                etl = torch.randint(0, ek + 1, (rows,), generator=gen, device=dev, dtype=torch.int32)
                ex = make_pool(extra.shape[0], pbs, gen, dev)
                args = (q, k_main, idx, tl, None, ex, eidx, etl)
                graphs, outs = {}, {}
                for flag in ("0", "1"):
                    run(flag, *args)                         # warm: buffers / module before capture
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        run(flag, *args)
                    torch.cuda.current_stream().wait_stream(s)
                    g = torch.cuda.CUDAGraph()
                    setflag(flag)
                    with torch.cuda.graph(g):
                        outs[flag] = call(*args)
                    graphs[flag] = g
                for rep in range(3):
                    q.copy_((torch.randn(rows, 16, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16))
                    idx.copy_(make_idx(rows, 128, MAIN_PAGES * 256, gen, dev))
                    eidx.copy_(make_idx(rows, ek, ptok, gen, dev))
                    tl.copy_(torch.randint(1, 129, (rows,), generator=gen, device=dev, dtype=torch.int32))
                    etl.copy_(torch.randint(0, ek + 1, (rows,), generator=gen, device=dev, dtype=torch.int32))
                    fill_pool(ex, gen)
                    torch.cuda.synchronize()
                    graphs["0"].replay()
                    graphs["1"].replay()
                    torch.cuda.synchronize()
                    r_off, r_on = outs["0"].clone(), outs["1"].clone()
                    ref = run("0", *args)
                    ok = bits_equal(r_on, ref) and bits_equal(r_off, ref)
                    res["replays"].append({"pbs": pbs, "rows": rows, "rewrite": rep, "on_eq_ref": bits_equal(r_on, ref),
                                           "off_eq_ref": bits_equal(r_off, ref), "ok": ok})
                    if not ok:
                        fail = True
                del graphs
        # 4. route evidence
        res["hit_lines"] = sorted(set(l for l in hits.lines if "DSV41_PCE_HIT" in l))
        for pbs in PBSX:
            if not any(f"extra_page_block_size={pbs} " in l for l in res["hit_lines"]):
                res["why"] = f"route evidence missing for page size {pbs}"
                fail = True
        res["extra_split_calls"] = {"on": state["extra_split_on"], "off": state["extra_split_off"]}
        if state["extra_split_on"] != 0:
            res["why"] = f"{state['extra_split_on']} extra split calls with the flag ON"
            fail = True
        if state["extra_split_off"] == 0 or ekey not in buffers:
            res["why"] = "stock path never split the extra cache (control dead)"
            fail = True
        res["extra_split_buffer_after_off_bytes"] = int(buffers[ekey].numel()) if ekey in buffers else None
        # 6. memory at the production pool (informational)
        res["prod_extra_split_bytes_formula"] = a.pool_pages * 4 * SPLIT_PAGE_BYTES
        if a.alloc_prod:
            torch.cuda.empty_cache()
            before = torch.cuda.memory_reserved()
            big = torch.empty(a.pool_pages * 4, SPLIT_PAGE_BYTES, dtype=torch.uint8, device=dev)
            res["prod_extra_split_bytes_measured"] = int(torch.cuda.memory_reserved() - before)
            del big
            torch.cuda.empty_cache()
        # 7. time (informational)
        extra = make_pool(64, 256, gen, dev)
        q = (torch.randn(4096, 16, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        idx = make_idx(4096, 128, MAIN_PAGES * 256, gen, dev)
        eidx = make_idx(4096, 512, 64 * 256, gen, dev)
        t = {}
        for flag, name in (("0", "off"), ("1", "on")):
            setflag(flag)
            for _ in range(3):
                call(q, k_main, idx, None, None, extra, eidx, None)
            ms = []
            for _ in range(20):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                call(q, k_main, idx, None, None, extra, eidx, None)
                e.record()
                e.synchronize()
                ms.append(s.elapsed_time(e))
            t[name] = statistics.median(ms)
        res["time_ms_4096rows"] = t
    except Exception as e:  # noqa: BLE001
        res["why"] = f"{type(e).__name__}: {e}"[:2000]
        res["n_cases"], res["n_replay"] = len(res["cases"]), len(res["replays"])
        done(1)
    res["n_cases"] = len(res["cases"])
    res["n_identical"] = sum(c["identical"] for c in res["cases"])
    res["n_replay"] = len(res["replays"])
    res["n_replay_ok"] = sum(r["ok"] for r in res["replays"])
    if res["n_cases"] != N_EAGER + N_EDGE or res["n_replay"] != N_REPLAY:
        res["why"] = f"case count {res['n_cases']} / {res['n_replay']} != {N_EAGER + N_EDGE} / {N_REPLAY}"
        fail = True
    res["verdict"] = "FAIL" if fail else "PASS"
    done(1 if fail else 0)


if __name__ == "__main__":
    main()
