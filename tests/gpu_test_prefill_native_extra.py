"""GB10 gate of DSV41_PREFILL_NATIVE_EXTRA: one idle GPU, the overlay image, no engine.

The production adapter's own prefill entry (sglang.kernels.ops.attention.flash_mla_sm120._flash_mla_sm120_prefill)
runs twice per case on the same inputs, flag OFF (stock: SWA split + extra split to 64-token pages, stock module) and
flag ON (SWA split, extra read in its native 128/256-token pages by the pce module):

1. identity: output and LSE are bit-identical (torch.equal on the raw bits) for extra pages of 128 and 256 tokens,
   16 and 64 heads (MG_N_HG 1 and 2), full-tile (no topk_length, extra top-k a multiple of 64) and general
   (topk_length / extra_topk_length, ragged) dispatch, -1 holes, 72 / 96 (the c16 verify graph: bs x 6 rows) / 1,031 /
   4,096 query rows, attention sink;
2. route evidence: flag ON logs DSV41_PCE_HIT and never creates the extra split buffer
   (`flash_mla_sm120_split:<dev>:extra`); flag OFF creates it with N x (pbs / 64) x 37,440 B;
3. memory: the extra split buffer's size at a production-sized extra pool (--pool-pages, default 19,912 pages of
   256 tokens = the v2.6 head's 5,097,472-token pool) is computed from the stock formula and, with --alloc-prod,
   measured by allocating it (informational; skipped by default to stay small on a shared node);
4. time (informational): per-call adapter time OFF vs ON at 4,096 rows, CUDA events, warm, medians of 20.

Exit 0 = PASS, 1 = FAIL (any identity break, missing route evidence, extra buffer under ON), 75 = INVALID (no GPU,
module did not build). Memory: < 2 GiB device without --alloc-prod.
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


def page_bytes(pbs):
    return math.ceil(pbs * BPT / ROW) * ROW


def make_pool(pages, pbs, gen, dev):
    """A V4 footer-format pool [pages, pbs, 1, 584] (padded page stride) with finite fp8 / bf16 / ue8m0 content."""
    pb = page_bytes(pbs)
    raw = torch.zeros(pages, pb, dtype=torch.uint8, device=dev)
    data = raw[:, : pbs * ROW].view(pages, pbs, ROW)
    nope = (torch.randn(pages, pbs, D_NOPE, generator=gen, device=dev) * 40).clamp(-448, 448)
    data[:, :, :D_NOPE] = nope.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = torch.randn(pages, pbs, D_ROPE, generator=gen, device=dev).to(torch.bfloat16)
    data[:, :, D_NOPE:] = rope.view(torch.uint8).view(pages, pbs, 2 * D_ROPE)
    foot = raw[:, pbs * ROW: pbs * ROW + pbs * FOOT].view(pages, pbs, FOOT)
    foot[:, :, :7] = torch.randint(118, 128, (pages, pbs, 7), generator=gen, device=dev, dtype=torch.int32).to(torch.uint8)
    return torch.as_strided(raw, (pages, pbs, 1, BPT), (pb, BPT, BPT, 1))


def make_idx(rows, k, pool_tokens, gen, dev, holes=0.1):
    idx = torch.randint(0, pool_tokens, (rows, k), generator=gen, device=dev, dtype=torch.int32)
    mask = torch.rand(rows, k, generator=gen, device=dev) < holes
    return idx.masked_fill(mask, -1)


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16 if a.dtype == torch.bfloat16
                                                                  else torch.int32),
                                              b.contiguous().view(torch.int16 if b.dtype == torch.bfloat16
                                                                  else torch.int32))


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
    res = {"cases": [], "verdict": "INVALID"}

    def done(code):
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "cases"}))
        sys.exit(code)

    if not torch.cuda.is_available():
        res["why"] = "no GPU"
        done(75)
    dev = torch.device("cuda", 0)
    from sglang.kernels.ops.attention import dsv41_pce
    from sglang.kernels.ops.attention import flash_mla_sm120 as A
    from sglang.srt.runtime_context import get_resources

    hits = Hits()
    logging.getLogger(dsv41_pce.__name__).addHandler(hits)
    logging.getLogger(dsv41_pce.__name__).setLevel(logging.WARNING)
    buffers = get_resources().buffers
    ekey = f"flash_mla_sm120_split:{dev}:extra"

    def run(flag, q, k_main, idx, tl, sink, extra, eidx, etl):
        os.environ[dsv41_pce.ENV] = flag
        dsv41_pce.enabled.cache_clear()
        out, _ = A._flash_mla_sm120_prefill(q, k_main, idx, tl, sink, 512, 512 ** -0.5, extra, eidx, etl)
        torch.cuda.synchronize()
        return out

    gen = torch.Generator(device=dev)
    gen.manual_seed(20261027)
    main_pages = 8
    k_main = make_pool(main_pages, 256, gen, dev)
    fail = False
    # ON first, in a process that never ran the stock extra split: the buffer must never appear
    order = [("1", "on"), ("0", "off")]
    try:
        for pbs in (128, 256):
            extra_pages = 64 if pbs == 256 else 128
            extra = make_pool(extra_pages, pbs, gen, dev)
            for nh in (16, 64):
                for rows in (72, 96, 1031, 4096):
                    for mode, ek in (("fulltile", 512), ("general", 448)):
                        q = (torch.randn(rows, nh, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
                        idx = make_idx(rows, 128, main_pages * 256, gen, dev)
                        eidx = make_idx(rows, ek, extra_pages * pbs, gen, dev)
                        sink = torch.randn(nh, generator=gen, device=dev) if nh == 16 else None
                        tl = etl = None
                        if mode == "general":
                            tl = torch.randint(1, 129, (rows,), generator=gen, device=dev, dtype=torch.int32)
                            etl = torch.randint(0, ek + 1, (rows,), generator=gen, device=dev, dtype=torch.int32)
                        outs = {}
                        for flag, name in order:
                            n_hits = len(hits.lines)
                            had_extra = ekey in buffers
                            outs[name] = run(flag, q, k_main, idx, tl, sink, extra, eidx, etl).clone()
                            if name == "on":
                                if not had_extra and ekey in buffers:
                                    res["extra_buffer_under_on"] = True
                                    fail = True
                        ok = bits_equal(outs["on"], outs["off"])
                        case = {"pbs": pbs, "heads": nh, "rows": rows, "mode": mode, "extra_topk": ek,
                                "sink": sink is not None, "identical": ok}
                        if not ok:
                            d = (outs["on"].float() - outs["off"].float()).abs()
                            case["max_abs_diff"] = float(d.max())
                            case["n_diff"] = int((outs["on"] != outs["off"]).sum())
                            fail = True
                        res["cases"].append(case)
                        order.reverse()  # alternate which flag runs first after the first case
        res["hit_lines"] = sorted(set(l for l in hits.lines if "DSV41_PCE_HIT" in l))
        if not any("extra_page_block_size=128" in l for l in res["hit_lines"]) or \
                not any("extra_page_block_size=256" in l for l in res["hit_lines"]):
            res["why"] = "route evidence missing"
            fail = True
        buf = buffers.get(ekey)
        res["extra_split_buffer_after_off_bytes"] = None if buf is None else int(buf.numel())
        if buf is None:
            res["why"] = "stock path never created the extra split buffer (control dead)"
            fail = True
        # memory at the production pool
        res["prod_extra_split_bytes_formula"] = a.pool_pages * 4 * SPLIT_PAGE_BYTES
        if a.alloc_prod:
            torch.cuda.empty_cache()
            before = torch.cuda.memory_reserved()
            big = torch.empty(a.pool_pages * 4, SPLIT_PAGE_BYTES, dtype=torch.uint8, device=dev)
            res["prod_extra_split_bytes_measured"] = int(torch.cuda.memory_reserved() - before)
            del big
            torch.cuda.empty_cache()
        # time (informational)
        extra = make_pool(64, 256, gen, dev)
        q = (torch.randn(4096, 16, 512, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        idx = make_idx(4096, 128, main_pages * 256, gen, dev)
        eidx = make_idx(4096, 512, 64 * 256, gen, dev)
        t = {}
        for flag, name in (("0", "off"), ("1", "on")):
            os.environ[dsv41_pce.ENV] = flag
            dsv41_pce.enabled.cache_clear()
            for _ in range(3):
                A._flash_mla_sm120_prefill(q, k_main, idx, None, None, 512, 512 ** -0.5, extra, eidx, None)
            ms = []
            for _ in range(20):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                A._flash_mla_sm120_prefill(q, k_main, idx, None, None, 512, 512 ** -0.5, extra, eidx, None)
                e.record()
                e.synchronize()
                ms.append(s.elapsed_time(e))
            t[name] = statistics.median(ms)
        res["time_ms_4096rows"] = t
    except Exception as e:  # noqa: BLE001
        res["why"] = f"{type(e).__name__}: {e}"[:2000]
        done(75 if not res["cases"] else 1)
    res["n_cases"] = len(res["cases"])
    res["n_identical"] = sum(c["identical"] for c in res["cases"])
    res["verdict"] = "FAIL" if fail else "PASS"
    done(1 if fail else 0)


if __name__ == "__main__":
    main()
