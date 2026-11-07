"""GB10 parity gate + IO-warp microbench of the DSV41_KV_LAYOUT=v41c readers (one idle GPU, the image, no engine).

The same latents are stored twice: as V4.1 FP4 records (the v41c extra cache; the model's rounding, encoded on the CPU by
tests/v41_fp4_ref.py, which the v41w window matched to SGLang's writer) and as the V4 pool v2.6 holds for them (the CPU
transcode, equal to SGLang's V4 store per v41w). Then, bit for bit:

1. transcode: the Triton kernel == the torch reference on every record of 128- and 256-token pages (data and scales);
2. decode (<= 64 rows): the v41c decode kernel (FP4, IO-warp transcode) == FlashInfer's stock decode-dsv4 (V4) for
   EVERY split-K tactic chunks_per_block in 1..num_splits (the tactic changes the reduction order; R1's claim is
   identity at a fixed tactic), rows 1 / 4 / 16 / 64, extra page sizes 128 / 256, extra top-k 512, with and without
   topk_length and sink;
3. prefill / wide verify (> 64 rows): the adapter's v41c prefill (FP4 -> V4 scratch -> stock prefill) == the stock prefill
   on the V4 pool, rows 72 / 96 (token mode) and 1,031 (page mode), and CUDA-graph replay of the 96-row token mode with
   inputs rewritten in place == eager;
4. microbench (informational, CUDA events, warm, median of 50): decode stock vs v41c at rows 1 / 16 and extra top-k
   512 / 2048 (the best stock tactic and the same tactic for v41c); "io_warp_bound" when v41c is > 10 % slower at any
   shape (then the two-IO-warp variant is the next step, not a KILL).
Exit 0 PASS, 1 FAIL (any bit difference), 75 INVALID (no GPU, the module did not verify / load).
usage: python3 gpu_test_v41c_reader.py [--out F.json]
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v41_fp4_ref as R  # noqa: E402

V4_PB = {p: -(-p * 584 // 576) * 576 for p in (64, 128, 256)}


def pools(pages, pbs, seed, dev):
    """(FP4 view [pages, pbs, 1, 288], V4 view [pages, pbs, 1, 584]) holding the same latents."""
    x = R.latents(pages * pbs, seed)
    fp4 = R.expected_pages(x, torch.arange(pages * pbs), pages, pbs).to(dev)
    d4, s4 = R.transcode_fp4_to_v4(*R.encode_rows(x))
    v4 = torch.zeros(pages, V4_PB[pbs], dtype=torch.uint8)
    v4[:, : pbs * 576] = d4.reshape(pages, pbs * 576)
    v4[:, pbs * 576: pbs * 584] = s4.reshape(pages, pbs * 8)
    v4 = v4.to(dev)
    pb4 = R.page_bytes(pbs)
    return (torch.as_strided(fp4, (pages, pbs, 1, 288), (pb4, 288, 288, 1)),
            torch.as_strided(v4, (pages, pbs, 1, 584), (V4_PB[pbs], 584, 584, 1)))


def split64(v4):
    """The stock decode reads 64-token pages: split a V4 pool like the adapter's _split_kv_pages_to_64."""
    pages, pbs = v4.shape[0], v4.shape[1]
    r = pbs // 64
    raw = torch.as_strided(v4, (pages, V4_PB[pbs]), (V4_PB[pbs], 1))
    out = torch.zeros(pages * r, V4_PB[64], dtype=torch.uint8, device=v4.device)
    for s in range(r):
        out[s::r, : 64 * 576] = raw[:, s * 64 * 576:(s + 1) * 64 * 576]
        out[s::r, 64 * 576: 64 * 584] = raw[:, pbs * 576 + s * 64 * 8: pbs * 576 + (s + 1) * 64 * 8]
    return torch.as_strided(out, (pages * r, 64, 1, 584), (V4_PB[64], 584, 584, 1))


def eq(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"transcode": [], "decode": [], "prefill": [], "bench": [], "verdict": "INVALID"}

    def done(code):
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("transcode", "decode", "prefill")}))
        sys.exit(code)

    if not torch.cuda.is_available():
        res["why"] = "no GPU"
        done(75)
    os.environ["DSV41_KV_LAYOUT"] = "v41c"
    dev = torch.device("cuda", 0)
    from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv4

    from sglang.kernels.ops.attention import dsv41_v41c as V
    from sglang.kernels.ops.attention import flash_mla_sm120 as A
    try:
        V.enabled.cache_clear()
        V._module()
        res["module"] = {"tag": V.check_sources(Path(os.environ.get(V.DIR_ENV, V.DEFAULT_DIR)),
                                                *V._flashinfer_dirs())["module_tag"]}
        runner = V._runner()
    except Exception as e:  # noqa: BLE001
        res["why"] = f"module: {type(e).__name__}: {e}"[:1500]
        done(75)
    fail = False
    g = torch.Generator(device=dev)
    g.manual_seed(1027)
    main_v4 = split64(pools(8, 256, 11, dev)[1])
    try:
        for pbs in (128, 256):
            fp4, v4 = pools(16 if pbs == 256 else 32, pbs, 20 + pbs, dev)
            ntok = fp4.shape[0] * pbs
            # 1. transcode
            toks = torch.arange(ntok, device=dev)
            n_pages = -(-ntok // 64)
            bt = torch.zeros(n_pages, V.SUB_PAGE_BYTES, dtype=torch.uint8, device=dev)
            br = torch.zeros_like(bt)
            V.transcode_tokens_triton(fp4, toks, bt, toks)
            V.transcode_tokens_torch(fp4, toks, br, toks)
            ok = torch.equal(bt, br)
            res["transcode"].append({"pbs": pbs, "records": ntok, "equal": ok})
            fail |= not ok
            # 2. decode, every tactic
            v4_64 = split64(v4)
            for rows in (1, 4, 16, 64):
                for variant in ("plain", "lengths_sink"):
                    q = (torch.randn(rows, 16, 512, generator=g, device=dev) * 0.3).to(torch.bfloat16)
                    idx = torch.randint(0, main_v4.shape[0] * 64, (rows, 128), generator=g, device=dev, dtype=torch.int32)
                    eidx = torch.randint(0, ntok, (rows, 512), generator=g, device=dev, dtype=torch.int32)
                    eidx[torch.rand(rows, 512, generator=g, device=dev) < 0.1] = -1
                    tl = etl = sink = None
                    if variant == "lengths_sink":
                        tl = torch.randint(1, 129, (rows,), generator=g, device=dev, dtype=torch.int32)
                        etl = torch.randint(0, 513, (rows,), generator=g, device=dev, dtype=torch.int32)
                        sink = torch.randn(16, generator=g, device=dev)
                    ns = 2 + 8
                    for cpb in range(1, ns + 1):
                        outs = []
                        for which in ("stock", "v41c"):
                            mid = torch.empty(rows, 16, ns, 512, dtype=torch.bfloat16, device=dev)
                            mlse = torch.empty(rows, 16, ns, dtype=torch.float32, device=dev)
                            out = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
                            lse = torch.empty(rows, 16, dtype=torch.float32, device=dev)
                            if which == "stock":
                                sparse_mla_sm120_decode_dsv4(q, main_v4, idx, mid, mlse, out, lse, 512 ** -0.5,
                                                             topk_length=tl, attn_sink=sink, extra_kv_cache=v4_64,
                                                             extra_indices=eidx, extra_topk_length=etl,
                                                             chunks_per_block=cpb)
                            else:
                                runner(inputs=[q, idx, mid, mlse, out, lse, tl, sink, eidx, etl], tactic=cpb,
                                       sm_scale=512 ** -0.5, kv_cache=main_v4, extra_kv_cache=fp4)
                            torch.cuda.synchronize()
                            outs.append((out.clone(), lse.clone()))
                        ok = eq(outs[0][0], outs[1][0]) and torch.equal(outs[0][1].view(torch.int32),
                                                                         outs[1][1].view(torch.int32))
                        res["decode"].append({"pbs": pbs, "rows": rows, "variant": variant, "cpb": cpb, "equal": ok})
                        fail |= not ok
            # 3. prefill through the adapter: FP4 (v41c) vs V4 (stock), same indices
            for rows in (72, 96, 1031):
                q = (torch.randn(rows, 1, 16, 512, generator=g, device=dev) * 0.3).to(torch.bfloat16)
                idx = torch.randint(0, main_v4.shape[0] * 64, (rows, 128), generator=g, device=dev, dtype=torch.int32)
                eidx = torch.randint(0, ntok, (rows, 512), generator=g, device=dev, dtype=torch.int32)
                eidx[torch.rand(rows, 512, generator=g, device=dev) < 0.1] = -1
                tl = torch.randint(1, 129, (rows,), generator=g, device=dev, dtype=torch.int32)
                etl = torch.randint(0, 513, (rows,), generator=g, device=dev, dtype=torch.int32)
                o4, _ = A._flash_mla_sm120_prefill(q, main_v4, idx, tl, None, 512, 512 ** -0.5, v4, eidx, etl)
                of, _ = A._flash_mla_sm120_prefill(q, main_v4, idx, tl, None, 512, 512 ** -0.5, fp4, eidx, etl)
                torch.cuda.synchronize()
                ok = eq(o4, of)
                res["prefill"].append({"pbs": pbs, "rows": rows, "mode": "token" if rows <= 96 else "page",
                                       "equal": ok})
                fail |= not ok
            # 3b. CUDA-graph replay of the 96-row token mode, inputs rewritten in place
            rows = 96
            q = (torch.randn(rows, 1, 16, 512, generator=g, device=dev) * 0.3).to(torch.bfloat16)
            idx = torch.randint(0, main_v4.shape[0] * 64, (rows, 128), generator=g, device=dev, dtype=torch.int32)
            eidx = torch.randint(0, ntok, (rows, 512), generator=g, device=dev, dtype=torch.int32)
            args = lambda ex: (q, main_v4, idx, None, None, 512, 512 ** -0.5, ex, eidx, None)  # noqa: E731
            A._flash_mla_sm120_prefill(*args(fp4))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                A._flash_mla_sm120_prefill(*args(fp4))
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gout, _ = A._flash_mla_sm120_prefill(*args(fp4))
            for rep in range(3):
                q.copy_((torch.randn(rows, 1, 16, 512, generator=g, device=dev) * 0.3).to(torch.bfloat16))
                eidx.copy_(torch.randint(0, ntok, (rows, 512), generator=g, device=dev, dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                ref, _ = A._flash_mla_sm120_prefill(*args(v4))
                torch.cuda.synchronize()
                ok = eq(gout, ref)
                res["prefill"].append({"pbs": pbs, "rows": rows, "mode": "graph-replay", "rewrite": rep, "equal": ok})
                fail |= not ok
        # 4. microbench (informational)
        fp4, v4 = pools(64, 256, 99, dev)
        v4_64 = split64(v4)
        ntok = 64 * 256
        for rows in (1, 16):
            for ek in (512, 2048):
                q = (torch.randn(rows, 16, 512, generator=g, device=dev) * 0.3).to(torch.bfloat16)
                idx = torch.randint(0, main_v4.shape[0] * 64, (rows, 128), generator=g, device=dev, dtype=torch.int32)
                eidx = torch.randint(0, ntok, (rows, ek), generator=g, device=dev, dtype=torch.int32)
                ns = 2 + ek // 64
                mid = torch.empty(rows, 16, ns, 512, dtype=torch.bfloat16, device=dev)
                mlse = torch.empty(rows, 16, ns, dtype=torch.float32, device=dev)
                out = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
                lse = torch.empty(rows, 16, dtype=torch.float32, device=dev)

                def t_of(fn):
                    for _ in range(5):
                        fn()
                    ms = []
                    for _ in range(50):
                        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        st.record()
                        fn()
                        en.record()
                        en.synchronize()
                        ms.append(st.elapsed_time(en))
                    return statistics.median(ms)
                best = None
                for cpb in range(1, ns + 1):
                    t = t_of(lambda: sparse_mla_sm120_decode_dsv4(q, main_v4, idx, mid, mlse, out, lse, 512 ** -0.5,
                                                                  extra_kv_cache=v4_64, extra_indices=eidx,
                                                                  chunks_per_block=cpb))
                    if best is None or t < best[1]:
                        best = (cpb, t)
                tv = t_of(lambda: runner(inputs=[q, idx, mid, mlse, out, lse, None, None, eidx, None], tactic=best[0],
                                         sm_scale=512 ** -0.5, kv_cache=main_v4, extra_kv_cache=fp4))
                res["bench"].append({"rows": rows, "extra_topk": ek, "cpb": best[0], "stock_ms": round(best[1], 4),
                                     "v41c_ms": round(tv, 4), "ratio": round(tv / best[1], 3)})
        res["io_warp_bound"] = any(b["ratio"] > 1.10 for b in res["bench"])
    except Exception as e:  # noqa: BLE001
        res["why"] = f"{type(e).__name__}: {e}"[:1500]
        done(1)
    res["counts"] = {k: [sum(x["equal"] for x in res[k]), len(res[k])] for k in ("transcode", "decode", "prefill")}
    res["verdict"] = "FAIL" if fail else "PASS"
    done(1 if fail else 0)


if __name__ == "__main__":
    main()
