# Measurement history

Every dated table, note and rejected experiment that used to live in the README, moved here unchanged when the README was reduced to the current state (2026-09-24). What changed in which commit is in [CHANGELOG.md](../CHANGELOG.md); the current numbers are in the [README](../README.md#current-results).

All rows below were taken with sparkDash up to 1.8.6. sparkDash 1.8.7 replaced the code prompts (c1 and the concurrent waves), so the code columns here are not comparable with the README's current table; the prose, structured and prefill prompts are unchanged.

## Production 2026-09-25 (v2, with the v2.1 c1 and prefill), sparkDash 1.8.8

Production stack (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](../.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes and measured on that image with sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-25. Engine start to healthy 160-170 s; KV pool 6.0-6.6M tokens (1M context). Raw output: [`docs/results/validation-20260925-v2.txt`](results/validation-20260925-v2.txt).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **87.7** | 120.4 (61.9) | 163.6 (41.4) | 237.6 (30.8) | 342.7 (22.2) |
| code | 124.8 | 175.0 (88.4) | 246.8 (63.2) | 309.8 (41.4) | 438.3 (29.3) |
| structured | 152.4 | 177.6 (103.3) | 240.4 (70.4) | 295.5 (44.2) | 572.2 (44.4) |
| json | 118.9 | 174.2 (89.8) | 301.7 (76.2) | 471.5 (60.2) | 659.9 (42.7) |

Prose and code c1 are from the latest fresh-clone boot (v2.1, prose median of five runs 86.7-87.7 after two discarded warm-ups; the v2 boot gave 86.5 and 122.6); the other columns are from the v2 sweep, whose decode path is identical. With the deterministic MoE reduction the greedy text is identical run to run, so the sparkDash numbers repeat within about ±1 tok/s; sparkDash's prose c1 is one prompt, and a stack that sums in a different order (another fabric, another all-reduce) follows a different greedy text there, so compare step time or a many-prompt benchmark across stacks. On 45 varied prompts (prose, structured and other catalogs, c1 greedy) the same image runs 58.4 / 94.3 / 71.8 tok/s; decode step 32.4-33.6 ms on prose and 38.9-39.4 ms on code at c1. sparkDash uses a different set of prompts at each concurrency for the non-prose types, so per-stream values are not comparable across columns. Sampled chat at the model card's T=1 / top_p=0.95 with thinking (c1, 18 requests x 800 tokens on two prompt sets) runs 67.2 / 65.9 tok/s (measured on the 2026-09-24 stack; sparkDash benches are greedy, where the draft temperature and block verification do not act).

**Prefill, cold, tok/s by prompt length** (two passes; fresh clone of the MXFP8 attention-gather commit, [`docs/results/validation-20260925-v21.txt`](results/validation-20260925-v21.txt), which also measured prose c1 87.7 (86.7-87.7), code c1 124.8, qeval 72/75 and the 1,011,084-token needle PASS in 325 s)

| 4k | 16k | 32k | 64k | 128k | 262k |
|---:|---:|---:|---:|---:|---:|
| 4119 / 4840 | 5891 / 5846 | 5893 / 5868 | 5936 / 5903 | 5793 / 5761 | 5355 / 5360 |

sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text (documentation and source code, a unique prefix per prompt so nothing comes from the prefix cache) the same image measured 4202-4685 / 4824-4922 / 4827-4830 / 4942-4970 / 4822-4871 tok/s at ~4k / ~15k / ~32k / ~62k / ~120k tokens. Needle retrieval (a single phrase in varied filler at 37 % depth): PASS at 99k, 198k, 746k and 1,011,084 tokens (the last one in 322 s, head `MemAvailable` low-water 7 GiB). A harder list-lookup needle ("value of item N" in a list of up to 197,000 items) passes for about half the keys at 129k and 259k on this stack and on the previous one alike, a limit of the model rather than of either stack.

## Production 2026-09-24 evening (EP1 release), sparkDash 1.8.8

Production stack (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](../.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes and measured on that image with sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-24. Engine start to healthy 170 s; KV pool 6.56M tokens (1M context). Raw output: [`docs/results/validation-20260924-ep1.txt`](results/validation-20260924-ep1.txt).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **84.6** | 118.0 (60.6) | 160.4 (40.5) | 231.6 (30.1) | 339.8 (22.1) |
| code | 120.9 | 172.7 (87.2) | 241.6 (61.9) | 303.3 (40.6) | 436.9 (29.2) |
| structured | 146.0 | 172.5 (100.6) | 232.8 (68.4) | 286.4 (42.9) | 565.5 (44.6) |
| json | 119.2 | 168.5 (86.9) | 299.4 (75.7) | 461.3 (58.9) | 658.7 (42.7) |

Prose c1 is the median of seven runs (84.2-85.2) after two discarded warm-ups; other boots of the same stack gave medians of 85.0-85.2. With the deterministic MoE reduction the greedy text is identical run to run, so the sparkDash numbers repeat within about ±1 tok/s. On 45 varied prompts (prose, structured and other catalogs, c1 greedy) the same image runs 57.4 / 92.9 / 70.8 tok/s. sparkDash uses a different set of prompts at each concurrency for the non-prose types, so per-stream values are not comparable across columns. Sampled chat at the model card's T=1 / top_p=0.95 with thinking (c1, 18 requests x 800 tokens on two prompt sets) runs 67.2 / 65.9 tok/s (the previous stack 62.3 / 59.8); sparkDash benches are greedy, where the draft temperature and block verification do not act.

**Prefill, cold, tok/s by prompt length** (two passes)

| 4k | 16k | 32k | 64k | 128k | 262k |
|---:|---:|---:|---:|---:|---:|
| 2449 / 3986 | 4784 / 4753 | 4870 / 4805 | 4815 / 4762 | 4721 / 4722 | 4264 / 4447 |

The 4k point of the first pass is the first request after the benches. sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text (documentation and source code, a unique prefix per prompt so nothing comes from the prefix cache) the same image measured 4003-4077 / 4154-4255 / 4144-4258 / 4138-4189 / 4092-4093 tok/s at ~4k / ~15k / ~29k / ~60k / ~113k tokens. Needle retrieval (a list lookup): passes at 129k tokens; at 259k some keys pass and some miss on both this stack and the previous FlashInfer/EP2 stack (key 17777 misses on both, 20001 and 3333 pass on both), a limit of the model at that length rather than of either stack. Engine start to ready is ~3 minutes with the fast loader.

## Production 2026-09-24 morning (EP2, FlashInfer MoE), sparkDash 1.8.8

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 74.8 | 101.9 (53.5) | 143.9 (37.4) | 192.9 (25.6) | 318.5 (20.8) |
| code | 113.0 | 151.4 (75.7) | 212.6 (55.3) | 280.7 (37.3) | 391.8 (26.7) |
| structured | 137.1 | 156.0 (88.1) | 196.1 (57.1) | 201.7 (35.7) | 517.2 (44.2) |
| json | 105.6 | 164.8 (82.4) | 261.1 (65.6) | 398.2 (52.1) | 648.9 (41.8) |

Prefill 4k / 16k / 32k / 64k / 128k / 262k: 3119 / 4001 / 4681 / 4675 / 4575 / 4206 tok/s. Raw output:
[`results/sweep-20260924-types.txt`](results/sweep-20260924-types.txt), [`results/prodbench-20260924-current.txt`](results/prodbench-20260924-current.txt).

## Production stack as of 2026-09-24, with the full rationale per switch

One image, one env file. Everything in the tables below labelled **production** is this stack. What changed when is in [CHANGELOG.md](../CHANGELOG.md):

| Layer | Setting | Status | Why |
|---|---|---|---|
| Image | `Dockerfile.canary-roce` = upstream `dsv4.1` branch at `f80c91a4b` + RoCEnante overlay + all adapters | **on** | fastest decode of the three images (branch kernels + RDMA all-reduce) |
| Slots | `MAX_RUNNING_REQUESTS=16` | on | adds the c16 tier; c1–c8 unchanged |
| Experts | `EP_SIZE=2`, `--enable-deepseek-v4-fp4-indexer` | on | straggler wait halved; kernel path |
| Engram | `DSV41_CACHE_GIB=4`, `DSV41_CACHE_WAYS=16`, `DSV41_ENGRAM_PREFETCH=1` | on | row cache on NVMe; the row lookups run on a side stream right after the hasher instead of stalling the graph before each gather: step 51.7 → 49.5 ms, real-text prose c1 +5 %, rows bit-identical ([docs/upstream-watch.md](upstream-watch.md)) |
| Draft | `DSPARK_BLOCK_SIZE=5`, `SGLANG_DSPARK_FOLDED_SAMPLING=2` | on | k=5 wins on code, ties on prose; forced fold keeps sampled decode equal to greedy on the branch |
| Prefill | `CHUNKED_PREFILL_SIZE=4096` + `DSV41_INDEXER_CHUNKED=1` (v3) + `SPARK_PREFILL_TP_SPLIT=1` | on | bounded indexer transient (sglang#39187) plus the SG18 row split across ranks; 985k prompt leaves 5 GiB on the head |
| Shared expert | `DSV41_SHARED_PAD_K=1` | on | keeps the K=576 shape on the b12x kernel, −0.9 ms/step, bit-identical |
| wo_a | `DSV41_WO_A_W8=1` | on | verify/draft `wo_a` reads the checkpoint's fp8 bytes (exact twin of the bf16 copy) in the stock tiling: 43 layers 3.43 → 2.20 ms, step 52.9 → 51.7 ms at c1 |
| Draft temperature | `DSV41_DRAFT_TAU=0.7` | on | sampled requests only; exact by construction (same q for proposal and acceptance). At T=1 / top_p=0.95 with thinking (c1, 18 x 800 tokens per arm), 0.7 beat 0.8 on two disjoint prompt sets: 62.3 vs 61.2 and 59.8 vs 58.6 tok/s (+1.8 %, +2.0 %; accepted tokens per step +3.3 %, +2.2 %); 0.6 and 0.9 were below 0.7. (0.8 was the offline pick on 2026-09-23, +1.2 % over no scaling.) |
| Draft LM head | `DSV41_DRAFT_HEAD_FP8=1` | on | the draft reads an fp8 copy of the shared LM head (target logits untouched): 1405 → 721 us per step, acceptance unchanged |
| Verification | `DSV41_BLOCK_VERIFY=1` | on | block verification for sampled rows: exact output distribution, +1.8 % (prose) / +2.5 % (coding with thinking) accepted tokens per step |
| Folded results | `DSV41_FOLDED_FENCE=1` | on | correctness: folded (all-greedy) verify results cloned before the overlapped D2H copy, closes the sglang#40919 race |
| Verify length | `DSV41_VERIFY_CAP=conf:0.1` | on | per request and step, only the leading drafts whose running product of the draft confidence head's survival stays >= 0.1 are verified; the other verify rows are routed to the anchor row's experts, so they add no expert reads (each such row saves ~2 ms at c1), and acceptance is capped at the verified drafts through the engine's own cutoff. Exact: greedy outputs identical, sampled rows go through block verification with the dropped positions removed. Prose c1 +5 %, prose c4 +6 %, sampled thinking traffic +5 %, code and structured flat |
| Engram cache (optional) | `DSV41_ENGRAM_DRM_NODE=/dev/dri/card0` | off by default | one Engram layer's row cache in the GB10 display reservation (outside `MemAvailable`): ~1.8 GiB of runtime headroom per node, same speed and hit rate; needs a host change, see [docs/display-reserve.md](display-reserve.md) |
| Autotune cache | `DSV41_AUTOTUNE_KEEP=1` | on | keeps FlashInfer's MoE autotune cache across boots under EP (sglang#40320: the stock gate deleted it on every boot and re-drew the tactics, 26 re-tunes per start); kept only while the launch configuration matches |
| wo_a at c2+ / KV | `DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1` | on | verify/draft `wo_a` at 9-192 rows also reads the fp8 twin (per-stream +5 % at c4, +3 % at c8 by verify step time); then the bf16 copy is released (722 MB per rank incl. the draft) and prefill dequantizes per call, bit-identical |
| Replicated linears | `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv` | on | `engram.wkv` (6144 → 25600, 183 MB MXFP8) and `wqkv_a` (5120 → 1792) are `ReplicatedLinear`: every rank streamed the whole weight for the same output (the two Engram projections alone 2 × 756 us per step at c1). Each rank now runs the same quantized linear on its 128-row weight tiles and the columns are all-gathered; enabled per layer only when the slice is bit-identical to the stock layer on all ranks (checked at boot). Step probe prose 38.2 → 36.8 ms, code 44.9 → 43.6 ms; greedy outputs identical |
| Router remap | `DSV41_ROUTER_LIVE=1` | on | the verify-length remap (dead rows take the anchor row's experts) folded into the router kernel: dead rows load the anchor's scores, so the router computes the anchor's ids and weights for them and the separate remap kernel (40 launches, 0.21 ms/step) is gone. Built from the engine's own router source with three text substitutions; bit-identical outputs (checked with and without per-token bias); greedy text identical. Varied prompts +0.3 %, sparkDash unchanged within noise |
| Draft main_proj | `DSV41_DRAFT_MAIN_PROJ_SPLIT=1` | on | the draft's replicated `main_proj` (15360 → 5120, 105 MB) as a 1/4 column shard in fp8 with the checkpoint's exact weight bytes + all-gather: 430 → ~150 us per step; draft only, the target is untouched |
| Transport | `SGLANG_ROCE_ALLREDUCE=1`, `SGLANG_ROCE_MAX_SIZE=2097152`, `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0`, `DSV41_ROCE_GATHER=2097152` | on | TP SUM all-reduces up to 2 MiB over RDMA on both rails; 2 MiB covers the 16-slot step (983 KB). `DSV41_ROCE_GATHER` also routes TP all-gathers up to 2 MiB per rank (the draft's vocab-parallel logits, 6 per step, and the splits above) through the one-shot kernel: 600 → 400 us per step at c1, a byte copy |
| NCCL | `IB_HCA=rocep1s0f0,roceP2p1s0f0` | on | neutral within noise, kept for the remaining collectives |
| Fabric | switched RoCE, tree reachable | on | every default assumes a switch; a switchless ring sets `NCCL_SWITCHLESS_RING_ONLY=1` instead ([optional-setups.md](optional-setups.md#switchless-ring-no-roce-switch)) |
| Serving | `--enable-cache-report`, `--sleep-on-idle`, `--min-free-slots-delay 1`, `DSV41_MAX_NEW_TOKENS`, loop abort, thinking alias | on | cached-token usage for clients; sleep-on-idle takes the head scheduler from 47 % to 14 % CPU when idle with no change to first-response latency (0.2 s) or decode; the rest is upstream's |
| Weight loading | `DSV41_FAST_LOAD=1` (+ `--model-loader-extra-config {"num_threads":1}`) | **on** | engine start 343 s → 111–124 s, bytes identical, decode/prefill/needle unchanged; costs 3–13 % of the KV pool (6.71–7.27 M vs 7.47–7.82 M tokens on the same image), the one trade-off in this table ([docs/fast-load.md](fast-load.md)) |
| Rust image processor | `SGLANG_RUST_BUILD_MODE=never` | off | the branch's `cargo` probe can hang the head before the HTTP server starts; PIL path is used |
| Adaptive chunk sizer | `DSV41_ADAPTIVE_CHUNK` | off | superseded by the bounded indexer; it would only shrink chunks needlessly |
| DSpark SPS table / ragged verify | `DSPARK_SPS_TABLE` | off (file absent) | crashes the Engram path on this model; verify-all schedule stays |
| NVFP4 checkpoint (`nvidia/DeepSeek-V4.1-Flash-NVFP4`) | – | not used | routed experts only, no bandwidth saved on GB10, +16 GiB, DSpark unvalidated |

Rollback to any earlier point is an env change: `DSV41_FAST_LOAD=0` restores the stock loader, `SGLANG_ROCE_ALLREDUCE=0` drops the RDMA transport, `SPARK_PREFILL_TP_SPLIT=0` the row split, `IMAGE=dsv41-4x-spark:canary` the RoCEnante overlay, `IMAGE=dsv41-4x-spark:local` the branch.

## Decode, prefill and boot time by stage

sparkDash decode bench, 256 new tokens, temperature 0, thinking off, idle fleet, no foreign traffic (checked against the engine's `#running-req` log). Four DGX Spark, TP4/EP2, driver 580.x, `lmsysorg/sglang:dev-dsv41` base. Run-to-run spread between boots of the same configuration is about ±2 % on c1, so differences inside that band are noise. Both images were rebuilt from a fresh clone of this repository on 2026-09-17 and re-measured (base: prose c1 52.7 / c8 170, code c1 100.5; canary: prose c1 55.2 / c8 177, code c1 100.8). Full record with every intermediate step: [`docs/window-20260916.md`](window-20260916.md), raw outputs in [`results/window-20260916/`](results/window-20260916/).

### Prose decode, aggregate tok/s (per stream in brackets)

| Profile | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| upstream TP4 example (from its README) | 45.4 | 72.9 | 103.1 (26.7) | 114.1 (23.2) | 134.2 (22.0) |
| this profile, `Dockerfile` (base image) | 51.6 | 76.7 | 109.3 (28.5) | 160.9 (20.8) | 248.8 (16.7) |
| this profile, `Dockerfile.canary` (upstream dsv4.1 branch) | 55.4 | 80.9 | 118.9 (30.7) | 178.2 (24.0) | 277.5 (18.6) |
| production on 2026-09-18 (`Dockerfile.canary-roce`, 2 MiB route, prefill TP split, both rails, fast load, Engram prefetch) | 61.0 | 85.9 | 125.2 (33.2) | 178.9 (24.3) | 292.9 (19.3) |
| production at noon 2026-09-23 (the 2026-09-18 stack + `wo_a` fp8 twin, fp8 draft LM head, draft temperature, block verification, folded fence; 2026-09-23) | 66.0 | 86.2 | 125.4 (32.9) | 190.6 (25.5) | 307.8 (20.4) |
| production 2026-09-23 evening (the noon stack + adaptive verify length `DSV41_VERIFY_CAP=conf:0.1` + kept autotune cache) | 69.7 | 91.4 | 133.5 (34.7) | 195.1 (25.7) | 310.3 (20.5) |
| production 2026-09-23 night (+ `wo_a` fp8 twin at 9-192 rows and the bf16 copy released) | 69.1 | 97.5 | 138.3 (36.2) | 189.6 (25.2) | 314.1 (20.5) |
| the same + RoCE all-gathers + draft `main_proj` split, uncapped GPU clock (2026-09-24; the rows above ran under a local 2200 MHz cap, which [costs ~1.5 % on prose](upstream-watch.md)) | 72.0 | 101.0 | 141.1 (36.7) | 192.9 (25.5) | 318.1 (20.8) |
| **production** (+ `engram.wkv` and `wqkv_a` column-split, bit-identical; 2026-09-24) | **73.9** | **102.4** | **142.3 (37.1)** | **191.5 (25.3)** | **319.2 (21.2)** |

### Code and structured decode, aggregate tok/s

| Profile | code c1 | code c8 | code c16 | structured c1 |
|---|---:|---:|---:|---:|
| this profile, base image | 96.7 | 446.7 | 595.3 | 104.5 |
| this profile, canary image | 100.4 | 513.3 | 838.6 | 108.0 |
| production on 2026-09-18 | 113.3 | 548.3 | 882.4 | 124.1 |
| production at noon 2026-09-23 | 118.3 | 543.8 | 879.7 | 128.8 |
| production 2026-09-23 evening | 114.5 | 540.8 | 878.9 | 126.4 |
| production 2026-09-23 night | 113.8 | 556.3 | 885.7 | 125.5 |
| RoCE all-gathers + draft `main_proj` split, uncapped clock (2026-09-24) | 120.8 | 565.9 | 900.5 | 129.5 |
| **production** (+ replicated-linear split, 2026-09-24) | **119.0** | **573.5** | **914.3** | **131.5** |

### Prefill, cold, tok/s by prompt length

| Profile | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| upstream example (chunk 1024) | 3350 | 3782 | 3768 | 3531 | 3251 | – |
| this profile, base image (chunk 4096 + indexer backport) | 3532 | 4006 | 4038 | 3917 | 3230 | 2724 |
| this profile, canary image | 3174 | 3982 | 4180 | 4010 | 3499 | 2701 |
| production on 2026-09-18 (canary-roce + prefill TP split + fast load + Engram prefetch) | 3202 | 3497 | 4665 | 4674 | 4539 | 4241 |
| production at noon 2026-09-23 | 3619 | 4516 | 4501 | 4500 | 4465 | 4169 |
| production 2026-09-23 evening (4k/16k repeated after the first cold pass) | 4086 | 4510 | 4566 | 4530 | 4446 | 4125 |
| production 2026-09-23 night (4k/16k/32k/128k from a repeated pass) | 3983 | 4592 | 4575 | 4539 | 4405 | 4082 |
| **production** (2026-09-24, single cold pass; the 4k point is the first request of the pass) | **3159** | **3638** | **4783** | **4804** | **4677** | **4337** |

**Caveat on the prefill table:** sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache (`DSV41_CACHE_GIB=4`) inflates those numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)). The same canary engine on random-word text, cold, one request per size, `prompt_tokens / TTFT`:

| random text | 11.8k | 23.8k | 47.3k | 94.3k | 188.7k |
|---|---:|---:|---:|---:|---:|
| canary, tok/s | 3330 | 3666 | 3347 | 3187 | 2657 |
| production, tok/s | 2879 | 3612 | 3776 | 4006 | 3092 |

Use these rows for real prompts; the sparkDash column overstates by 9–20 % at 16k–128k. The v3 row's 12k value is a single cold request right after boot (the split does not engage below 32k).

Long-context checks: needle retrieval PASS at 131k, 262k and **985k** tokens on the canary image (985k cold prefill 732 s, head `MemAvailable` low-water 6.6 GiB) and on production (131k 25.6 s, 262k 58 s, **985k 585 s**, low-water 5.0 GiB). The split without the chunked scoring (SG18 as published, base image) reached 503 s at 985k but left only 1.9 GiB on the head, which is why v3 keeps the 2 GiB logits budget inside each rank's partition.

### Boot time

| | stock loader | fast load (`DSV41_FAST_LOAD=1`) |
|---|---:|---:|
| target `load_weight` (rank 0 / 1 / 2 / 3) | 225–246 / 95 / 246 / 114 s | 71–74 / 83 / 73 / 71 s |
| draft `load_weight` | 38–50 s | 3–8 s |
| engine start to ready (`scheduler_e2e`) | 343–354 s | 124–129 s |
| `max_total_num_tokens` (KV pool, same image, same night) | 7.47–7.82 M | 7.27 M and 6.71 M on two boots (pinned buffers); 6.2–6.8 M with the earlier mmap buffers |

Same image, gate on versus off, 2026-09-18. Decode, prefill and the needle test are unchanged. The KV pool is 3–13 % smaller (it also varies more from boot to boot): SGLang sizes it from the head's `MemAvailable` right after the loads, and ~0.8 GB less is available then with the fast loader (with pageable buffers it was 1.5 GB, traced to driver staging memory; the remainder shows only as mapped file pages of the scheduler process). Flip `DSV41_FAST_LOAD=0` if the last 0.5–1 M tokens of pool matter more than 220 s per boot. Profile, dead ends and raw snapshots: [docs/fast-load.md](fast-load.md).

## Adaptive verify length (2026-09-23 evening)

Same boots, sparkDash c1 medians of three; step probe = greedy 400-token requests with the engine's own `spec_verify_ct`:

| `DSV41_VERIFY_CAP` | prose c1 | code c1 | structured c1 | step probe prose (ms/step, tok/s) | step probe code (ms/step, tok/s) |
|---|---:|---:|---:|---:|---:|
| unset | 66.1 / 65.5 | 118.2 / 115.8 | 126.4 / 124.9 | 47.4, 48.3 | 48.5, 82.0 |
| `5` (machinery, all rows live) | 65.1 | 116.6 | 124.7 | 47.7, 48.1 | 48.8, 81.4 |
| `3` (fixed) | 60.5 | 86.0 | 72.0 | 44.9, 48.4 | 43.7, 71.3 |
| `2` (fixed) | 56.5 | 69.2 | 70.4 | 40.7, 49.9 | 41.2, 63.9 |
| `conf:0.05` | 67.5 | 115.7 | 123.9 | 42.4, 51.9 | 48.0, 82.8 |
| `conf:0.1` (two boots) | **69.3 / 68.8** | 115.9 / 113.6 | 124.3 | 41.0, **52.4** | 47.4, 82.6 / 80.9 |
| `conf:0.2` | 65.7 | 115.9 | 124.5 | 40.1, 49.1 | 48.0, 78.4 |
| `conf:0.35` | 61.8 | 116.4 | 124.3 | 38.6, 48.9 | 44.5, 80.0 |

Sampled thinking traffic (T=1, top_p 0.95, 6 technical prompts x 2, 800 tokens, c1), boots in the order conf / unset / conf / unset: 53.8, 53.2, 56.7, 52.0 tok/s, i.e. 55.3 vs 52.6 (+5 %); accepted tokens per step 2.62 vs 2.72, step 47.4 vs 51.9 ms. qeval 71 and 72 of 75 (primary 53 of 55; `math_m9` hits the 640-token cap on most images, `prose_p2` as noted below). A fixed cap loses: the head is what makes the cut pay. Rebuilt from a fresh clone of this repository and booted with the `.env.tp4.example` production line: prose c1 70.2 (median of 62.8 / 70.2 / 70.2), code c1 117.9, structured 125.7; qeval 73 of 75.

`DSV41_REPLICATED_SPLIT`, `DSV41_DRAFT_MAIN_PROJ_SPLIT`, `DSV41_ROCE_GATHER` (2026-09-24, uncapped clock, greedy step probe = 3 prompts x 400 tokens, median of the 2nd/3rd run after boot): the six per-step draft vocab all-gathers over RoCE instead of NCCL 600 -> 400 us, prose step 38.9 -> 38.4 ms; the draft `main_proj` shard prose 38.4 -> 38.2 ms, code 45.3 -> 44.9 ms; `engram.wkv` split prose 38.2 -> 37.1 ms, code 44.9 -> 43.9 ms (sparkDash prose c1 71.8 -> 74.1, code c1 118 -> 121); `wqkv_a` split prose 37.1 -> 36.8 ms (sparkDash prose c1 74.1 -> 74.7). Every split layer was bit-identical on all four ranks at boot (40 `wqkv_a` + 2 `engram.wkv`, plus the draft's), and greedy outputs match the unsplit engine byte for byte. Splitting the indexer's `wq_b` and the compressor's `wkv_gate` the same way gave nothing (prose 36.8 -> 36.9 ms) and is not in the production line. Cost: the column slices are extra copies next to the full weights (which prefill still uses), ~210 MB per rank allocated after the KV pool is sized. Fresh clone of this state, built on all four nodes and booted with the `.env.tp4.example` production line: every split layer ON and bit-identical, greedy outputs identical to the pre-split engine, sparkDash prose c1 74.7 (74.72 / 74.68 / 74.05), code c1 120.1 / 124.4; qeval 71 of 75 (`code_interval_intersect`, `json_escape`, `math_m9` fail on every image here, `json_count` has failed before).

`DSV41_WO_A_W8_MID`, verify step time with 400-token greedy requests on different prompts (two boots per arm, on / off): c2 59.0 / 58.4 ms, c4 78.3 / 82.1 ms, c8 118.1 / 120.0 ms; greedy c1 text identical. sparkDash's fixed prompts are not a usable A/B for kernels that change rounding order: any change to the prefill numerics flips a near-tie somewhere in the 256 tokens and moves c1/c2 by +-10 %, which is why the mid path leaves prefill on the stock kernel. Fresh clone of this state (all switches of the production line): prose c1 69.2 (69.6 / 66.8 / 69.2), code c1 116.9, structured 123.1, prose c4 137.8; qeval 72 of 75. `DSV41_WO_A_W8_DROP`: greedy outputs byte-identical to the bf16 path (after the first request of a boot, which drifts with or without it), prefill and decode unchanged; KV pool 6.10-6.43 M tokens over four boots against 5.93-6.34 M without.

Note on measuring: a dashboard polling `nvidia-smi` every 2 s on every node cost 0.6 ms per decode step here (47.1 vs 46.5 ms/step with it paused); the sparkDash instance used for these tables polls every 10 s since (`POLL_INTERVAL_GPU=10000`, `POLL_INTERVAL_BANDWIDTH=10000`).

## Measured 2026-09-23 (sparkDash, same method as the tables above)

After `DSV41_WO_A_W8`, `DSV41_DRAFT_TAU`, `DSV41_DRAFT_HEAD_FP8`, `DSV41_BLOCK_VERIFY`, `DSV41_FOLDED_FENCE`: **prose c1 65.3** (65.26 / 65.37 / 65.28, was 61.0), code c1 113.8 (113.3), structured c1 124.7 (124.1). sparkDash benches run greedy, where the draft temperature and block verification do not act; at the model card's T=1 / top_p=0.95 (6 technical prompts, thinking on, 12 × 800 tokens, c1) the same stack went 50.9 → 54.2 tok/s end to end. qeval over three runs (two on the production checkout, one from a fresh clone of this repository): 72, 71 and 73 of 75; primary (code + reasoning + math) 54, 53 and 54 of 55 (53 before these adapters); `json_escape` fails as it also does on earlier production images; `prose_p2` came out at 88 words against a 100-word minimum on the first two runs and passed on the third. Fresh-clone boot: sparkDash prose c1 65.42 / 65.47 / 65.34, code c1 107.1 / 115.9 / 117.0 / 111.0 (median 113.5; the code tier is the noisiest), structured c1 124.7, prose c4 32.3 per stream (122.9 aggregate).

## Quality gate, 2026-09-17

| Run (2026-09-17, same day, same fleet) | pass | broke | fixed | p |
|---|---:|---:|---:|---:|
| base image, same env (`dsv41-4x-spark:local`, chunk 4096, indexer backport) | 71/75 | – | – | – |
| **production** (branch + RoCEnante + prefill TP split) | **72/75** | 0 | 1 (`json_count`) | 1.000 |
| reference: upstream example, 2026-09-11 | 71/75 | | | |

The three tasks that fail on every stack (`code_interval_intersect`, `json_escape`, `math_m9`) fail identically on the upstream example. Raw results: [`results/quality-20260917/`](results/quality-20260917/). Run it from a worker, not from the head (it executes model-generated Python).

## Measurement notes for the dated rows

- `scripts/window-20260916.sh` is the runbook that produced the tables (preflight, build, two boots, benches, rollback).
- The 2026-09-23 production rows come from one boot right after a power cycle (`prodbench` as below; prose c1 is the median of 65.24 / 65.97 / 66.23). The draft temperature and block verification act only on sampled requests, so they are not visible in these greedy benches; the `wo_a` and draft-head kernels take 2.0 ms (~4 %) off the step, and the rest of the prose c1 difference is within the boot-to-boot spread (FlashInfer re-draws its MoE tactics on every boot under EP, sglang#40320).
- The 2026-09-18 production rows were re-measured on 2026-09-18 on the Engram-prefetch boot (`results/fastload-20260918/prodbench-prefetch-20260918.txt`): two warm-up prose c1 runs discarded, then one run per cell; prose c1 is the median of three runs (61.0, 61.2, 61.0); the 4k and 16k prefill cells are single cold points that swing 2.4–3.8k between boots.

## Tested and not adopted

### 2026-09-25

Measured on the fleet while building v2.2.

- **L2 prefetch v3/v4 sub-gates**: `AHEAD`, `ENGRAM`, `DRAFT` and `LMHEAD` flat; `SKIP_N` +0.17 ms/step; `WOB_MB=4` +0.12 ms/step; `MOE` slows the MoE stream it overlaps (~12.6 us per 10 MB) and its lines do not survive to the next layer. Only `WOA` shipped.
- **Fused MXFP8 quantization into q_norm / wo_a / hc**: flat.
- **Layer-14 Engram lookup on a side stream during verify**: ~0.
- **Fused hc prefill kernel**: slower than the current kernels.
- **MoE restructures**: closed; the MoE phases already run at 223 GB/s live, near the DRAM limit.
- **CPU pinning**: flat.

### 2026-09-23

All measured against the live engine or with an offline draft forward whose per-position acceptance matches the engine within 0.016.

- **Fine-tuning the DSpark draft on our own traffic** (~10 M captured tokens, markov + norms + hyper-connection mixers + attention, fp8 quantization-aware, exported in the checkpoint format and swapped in at load): +3.4 % accepted tokens on held-out requests, but 0 % on new prompts and on new coding problems. Agent sessions resend near-identical context, so a request-level split leaks; split by session or date.
- **Multi-pass drafting** (a second draft forward that sees the tokens drafted so far, trained for it): +2.4 % (2 passes) to +3.9 % (5) accepted tokens; each extra pass costs ~9 % of a step.
- **Expert-sharing routing** (a verify row's last two experts swapped to one another row already streams when the router scored it within 0.05): 20.5 → 18.4 experts per layer at bs 1, but the step went 52.9 → 56.6 ms from the extra ops, and it changes routing. Reverted.
- **n-gram lookup over the request's own output** on coding-with-thinking traffic: even an oracle choosing per step is +1 % (3.17 → 3.20 tokens/step); DSpark already covers the repeats.
- **Relaxed acceptance** (not exact): accepting any draft token inside the target's top-p is only +20 % accepted tokens; on free text the draft's deeper proposals are mostly wrong, not merely differently distributed.
- **Kernels**: split-K MXFP8 for the small-M dense projections was 1.5–3x slower than b12x (which already reads at ~200 GB/s); the hyper-connection mix kernel stays at 23–28 µs whatever the slicing or weight layout; the draft's LM head in fp8 would save ~0.5 ms with no acceptance loss (3.0882 → 3.0875), not done yet.
- Where a coding answer with thinking spends its steps: 99.4 % inside the thinking at 3.15 tokens/step, 0.6 % in the code at 5.6 tokens/step.

### 2026-09-18

- **`SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=0`** (Markov W2 replicated instead of TP-sharded, drops the 0.5 ms vocab all-gather): the bf16 vocab GEMM grows 3.04 → 4.31 ms/step, net step 49.7 → 50.3 ms, prose c1 unchanged. Kept sharded.
- **Fast-load knobs `DSV41_FAST_LOAD_INFLIGHT_GB=12` + `num_threads=2`**: load_weight 79 → 69 s but start-to-ready only 115 → 113 s. Defaults kept.
- **Parallel Engram misses in `row_store.cpp`** (probe first, pool for ≥2 misses): gather gap 2.44 → 2.38 ms, i.e. nothing; the side-stream prefetch above is what removed it.

### 2026-09-17

- **sglang#39704** (mHC/metadata overhead for medium batches) applied onto the pinned branch: every column within ±2 % of production on this fleet (its gain is at 32–64 concurrent requests on GB300). Kept out.
- **Newer `dsv4.1` heads (from 2026-09-16 22:15, #39671)** drop the torch candidate indexer and gate the DeepGEMM one on SM100; on SM121 DeepGEMM then rejects the 256-token KV pages (`block_kv == 64`). The pin stays at `f80c91a4b` until upstream has an SM12x candidate path again.
- `CHUNKED_PREFILL_SIZE=8192`, split threshold 16k, NVFP4 experts, `DSV41_CACHE_GIB` above 4, k≠5, NCCL channel/algorithm tuning: measured, no gain or worse.
