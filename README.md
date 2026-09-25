<h1 align="center">DeepSeek-V4.1-Flash on 4x DGX Spark — tuned TP4 profile</h1>

<p align="center">A measured TP4 serving profile for <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">deepseek-ai/DeepSeek-V4.1-Flash</a> (native weights, SGLang, DSpark) on four NVIDIA DGX Spark (GB10) nodes, built on top of the MiaAI-Lab recipe.</p>

## Credits

This repository is a downstream profile of **[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)** by Mia (MiaAI-Lab). The launcher (`start.sh`, `start-tp4.sh`, `boot.py`), the Engram NVMe row store, the MXFP8 b12x routing, the memory model in `docs/chunked-prefill-memory.md`, the thinking alias, the output cap, the loop abort and the overall deployment design are her work, kept here with full git history and under the same licence. The original README is preserved as [`docs/README-upstream.md`](docs/README-upstream.md); read it first for the fleet setup (NFS share, Engram packing, fabric, 3-node profile).

Other work this profile builds on:

- **kpham-sgl**, [sgl-project/sglang#39187](https://github.com/sgl-project/sglang/pull/39187): the bounded dense-indexer prefill transient, backported here as `adapter/indexer_chunked*.py`.
- **BBuf** and the SGLang `dsv4.1` branch contributors ([#39370](https://github.com/sgl-project/sglang/pull/39370), [#39646](https://github.com/sgl-project/sglang/pull/39646), [#39648](https://github.com/sgl-project/sglang/pull/39648), [#39653](https://github.com/sgl-project/sglang/pull/39653)): the decode kernel work in the optional `Dockerfile.canary` image.
- **hushengkai**, for independently reproducing the EP2 / Engram cache / shared-expert padding changes on a second 4x GB10 fleet.
- **rhys101**, [DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8): the SG17 SGLang overlay that routes small tensor-parallel all-reduces to RoCEnante (reused with a TP4 adaptation in `Dockerfile.canary-roce`) and the SG18 native prefill TP split (`adapter/spark_prefill_dense.py`, combined with the indexer backport in `adapter/indexer_chunked_v3.py`).
- **local-inference-lab / Luke Alonso and Jason (original-el8)**, [b12x](https://github.com/local-inference-lab/b12x): RoCEnante, the one-shot RDMA all-reduce (`runtime/b12x`, Apache-2.0, frozen at the SG17 revision), and the fused MoE kernels that run the routed experts (`runtime/b12x_next`: b12x main at `a7d7d29b`, renamed so both revisions live in one image, with a two-line patch that admits 64-row tiles for 576-wide experts at prefill sizes).
- **luxingcom (LuZ)**, [LuZ DGX Spark TP4 ring](https://github.com/luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring): the first integration of b12x's fused MoE into SGLang on a four-Spark fleet, which showed the route.
- **FujitsuPolycom and the sparkring contributors**, [sparkring](https://github.com/FujitsuPolycom/sparkring): the hardware-forwarded opposite-node paths and the path-aware RoCEnante (`runtime/b12x/b12x/comm/roce_ring`) that run the production line on a switchless ring.
- **sumsliu**, [dgx-spark-deepseek-v41](https://github.com/sumsliu/dgx-spark-deepseek-v41): the eight-Spark measurement that moving to expert tensor parallelism removes most of the all-reduce wait.
- **MiaAI-Lab/sparkDash**, the benchmark used for every number below.

## Current results

Production stack (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes and measured on that image with sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-25. Engine start to healthy 160-170 s; KV pool 6.0-6.6M tokens (1M context). Raw output: [`docs/results/validation-20260925-v2.txt`](docs/results/validation-20260925-v2.txt).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **87.7** | 120.4 (61.9) | 163.6 (41.4) | 237.6 (30.8) | 342.7 (22.2) |
| code | 124.8 | 175.0 (88.4) | 246.8 (63.2) | 309.8 (41.4) | 438.3 (29.3) |
| structured | 152.4 | 177.6 (103.3) | 240.4 (70.4) | 295.5 (44.2) | 572.2 (44.4) |
| json | 118.9 | 174.2 (89.8) | 301.7 (76.2) | 471.5 (60.2) | 659.9 (42.7) |

Prose and code c1 are from the latest fresh-clone boot (v2.1, prose median of five runs 86.7-87.7 after two discarded warm-ups; the v2 boot gave 86.5 and 122.6); the other columns are from the v2 sweep, whose decode path is identical. With the deterministic MoE reduction the greedy text is identical run to run, so the sparkDash numbers repeat within about ±1 tok/s; sparkDash's prose c1 is one prompt, and a stack that sums in a different order (another fabric, another all-reduce) follows a different greedy text there, so compare step time or a many-prompt benchmark across stacks. On 45 varied prompts (prose, structured and other catalogs, c1 greedy) the same image runs 58.4 / 94.3 / 71.8 tok/s; decode step 32.4-33.6 ms on prose and 38.9-39.4 ms on code at c1. sparkDash uses a different set of prompts at each concurrency for the non-prose types, so per-stream values are not comparable across columns. Sampled chat at the model card's T=1 / top_p=0.95 with thinking (c1, 18 requests x 800 tokens on two prompt sets) runs 67.2 / 65.9 tok/s (measured on the 2026-09-24 stack; sparkDash benches are greedy, where the draft temperature and block verification do not act).

**Prefill, cold, tok/s by prompt length** (two passes; fresh clone of the MXFP8 attention-gather commit, [`docs/results/validation-20260925-v21.txt`](docs/results/validation-20260925-v21.txt), which also measured prose c1 87.7 (86.7-87.7), code c1 124.8, qeval 72/75 and the 1,011,084-token needle PASS in 325 s)

| 4k | 16k | 32k | 64k | 128k | 262k |
|---:|---:|---:|---:|---:|---:|
| 4119 / 4840 | 5891 / 5846 | 5893 / 5868 | 5936 / 5903 | 5793 / 5761 | 5355 / 5360 |

sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text (documentation and source code, a unique prefix per prompt so nothing comes from the prefix cache) the same image measured 4202-4685 / 4824-4922 / 4827-4830 / 4942-4970 / 4822-4871 tok/s at ~4k / ~15k / ~32k / ~62k / ~120k tokens. Needle retrieval (a single phrase in varied filler at 37 % depth): PASS at 99k, 198k, 746k and 1,011,084 tokens (the last one in 322 s, head `MemAvailable` low-water 7 GiB). A harder list-lookup needle ("value of item N" in a list of up to 197,000 items) passes for about half the keys at 129k and 259k on this stack and on the previous one alike, a limit of the model rather than of either stack.

**Quality.** `scripts/qeval.py` runs 75 auto-scored tasks (code executed against hidden asserts, JSON schema-checked, numeric answers matched, format constraints enforced, prose checked for degeneration; no LLM judge), one request at a time, temperature 0. The fresh-clone image scores 72 of 75 (the previous stack 71-72); the three misses (`code_interval_intersect`, `json_escape`, `math_m9`) fail identically on the upstream example. Every speed change here is meant to be lossless: same weights, every draft token verified by the target, and each adapter either bit-identical to the stock path (checked at boot or in the in-image tests) or exact in distribution (draft temperature, block verification). RoCEnante sums in a different order than NCCL, so the numerics are not bit-identical to an NCCL run, which is why the profile is also scored. Run qeval from a worker, not from the head (it executes model-generated Python).

Every earlier measurement, the per-stage tables and the experiments that were tried and not adopted are in [docs/history.md](docs/history.md); what changed when is in [CHANGELOG.md](CHANGELOG.md).

## What runs in production

| Layer | Setting | What it does |
|---|---|---|
| Image | `Dockerfile.canary-roce` | upstream SGLang `dsv4.1` branch at `f80c91a4b` + rhys101's RoCEnante overlay + all adapters |
| Slots | `MAX_RUNNING_REQUESTS=16` | adds the c16 tier |
| Experts | `EP_SIZE=1`, `DSV41_MOE_B12X_NEXT=1`, `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1` | every rank holds a quarter of all 384 experts and the routed MoE runs on b12x main: no expert-group straggler at the MoE all-reduce (all-reduce 5.9 → ~2.6 ms per step), MoE time unchanged. The deterministic reduction (per-slot buffer and a fixed-order sum instead of atomics) costs nothing measurable and makes greedy output bit-identical run to run |
| Engram | `DSV41_CACHE_GIB=4`, `DSV41_CACHE_WAYS=16`, `DSV41_ENGRAM_PREFETCH=1` | row cache (67–76 % hits) and row lookups on a side stream, rows bit-identical |
| Draft | `DSPARK_BLOCK_SIZE=5`, `SGLANG_DSPARK_FOLDED_SAMPLING=2` | k=5 wins on code, ties on prose; sampled requests stay in the CUDA graph |
| Draft sampling | `DSV41_DRAFT_TAU=0.7`, `DSV41_BLOCK_VERIFY=1` | sharper draft proposals and block verification for sampled rows, both exact in distribution |
| Verify length | `DSV41_VERIFY_CAP=conf:0.1`, `DSV41_ROUTER_LIVE=1` | verifies only the drafts the confidence head expects to survive; the other verify rows reuse the anchor row's experts inside the router kernel, so they read no extra weights; greedy outputs identical |
| Kernels | `DSV41_SHARED_PAD_K=1`, `DSV41_WO_A_W8=1`, `DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1`, `DSV41_DRAFT_HEAD_FP8=1` | shared expert kept on the b12x kernel; `wo_a` and the draft LM head read exact fp8 twins of their weights |
| Replicated linears | `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv`, `DSV41_DRAFT_MAIN_PROJ_SPLIT=1` | layers every rank computed in full are split by columns across ranks and all-gathered; enabled only where bit-identical on all ranks |
| Transport | `SGLANG_ROCE_ALLREDUCE=1`, `SGLANG_ROCE_MAX_SIZE=2097152`, `DSV41_ROCE_GATHER=2097152`, `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0` | TP all-reduces and all-gathers up to 2 MiB over the one-shot RDMA kernel on both rails |
| Prefill | `CHUNKED_PREFILL_SIZE=4096`, `DSV41_INDEXER_CHUNKED=1`, `SPARK_PREFILL_TP_SPLIT=1` | bounded indexer transient (sglang#39187) plus the SG18 row split across ranks from 32k context |
| Loading | `DSV41_FAST_LOAD=1`, `DSV41_FAST_LOAD_TP_SLICE=auto`, `DSV41_AUTOTUNE_KEEP=1` | engine start 343 s → ~120 s; at EP1 each rank reads only its slice of every expert, into 256 MiB pinned slabs (330 allocations instead of ~94k, which gave back ~0.7M tokens of KV pool); MoE autotune cache kept across boots |
| Prefill hc | `DSV41_HC_FUSED=1` | the hyper-connection mix statistics of prefill chunks in one pass over K instead of 80 partial slices: ~1.46 → ~0.83 ms per call at 4096 rows, bit-identical to the stock kernels (checked on the first live call) |
| Prefill sequence parallel | `DSV41_PREFILL_SP=1` | at prefill chunks of >= 2048 rows the per-layer all-reduces become reduce-scatter + all-gather, and the per-row work between them (hyper-connection mixing and norms, the Engram `wkv` projection and gate) runs on each rank's quarter of the rows instead of all four ranks repeating it: prefill ~+20 % from 16k to 262k. Decode is untouched. The partial sums are added in a different order than the all-reduce's (the same kind of change as RoCEnante); `DSV41_PREFILL_SP_EXACT=1` keeps the all-reduce and is bit-identical to the unsharded path (checked op by op on the fleet) for ~+10 %. `DSV41_PREFILL_SP_FP8=1` gathers the attention input of 17 of the 21 full-row layers as the MXFP8 bytes and scales `wqkv_a` would compute itself (checked bit-identical per chunk size on every rank): about half the gather bytes there, +1.5-3 % prefill |
| L2 prefetch | `DSV41_L2_PREFETCH=1` | during each RoCE all-reduce and all-gather of a decode step (SMs and DRAM mostly idle for 15-30 us) a side-stream kernel prefetches the first 6 MB of the weights that follow into L2; plans are learned from the warm-up forwards and captured in the CUDA graphs. Data is untouched (outputs bit-identical); decode step -1 ms at c1, +3 % on varied prompts, flat at c16 |
| Correctness | `DSV41_FOLDED_FENCE=1` | closes the sglang#40919 D2H race on the folded verify path |
| Serving | `--enable-cache-report`, `--sleep-on-idle`, `SGLANG_RUST_BUILD_MODE=never` | cached-token usage for clients, idle CPU 47 % → 14 %, avoids a `cargo` hang at start |
| Optional | `DSV41_ENGRAM_DRM_NODE=/dev/dri/card0` | one Engram layer's cache in the GB10 display reservation, ~1.8 GiB more headroom; needs a host change ([docs/display-reserve.md](docs/display-reserve.md)) |

Not used: `DSV41_ADAPTIVE_CHUNK` (superseded by the bounded indexer), `DSPARK_SPS_TABLE` (crashes the Engram path), the NVFP4 checkpoint (no bandwidth saved on GB10). Each adapter is described in [docs/adapters.md](docs/adapters.md); the measured effect of each switch is in [docs/history.md](docs/history.md#production-stack-as-of-2026-09-24-with-the-full-rationale-per-switch).

## What this profile changes

Relative to the upstream TP4 example, all of it in `.env.tp4.example` plus gated adapters:

| Setting | Upstream | Here | Why (measured) |
|---|---|---|---|
| `EP_SIZE` | 4 | **1** | On the production image (`DSV41_MOE_B12X_NEXT`): every rank streams a 576-wide slice of every touched expert, so no expert group waits for the other at the MoE all-reduce. The base and canary images stay at `EP_SIZE=2` (FlashInfer's MXFP4 MoE needs the per-rank width to be a multiple of 128), which already halves the EP4 straggler wait: NCCL time per step 16.5 → 10.2 ms |
| `DSV41_CACHE_GIB`/`WAYS` | 0/4 | **4/16** | Engram rows do repeat (bigram/trigram heads): 67–76 % hit rate, 4x fewer NVMe reads; 16 ways are free |
| `--min-free-slots-delay 1` | on | on | Without it the admission delayer never fills the last slot |
| `--enable-deepseek-v4-fp4-indexer` | off | on | no effect on V4.1: the model has no ratio-4 layers and its only indexer is fp4 by design; a four-boot A/B (off/on/off/on) gave identical greedy output and speed. Kept only so the argument line matches earlier runs |
| `DSPARK_BLOCK_SIZE` | 3 | **5** | k=3 is a 3-node prose result; on TP4 k=5 wins on code by ~10 % and ties on prose |
| `MAX_RUNNING_REQUESTS` | 8 | **16** | CUDA graphs to bs 16 cost ~5.6 GB and add a c16 tier (+64 % aggregate over c8) |
| `CHUNKED_PREFILL_SIZE` | 1024 | **4096** | Safe only together with the indexer backport below |
| `DSV41_SHARED_PAD_K=1` | – | **on** | Pads the shared expert's K 576 → 640 so it stops falling off the b12x MXFP8 kernel: −0.9 ms/step, bit-identical |
| `DSV41_INDEXER_CHUNKED=1` | – | **on** | sglang#39187: indexer logits scored in ≤ 2 GiB row chunks, tail-only candidate masks; 262k cold prefill keeps ≥ 7 GiB free on the head |
| prefill TP split (`SPARK_PREFILL_TP_SPLIT=1`, canary images) | – | **on** | SG18: the dense prefill indexer's query rows are partitioned across the four ranks from 32k context; each rank scores a quarter, the top-k and candidate block ids are all-gathered as ints. sparkDash prefill 128k 3499 → 4364, 262k 2701 → 3893 |
| `--enable-cache-report` | off | **on** | `usage.prompt_tokens_details.cached_tokens` on every response (also in streaming `usage`), so clients can see prefix-cache hits |

Everything else (memory fraction 0.80, 8M-token KV pin, 1M context, NFS/Engram layout, the OpenAI serving fixes) is upstream's.

## Quick start

Same launcher as upstream; read [`docs/README-upstream.md`](docs/README-upstream.md) first for the fleet setup (NFS share, Engram packing, fabric).

```bash
cp .env.tp4.example .env.tp4          # fill in HEAD_IP / WORKER_* / MODEL_DIR / fabric
# production: uncomment IMAGE=dsv41-4x-spark:canary-roce and the last EXTRA_CONTAINER_ENV line,
# set BUILD_DOCKERFILE=Dockerfile.canary-roce and EP_SIZE=1 (the file default EP_SIZE=2 is for the
# base/canary images; the production line was measured at EP_SIZE=1)
scripts/fetch-sglang-canary.sh        # stages the pinned dsv4.1 branch tree
./start-tp4.sh doctor
./start-tp4.sh build                  # builds on every node, bakes the adapters, runs the in-image tests
./start-tp4.sh share && ./start-tp4.sh pack   # first time only, see upstream README
./start-tp4.sh serve                  # ./start-tp4.sh stop | status | logs | smoke
```

The boot log must show these lines, otherwise the profile is not active:

```
DSV41 shared-expert padding K: ... (5120, 576) -> (5120, 640)
DSV41 indexer chunked (sglang#39187 backport) ARMED: ...
Initialized DSpark draft runner. ... gamma=5, verify_num_draft_tokens=6
max_total_num_tokens=..., chunked_prefill_size=4096, ... max_running_requests=16
RoCEnante ready: world=4 hcas=...
[moe_b12x_next] armed: routed MoE on b12x_next a7d7d29b ...
[moe_b12x_next] INFO: routed MoE at EP_SIZE=1 ...
DSV41 hc_fused: first call (4096 rows) bit-identical to stock
```

The simpler images (`Dockerfile` on the stock base, `Dockerfile.canary` without RoCEnante), each layer on its own, and the **switchless ring** for fleets without a RoCE switch are in [docs/optional-setups.md](docs/optional-setups.md). What the image is pinned to and what would let it move: [docs/upstream-watch.md](docs/upstream-watch.md).

## Measuring

- Bench with sparkDash's decode and prefill benches, never with a hand-rolled loop; discard the first two runs after a boot (cold Engram row cache, first-run warm-up).
- Any `#running-req` in the engine log above the concurrency being benched means foreign traffic landed in the window.
- A dashboard polling `nvidia-smi` every 2 s on every node costs ~0.6 ms per decode step; poll every 10 s (`POLL_INTERVAL_GPU=10000`, `POLL_INTERVAL_BANDWIDTH=10000`).
- sparkDash's fixed prompts are greedy: a change that alters rounding can flip a near-tie early in the 256 tokens and move c1 by ±10 % without changing speed. Kernel A/Bs here use step time on several prompts as well.
- Greedy text equality is not a correctness gate for prefill changes: identical cold prompts of ~70k tokens produce different greedy continuations run to run. Correctness of the backports rests on the bitwise CPU tests, the needle tests and qeval.

## Rollback

Every layer is an env change: `EP_SIZE=2` without `DSV41_MOE_B12X_NEXT` returns the routed MoE to FlashInfer, `DSV41_FAST_LOAD=0` restores the stock loader, `SGLANG_ROCE_ALLREDUCE=0` drops the RDMA transport, `SPARK_PREFILL_TP_SPLIT=0` the row split, `IMAGE=dsv41-4x-spark:canary` the RoCEnante overlay, `IMAGE=dsv41-4x-spark:local` the branch. Upstream's profile: `MAX_RUNNING_REQUESTS=8`, `CHUNKED_PREFILL_SIZE=1024`, `EXTRA_CONTAINER_ENV=""` (and `EP_SIZE=4`, `DSV41_CACHE_GIB=0` for the exact upstream example). The adapters stay in the image but do nothing when their gate is unset.

## Known limits

- The numbers above are on a switched RoCE fabric. On a switchless ring RoCEnante needs a path to the opposite node, built in the neighbours' ConnectX-7 hardware ([docs/switchless-ring.md](docs/switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths)); without it the collectives go through NCCL and decode is slower. Prefill is lower on a ring either way (one-link bisection).
- The EP1 production line needs the `Dockerfile.canary-roce` image (it carries `runtime/b12x_next`). The routed MoE on b12x is numerically equivalent to FlashInfer's, not bit-identical (relative error against an fp32 reference 4.5–4.8 % for both); qeval and the needle tests are unchanged.
- Single-stream prose speed is bounded by DSpark acceptance (~3 accepted tokens per step on prose against ~6 on code); no configuration changes that.
- The fast loader leaves the KV pool 3–13 % smaller than the stock loader (6.7–7.3 M vs 7.5–7.8 M tokens) and more variable between boots; `DSV41_FAST_LOAD=0` restores it at ~220 s per boot ([docs/fast-load.md](docs/fast-load.md)).
- RoCEnante is a new transport in the decode path; b12x has one open report of a rank wedging under long mixed-context traffic on an earlier revision ([b12x#313](https://github.com/local-inference-lab/b12x/issues/313)). The overlay's result-boundary health check fails the step instead of hanging, and NCCL is one env change away.

## License

Same as upstream: see [`LICENSE`](LICENSE). Model weights are MIT (DeepSeek).
