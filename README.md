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
- **rsync** (rchmagos), [#8](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/8): got RoCEnante running on a four-node switchless ring with sparkring's opposite-node paths and contributed the integration (`DSV41_ROCE_RING`, the `scripts/ring_mesh/` planner).
- **sumsliu**, [dgx-spark-deepseek-v41](https://github.com/sumsliu/dgx-spark-deepseek-v41): the eight-Spark measurement that moving to expert tensor parallelism removes most of the all-reduce wait.
- **Saolence**, [#1](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/1), [#2](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/2), [#5](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/5), [#6](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/6): the switchless ring as an opt-in switch, the worker NCCL and build fixes, and the first-deployer review in [#4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/4) behind the host prerequisites, the download step, the patched-NCCL section and the ring addressing notes.
- **MiaAI-Lab/sparkDash**, the benchmark used for every number below.

## Current results

Production stack v2.3 (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes, booted once with empty caches (healthy in 556 s; KV pool 6.80M tokens, 1M context), sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-29. Raw output: [`docs/results/validation-20260929-v23.txt`](docs/results/validation-20260929-v23.txt).

### sparkDash, GPU clock capped at 2200 MHz

All results here are at the 2200 MHz SM clock cap this fleet runs with (`nvidia-smi -lgc 0,2200`, re-applied by a timer), which keeps the nodes cooler; DeepSeek-V4.1 decode gains almost nothing from higher clocks. SM clock under load median 2184 MHz (p90 2190), hottest GPU 68 °C, no throttling samples.

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **89.7** | 125.2 (64.2) | 165.9 (41.9) | 244.6 (31.7) | 357.2 (23.2) |
| code | 131.9 | 181.4 (91.7) | 248.0 (63.5) | 328.6 (44.0) | 456.3 (30.5) |
| structured | 157.8 | 184.6 (106.9) | 242.1 (70.6) | 300.6 (44.6) | 603.4 (47.1) |
| json | 130.4 | 179.9 (92.8) | 319.8 (80.9) | 477.3 (61.0) | 693.2 (44.8) |

**Prefill, cold, tok/s**

| 4k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 4688 | 5583 | 5650 | 5664 | 5398 |

**How these were measured.** sparkDash DecodeBench after two discarded warm-up prose c1 runs; c1 is the median of three runs, c4 the median of two, c2 / c8 / c16 one run each (c1 runs: prose 89.7 / 89.0 / 90.2; code 131.5 / 131.9 / 132.1; structured 157.8 / 160.1 / 155.7; json 130.7 / 130.4 / 130.3). sparkDash's prose c1 is one prompt: greedy text is identical run to run on this stack, but a change in rounding anywhere (another stack, another clock) can follow a different greedy text there, so compare step time or many prompts across stacks. The non-prose types use a different prompt at each concurrency, so per-stream values are not comparable across columns. Prefill: sparkDash prefill bench, cold (salted prompts), after a 4k warm-up, median of the second and third of three runs per length. sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers ([MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text v2.2 measured 3915 / 4743 / 4822 / 4946 / 4800 tok/s at ~4k / ~15k / ~30k / ~56k / ~123k tokens, and v2.3 does not change prefill.

**Quality and correctness gates** (same boot, against the v2.2 production tree booted the same afternoon as the reference):

- `scripts/qeval.py` (75 auto-scored tasks: code run against hidden asserts, JSON schema checks, numeric answers, format constraints; no LLM judge; c1, temperature 0, number extractor fixed in this release): 72 / 75 on v2.3; the v2.2 reference scored 72 / 72 / 72 in three runs. Both miss the same three tasks (`code_interval_intersect`, `json_escape`, `json_count`).
- Greedy-continuation KL against the reference (`scripts/ds_gate.py kld`, 256 greedy tokens after cold prompts of 16.5k-40k tokens and of 1-3k tokens, KL up to and including the first divergent token): mean 0.0031 over 178 positions (long) and 0.0071 over 190 positions (short), PASS against a 0.035 limit and at the A/A floor: the same candidate collected twice in one boot gives 0.0046 / 0.0035, because cold prefills of these prompts are not bit-reproducible run to run and the greedy texts part within a few tokens either way. Prompt-token logprobs are not available on this launch (see Known limits), so the panel scores the continuation.
- T > 0 scan (`ds_gate.py tscan`, 35 outputs at T 1.0 / 0.6 and mixed c4 batches, thinking on): 0 garbled outputs, 0 errors (reference: 0 of 35).
- Prefix cache (`scripts/prefix_scan.py`, ~100k-token shared prefix, c8 cold then warm at T 0.8, plus a greedy drift test with 4 full and 4 mid-document cache hits): PASS; cold-vs-warm logprob drift 0.0 over 128 greedy tokens on all eight hits, 0 degenerate outputs.
- The certified head is exact by construction and was checked on the device: its exact kernel matches the stock matmul bit for bit on every logit of all four rank shards, and check mode counted 0 mismatches over 3,355 live verify steps; armed on all four ranks in this boot. Prompt-logprob requests answered HTTP 400 and the server kept serving.

Every earlier measurement, the per-stage tables and the experiments that were tried and not adopted are in [docs/history.md](docs/history.md); what changed when is in [CHANGELOG.md](CHANGELOG.md).

## What runs in production

| Layer | Setting | What it does |
|---|---|---|
| Image | `Dockerfile.canary-roce` | upstream SGLang `dsv4.1` branch at `f80c91a4b` + rhys101's RoCEnante overlay + all adapters |
| Slots | `MAX_RUNNING_REQUESTS=16` | adds the c16 tier |
| Experts | `EP_SIZE=1`, `DSV41_MOE_B12X_NEXT=1`, `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1` | every rank holds a quarter of all 384 experts and the routed MoE runs on b12x main: no expert-group straggler at the MoE all-reduce (all-reduce 5.9 → ~2.6 ms per step), MoE time unchanged. The deterministic reduction (per-slot buffer and a fixed-order sum instead of atomics) costs nothing measurable and makes greedy output bit-identical run to run; its <= 8-row plans use a Triton route planner and zero the barrier words in one fill (bit-identical to b12x's internal planner) |
| Engram | `DSV41_CACHE_GIB=4`, `DSV41_CACHE_WAYS=16`, `DSV41_ENGRAM_PREFETCH=1` | row cache (67–76 % hits) and row lookups on a side stream, rows bit-identical |
| Draft | `DSPARK_BLOCK_SIZE=5`, `SGLANG_DSPARK_FOLDED_SAMPLING=2` | k=5 wins on code, ties on prose; sampled requests stay in the CUDA graph |
| Draft sampling | `DSV41_DRAFT_TAU=0.7`, `DSV41_BLOCK_VERIFY=1` | sharper draft proposals and block verification for sampled rows, both exact in distribution |
| Verify length | `DSV41_VERIFY_CAP=conf:0.1`, `DSV41_ROUTER_LIVE=1` | verifies only the drafts the confidence head expects to survive; the other verify rows reuse the anchor row's experts inside the router kernel, so they read no extra weights; greedy outputs identical |
| Kernels | `DSV41_SHARED_PAD_K=1`, `DSV41_WO_A_W8=1`, `DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1`, `DSV41_DRAFT_HEAD_FP8=1` | shared expert kept on the b12x kernel; `wo_a` and the draft LM head read exact fp8 twins of their weights |
| Replicated linears | `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv`, `DSV41_DRAFT_MAIN_PROJ_SPLIT=1`, `DSV41_SPLIT_COMPACT_GATHER=1` | layers every rank computed in full are split by columns across ranks and all-gathered; enabled only where bit-identical on all ranks. A rank with a narrower slice runs its GEMM on a window of the widest slice's width, so no pad precedes the gather, and with the compact gather each rank sends only its own columns, which RoCEnante writes in place (no pad, reorder or cat; checked byte for byte against the old path per layer and row count at boot) |
| Transport | `SGLANG_ROCE_ALLREDUCE=1`, `SGLANG_ROCE_MAX_SIZE=2097152`, `DSV41_ROCE_GATHER=2097152`, `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0` | TP all-reduces and all-gathers up to 2 MiB over the one-shot RDMA kernel on both rails |
| Prefill | `CHUNKED_PREFILL_SIZE=4096`, `DSV41_INDEXER_CHUNKED=1`, `SPARK_PREFILL_TP_SPLIT=1` | bounded indexer transient (sglang#39187) plus the SG18 row split across ranks from 32k context |
| Loading | `DSV41_FAST_LOAD=1`, `DSV41_FAST_LOAD_TP_SLICE=auto`, `DSV41_AUTOTUNE_KEEP=1` | engine start 343 s → ~120 s; at EP1 each rank reads only its slice of every expert, into 256 MiB pinned slabs (330 allocations instead of ~94k, which gave back ~0.7M tokens of KV pool); MoE autotune cache kept across boots |
| Prefill hc | `DSV41_HC_FUSED=1` | the hyper-connection mix statistics of prefill chunks in one pass over K instead of 80 partial slices: ~1.46 → ~0.83 ms per call at 4096 rows, bit-identical to the stock kernels (checked on the first live call) |
| Prefill sequence parallel | `DSV41_PREFILL_SP=1` | at prefill chunks of >= 2048 rows the per-layer all-reduces become reduce-scatter + all-gather, and the per-row work between them (hyper-connection mixing and norms, the Engram `wkv` projection and gate) runs on each rank's quarter of the rows instead of all four ranks repeating it: prefill ~+20 % from 16k to 262k. Decode is untouched. The partial sums are added in a different order than the all-reduce's (the same kind of change as RoCEnante); `DSV41_PREFILL_SP_EXACT=1` keeps the all-reduce and is bit-identical to the unsharded path (checked op by op on the fleet) for ~+10 %. `DSV41_PREFILL_SP_FP8=1` gathers the attention input of 17 of the 21 full-row layers as the MXFP8 bytes and scales `wqkv_a` would compute itself (checked bit-identical per chunk size on every rank): about half the gather bytes there, +1.5-3 % prefill |
| L2 prefetch | `DSV41_L2_PREFETCH=1`, `DSV41_L2_PREFETCH_WOA=1` | during each RoCE all-reduce and all-gather of a decode step (SMs and DRAM mostly idle for 15-30 us) a side-stream kernel prefetches the first 6 MB of the weights that follow into L2; plans are learned from the warm-up forwards and captured in the CUDA graphs. Data is untouched (outputs bit-identical); decode step -1 ms at c1, +3 % on varied prompts, flat at c16. `WOA` adds a window after `wq_b` that prefetches `wo_a` while the attention core runs: -0.4 ms/step |
| Decode-step sync | `DSV41_SPEC_SYNC_FREE=all` | drops the per-step rank-0 broadcasts whose values are already identical on every rank (the draft token of each Markov step, the verify epilogue's three, verify_cap's live length); the draft noise comes from a counter-based stream every rank computes alike. Audit mode counted 0 rank mismatches in ~60k checks; greedy output byte-identical; -0.53 to -0.59 ms/step in an in-boot A/B (~-0.2 ms across reboots) |
| Decode-step glue | `DSV41_EAGER_GLUE=all` | fewer eager kernels between the draft and verify CUDA graphs (the folded fence's six copies as one, sampling-param staging skipped while unchanged, verify_cap's update inside the draft graph as one kernel); bit-identical, ~-0.1 to -0.2 ms/step |
| Target LM head | `DSV41_CERT_HEAD=1`, `DSV41_CERT_HEAD_M=6,...,48` | greedy verify steps read an 8-bit (MXINT8) screen of the head shard (173 MB instead of 331 MB per rank) with a proven error bound, and compute exactly only the 16-row head tiles that can hold the argmax; the argmax is the stock one (exact kernel bit-identical to the stock matmul, 0 mismatches over 3,355 checked steps), and sampled rows, logprobs, grammar and penalties get the full exact logits. -0.5 ms per c1 step; slower from 10 requests up, so it stops at 8 (M = 48 verify rows); +173 MB per rank |
| Correctness | `DSV41_FOLDED_FENCE=1` | closes the sglang#40919 D2H race on the folded verify path |
| Request guard | `DSV41_REPLAY_GUARD` (default on) | the launch runs SGLang's decoder SWA bounded replay (late layers see only the last window of a prefill), which cannot return prompt-token logprobs; such a request (`/generate` with `logprob_start_len` below the prompt length, `/v1/completions` with `echo` + `logprobs`) used to raise inside the forward and end every rank. It now gets HTTP 400 and the server keeps serving |
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

## Host prerequisites

`./start-tp4.sh doctor` checks only that `docker` and `nvidia-smi` exist; the rest of this table is on you.

| | |
|---|---|
| Hardware | Four DGX Sparks (GB10, aarch64, ~121 GiB unified memory each) with ConnectX-7: switched RoCE for the production profile, or four DACs in a ring for the optional one |
| Driver / runtime | NVIDIA driver 580.x with `nvidia-smi` on every node, plus **nvidia-container-toolkit** (every `docker run` uses `--gpus all`) |
| Docker | Engine on every node, usable without `sudo` (including `docker volume`); containers run with `--privileged`, `--ipc host`, `--shm-size` and `--ulimit memlock=-1` |
| Host CLI (head) | `python3` with `pexpect` (`scripts/remote.py`), `ssh` to every worker with the key at `SSH_IDENTITY` (and its `.pub` next to it), `rsync` (build pushes the tree to the workers), `curl`, `tar`, `base64`, and `rpcinfo` (`nfs-common`) while `NFS_SHARE=1` |
| Weights | `deepseek-ai/DeepSeek-V4.1-Flash` at the pinned revision: 476 GiB, 48 shards. `./start-tp4.sh download` fetches it on the head with the `hf` CLI (`pip install -U "huggingface_hub[cli]"`); the switched profile shares the head's copy over NFS, the ring profile keeps a copy on every node |
| Disk | head: 476 GiB for the checkpoint; every node: ~48 GiB of packed Engram rows (`pack`, TP4) and ~49 GB for the image |
| Ports | 8888 (API), 20000 (torch distributed init), and 2049/111 (NFS, only with `NFS_SHARE=1`) between the nodes |
| Credentials | none for the weights. An empty `API_KEY` (or `none`, `off`, `dummy`, `0`) leaves the endpoint open, so keep the port private when it is unset |

With `NFS_REUSE_EXPORT=1` the checkpoint is published into the export with hardlinks (`files/nfs-share.sh`), so `MODEL_DIR` and the export root must be on the same filesystem.

## Quick start

Same launcher as upstream; read [`docs/README-upstream.md`](docs/README-upstream.md) first for the fleet setup (NFS share, Engram packing, fabric). `pack` needs the checkpoint, so download it before the first `pack`.

```bash
cp .env.tp4.example .env.tp4          # fill in HEAD_IP / WORKER_* / MODEL_DIR / fabric
# production: uncomment IMAGE=dsv41-4x-spark:canary-roce and the last EXTRA_CONTAINER_ENV line,
# set BUILD_DOCKERFILE=Dockerfile.canary-roce and EP_SIZE=1 (the file default EP_SIZE=2 is for the
# base/canary images; the production line was measured at EP_SIZE=1)
./start-tp4.sh download               # 476 GiB at the pinned revision, needs the hf CLI; skips when complete
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
DSV41_L2_PREFETCH: model forward bracketed (decode/verify only); v3: wo_a windows (6.0 MB)
[spec_sync_free] armed (all): skip draft=True accept=True vcap=True ...
[eager_glue] vcap: verify_cap live length captured into the draft graph ...
[replicated_split] model.layers.0.self_attn.wqkv_a: compact gather (roce) bit-identical, ON ...
```

The simpler images (`Dockerfile` on the stock base, `Dockerfile.canary` without RoCEnante), each layer on its own, and the **switchless ring** for fleets without a RoCE switch are in [docs/optional-setups.md](docs/optional-setups.md). What the image is pinned to and what would let it move: [docs/upstream-watch.md](docs/upstream-watch.md).

## Independent deployment profiles

[R5 wired switchless-ring deployment and complete parameters](docs/r5-wired-profile.md). This optional contributor profile does not replace the upstream production profile above.

### sparkDash results: contributor R5 vs the author's published v2.3

The following uses the author's table layout. **These are different fleets and sampling schedules: percentages are observed differences from published values, not an isolated optimization gain or a formal benchmark ranking.** Negative differences are retained as well as positive ones. R5 data are the archived initial run already included in this PR; no new measurement is implied.

Author source: [README at 58f23215](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/58f232155917d388b6053eca079c617fd306c33c/README.md#current-results), measured 2026-09-29. Contributor source: [sanitized R5 result JSON](docs/results/r5-wired-20261007.json), measured 2026-10-03. Difference = (R5 / author − 1) × 100, calculated from the displayed values.

**Contributor R5 Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 84.44 | 119.90 (61.10) | 156.30 (40.58) | 224.74 (29.53) | 347.41 (22.98) |
| code | 129.86 | 182.57 (92.21) | 238.28 (62.85) | 323.61 (43.34) | 453.12 (30.19) |
| structured | 156.14 | 174.54 (89.39) | 267.76 (74.12) | 257.64 (44.93) | 537.83 (43.17) |
| json | 135.26 | 188.32 (96.68) | 295.06 (74.49) | 440.73 (56.85) | 673.80 (44.15) |

**Author Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 89.7 | 125.2 (64.2) | 165.9 (41.9) | 244.6 (31.7) | 357.2 (23.2) |
| code | 131.9 | 181.4 (91.7) | 248.0 (63.5) | 328.6 (44.0) | 456.3 (30.5) |
| structured | 157.8 | 184.6 (106.9) | 242.1 (70.6) | 300.6 (44.6) | 603.4 (47.1) |
| json | 130.4 | 179.9 (92.8) | 319.8 (80.9) | 477.3 (61.0) | 693.2 (44.8) |

**Aggregate Decode: R5 vs author, observed difference**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | -5.86% | -4.23% | -5.79% | -8.12% | -2.74% |
| code | -1.55% | +0.64% | -3.92% | -1.52% | -0.70% |
| structured | -1.05% | -5.45% | +10.60% | -14.29% | -10.87% |
| json | +3.73% | +4.68% | -7.74% | -7.66% | -2.80% |

**Per-stream Decode: R5 vs author, observed difference**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | -5.86% | -4.83% | -3.15% | -6.85% | -0.95% |
| code | -1.55% | +0.56% | -1.02% | -1.50% | -1.02% |
| structured | -1.05% | -16.38% | +4.99% | +0.74% | -8.34% |
| json | +3.73% | +4.18% | -7.92% | -6.80% | -1.45% |

**Prefill, salted/cold-prefix repeated-token prompts, tok/s**

| fleet | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| Author, switched | 4688.00 | 5583.00 | 5650.00 | 5664.00 | 5398.00 | Not published in this table |
| Contributor R5, ring | 3142.73 | 5287.15 | 5694.98 | 5888.80 | 5462.13 | 5112.71 |
| R5 vs author, observed difference | -32.96% | -5.30% | +0.80% | +3.97% | +1.19% | N/A |

“Cold” refers to salted prefix prompts, not an empty Engram row cache. The repeated-token filler can favor Engram caching on both fleets. The 4k result is displayed for completeness and should not determine the overall conclusion; 262k has no matching entry in the author's current table. See the JSON for actual prompt tokens and TTFT. A positive aggregate difference does not necessarily mean a positive per-stream difference.

**Configuration and measurement differences**

| Item | Author's published v2.3 | Contributor R5 |
|---|---|---|
| Hardware / parallelism | Four DGX Spark GB10, TP4/EP1 | Four DGX Spark GB10, TP4/EP1 |
| Source line | v2.3 at 58f23215 | Same upstream source ref, local wired capacity profile |
| Fabric | Switched RoCE | Four-cable switchless ring, hardware-forwarded diagonal paths |
| Prefill chunk | 4096 | 8192 |
| RoCEnante all-reduce/gather cap | 2097152 bytes | 262144 bytes |
| Context / token pool | 1M context; reported boot pool 6.80M | 1048576 context; configured max-total-tokens 6500000 |
| GPU clock | Explicit 2200-MHz SM cap | Equivalence to that cap not established in the exported measurement record |
| Benchmark / generation | sparkDash1.8.8, max256 tokens, temperature0, thinking off | Same core benchmark/version and generation settings |
| Decode discarded warmup | Two prose c1 runs before sweep | Manager-internal32-token warmup each job; two extra discarded jobs at each type's c2/c8 |
| Decode repetitions | Each c1 median of3, c4 median of2; c2/c8/c16 once | Prose c1 median of5, code c1 median of3, remaining cells once |
| Prefill warmup / samples | 4k warmup; median of runs2 and3 of three runs per length | Manager512-token warmup; one six-length sweep |
| Native initialization | Not specified in the published comparison protocol | WARMUP=1, head quick smoke, workers skip smoke |
| Monitoring | README recommends10-second GPU/bandwidth polling | sparkDash remained running; polling equivalence not established here |

## Measuring

- Bench with sparkDash's decode and prefill benches, never with a hand-rolled loop; discard the first two runs after a boot (cold Engram row cache, first-run warm-up).
- Any `#running-req` in the engine log above the concurrency being benched means foreign traffic landed in the window.
- A dashboard polling `nvidia-smi` every 2 s on every node costs ~0.6 ms per decode step; poll every 10 s (`POLL_INTERVAL_GPU=10000`, `POLL_INTERVAL_BANDWIDTH=10000`).
- sparkDash's fixed prompts are greedy: a change that alters rounding can flip a near-tie early in the 256 tokens and move c1 by ±10 % without changing speed. Kernel A/Bs here use step time on several prompts as well.
- Release gate tools: `scripts/qeval.py` (75 tasks), `scripts/ds_gate.py kld` (greedy-continuation KL against a reference stack, long and short panels) and `ds_gate.py tscan` (T > 0 garble scan), `scripts/prefix_scan.py` (shared long prefix at c8, cold vs warm, plus cold-vs-warm logprob drift on full and mid-document cache hits).
- Greedy text equality is not a correctness gate for prefill changes: identical cold prompts of ~70k tokens produce different greedy continuations run to run. Correctness of the backports rests on the bitwise CPU tests, the needle tests and qeval.

## Rollback

Every layer is an env change: `EP_SIZE=2` without `DSV41_MOE_B12X_NEXT` returns the routed MoE to FlashInfer, `DSV41_FAST_LOAD=0` restores the stock loader, `SGLANG_ROCE_ALLREDUCE=0` drops the RDMA transport, `SPARK_PREFILL_TP_SPLIT=0` the row split, `IMAGE=dsv41-4x-spark:canary` the RoCEnante overlay, `IMAGE=dsv41-4x-spark:local` the branch; unsetting `DSV41_SPEC_SYNC_FREE`, `DSV41_EAGER_GLUE`, `DSV41_SPLIT_COMPACT_GATHER` and `DSV41_L2_PREFETCH_WOA` turns off the v2.2 decode-step changes, unsetting `DSV41_CERT_HEAD` the v2.3 certified head (`DSV41_REPLICATED_SPLIT_WINDOW=0` and `DSV41_MOE_B12X_NEXT_DET_TRITON=0` the two that are on by default). Upstream's profile: `MAX_RUNNING_REQUESTS=8`, `CHUNKED_PREFILL_SIZE=1024`, `EXTRA_CONTAINER_ENV=""` (and `EP_SIZE=4`, `DSV41_CACHE_GIB=0` for the exact upstream example). The adapters stay in the image but do nothing when their gate is unset.

## Known limits

- The numbers above are on a switched RoCE fabric. On a switchless ring RoCEnante needs a path to the opposite node, built in the neighbours' ConnectX-7 hardware ([docs/switchless-ring.md](docs/switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths)); without it the collectives go through NCCL and decode is slower. Prefill is lower on a ring either way (one-link bisection).
- The EP1 production line needs the `Dockerfile.canary-roce` image (it carries `runtime/b12x_next`). The routed MoE on b12x is numerically equivalent to FlashInfer's, not bit-identical (relative error against an fp32 reference 4.5–4.8 % for both); qeval and the needle tests are unchanged.
- Prompt-token logprobs and all-token hidden states are not available: `boot.py` launches with `--enable-decoder-swa-bounded-replay` (Mia's recipe), so the 19 layers after the last KV-source layer run only over the last 128 rows of each prefill, and such requests get HTTP 400. Output logprobs work. Cache hits are exact: a hit that lands on a prefill chunk boundary gives bit-identical logprobs to a cold request; a hit elsewhere computes the tail with a different batch shape, which moves low-probability logprobs by the same amount as the run-to-run nondeterminism of cold prefills on some prompts (`scripts/prefix_scan.py`, `docs/results/validation-20260929-v23.txt`).
- Single-stream prose speed is bounded by DSpark acceptance (~3 accepted tokens per step on prose against ~6 on code); no configuration changes that.
- The fast loader leaves the KV pool 3–13 % smaller than the stock loader (6.7–7.3 M vs 7.5–7.8 M tokens) and more variable between boots; `DSV41_FAST_LOAD=0` restores it at ~220 s per boot ([docs/fast-load.md](docs/fast-load.md)).
- RoCEnante is a new transport in the decode path; b12x has one open report of a rank wedging under long mixed-context traffic on an earlier revision ([b12x#313](https://github.com/local-inference-lab/b12x/issues/313)). The overlay's result-boundary health check fails the step instead of hanging, and NCCL is one env change away.

## License

Same as upstream: see [`LICENSE`](LICENSE). Model weights are MIT (DeepSeek).
