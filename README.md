<h1 align="center">DeepSeek-V4.1-Flash on 4x DGX Spark — tuned TP4 profile</h1>

<p align="center">A measured TP4 serving profile for <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">deepseek-ai/DeepSeek-V4.1-Flash</a> (native weights, SGLang, DSpark) on four NVIDIA DGX Spark (GB10) nodes, built on top of the MiaAI-Lab recipe.</p>

## Current results

v2.6.2 is v2.6 plus a routed-MoE race fix and a prefill memory fix (see [below](#new-in-v262)); the tables are v2.6's (v2.6.2 changes speed only at long prefill: cold 1M about 5 % faster). v2.6: image `Dockerfile.canary-roce` with this release's adapters and RoCE runtime (SGLang `dsv4.1` `f80c91a4b` + the credited backports, without #45251), the "Production profile" block of [`.env.tp4.example`](.env.tp4.example) with `EP_SIZE=1`, 1M context, checkpoint [`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) at `517efab`, GPU clock capped at 2200 MHz, measured 2026-10-26 on fresh boots with the host memory compacted first and the first run after each boot discarded. Speculative decoding makes tok/s depend on the generated text, so other DeepSeek releases give somewhat different numbers on the same prompts.

### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories (prompt set speedbench-v1), c1, temperature 0, 256 output tokens, thinking off, run on the head. Decode tok/s, geometric mean over the prompts of each category, median of two full runs on the release-gate boot:

| tok/s | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | **65.6** | 112.2 | 107.0 | 103.8 | 70.1 | 92.7 | **89.3** |

Runs: all 88.84 / 89.80, prose 65.04 / 66.14, code 111.57 / 112.84, structured 106.36 / 107.71, agentic 103.4 / 104.2, multilingual 69.51 / 70.70, long context 92.96 / 92.40. Median time to first token 206 / 208 ms; 73 of 73 prompts valid in both runs, none failed.

**Under load** (speedbench load: closed loop, 90 s per level, 256 output tokens, thinking off; one fresh boot of the same image and profile)

| concurrency | block tok/s | per stream, median tok/s | TTFT, median |
|---|---:|---:|---:|
| c1 | 77.3 | 83.3 | 207 ms |
| c4 | 131.7 | 38.7 | 310 ms |
| c16 | **211.0** | 16.0 | 539 ms |

Block tok/s is all completion tokens over the wall time of the level, including request start and prefill, so the c1 value sits below the decode geomean above.

**Prefill, cold, tok/s** (4k and 64k: speedbench longctx, unique salt per request, median of three documents per size, same boot as the load table; 388K: one cold varied-text prompt on the release-gate boot)

| 4k | 64k | 388K |
|---:|---:|---:|
| 5309 | 5970 | 5770 (65.2 s) |

The official checkpoint ([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) `1bae0e1`, the default weights profile in `.env.tp4.example`) was not re-measured on this release. On the v2.5 image on 2026-10-26, same speedbench protocol: c1 all 88.4 (prose 66.0, code 108.4, structured 111.0), c4 132.3, c16 200.0 block tok/s, cold prefill 4k 4,605 and 128k 5,783 tok/s. This release's switches leave c1 decode unchanged (T = 0 outputs byte-identical), so its c1 carries over; its c16 and short-prefill gains were measured on the abliterated checkpoint only. The abliterated edit touches only `attn.wo_b` and `shared_experts.w2` of the main layers (the DSpark draft layers are identical), so the two run within a few percent of each other.

**Correctness checks.** On the release-gate boot: needle retrieval PASS at 124k and 388k tokens (passphrase at 50 % depth; 19.8 s and 66.6 s); cold prefill of a 388K-token varied-text prompt 65.2 s (5,770 tok/s); `scripts/qeval.py` 71 / 75 (code 24/25, format 7/7, json 6/8, math 13/14, prose 5/5, reason 16/16), the same score and split as v2.5; T > 0 scan (`ds_gate.py tscan`, 35 outputs) 0 garbled, 0 errors; `/v1/messages` with the image placeholder token in text and with thinking enabled without a budget: both answered; the issue #16 accelerated reproducer (`scripts/issue16_repro.py`: 190K-900K cached prefixes with 512-4096 fresh tokens, alternating c1/c4) 410 requests in 15 minutes, 0 failures ([docs/issue-16-swa-crash.md](docs/issue-16-swa-crash.md)). Exactness against v2.5: 75 qeval prompts at T = 0 with top-20 logprobs and cold prefills, on four fresh boots (v2.5, two-shot only, the full v2.6 set, v2.5 again): every v2.6 output byte-identical to v2.5. Sampling at T > 0 (v2.5 profile, 12,000 samples): the second token's counts match the verify row's own probabilities (chi-square p 0.61).

### New in v2.6.2

**Routed-MoE claim race fixed.** In the routed-MoE dynamic kernel, the CTA leader stores each batch claim in one
shared-memory word, and every thread then reads it. Nothing made those reads finish before the leader stored the next
claim, so a warp that was late to its read could pick up the next batch's claim: tokens were skipped or routed twice,
or the loop stopped early.
- **The fix:** one CTA barrier after the read, backported from b12x `d7937d7f` (Martin Vit). It is always on in the
  image.
- **On the fleet:** T = 0 outputs equal to v2.6 on all 75 qeval prompts, and unchanged speed (c1 / c4 / c16 load at
  0.999 / 1.002 / 1.011 of the slower v2.6 boot).

**Prefill reads the extra cache's native pages** (`DSV41_PREFILL_NATIVE_EXTRA=1`, now in the production profile).
v2.6 kept a persistent 64-token copy of the whole compressed extra pool, about 3 GB per rank allocated after the KV
pool is sized, because FlashInfer's SM120 prefill was instantiated only for 64-token extra pages. A module generated at
image build from FlashInfer's own sources reads SGLang's 128 / 256-token pages in place: the kernel is generic in the
page size, so only the dispatch changes. It is compiled in the image build, sha256-pinned and refused on any source
drift.
- **On the fleet:** 2.45-2.76 GiB less memory allocated after pool sizing on every node; cold 1M prefill 194 s instead
  of 205 s (-5 %); T = 0 outputs equal to v2.6; c16 unchanged.

<!-- v2.6.2 release gate: filled in from the receipts before the PR. -->
Release gate (v2.6 vs v2.6.2 vs v2.6, fresh boots, production profile): PENDING.

**Fixed: logprob decode hang.** Images built from `57a9888` and earlier main commits hang every rank on a greedy native `/generate` with `return_logprob` on long code prompts. The cause was the SGLang #45251 backport, now removed; see the CHANGELOG.

Earlier measurements, including the v2.5 sparkDash tables (sparkDash is no longer used here), the v2.4 table on the original checkpoint, the per-stage tables and the experiments that were tried and not adopted, are in [docs/history.md](docs/history.md); what changed when, with the v2.6 A / B / A' table, is in [CHANGELOG.md](CHANGELOG.md).

## What the production profile runs

| Layer | Setting | What it does |
|---|---|---|
| Image | `Dockerfile.canary-roce` | upstream SGLang `dsv4.1` branch at `f80c91a4b` + credited #41303/#42901 backports + rhys101's RoCEnante overlay + all adapters |
| Slots | `MAX_RUNNING_REQUESTS=16` | adds the c16 tier |
| Experts | `EP_SIZE=1`, `DSV41_MOE_B12X_NEXT=1`, `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1` | every rank holds a quarter of all 384 experts and the routed MoE runs on b12x main: no expert-group straggler at the MoE all-reduce (all-reduce 5.9 → ~2.6 ms per step), MoE time unchanged. The deterministic reduction (per-slot buffer and a fixed-order sum instead of atomics) costs nothing measurable and makes greedy output bit-identical run to run; its <= 8-row plans use a Triton route planner and zero the barrier words in one fill (bit-identical to b12x's internal planner) |
| Engram | `DSV41_CACHE_GIB=4`, `DSV41_CACHE_WAYS=16`, `DSV41_ENGRAM_PREFETCH=1`, `DSV41_ENGRAM_KEEPALIVE_MS=25` (default) | row cache (67–76 % hits) and row lookups on a side stream, rows bit-identical. While lookups arrive, one 4 KiB read every 25 ms keeps the NVMe drive out of its power-saving states: at c16 the drive slept between steps and each step's row reads paid ~10 ms to wake it (c16 step 149 → 140 ms, +5-6 % c16 aggregate; c1 unchanged) |
| Draft | `DSPARK_BLOCK_SIZE=5`, `SGLANG_DSPARK_FOLDED_SAMPLING=2` | k=5 wins on code, ties on prose; sampled requests stay in the CUDA graph |
| Draft sampling | `DSV41_DRAFT_TAU=0.7`, `DSV41_BLOCK_VERIFY=1` | sharper draft proposals and block verification for sampled rows, both exact in distribution. Since v2.6 the acceptance threshold forms `1 - w` before adding `S`, so FP32 no longer rounds `S` against 1 (sampled rows only; greedy unaffected) |
| Verify length | `DSV41_VERIFY_CAP=conf:0.1`, `DSV41_ROUTER_LIVE=1` | verifies only the drafts the confidence head expects to survive; the other verify rows reuse the anchor row's experts inside the router kernel, so they read no extra weights; greedy outputs identical |
| Kernels | `DSV41_SHARED_PAD_K=1`, `DSV41_WO_A_W8=1`, `DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1`, `DSV41_DRAFT_HEAD_FP8=1` | shared expert kept on the b12x kernel; `wo_a` and the draft LM head read exact fp8 twins of their weights |
| Replicated linears | `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv`, `DSV41_DRAFT_MAIN_PROJ_SPLIT=1`, `DSV41_SPLIT_COMPACT_GATHER=1` | layers every rank computed in full are split by columns across ranks and all-gathered; enabled only where bit-identical on all ranks. A rank with a narrower slice runs its GEMM on a window of the widest slice's width, so no pad precedes the gather, and with the compact gather each rank sends only its own columns, which RoCEnante writes in place (no pad, reorder or cat; checked byte for byte against the old path per layer and row count at boot) |
| Transport | `SGLANG_ROCE_ALLREDUCE=1`, `SGLANG_ROCE_MAX_SIZE=2097152`, `DSV41_ROCE_GATHER=2097152`, `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0` | TP all-reduces and all-gathers up to 2 MiB over the one-shot RDMA kernel on both rails |
| Two-shot all-reduce | `DSV41_ROCE_TWOSHOT=1`, `DSV41_ROCE_TWOSHOT_ROWS=18`, `B12X_ROCE_PIPE_BLOCKS=4`, `B12X_ROCE_PIPE_CHUNK_BYTES=65536` | all-reduces of 18 rows or more (the wide verify and draft steps at higher concurrency) take a pipelined two-shot path: each rank sends its quarter to the row owner, which sums the four partials in FP32 in rank order and rounds once, as the one-shot does, then the reduced rows are gathered back. Half the bytes on the wire, bit-identical output; smaller all-reduces stay one-shot. c16 +3.7 % alone (+4.9 % with the two prefill switches below), c1 / c4 unchanged |
| Prefill | `CHUNKED_PREFILL_SIZE=8192`, `MEM_FRACTION_STATIC=HEAD_MEM_FRACTION_STATIC=0.78`, `DSV41_INDEXER_CHUNKED=1`, `SPARK_PREFILL_TP_SPLIT=1` | bounded indexer transient (sglang#39187) plus the SG18 row split across ranks from 32k context; 8192-token chunks are 17-28 % faster on 400K-1M cold prompts, and the 0.78 fraction gives back the ~3 GiB the larger chunk costs at 1M (at 0.80 the head dropped to 2.7 GiB available) |
| Prefill attention pages | `DSV41_PREFILL_NATIVE_EXTRA=1` (since v2.6.2) | SM120 sparse-MLA prefill reads the extra (compressed) cache in SGLang's own 128 / 256-token pages through a module generated from FlashInfer's sources at image build (approved source closure, sha256-pinned .so, no JIT at boot) instead of a persistent 64-token copy of the whole pool: 2.45-2.76 GiB less allocated after pool sizing per node, cold 1M prefill -5 %, bit-identical |
| Prefill MoE launch | `DSV41_MOE_LAUNCH_PREPARED=plan` | eager (prefill) b12x MoE launches reuse a launch plan prepared once per launch contract instead of redoing the host-side quant, routing, tile and kernel-lookup decisions on every call; barrier fill, route planner, pointers and stream stay per call, and the plan is dropped on any `B12X_NEXT_*` environment change or kernel-cache change. Same kernels and arguments, bit-identical |
| Prefill indexer | `DSV41_INDEXER_NOSYNC=1` | the eager prefill indexer of the eight index-source layers builds its compressed-slot ids with one device gather and pinned non-blocking copies instead of a per-request `.item()` and a pageable copy (two host syncs per request per layer). Integer ops in the same order, bit-identical. With the MoE launch plan: cold 4k prefill +16 %, 64k +3.4 %, TTFT -4 % (c1 214 → 206 ms, c16 563 → 539 ms) |
| Multi-user | `--prefill-decode-interval 2`, `DSV41_PREFILL_BYPASS=256` | a short request that arrives during a long cold prefill is admitted next to the in-flight chunk and decoded between chunks: answered in 4.6 s instead of after the prefill (80 s at 400K); the admission adapter only acts after the engine's own memory check and reads only rank-identical state |
| Prefix cache | `--swa-prefix-tails 128` | room for 128 sliding-window prefix tails instead of 64: a long session stays cached when another long session runs (388K revisit 1.5 s instead of an 89 s re-prefill); costs KV pool |
| Loading | `DSV41_FAST_LOAD=1`, `DSV41_FAST_LOAD_TP_SLICE=auto`, `DSV41_AUTOTUNE_KEEP=1` | engine start 343 s → ~120 s; at EP1 each rank reads only its slice of every expert, into 256 MiB pinned slabs (330 allocations instead of ~94k, which gave back ~0.7M tokens of KV pool); MoE autotune cache kept across boots |
| Prefill hc | `DSV41_HC_FUSED=1` | the hyper-connection mix statistics of prefill chunks in one pass over K instead of 80 partial slices: ~1.46 → ~0.83 ms per call at 4096 rows, bit-identical to the stock kernels (checked on the first live call) |
| Prefill sequence parallel | `DSV41_PREFILL_SP=1` | at prefill chunks of >= 2048 rows the per-layer all-reduces become reduce-scatter + all-gather, and the per-row work between them (hyper-connection mixing and norms, the Engram `wkv` projection and gate) runs on each rank's quarter of the rows instead of all four ranks repeating it: prefill ~+20 % from 16k to 262k. Decode is untouched. The partial sums are added in a different order than the all-reduce's (the same kind of change as RoCEnante); `DSV41_PREFILL_SP_EXACT=1` keeps the all-reduce and is bit-identical to the unsharded path (checked op by op on the fleet) for ~+10 %. `DSV41_PREFILL_SP_FP8=1` gathers the attention input of 17 of the 21 full-row layers as the MXFP8 bytes and scales `wqkv_a` would compute itself (checked bit-identical per chunk size on every rank): about half the gather bytes there, +1.5-3 % prefill |
| L2 prefetch | `DSV41_L2_PREFETCH=1`, `DSV41_L2_PREFETCH_WOA=1` | during each RoCE all-reduce and all-gather of a decode step (SMs and DRAM mostly idle for 15-30 us) a side-stream kernel prefetches the first 6 MB of the weights that follow into L2; plans are learned from the warm-up forwards and captured in the CUDA graphs. Data is untouched (outputs bit-identical); decode step -1 ms at c1, +3 % on varied prompts, flat at c16. `WOA` adds a window after `wq_b` that prefetches `wo_a` while the attention core runs: -0.4 ms/step |
| Decode-step sync | `DSV41_SPEC_SYNC_FREE=all` | drops the per-step rank-0 broadcasts whose values are already identical on every rank (the draft token of each Markov step, the verify epilogue's three, verify_cap's live length); the draft noise comes from a counter-based stream every rank computes alike. Audit mode counted 0 rank mismatches in ~60k checks; greedy output byte-identical; -0.53 to -0.59 ms/step in an in-boot A/B (~-0.2 ms across reboots) |
| Decode-step glue | `DSV41_EAGER_GLUE=all` | fewer eager kernels between the draft and verify CUDA graphs (the folded fence's six copies as one, sampling-param staging skipped while unchanged, verify_cap's update inside the draft graph as one kernel); bit-identical, ~-0.1 to -0.2 ms/step |
| Target LM head | `DSV41_CERT_HEAD=1`, `DSV41_CERT_HEAD_M=6,...,48` | greedy verify steps read an 8-bit (MXINT8) screen of the head shard (173 MB instead of 331 MB per rank) with a proven error bound, and compute exactly only the 16-row head tiles that can hold the argmax; the argmax is the stock one (exact kernel bit-identical to the stock matmul, 0 mismatches over 3,355 checked steps), and sampled rows, logprobs, grammar and penalties get the full exact logits. -0.5 ms per c1 step; slower from 10 requests up, so it stops at 8 (M = 48 verify rows); +173 MB per rank |
| Correctness | `DSV41_FOLDED_FENCE=1` | closes the sglang#40919 D2H race on the folded verify path |
| Request guard | `DSV41_REPLAY_GUARD` (default on) | the launch runs SGLang's decoder SWA bounded replay (late layers see only the last window of a prefill), which cannot return prompt-token logprobs; such a request (`/generate` with `logprob_start_len` below the prompt length, `/v1/completions` with `echo` + `logprobs`) used to raise inside the forward and end every rank. It now gets HTTP 400 and the server keeps serving |
| Serving | `--enable-cache-report`, `--sleep-on-idle`, `SGLANG_RUST_BUILD_MODE=never` | cached-token usage for clients, idle CPU 47 % → 14 %, avoids a `cargo` hang at start; engine `/metrics` is documented as an opt-in |
| Optional | `DSV41_ENGRAM_DRM_NODE=/dev/dri/card0` | one Engram layer's cache in the GB10 display reservation, ~1.8 GiB more headroom; needs a host change ([docs/display-reserve.md](docs/display-reserve.md)) |

Not used: `DSV41_ADAPTIVE_CHUNK` (superseded by the bounded indexer), `DSPARK_SPS_TABLE` (ignored by `boot.py` because compact ragged verify crashes the Engram path; `DSPARK_SPS_ALLOW_ENGRAM=1` is an explicit future-fix override), the NVFP4 checkpoint (no bandwidth saved on GB10). Each adapter is described in [docs/adapters.md](docs/adapters.md); the measured effect of each switch is in [docs/history.md](docs/history.md#production-stack-as-of-2026-09-24-with-the-full-rationale-per-switch).

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
| Weights | `deepseek-ai/DeepSeek-V4.1-Flash-1017` (or its abliterated variant, see `.env.tp4.example`) at the pinned revision: 476 GiB, 48 shards. `./start-tp4.sh download` fetches it on the head with the `hf` CLI (`pip install -U "huggingface_hub[cli]"`); the switched profile shares the head's copy over NFS, the ring profile keeps a copy on every node |
| Disk | head: 476 GiB for the checkpoint; every node: ~48 GiB of packed Engram rows (`pack`, TP4) and ~49 GB for the image |
| Ports | 8888 (API), 20000 (torch distributed init), and 2049/111 (NFS, only with `NFS_SHARE=1`) between the nodes |
| Credentials | none for the weights. An empty `API_KEY` (or `none`, `off`, `dummy`, `0`) leaves the endpoint open, so keep the port private when it is unset |

With `NFS_REUSE_EXPORT=1` the checkpoint is published into the export with hardlinks (`files/nfs-share.sh`), so `MODEL_DIR` and the export root must be on the same filesystem.

## Quick start

Same launcher as upstream; read [`docs/README-upstream.md`](docs/README-upstream.md) first for the fleet setup (NFS share, Engram packing, fabric). `pack` needs the checkpoint, so download it before the first `pack`.

```bash
cp .env.tp4.example .env.tp4          # fill in HEAD_IP / WORKER_* / MODEL_DIR / fabric
# production: uncomment IMAGE=dsv41-4x-spark:canary-roce, BUILD_DOCKERFILE=Dockerfile.canary-roce
# and the "Production profile" block (EXTRA_CONTAINER_ENV, CHUNKED_PREFILL_SIZE, the memory
# fractions, EXTRA_SGLANG_ARGS), and set EP_SIZE=1 (the file default EP_SIZE=2 is for the
# base/canary images); the "Conservative fallback profile" block is the v2.1 feature set
./start-tp4.sh download               # 476 GiB at the pinned revision, needs the hf CLI; skips when complete
scripts/fetch-sglang-canary.sh        # stages the pinned dsv4.1 branch tree
./start-tp4.sh doctor
./start-tp4.sh build                  # builds on every node, bakes the adapters, runs the in-image tests
./start-tp4.sh share && ./start-tp4.sh pack   # first time only, see upstream README
./start-tp4.sh serve                  # ./start-tp4.sh stop | status | logs | smoke
```

The canary images have about 148 layers. A node whose Docker storage driver is `overlay2` (`docker info -f '{{.Driver}}'`) refuses images deeper than 128 layers ("max depth exceeded"); use the containerd image store there, or load a flattened copy (`docker export` of a container from the image, `docker import`, then restore `ENV`, `WORKDIR`, `HEALTHCHECK`, `ENTRYPOINT` and `CMD` in a small Dockerfile on top).

The boot log must show these lines, otherwise the profile is not active:

```
DSV41 shared-expert padding K: ... (5120, 576) -> (5120, 640)
DSV41 indexer chunked (sglang#39187 backport) ARMED: ...
Initialized DSpark draft runner. ... gamma=5, verify_num_draft_tokens=6
max_total_num_tokens=..., chunked_prefill_size=4096, ... max_running_requests=16
RoCEnante ready: world=4 hcas=...
DSV41_ROCE_TWOSHOT: two-shot all-reduce armed, DSV41_ROCE_TWOSHOT_ROWS=18 at install ...
[moe_launch_prepared] armed: prepared=plan direct-op=no timing=off sources pinned ok
DSV41 indexer nosync: DSV41_INDEXER_NOSYNC=1 ...
[moe_b12x_next] armed: routed MoE on b12x_next a7d7d29b ...
[moe_b12x_next] INFO: routed MoE at EP_SIZE=1 ...
DSV41 hc_fused: first call (4096 rows) bit-identical to stock
DSV41_L2_PREFETCH: model forward bracketed (decode/verify only); v3: wo_a windows (6.0 MB)
[spec_sync_free] armed (all): skip draft=True accept=True vcap=True ...
[eager_glue] vcap: verify_cap live length captured into the draft graph ...
[replicated_split] model.layers.0.self_attn.wqkv_a: compact gather (roce) bit-identical, ON ...
```

The simpler images (`Dockerfile` on the stock base, `Dockerfile.canary` without RoCEnante), each layer on its own, and the **switchless ring** for fleets without a RoCE switch are in [docs/optional-setups.md](docs/optional-setups.md). What the image is pinned to and what would let it move: [docs/upstream-watch.md](docs/upstream-watch.md).

## Measuring

- Bench with [speedbench](https://github.com/knapcio/speedbench) (full, load and longctx modes), never with a hand-rolled loop; discard the first run after a boot (cold Engram row cache, first-run warm-up).
- Any `#running-req` in the engine log above the concurrency being benched means foreign traffic landed in the window.
- A dashboard polling `nvidia-smi` every 2 s on every node costs ~0.6 ms per decode step; poll every 10 s (`POLL_INTERVAL_GPU=10000`, `POLL_INTERVAL_BANDWIDTH=10000`).
- Fixed benchmark prompts are greedy: a change that alters rounding can flip a near-tie early in the 256 tokens and move c1 by ±10 % without changing speed. Kernel A/Bs here use step time on several prompts as well.
- Release gate tools: `scripts/qeval.py` (75 tasks), `scripts/ds_gate.py kld` (greedy-continuation KL against a reference stack, long and short panels) and `ds_gate.py tscan` (T > 0 garble scan), `scripts/prefix_scan.py` (shared long prefix at c8, cold vs warm, plus cold-vs-warm logprob drift on full and mid-document cache hits).
- Greedy text equality is not a correctness gate for prefill changes: identical cold prompts of ~70k tokens produce different greedy continuations run to run. Correctness of the backports rests on the bitwise CPU tests, the needle tests and qeval.

## Rollback

Every layer is an env change: `EP_SIZE=2` without `DSV41_MOE_B12X_NEXT` returns the routed MoE to FlashInfer, `DSV41_FAST_LOAD=0` restores the stock loader, `SGLANG_ROCE_ALLREDUCE=0` drops the RDMA transport, `SPARK_PREFILL_TP_SPLIT=0` the row split, `IMAGE=dsv41-4x-spark:canary` the RoCEnante overlay, `IMAGE=dsv41-4x-spark:local` the branch; unsetting `DSV41_SPEC_SYNC_FREE`, `DSV41_EAGER_GLUE`, `DSV41_SPLIT_COMPACT_GATHER` and `DSV41_L2_PREFETCH_WOA` turns off the v2.2 decode-step changes, unsetting `DSV41_CERT_HEAD` the v2.3 certified head (`DSV41_REPLICATED_SPLIT_WINDOW=0` and `DSV41_MOE_B12X_NEXT_DET_TRITON=0` the two that are on by default). Unsetting `DSV41_ROCE_TWOSHOT`, `DSV41_MOE_LAUNCH_PREPARED` and `DSV41_INDEXER_NOSYNC` returns to the v2.5 profile. Upstream's profile: `MAX_RUNNING_REQUESTS=8`, `CHUNKED_PREFILL_SIZE=1024`, `EXTRA_CONTAINER_ENV=""` (and `EP_SIZE=4`, `DSV41_CACHE_GIB=0` for the exact upstream example). The adapters stay in the image but do nothing when their gate is unset.

## Known limits

- The numbers above are on a switched RoCE fabric. On a switchless ring RoCEnante needs a path to the opposite node, built in the neighbours' ConnectX-7 hardware ([docs/switchless-ring.md](docs/switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths)); without it the collectives go through NCCL and decode is slower. Prefill is lower on a ring either way (one-link bisection).
- The EP1 production line needs the `Dockerfile.canary-roce` image (it carries `runtime/b12x_next`). The routed MoE on b12x is numerically equivalent to FlashInfer's, not bit-identical (relative error against an fp32 reference 4.5–4.8 % for both); qeval and the needle tests are unchanged.
- Prompt-token logprobs and all-token hidden states are not available: `boot.py` launches with `--enable-decoder-swa-bounded-replay` (Mia's recipe), so the 19 layers after the last KV-source layer run only over the last 128 rows of each prefill, and such requests get HTTP 400. Output logprobs work. Cache hits are exact: a hit that lands on a prefill chunk boundary gives bit-identical logprobs to a cold request; a hit elsewhere computes the tail with a different batch shape, which moves low-probability logprobs by the same amount as the run-to-run nondeterminism of cold prefills on some prompts (`scripts/prefix_scan.py`, `docs/results/validation-20260929-v23.txt`).
- v2.4's KV pool is 3.67M tokens instead of ~5.8M: the 0.78 memory fraction and the larger SWA prefix-tail reserve both come out of it. That still holds three full 1M-token contexts; with fewer long sessions, `MEM_FRACTION_STATIC=0.80` with `CHUNKED_PREFILL_SIZE=4096` restores the pool at the v2.3 prefill speed. Never raise the chunk to 16384: at 1M it took a worker to 2.5 GiB available and hung the node on another fleet.
- One of the eight v2.4 boots on 2026-10-12 hung in the server's internal warm-up request and was restarted by the boot timeout; the retry and the seven other boots were clean.
- Single-stream prose speed is bounded by DSpark acceptance (~3 accepted tokens per step on prose against ~6 on code); no configuration changes that.
- The fast loader leaves the KV pool 3–13 % smaller than the stock loader (6.7–7.3 M vs 7.5–7.8 M tokens) and more variable between boots; `DSV41_FAST_LOAD=0` restores it at ~220 s per boot ([docs/fast-load.md](docs/fast-load.md)).
- RoCEnante is a new transport in the decode path; b12x has one open report of a rank wedging under long mixed-context traffic on an earlier revision ([b12x#313](https://github.com/local-inference-lab/b12x/issues/313)). The overlay's result-boundary health check fails the step instead of hanging, and NCCL is one env change away.
- The two-shot all-reduce is measured on a switched fabric only and is not available with `DSV41_ROCE_RING=1`; on the ring leave `DSV41_ROCE_TWOSHOT` unset.

## License

Same as upstream: see [`LICENSE`](LICENSE). Model weights are MIT (DeepSeek).

## Credits

This repository is a downstream profile of **[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)** by Mia (MiaAI-Lab). The launcher (`start.sh`, `start-tp4.sh`, `boot.py`), the Engram NVMe row store, the MXFP8 b12x routing, the memory model in `docs/chunked-prefill-memory.md`, the thinking alias, the output cap, the loop abort and the overall deployment design are her work, kept here with full git history and under the same licence. The original README is preserved as [`docs/README-upstream.md`](docs/README-upstream.md); read it first for the fleet setup (NFS share, Engram packing, fabric, 3-node profile).

Other work this profile builds on:

- **kpham-sgl**, [sgl-project/sglang#39187](https://github.com/sgl-project/sglang/pull/39187): the bounded dense-indexer prefill transient, backported here as `adapter/indexer_chunked*.py` (whose prefill slot ids `DSV41_INDEXER_NOSYNC` builds without host syncs).
- **BBuf** and the SGLang `dsv4.1` branch contributors ([#39370](https://github.com/sgl-project/sglang/pull/39370), [#39646](https://github.com/sgl-project/sglang/pull/39646), [#39648](https://github.com/sgl-project/sglang/pull/39648), [#39653](https://github.com/sgl-project/sglang/pull/39653)): the decode kernel work in the optional `Dockerfile.canary` image.
- **Vineel Bhat, Claude Fable 5.1 and hnyls2002**, [SGLang #41303](https://github.com/sgl-project/sglang/pull/41303): keeping a full SWA window for short decoder bounded-replay extends and holding the window back from ordinary prefix-cache matches; backported to the pinned tree in `patches/sglang-41303-bounded-replay-window.patch`.
- **Yi (Vincent) Zhong**, [SGLang #42901](https://github.com/sgl-project/sglang/pull/42901): avoiding discarded layer-20 indexer, attention and MoE work on early prompt rows under decoder bounded replay. The adaptation for the pinned pre-`HcState` tree is in `patches/sglang-42901-layer20-tail.patch`; it is source-qualified but has not yet been benchmarked on GB10.
- **Shangming Cai**, [SGLang #44001](https://github.com/sgl-project/sglang/pull/44001): gating `--prefill-decode-interval` on the live batch size. The zero-default backport in `patches/sglang-44001-prefill-interval-min-bs.patch` preserves the current scheduler unless `--prefill-decode-interval-min-bs` is set explicitly.
- **Xiaofei Zheng** and **kano-zheng**, [SGLang #41259](https://github.com/sgl-project/sglang/pull/41259) and [#45410](https://github.com/sgl-project/sglang/pull/45410): fail-closed parallel tokenization for long rendered prompts and its native DeepSeek V4/V4.1/V3.2 integration. The helper is fetched byte-exact from the merged upstream commit and its SHA-256 plus all four call sites are verified during staging and image build.
- **nankingjing**, [SGLang #43549](https://github.com/sgl-project/sglang/pull/43549): preserve mid-conversation system messages in the native V4.1 encoder instead of hoisting them into the leading system block. This keeps Claude/Codex-style growing conversations prefix-cacheable; the adaptation for the older Anthropic frontend is fail-closed on the exact `dsv41` encoding spec.
- **Justin Tong (JustinTong0323)**, [SGLang #44751](https://github.com/sgl-project/sglang/pull/44751) commit [`a038a0dc`](https://github.com/sgl-project/sglang/commit/a038a0dca05cefca1e0edc4ac43e3bfbec2e064c): bring the official DeepSeek 1017 encoder semantics to the pinned runtime. Tool-result images now follow tool-call order, structured tool content remains structured, and a tool-result-only user message no longer starts a new turn that discards the current assistant reasoning. The local adaptation makes private copies before placeholder sanitization so request objects remain immutable.
- **Liangsheng Yin (hnyls2002)**, [SGLang #45301](https://github.com/sgl-project/sglang/pull/45301) commit [`117247ea`](https://github.com/sgl-project/sglang/commit/117247ea2f9ba77cc8d47005553d3eed18e3a7bc): keep TP-padded V4.1 query storage pointer-stable across decoder-tail graph buckets. The pinned-tree adaptation grows one zeroed buffer before capture, reuses row slices for smaller buckets and refuses any post-capture replacement, avoiding a freed CUDA-graph address and the reported MMU fault after cache flush.
- **JiaShuai0209 (ssJ) and Liangsheng Yin (hnyls2002)**, [SGLang #40424](https://github.com/sgl-project/sglang/pull/40424) commit [`720857de`](https://github.com/sgl-project/sglang/commit/720857de766ad8855286da9246418da66724596c): keep DP-attention padding out of the V4.1 low-ratio compressors and return throwaway zero hashes for padded idle Engram batches. This prevents the upstream-reproduced uninitialized request-index/device-assert and idle-mode scheduler crash without changing live-token outputs.
- **maocheng23 and Claude Opus 5.5**, [SGLang #43448](https://github.com/sgl-project/sglang/pull/43448) commit [`42228038`](https://github.com/sgl-project/sglang/commit/422280382336eba5daa5233f16a26f904646442d): make V4.1 MHC prewarm use the configured prefill-token budget when chunked prefill is disabled (`-1`, `0` or unset), instead of constructing a negative/invalid warmup bucket and failing during weight load.
- **Chen Cheng (ischencheng)**, [SGLang #45457](https://github.com/sgl-project/sglang/pull/45457): encode control-token spellings from user content, tool results, tool-call arguments and tool schemas as plain text rather than letting request data forge V4.1 turn boundaries. The exact upstream helper and test suite are content-pinned; the local adaptation retains #41259's parallel path for ordinary prompts.
- **Chen Cheng (ischencheng)**, [SGLang #45458](https://github.com/sgl-project/sglang/pull/45458): end V4/V4.1 reasoning on the tokenizer's real `</think>` token rather than identical text spelled with ordinary tokens. Chat and Responses pass output IDs in streaming and non-streaming paths; the pinned upstream parser suite verifies both modes.
- **Kevin Mi (kevin-mii) and Claude Opus 5.5**, [SGLang #45663](https://github.com/sgl-project/sglang/pull/45663): fuse per-row DSpark draft temperatures, greedy masks and step temperatures into one Triton launch, then have the accept/finalize kernels write their graph-static outputs directly instead of launching six post-kernel copies. The exact content-pinned kernel and zero-fuzz accept hunks are adapted to the older sampler; these canary optimizations are source-qualified and not yet assigned a GB10 TPS gain.
- **liweizheng (LevyJeng)**, [SGLang #45518](https://github.com/sgl-project/sglang/pull/45518) commit [`8e85d6cf`](https://github.com/sgl-project/sglang/commit/8e85d6cfa9fc9025942c4faf07c362c1ce3d06f4): clamp FlashInfer's fresh DSpark softmax output to the valid probability ceiling in place. This prevents rare rounding above 1.0 from being treated as an unconditional rejection and collapsing speculative acceptance at affected non-greedy temperatures.
- **Yangmin Li (YAMY1234)**, [SGLang #45450](https://github.com/sgl-project/sglang/pull/45450) commits [`20189751`](https://github.com/sgl-project/sglang/commit/20189751de186f394abe850b1ad48087df32eb77) and [`1f8e4f3e`](https://github.com/sgl-project/sglang/commit/1f8e4f3e6cdc378d94a8a08630a9763c648e88eb): restage the draft graph's temperature/top-k buffers only when their sources change, and capture the eligible static DSpark draft metadata prep in a glue graph. The second port deliberately uses the runner registry's stable `out_cache_loc` address and persistent device extend-length constants; unsupported backends/configurations remain eager.
- **Yangmin Li (YAMY1234)**, [SGLang #45630](https://github.com/sgl-project/sglang/pull/45630) commit [`741aea0a`](https://github.com/sgl-project/sglang/commit/741aea0ab6c51ccfbb862203a5a4e4aeaf7e8466): let startup-autotuned FlashInfer CuTeDSL MXFP8 tactics serve block-FP8 decode/verify row counts instead of falling back to the backend heuristic below 4,096 rows. Upstream measured about 2% shorter V4.1 steps at 16-64 requests per rank; deterministic inference remains pinned.
- **05yuki and Claude Opus 5.5**, [SGLang #45745](https://github.com/sgl-project/sglang/pull/45745) commit [`3458c1f1`](https://github.com/sgl-project/sglang/commit/3458c1f1900fbbaffc237d74491f30e6f8cf288c): apply the V4.1 Torch prefill indexer's ReLU and head weighting in place on its fresh einsum result. The identical arithmetic order removes two score-sized transient tensors on SM12x; this is a memory/stability improvement, not a decode-TPS claim.
- **Shuwen Wang (alphabetc1)**, [SGLang #45793](https://github.com/sgl-project/sglang/pull/45793) commit [`0de6dd95`](https://github.com/sgl-project/sglang/commit/0de6dd957f4da0b0655ef6d41bfd4b0593a08d12): process eight V4.1 FP8/FP4 cache tokens per Triton program during sparse prefill instead of one. The pinned-tree adaptation keeps the upstream masked-tail/addressing/output-order design and is source-qualified only pending GB10 parity and timing.
- **Bingjia Wang (BJWang-ant)**, [SGLang #45794](https://github.com/sgl-project/sglang/pull/45794) commit [`b7344253`](https://github.com/sgl-project/sglang/commit/b7344253b3136384c522aded59015f79c9bb7a08): pad CUDA BF16 V4.1 Torch-indexer score widths to a 64-row boundary, budget against the padded width and slice back to the logical context before masking/selection. The pinned TP path adapts only this performance commit; the PR's separate CP row-mapping fix is not needed here.
- **hantenglong (huangxiaobai-ydmy)**, [SGLang #45762](https://github.com/sgl-project/sglang/pull/45762) commit [`f2aad50c`](https://github.com/sgl-project/sglang/commit/f2aad50c4163d047d740fd61defea2d122ac5d30): make the V4/V4.1 reasoning parser follow `SGLANG_DEFAULT_THINKING` when `thinking` is omitted, matching the native encoder while preserving explicit per-request overrides.
- **potoior**, [SGLang #45804](https://github.com/sgl-project/sglang/pull/45804) commit [`af882cfe`](https://github.com/sgl-project/sglang/commit/af882cfe1e7ff244db2a49bb70fdc72199c68bb9): scope that default-thinking parity to native `dsv4`/`dsv41`/`dsv32` encoders, keep Jinja-driven parsers independent of the environment default, and apply the same rule to the Responses API.
- **hnyls2002**, [SGLang #45579](https://github.com/sgl-project/sglang/pull/45579) commit [`ad55c50b`](https://github.com/sgl-project/sglang/commit/ad55c50bc3a45bdf957626eb0a9350ca0f4f5176): compute request-window SWA history source rows once per prefill layout and reuse them across layers; the pinned-tree adaptation preserves the original address and zero-row semantics.
- **alphabetc1**, [SGLang #45726](https://github.com/sgl-project/sglang/pull/45726) merge [`9d5386fd`](https://github.com/sgl-project/sglang/commit/9d5386fd13f7a91e9e84dc0c0ab61c5b1fbaba98): fuse CUDA request-window query and history metadata construction into two Triton kernels. The exact upstream kernel is content-pinned; only its older-tree call site is adapted locally.
- **hnyls2002**, [SGLang #45586](https://github.com/sgl-project/sglang/pull/45586) and [#45699](https://github.com/sgl-project/sglang/pull/45699): route idle SWA stores before reading stale/unset attention metadata, using a cached idle request-window layout when encoder replay is enabled and the translated dummy slot otherwise.
- **Divy (divyvasal) and Claude Opus 5.5**, [SGLang #44923](https://github.com/sgl-project/sglang/pull/44923) commit [`6e0355d9`](https://github.com/sgl-project/sglang/commit/6e0355d972cb926f137d2b8bb70caf5c9af5d129): keep per-row greedy flags on the host and recompute `is_all_greedy` / `is_any_greedy` when finished requests leave the batch, so a greedy decode batch that once held a sampled request returns to DSpark's folded verify path instead of staying on the eager accept path until it drains. Ported from the open PR with `patches/sglang-44923-greedy-flags-refresh.patch`; not yet measured on GB10.
- **DeepSeek**, [`inference/model.py` revision `4dd75a8b`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/4dd75a8b448d95ae8e284d5c5a1e4b8808bd583b) of DeepSeek-V4.1-Flash (2026-10-12; the same code ships with [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)): the reference rounds the attention output to FP8 (32-value blocks, power-of-two scales) before `wo_a`, in the target and the DSpark draft attention. Mirrored by the opt-in `adapter/wo_a_input_fp8.py` (`DSV41_WO_A_INPUT_FP8=1`, off by default: measured ~3 % slower decode with the same eval results).
- **05yuki**, [SGLang #45185](https://github.com/sgl-project/sglang/pull/45185) (open): the same quantize-dequantize of `o` before the bf16 `wo_a` in SGLang's V4 attention and DSpark draft, including the fp32 emulation of the e4m3 rounding that `DSV41_WO_A_INPUT_FP8_EMULATE=1` follows.
- **christopherowen**, [spark-ds41f](https://github.com/christopherowen/spark-ds41f): adopted the reference's WO-A input quantization in its vLLM recipe for DGX Spark (first inside the W8A16 GEMM's tile load, then in the sparse MLA kernels' output store), which pointed us to the change.
- **Barron, Gnanapragasam and Tschannen**, ["A Fix for Softmax and Cross-Entropy"](https://jonbarron.info/softmaxfix/) (Eq. 5), as applied to verification kernels in **Christopher Owen** (christopherowen)'s [spark-ds41f #584](https://github.com/christopherowen/spark-ds41f/tree/9065a471a6b0e63529211931fab5793fce9f27a1/experiments/2026-10-25-softmax-fix): the rounding of probabilities against 1 that led us to audit block verification. `adapter/block_verify.py` now forms `1 - w` before adding `S` in the acceptance threshold (`tests/test_block_verify_numerics.py`, against an FP64 oracle); the full log-domain form of Eq. 5 is not ported.
- **hushengkai**, for independently reproducing the EP2 / Engram cache / shared-expert padding changes on a second 4x GB10 fleet.
- **rhys101**, [DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8): the SG17 SGLang overlay that routes small tensor-parallel all-reduces to RoCEnante (reused with a TP4 adaptation in `Dockerfile.canary-roce`, which the two-shot all-reduce plugs into) and the SG18 native prefill TP split (`adapter/spark_prefill_dense.py`, combined with the indexer backport in `adapter/indexer_chunked_v3.py`).
- **local-inference-lab / Luke Alonso and Jason (original-el8)**, [b12x](https://github.com/local-inference-lab/b12x): RoCEnante, the one-shot RDMA all-reduce (`runtime/b12x`, Apache-2.0, frozen at the SG17 revision), and the fused MoE kernels that run the routed experts (`runtime/b12x_next`: b12x main at `a7d7d29b`, renamed so both revisions live in one image, with the local M64 prefill admission, Luke Alonso's [scratch-alignment fix](https://github.com/local-inference-lab/b12x/commit/3f8b71d991df31a9aa851604fc999fd72f08aa93) and, since v2.6.2, **Martin Vit**'s [producer-loop claim barrier](https://github.com/local-inference-lab/b12x/commit/d7937d7fc2e5511a0b61eb3572cd149f563f9b4f) (backport); b12x is now developed in [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) under `flashinfer/experimental/b12x`); the prepared eager MoE launch (`adapter/moe_launch_prepared.py`, `DSV41_MOE_LAUNCH_PREPARED`) mirrors b12x main's `_launch_dynamic_flat`.
- **Luke Alonso**, [flashinfer-ai/flashinfer#6915](https://github.com/flashinfer-ai/flashinfer/pull/6915): the pipelined two-shot RoCE all-reduce (reduce-scatter to the row owner, FP32 sum in rank order, all-gather back, staging and the wire overlapped per chunk) behind the opt-in `DSV41_ROCE_TWOSHOT` switch, ported onto rhys101's RoCEnante path in `runtime/b12x/b12x/comm/roce_twoshot` (Apache-2.0; `runtime/b12x/twoshot-provenance.json`). In the production profile since v2.6 (c16 +3.7 %, bit-identical).
- **Gabriel Wu** and the contributors to FlashInfer's SM120 sparse-MLA kernels ([flashinfer-ai/flashinfer #3395](https://github.com/flashinfer-ai/flashinfer/pull/3395) and later; **Ka-Hyun Nam**'s [#4732](https://github.com/flashinfer-ai/flashinfer/pull/4732) fix for issue #3700 is in the version we build on): `prefill_kernel.cuh` is generic in the page size, which is what lets v2.6.2's `DSV41_PREFILL_NATIVE_EXTRA` module read 128 / 256-token extra pages with the stock templates (`runtime/flashinfer_pce/PROVENANCE.json`).
- **luxingcom (LuZ)**, [LuZ DGX Spark TP4 ring](https://github.com/luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring): the first integration of b12x's fused MoE into SGLang on a four-Spark fleet, which showed the route.
- **FujitsuPolycom and the sparkring contributors**, [sparkring](https://github.com/FujitsuPolycom/sparkring): the hardware-forwarded opposite-node paths and the path-aware RoCEnante (`runtime/b12x/b12x/comm/roce_ring`) that run the production line on a switchless ring, plus the maintained [atomic-counter](https://github.com/FujitsuPolycom/sparkring/commit/2d83bb60402fc2de5708ddcf3f962a007772600f), [memory-scope](https://github.com/FujitsuPolycom/sparkring/commit/e4d092181007f9a1060b9c172804b37f37c782f3), and [sequence-wrap timeout](https://github.com/FujitsuPolycom/sparkring/commit/f91b5c87ef2352014b8724388ab26c6ad441a0e1) fixes.
- **Stanislav Bardyuk**, [NVIDIA/nccl #2393](https://github.com/NVIDIA/nccl/pull/2393): the AArch64 acquire fence for the IB CTS FIFO, backported as an additional patch in the recommended switchless-ring NCCL build.
- **rsync** (rchmagos), [#8](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/8): got RoCEnante running on a four-node switchless ring with sparkring's opposite-node paths and contributed the integration (`DSV41_ROCE_RING`, the `scripts/ring_mesh/` planner).
- **sumsliu**, [dgx-spark-deepseek-v41](https://github.com/sumsliu/dgx-spark-deepseek-v41): the eight-Spark measurement that moving to expert tensor parallelism removes most of the all-reduce wait.
- **Saolence**, [#1](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/1), [#2](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/2), [#5](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/5), [#6](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/6): the switchless ring as an opt-in switch, the worker NCCL and build fixes, and the first-deployer review in [#4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/4) behind the host prerequisites, the download step, the patched-NCCL section and the ring addressing notes.
- **adapt-ai-systems**, [#12](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/12), [#13](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/13), [#14](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/14), [#15](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/15): `ring_mesh/plan.py --reversed` for a node cabled the other way round, the per-port GID index under `NCCL_IB_GID_AUTO=1` in the ring preflight, and the field notes on a mesh hop through an L2 switch and on link-local IPv6 taking GID index 3.
- **ChrisLou-bioinfo**, **ZackO2o** and **hushengkai** (Mia's repo [#40](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/40), [#41](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/41), [#45](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/45)): the ring recovery notes and the bring-up pitfalls in `docs/switchless-ring.md` and `.env.tp4.example`; the people who published their own measurements are listed in [`docs/community-results.md`](docs/community-results.md).
- **hushengkai** ([Mia #44](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/44), [#9](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/9), [Mia PR #15](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/15), [#10](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/10)): the diagnosis of short-request starvation behind chunked prefill (the opt-in `DSV41_PREFILL_BYPASS` adapter is our own implementation, pending fleet qualification), the image-placeholder sanitizer patched into the images, and the offline-load and dead-proxy reports behind `SKIP_PULL` and the proxy pre-check.
- **100menotu001** ([Mia PR #46](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/46), [PR #47](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/47)): public-key recovery and key-blob matching in `ensure_ssh_keys`, the share-before-build error, and Messages `thinking: enabled` without a budget.
- **colonel-otto** (ported from Mia upstream commits `5685a5d` / `dd02792`): per-host SSH users through `WORKER_USER=""` and `~/.ssh/config`, including host-name-based spec-table distribution and safe worker paths.
- **joeym82956**, [#54](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/54): the `scripts/ring_mesh/` checks no longer re-add existing rules when `grep -q` closes the pipe under `pipefail` (intermittent "Filter already exists" on ring restarts).
- **SvangenStudios**, [#57](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/57): `stop.sh` no longer reports the shared NFS exporter it keeps on purpose as left over, names the exporter it keeps, and counts the configured nodes in its banner.
- **Scott Woolsey Biggart and Claude Opus 5.5**, [Mia PR #58](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/58), and the authors of [SGLang #39665](https://github.com/sgl-project/sglang/pull/39665): the fail-closed V4.1 structural-tag backport that fixes required and named tool choice on the default image. The pinned canary source already contains that upstream detector.
- **yalindogusahin** ([Mia PR #4](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/4)), **RustyAiLab** ([PR #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/5), [PR #8](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/8)) and **BHCC2025** ([PR #26](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/26)): NFS exporter reuse, the single-rail shell fixes, `remote.py` reading `.env` defaults, and re-sharing before `pack` (GID auto-selection is opt-in here).
- **bertholomus** ([Mia PR #10](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/10)): reading each node's HCA port state, used here as a `doctor` check.
- **motzudao** ([Mia #31](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/31)): the report that a custom `EXTRA_SGLANG_ARGS` dropped the required FP8 GEMM backend.
- **MiaAI-Lab/sparkDash**, the benchmark behind the README numbers up to v2.5 (now in [docs/history.md](docs/history.md)).
