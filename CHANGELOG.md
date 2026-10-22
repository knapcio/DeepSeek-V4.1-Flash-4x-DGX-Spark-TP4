# Changelog

## Unreleased

- Adapt Shuwen Wang's SGLang #45793 for the pinned V4.1 KV layout: sparse
  prefill now dequantizes eight cache tokens per Triton program instead of
  launching one program per token. FP8 and FP4 share a masked page/address
  mapping, and the FP4 low/high nibbles remain interleaved per output row.
  The upstream GB300 kernel result (1.5-2.34x) is context only; this canary
  carries no GB10 or end-to-end claim before a GPU parity/timing gate.
- Backport 05yuki and Claude Opus 5.5's SGLang #45745: the SM12x-compatible
  Torch V4.1 prefill indexer now applies ReLU and head weights in place on the
  fresh einsum result before the unchanged head reduction. This preserves the
  arithmetic order while removing two score-sized temporary tensors (about
  2 GiB at the upstream 1-GiB score-chunk cap). The pinned adaptation has a
  zero-fuzz patch and a fail-closed AST verifier.
- Backport Yangmin Li's first SGLang #45450 commit (`20189751`): the draft
  graph now restages temperature/top-k buffers only when its sampler, batch
  size or source tensor identities change. `SamplingBatchInfo` rebinds those
  tensors on filter/merge and does not mutate them in place, so steady decode
  skips the #45663 staging kernel without stale values. A focused harness
  extracts and executes the staged production method without importing the
  accelerator stack.
- Continue the bounded SGLang #45663 port: DSpark's folded accept path now
  writes `correct_len`, bonus, finalized lengths and output tokens directly
  into its graph-static buffers. This removes six post-kernel `copy_` launches
  without changing eager behavior. The two kernel hunks are upstream-exact,
  the older epilogue adaptation applies with zero fuzz, and source checks fail
  closed if a copy path returns. GB10 correctness/timing remain a separate
  promotion gate.
- Adapt the portable part of SGLang #45663 (Kevin Mi / kevin-mii and Claude
  Opus 5.5): DSpark sampling parameters now use the exact upstream fused
  Triton helper instead of separate temperature-clamp and greedy-mask GPU
  launches. The older sampler supplies the helper's otherwise-unused
  step-temperature output from a static scratch buffer. The helper is
  SHA-256-pinned and the integration fails closed on source drift; expected
  GB10 TPS impact remains unmeasured.
- Adapt SGLang #45458 (Chen Cheng / ischencheng): DeepSeek V4/V4.1
  reasoning now ends only on the tokenizer's single `</think>` token when
  output IDs are available. Literal `</think>` text spelled with ordinary
  tokens stays inside reasoning; Chat and Responses cover incremental,
  cumulative and non-streaming output, with text-only fallback preserved.
- Adapt SGLang #45457 (Chen Cheng / ischencheng) onto the pinned V4.1 tree.
  Added-token spellings supplied in user/tool data now encode as plain text,
  while template-written turn, reasoning and DSML tokens keep their control
  IDs. The exact upstream helper and test suite are SHA-256-pinned; clean
  prompts retain #41259's parallel tokenizer rather than regressing to serial
  encode. `SGLANG_ENABLE_DSV41_PLAIN_TEXT_ENCODE=false` is the rollback.
- Backport SGLang #43549 (nankingjing) onto the pinned Anthropic frontend:
  native `dsv41` conversations retain mid-conversation system messages in
  place, so a new turn does not rewrite the leading prompt and destroy radix
  prefix-cache reuse. Other encoding specs retain the old Jinja-probe policy.
- Backport SGLang #41259's exact, content-verified parallel prompt encoder and
  #45410's native DeepSeek V4/V4.1/V3.2 call sites. Long agent-shaped prompts
  can use the tokenizers/Rayon pool instead of blocking one HTTP thread in a
  single encode call; unsupported tokenizer pipelines and short prompts retain
  the original path. Upstream measured identical token IDs and roughly 9x
  faster tokenization on a 947K-token V4.1 prompt. This is source-qualified,
  not yet measured end-to-end on this GB10 deployment.
- Backport b12x `3f8b71d9` (Luke Alonso): when sequential prepared-kernel
  variants have equal 256-byte-aligned scratch totals, retain the variant with
  the larger raw requirement. This prevents an undersized arena for affected
  exact-token-count MXFP8 MoE capacities; the regression test covers both
  variant orderings. The b12x source pin remains `a7d7d29b`.

Newest first. Each entry says what changed in the production stack and what was measured; the
raw results live under `docs/results/`.

## 2026-10-22 issue #16 resolution

- Backported the runtime part of [SGLang #44001](https://github.com/sgl-project/sglang/pull/44001)
  (Shangming Cai): `--prefill-decode-interval-min-bs N` admits queued prefills without the configured
  decode-only delay while fewer than `N` requests are running, then restores the interval at/above `N`.
  Its default is zero, so rebuilt images preserve the existing scheduler. The patch applies with zero fuzz,
  has source-order and behavior regressions in both canary builds and CI, and remains opt-in pending a GB10
  multi-user TTFT/prefill-throughput A/B.
- Added a credited, zero-fuzz adaptation of [SGLang #42901](https://github.com/sgl-project/sglang/pull/42901)
  for the pinned pre-`HcState` engine. Under decoder bounded replay it writes layer 20's compressed KV for all
  prompt rows, then limits that layer's otherwise discarded indexer/attention/MoE work to the final 128 rows.
  The fetch path, image builds and CI fail closed on source-order invariants. Upstream measured 5-10% lower TTFT
  for 16K-200K cold prompts on GB300; GB10 performance and model-level correctness remain unmeasured, so this is
  a source-qualified canary change rather than a promoted serving result.
- Added an Ubuntu/macOS offline CI gate for the launcher, SGLang backport verifier, Engram/SPS guard,
  per-host remote configuration, forced-tool detector installer, syntax checks and vendored-source identity.
- Ported MiaAI-Lab's Engram/SPS safety guard (`6eb9a7a`, MiaAI-Lab and Claude Sonnet 5.5): an existing
  `DSPARK_SPS_TABLE` can no longer silently arm compact ragged verification and crash CUDA-graph capture.
  `DSPARK_SPS_ALLOW_ENGRAM=1` is the explicit future-engine-fix override. Documented SGLang's opt-in metrics
  flags so operators know why `/metrics` otherwise returns 404 and how to expose all schedulers' `sglang:*`
  series (MiaAI-Lab `786859e`, original verification by Zbigniew).
- Ported Scott Woolsey Biggart and Claude Opus 5.5's Mia PR #58 backport of SGLang #39665: the
  default image now emits typed `tool_calls` for required and named tool choice instead of returning bare DSML
  in `message.content`. The installer is content-pinned and fails closed on base-image drift; all three image
  recipes run the real SGLang/xgrammar grammar-and-parser regression during build.
- Added Stanislav Bardyuk's NVIDIA/nccl #2393 acquire-fence patch to the recommended AArch64
  switchless-ring NCCL build. It orders the CTS `nreqs` load after `idx`, preventing a reused FIFO slot's
  stale count from hanging or truncating the IB send path on weakly ordered CPUs. The previously measured
  sparkring release binary remains documented as historical reproduction material because it predates the fix.
- Marked the original v2.4 image as pre-fix after a second fleet reported repeated CUDA Xid 13/43 faults under large
  prefix-cache hits. Four reports now include two with `DSV41_PREFILL_BYPASS` disabled, so bypass is not treated as
  the root cause. The v2.1 feature set is the documented fallback while the candidate is soaked.
- Backported [SGLang #41303](https://github.com/sgl-project/sglang/pull/41303) to the pinned `f80c91a4b` tree. The
  patch keeps the full SWA window for short decoder bounded-replay extends and holds one window back from ordinary
  radix-cache matches. It is credited in the patch and README, applies with zero fuzz, and is source-verified both
  after staging and during image build.
- PR #18 merged the source fix and closed issue #16. No GPU or fleet action was part of the repository repair;
  the focused two-boot, 60-90 minute regression procedure remains in
  [`docs/issue-16-swa-crash.md`](docs/issue-16-swa-crash.md) for deployments rebuilding old v2.4 images.
- Ported Mia upstream's per-host SSH-user support (`5685a5d`, `dd02792`, colonel-otto): an explicit empty
  `WORKER_USER` now lets each `WORKER_HOSTS` alias select its own `User` from `~/.ssh/config`; rsync, stop,
  Engram packing and DSpark table distribution follow the aliases, while an implicit `/home//...` worker path
  fails closed.

## 2026-10-12 v2.4

Long context and multi-user behaviour; decode unchanged (same-session A/B within noise). Release gate on the final
image against the v2.3 settings booted the same morning, raw output:
[`docs/results/validation-20261012-v24.txt`](docs/results/validation-20261012-v24.txt).

- **Prefill chunk 4096 -> 8192 with `MEM_FRACTION_STATIC=HEAD_MEM_FRACTION_STATIC=0.78`.** Cold 388K prompt
  93.4 -> 77.8 s, 969K needle 325 -> 235 s. At 0.80 the larger chunk left the head with 2.7 GiB available during
  a 1M prefill. 0.78 restores 5.7 GiB, at the cost of KV pool. Reported by hushengkai (MiaAI-Lab#44).
- **`--prefill-decode-interval 2` + `DSV41_PREFILL_BYPASS=256` (`adapter/prefill_bypass.py`).** A short request
  arriving during a long cold prefill is answered in 4.6 s instead of 80 s at 400K. The adapter is our own
  implementation of hushengkai's diagnosis (#9, MiaAI-Lab#44). It admits small requests beside the in-flight chunk
  only after the engine's memory check and only from rank-identical state, and it refuses to arm on source drift or
  with the prefill delayer.
- **`--swa-prefix-tails 128`.** A 388K session stays cached while another 388K session runs: the revisit takes
  1.5 s instead of an 89 s re-prefill. The default of 64 tails evicted it (MiaAI-Lab#43, LECYWZA). KV pool
  5.77M -> 3.67M tokens with both settings.
- **Images.** `runtime/patch_encoding_dsv41.py` sanitizes the image placeholder token in message text instead of
  failing the request (MiaAI-Lab PR #15, hushengkai). `/v1/messages` accepts `thinking: {"type": "enabled"}`
  without `budget_tokens` (MiaAI-Lab PR #47, 100menotu001). Both are fingerprinted against the pinned source.
- **Launcher.**
  - The required `--fp8-gemm-backend flashinfer_cutlass` survives a custom `EXTRA_SGLANG_ARGS` (MiaAI-Lab#31).
  - `doctor` checks every node's listed RDMA ports are ACTIVE and the socket interfaces exist (MiaAI-Lab PR #10, #45).
  - `SKIP_PULL=1` builds from a local base tag; a dead docker proxy is reported before a pull (#10).
  - SSH public-key recovery (MiaAI-Lab PR #46).
  - NFS and `remote.py` fixes (MiaAI-Lab PRs #4, #5, #8, #26).
  - Opt-in `DSV41_TRITON_CACHE=1` and `NCCL_IB_GID_AUTO=1`.
- **Gate.** qeval 72/72/72 (reference 72/72/72), KL 0.0045 / 0.0061 (limit 0.035), T > 0 scan 0/35, prefix scan
  PASS, 60-minute c16 soak with 2,285 completions and 0 errors / restarts / OOM / Xid. The new long-context probes
  are `scripts/v24_probe.py` and `scripts/v24_soak.py`.

## 2026-09-29 v2.3

Lossless: the certified head returns the stock argmax (in-boot A/B: greedy outputs identical 24/24), the guard only
rejects requests. Fresh-clone release gate, raw output: [`docs/results/validation-20260929-v23.txt`](docs/results/validation-20260929-v23.txt).

- **`adapter/cert_head.py`, `DSV41_CERT_HEAD=1 DSV41_CERT_HEAD_M=6,12,18,24,30,36,42,48`, on.** Certified
  target LM head for greedy verify steps: an MXINT8 screen of the head shard with a proven error bound decides
  which 16-row tiles can hold the argmax; only those are computed with the exact kernel (bit-identical to the stock
  matmul), the rest are -inf. Sampled rows, logprobs, grammar and penalties take the full exact logits. GPU
  qualification (`tests/gpu_test_cert_ds.py`): exact kernel bit-identical to `torch.matmul` on every logit of all
  four rank shards for every verify row count 6..96; check mode on the fleet: 0 mismatches over 3,355 certified
  steps. In-boot A/B c1: step -0.50 ms (CI -0.59..-0.37), outputs identical 24/24; sparkDash prose c1 +1.2 % /
  +1.9 % (two windows), c4 +0.5 %, c8 flat. Slower from 10 requests up (M=96: +0.44 ms), so M stops at 48.
  +173 MB per rank.
- **`adapter/replay_guard.py`, `DSV41_REPLAY_GUARD` (default on).** Under the decoder SWA bounded replay that
  `boot.py` always enables, a request for prompt-token logprobs (`/generate` with `logprob_start_len` below the
  prompt length, `/v1/completions` with `echo` + `logprobs`) or for every prompt token's hidden states raised a
  `ValueError` inside the forward and ended all four ranks. Such requests now get HTTP 400 and the engine keeps
  serving; output logprobs are unaffected.
- **`scripts/prefix_scan.py`** (from the GLM-5.3 TP4 recipe, adapted to SGLang): shared-long-prefix gate, c8 cold
  vs warm at T > 0 plus a greedy drift test (full and mid-document cache hits, cold-vs-warm against cold-vs-cold
  logprob drift). **`scripts/ds_gate.py`**: greedy-continuation KL panels (long 16.5k-40k prompts, short 1-3k) and
  the T > 0 garble scan. **`scripts/qeval_tasks.py`**: `extract_final_number` prefers the last line that holds only a
  number (the form the prompts ask for) before falling back to the last number.
- Release gate (fresh clone of 5f11a97 built on all four nodes, cold caches, healthy in 556 s; reference = the v2.2
  tree booted the same afternoon): qeval 72 / 75 (v2.2: 72 / 72 / 72, the same three misses); greedy-continuation
  KL vs v2.2 0.0031 (long prompts) / 0.0071 (short), A/A in one boot 0.0046 / 0.0035; T > 0 scan 0 of 35 garbled; `prefix_scan` PASS (drift 0.0 on
  4 full and 4 mid-document hits); certified head armed on 4 / 4 ranks; prompt-logprob requests answered 400 with the
  server staying up. sparkDash at the 2200 MHz cap: prose c1 89.7 (v2.2 reference the same afternoon: 89.3 / 87.7 / 89.3), code c1 131.9, structured 157.8, json 130.4; prefill 4.7k-5.7k tok/s.
- Measured and not adopted: an L2 prefetch of the fp32 hyper-connection weights at the front of each collective
  window (in-boot A/B +0.64 ms/step, -1.8 % tok/s: the stats kernels run on a side stream, off the critical path).
  Checked and not applicable on this stack: a host sync before the target launch (draft end to verify start is
  2-3 us), single-CTA fp32 GEMVs (none in the decode trace), large prefill gathers over RoCE (prefill gathers already
  use NCCL).

## 2026-09-25 v2.2

Decode-step overhead removed, all lossless: greedy output byte-identical to v2.1. Fresh-clone release gate,
raw output: [`docs/results/validation-20260925-v22.txt`](docs/results/validation-20260925-v22.txt).

- **`adapter/l2_prefetch.py`, `DSV41_L2_PREFETCH_WOA=1`, on.** The MXFP8 linear before `wo_a` opens an L2
  prefetch window for `wo_a` while the attention core runs: -0.4 ms/step. The other v3/v4 sub-gates (AHEAD,
  DRAFT, ENGRAM, LMHEAD, SKIP_N, WOB_MB, MOE) measured flat or slower and stay off. The ring package's
  collectives are hooked at both install sites.
- **`adapter/spec_sync_free.py`, `DSV41_SPEC_SYNC_FREE=all`, on.** The per-step rank-0 broadcasts of the draft
  token, the verify epilogue and verify_cap's length are dropped (their values are identical on every rank; the
  draft noise is a counter-based stream shared by the ranks). Audit mode: 0 rank mismatches over ~60k checks.
  In-boot A/B -0.53 to -0.59 ms/step; the reboot suite gave ~-0.2 ms.
- **`adapter/eager_glue.py`, `DSV41_EAGER_GLUE=all` (fence, stage, vcap, vcapk), on.** Fewer eager kernels
  between the draft and verify graphs, bit-identical: ~-0.1 to -0.2 ms/step. `tvglue` / `dglue` dropped after
  a fleet boot's check failed (stale `out_cache_loc` pointer in a glue graph).
- **Bit-exact bundle, `DSV41_SPLIT_COMPACT_GATHER=1`, on:** compact RoCE columns gather for the replicated
  splits (new `all_gather(columns=...)` kernel in `runtime/b12x`), the `wqkv_a` window GEMM
  (`DSV41_REPLICATED_SPLIT_WINDOW`, default on), and the b12x_next barrier single fill + Triton route planner
  under determinism (`SOURCE_PATCH` `da662ccc7e2372b5`). -0.40 ms/step prose, -0.45 ms code. 4-rank gather
  test PASS (M=6: compact 11.31 us vs 14.88 us for the old path).
- Includes the ring entry below (`DSV41_ROCE_RING`, #8, off by default).
- `DSV41_PREFILL_SP_FP8_MOE=1` (MXFP8 MoE-input gathers in prefill SP, b12x_next pre-quantized input): bit-exact
  on every check but flat on real-text prefill. In the code, off, not in the production line.
- In-boot A/B harness (`adapter/ab_variant.py`, `scripts/ab_inboot.py`, `scripts/repeat_sha.py`), test only, off
  unless `DSV41_AB_VARIANTS` >= 2.
- Fresh clone, sparkDash 1.8.8: prose c1 87.67 -> 89.76 (median of five, 88.97-90.21; second boot 89.04-90.42),
  code c1 124.76 -> 132.37 (median of three: 124.95 / 132.62 / 132.37), structured c1 156.9, json c1 124.3; c4 per stream
  prose 41.5, code 64.1, structured 70.2, json 78.7; c16 aggregate prose 340.9, code 436.4, structured 565.5,
  json 653. Decode step c1 prose 33.26 -> 31.24-32.5 ms, code 38.88 -> 37.4-37.84 ms. Varied prompts
  58.4 / 94.3 / 71.8 -> 61.1 / 99.0 / 74.7. Prefill 16k-128k 5734-5855, 262k 5286 (one pass; v2.1
  5793-5936, 5355); real text 4743-4946 from 15k to 123k. 1,011,084-token needle PASS in 324.3 s (head low-water 6,029 MiB). qeval not
  re-run (greedy output byte-identical to v2.1, which scores 72/75). Two boots, both healthy on the first try.
- Tested and not adopted: the L2 sub-gates above, fused MXFP8 quantization into q_norm / wo_a / hc (flat),
  layer-14 Engram lookup on a side stream during verify (~0), fused hc prefill kernel (slower), MoE restructures
  (the phases already run at 223 GB/s live), CPU pinning (flat). Details in
  [docs/history.md](docs/history.md#2026-09-25).

## 2026-09-25 (ring)

- **RoCEnante on a switchless ring, `DSV41_ROCE_RING=1`, off by default, research-only.** A four-node ring has no link between
  opposite nodes, so RoCEnante could not run there and the production line's collectives went through NCCL. The
  opposite-node path is now built in the neighbours' ConnectX-7 hardware with FujitsuPolycom/sparkring's
  `cx7_hairpin_diagonal` (commit `f16b5f4`: an RDMA-TX marker re-tags the opposite-node queue pairs' packets, a
  `skip_sw` tc rule on the neighbour restores and redirects them, no CPU or kernel forwarding), and
  `DSV41_ROCE_RING=1` makes the SG17 overlay load `runtime/b12x/b12x/comm/roce_ring`, sparkring's path-aware
  RoCEnante (two paths per peer over all four RDMA functions; origin and local changes in
  `runtime/b12x/roce_ring-provenance.json`). `scripts/ring_mesh/` inventories the nodes, reads the cabling,
  runs sparkring's planner and writes the per-node routes / rules / markers, the boot unit and the
  `EXTRA_CONTAINER_ENV` line with per-rank peer maps (`B12X_ROCE_PEER_HCA_MAPS`) for the TP rank order. On
  the ring, same day, one change at a time: NCCL -> RoCEnante decode step -5.2 % (51 qeval tasks, faster on 50),
  qeval 76.3 -> 79.5; with v2 on top qeval 81.8 / 84.7 / 85.9 over three runs (71-72/75), decode step within a few percent of the
  switched README, prefill 5-5.5k, phrase needle PASS at 1,030,651 tokens. Details:
  [switchless-ring.md](docs/switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths),
  raw output: [`docs/results/ring-mesh-20260925.txt`](docs/results/ring-mesh-20260925.txt).
- `roce_ring` keeps the SG17 proxy's `ROCE_IDLE_SPINS` (20,000,000; sparkring ships 200,000, at which the
  proxy slept in 27 % of samples during decode): c1 step 40.6 -> 37.1 ms, c2 -7 %, c4 -6 %.
- `hairpin_queue_size` 8192 on the fabric functions (`scripts/ring_mesh/hairpin.sh`, re-applied at boot): at
  the default 1024 the neighbour's forwarding queue overflows above ~100 KB per message; at 8192 all-reduces
  up to 480 KB run without a drop, so `SGLANG_ROCE_MAX_SIZE`/`DSV41_ROCE_GATHER` go to 256 KB (c2-c4 -1 %) with
  `B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0`.
- `runtime/roce_tp4_adapt.py`: the package is chosen at runtime from `DSV41_ROCE_RING`; the `ROCE single-rail`
  warning fires for one HCA only (it fired for any count but two). `adapter/l2_prefetch.py` and
  `adapter/sitecustomize.py` hook either package. `tests/test_roce_ring.py` (CPU) runs in the image build.
- Tested and not adopted: the scheduler pinned to the X925 cores (qeval -5.5 %), the RoCE proxy threads
  alone on dedicated X925 cores (within noise), the two-wave schedule once nothing drops.

## 2026-09-25

- **`DSV41_PREFILL_SP_FP8=1`, on:** the prefill attention-input gather of 17 of the 21 full-row layers carries
  the MXFP8 bytes and scales `wqkv_a` computes itself (about half the bytes; checked bit-identical per chunk
  size on every rank at runtime). Fresh clone: prefill 16k-262k 5214-5818 -> 5355-5936 (+2-2.6 %), real text
  +1-2 %; qeval 72/75; 1,011,084-token needle PASS (325 s).
- **`adapter/prefill_sp.py`, `DSV41_PREFILL_SP=1`, on.** Prefill sequence parallel: at chunks of >= 2048 rows the
  per-layer all-reduces become reduce-scatter + all-gather and the per-row work between them (hyper-connection
  mixing and norms, Engram `wkv` + gate) runs on each rank's quarter of the rows. sparkDash prefill 16k-262k
  4447-4870 -> 5214-5818 (+18-21 %), real text +15 %; decode untouched. The reduce-scatter adds the partials in
  a different order than the all-reduce (the same kind of change as RoCEnante); `DSV41_PREFILL_SP_EXACT=1`
  keeps the all-reduce and was checked bit-identical to the unsharded path op by op on the fleet (174 ops per
  4096-row chunk, 0 differing) for ~+10 %.
- **`adapter/l2_prefetch.py`, `DSV41_L2_PREFETCH=1`, on.** During each RoCE all-reduce / all-gather of a decode
  step a side-stream kernel prefetches the first 6 MB of the following weights into L2 (plans learned from the
  warm-up forwards, captured in the CUDA graphs; data untouched). ABAB on the fleet: decode step -1 ms at c1,
  sparkDash prose c1 84.5 -> 87.6, varied prompts +3 %, c8 +2 %, c16 flat.
- `adapter/draft_head_fp8.py`: 16/64-row tiles so the fp8 draft head is streamed once per call (it was re-read
  per 16-row tile and slower than bf16 from ~20 rows): +1-2 % at c4-c8, bit-identical logits; rows above 64 stay
  on the bf16 head as before.
- `adapter/moe_b12x_next.py`: `DSV41_MOE_B12X_NEXT_PLAN_TABLE` (per-capacity plan overrides); the default keeps
  today's plans: every alternative at 10-96 rows was measured and none beat b12x's default (the MoE already
  reads at the GB10 limit).
- Fresh clone of this state: prose c1 86.5, c16 342.7, code c1 122.6, structured 152.4; prefill 16k-262k
  5214-5818; qeval 72/75; phrase needle PASS up to 1,011,084 tokens (322 s, head low-water 7 GiB).
- Docs: the production line on a switchless ring (RoCEnante needs a path to every peer; rsync ran it on a ring
  with sparkring's opposite-node path).
- Tested and not adopted: batch-size-dependent verify threshold, locked GPU clocks, PDL on the dense GEMMs,
  a fused hc post/combine/stats prefill kernel (bit-identical but slower than the current kernels), MoE plan
  alternatives at 10-96 rows.

## 2026-09-24 (evening)

- **`EP_SIZE=1` with the routed MoE on b12x main, `DSV41_MOE_B12X_NEXT=1`, on.** Every rank holds a
  576-wide slice of all 384 experts, so no expert group waits for the other at the MoE all-reduce
  (all-reduce 5.9 -> ~2.6 ms per decode step at c1, the wait was ~3.5 ms of it). FlashInfer's MXFP4
  MoE cannot run 576-wide experts; `adapter/moe_b12x_next.py` runs them on b12x main
  (`runtime/b12x_next`, commit a7d7d29b, renamed beside the SG17 b12x that RoCEnante needs). Two plan
  fixes were needed: `dynamic` instead of `micro` at <= 8 rows (523 -> 349 us per MoE call) and 64-row
  tiles from 2048 rows at prefill (b12x pinned 576-wide experts to 16-row tiles; two-line patch,
  output bit-identical to the 16-row path). `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1`: fixed-order
  reduction instead of atomics, no measurable cost, greedy output bit-identical run to run.
  Fresh clone: sparkDash prose c1 74.8 -> 84.6, c4 143.9 -> 160.4, c16 319 -> 340; code c1 108 -> 121;
  structured 133 -> 146; json 104 -> 119; prefill 32k-262k 4206-4681 -> 4264-4870; qeval 72/75.
  Tested and not adopted: folding the fp8 shared expert into the b12x call (b12x has no fp8 weight
  source; a byte-equivalent proxy was slower than the side stream), the CPU deep-idle states and
  batched AIO Engram lookups (flat once warm), a warm-started Engram row cache (flat), and
  `--enable-deepseek-v4-fp4-indexer` (a no-op on V4.1). `DSV41_VERIFY_CAP=conf:0.1` re-checked at EP1
  against 0.05 / 0.15 / 0.2: kept.
- **`adapter/fast_load.py`:** at EP1 each rank reads only its slice of every expert
  (`DSV41_FAST_LOAD_TP_SLICE=auto`; 144.5 GiB read, 83 GB kept) into 256 MiB pinned slabs (330
  allocations instead of ~94k), which gave back ~0.7M tokens of KV pool. Engine start ~170 s.
- **`adapter/hc_fused.py`, `DSV41_HC_FUSED=1`:** prefill-size hyper-connection mix statistics in one
  pass over K, bit-identical to the stock kernels (checked on the first live call): ~1.46 -> ~0.83 ms
  per call at 4096 rows.

## 2026-09-24

- README decode table: all four sparkDash prompt types at c1-c16 on one boot
  (`docs/results/sweep-20260924-types.txt`). `--enable-deepseek-v4-fp4-indexer` measured as a no-op on
  V4.1 (four-boot off/on/off/on A/B: identical greedy output, speed and prefill; the model's only
  indexer pool is fp4 by construction); the README no longer lists it as a kernel change.
- README reduced to the current state; every dated table, note and rejected experiment moved
  unchanged to `docs/history.md`, the adapter reference to `docs/adapters.md`, the optional images and
  the switchless ring to `docs/optional-setups.md`. Current numbers re-measured on sparkDash 1.8.8:
  prose c1 74.4 / c16 319.4, code c1 108.1, structured 133.4, json 104.3, prefill 262k 4206
  (`docs/results/prodbench-20260924-current.txt`).
- **`adapter/replicated_split.py`, `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv`, on.** `ReplicatedLinear`
  layers made every rank stream the whole weight for the same output; the Engram `wkv` (183 MB) alone
  cost 2 x 756 us per c1 step. Each rank now runs the same quantized linear on its 128-row weight
  tiles and the columns are all-gathered; per layer, and only if bit-identical to the stock layer on
  every rank (checked at boot). Step probe prose 38.2 -> 36.8 ms; sparkDash prose c1 71.8 -> 74.7;
  greedy outputs identical.
- **`adapter/draft_main_proj.py`, `DSV41_DRAFT_MAIN_PROJ_SPLIT=1`, on.** The draft's replicated
  `main_proj` as a 1/4 fp8 column shard (exact weight bytes) + all-gather: -0.2..-0.4 ms per step.
- **`adapter/roce_gather.py`, `DSV41_ROCE_GATHER=2097152`, on.** TP all-gathers up to 2 MiB per rank
  over the RoCEnante one-shot kernel (the overlay built it with gathers disabled): draft logits
  gathers 600 -> 400 us per step.
- **`adapter/router_live.py`, `DSV41_ROUTER_LIVE=1`, on.** The dead-row remap of the verify cap folded
  into the router kernel (dead rows load the anchor row's scores): 40 launches per step fewer, bit-identical.
  Step probe prose 36.76 -> 36.65 ms, code 43.55 -> 43.46 ms; 45 varied prompts +0.3 %.
  Fresh clone of this state on all four nodes with the `.env.tp4.example` production line: in-image
  tests pass, greedy outputs identical, sparkDash 1.8.8 prose c1 74.9 (74.93 / 74.93 / 74.81), code c1
  108 / 113, qeval 72 of 75 (the usual three).
- Measured and kept off: a dynamic shared-expert split between the EP groups (each rank holds half the
  shared expert, the group with fewer routed experts takes more columns per step, range-limited Triton
  kernels): step -1.7 % prose / -2.4 % code and +2-4 % on structured and sampled traffic, but not
  bit-identical and -15 % on sparkDash's prose prompt (the greedy text diverges into a continuation the
  draft predicts worse). Not in the repository.
- `adapter/autotune_keep.py`: a cache whose launch fingerprint does not match is now deleted, not just
  reported as non-matching (with every rank reporting "" the stock gate agreed and FlashInfer loaded
  each rank's stale file anyway; found independently in MiaAI-Lab's port). Decode-side switches that
  change neither the tuned MoE shapes nor their kernels (draft head fp8, Engram prefetch, the
  replicated/draft splits, RoCE gathers, the DRM row cache, the logging switches) are volatile, so
  toggling one reuses the tactics instead of re-drawing them for every A/B boot.
  `SGLANG_RUN_ID` (a per-boot timestamp the engine sets in its own processes) is excluded too: with it
  no fingerprint ever matched and every boot re-tuned (found by MiaAI-Lab).
- `DSV41_DRAFT_TAU` 0.8 -> 0.7: sampled thinking traffic (T=1, top_p 0.95) +1.8 % and +2.0 % on two
  disjoint prompt sets; greedy unaffected. The verify-length threshold was re-checked on the same
  sampled traffic (0.07 / 0.1 / 0.15: 61.3 / 61.2 / 61.3 tok/s) and on 45 varied greedy prompts (flat):
  0.1 stays.
- Production rows re-measured at the uncapped GPU clock: prose c1 73.9 (72.2 / 73.9 / 74.4), c16 319,
  code c1 119, structured 131.5.
- Measured and not adopted: splitting the indexer `wq_b` / compressor `wkv_gate` (no gain); a
  logistic verify-length policy on draft-distribution features (+0.9 % modelled, flat live); Markov W2
  on a Triton GEMV (slower than cuBLAS at 19 us); static expert re-placement between the EP groups
  was not built: routing counts show EP group 1 streaming 3.8 % more experts, but the per-layer
  imbalance is mostly step-to-step noise (estimate ~0.2 ms/step recoverable, untested).

## 2026-09-23 (evening)

- **`adapter/verify_cap.py`, `DSV41_VERIFY_CAP=conf:0.1`, on.** Adaptive verify length without changing
  the verify layout: dead rows reuse the anchor row's experts (in-graph Triton remap), acceptance is
  capped through the engine's cutoff, the draft confidence head (force-built in static mode) picks the
  length as a stopping rule. sparkDash prose c1 66.0 -> 69.7, prose c4 125.4 -> 133.5, sampled
  thinking traffic +5 %, code/structured flat; greedy outputs identical; toy-LM chi-square test in the
  image build.
- **`adapter/autotune_keep.py`, `DSV41_AUTOTUNE_KEEP=1`, on.** The FlashInfer MoE autotune cache is kept
  across boots under EP (sglang#40320); 0 re-tunes from the second boot, was 26.
- `adapter/block_verify.py`: `live=` for shortened blocks.
- Fixed verify caps (1..4) measured and rejected: prose -8..-15 %, code up to -40 %.
- **`DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1`, on.** wo_a at 9-192 verify/draft rows from the fp8
  twin (verify step c4 82.1 -> 78.3 ms, c8 120.0 -> 118.1 ms), and the bf16 copy released (722 MB per
  rank) with bit-identical prefill through per-call dequantization.
- `verify_cap`: live lengths broadcast from rank 0 each step (the confidence head reads all-reduced
  activations, which may differ in the last bits between ranks); no measurable cost.
- Optional `DSV41_ENGRAM_DRM_NODE`: one Engram layer's row cache in the GB10 display reservation
  (DRM dumb buffer, outside MemAvailable), ~1.8 GiB runtime headroom per node, no speed change;
  host setup in docs/display-reserve.md.

## 2026-09-23

- **`adapter/wo_a_w8.py`, `DSV41_WO_A_W8=1`, on.** The verify/draft `wo_a` (2–8 rows) reads the
  checkpoint's fp8 bytes instead of the bf16 copy made at load: exact fp8 twins for 43 of 43
  layers, same Triton tiling with the bf16 tile rebuilt in registers. 43 layers at M=6
  3.43 → 2.20 ms, decode step at c1 52.9 → 51.7 ms (bench of 12 × 800-token answers, warm).
  MXFP8-epilogue path bitwise identical; plain bf16 path differs in fp32 accumulation order only.
  Quality gate 72/75 twice against 73/75: primary code+reason+math 53/55 both, json 15/15; the one
  flip is `prose_p2` coming in at 88 words against a 100-word minimum.
- **`adapter/draft_tau.py`, `DSV41_DRAFT_TAU=0.8`, on.** Draft proposal temperature for sampled
  requests; exact by construction. +1.2 % accepted tokens per step at T=1 / top_p=0.95 offline
  (3.242 → 3.280 on held-out traffic); 0.7–0.8 is the optimum.
- **`adapter/draft_head_fp8.py`, `DSV41_DRAFT_HEAD_FP8=1`, on.** The draft's block logits come from an
  fp8 copy of the shared LM head (target head untouched): 1405 → 721 us, live step 51.7 → 50.9 ms.
- **`adapter/block_verify.py`, `DSV41_BLOCK_VERIFY=1`, on.** Block verification (Sun et al., ICLR 2025)
  for sampled rows. Lossless (chi-square test on a toy LM in the image build). Offline on
  engine-exact p and q: +1.8 % tokens/step on prose (2.705 -> 2.754), +2.5 % on coding with thinking
  (2.927 -> 2.999); live, the engine accepted 1.766 drafts/step on steps where token verification
  expects 1.705.
- **`adapter/folded_result_fence.py`, `DSV41_FOLDED_FENCE=1`, on.** Folded (all-greedy) verify results are
  cloned before the overlapped D2H copy (sglang#40919); sampled batches were never exposed.
- `adapter/draft_capture.py` schema 4: also the committed tokens, the drafted tokens and the draft's
  top-64 logits with its full-vocab normaliser, so verification rules can be evaluated offline.
- sparkDash after all of the above: prose c1 65.3 (was 61.0), code c1 113.8, structured 124.7.
- `sitecustomize`: `dspark_verify` was missing from the hooked modules, so `DSV41_DRAFT_CAPTURE`
  never installed; added together with `dspark_draft_sampler`.
- Tested and not adopted: draft fine-tune on own traffic, multi-pass drafting, expert-sharing
  routing, n-gram lookup, relaxed acceptance, split-K MXFP8 and hyper-connection kernel variants
  (numbers in the README).

## 2026-09-18

- Tested and not adopted: Markov W2 unsharded (+0.6 ms/step), fast-load knobs 12 GB / 2 threads
  (−2 s to ready), parallel Engram misses in the row store (no change).
- **Engram prefetch on a side stream (`adapter/engram_prefetch.py`, `DSV41_ENGRAM_PREFETCH=1`, on).**
  The NVMe misses of the Engram row lookup no longer stall the graph before each gather.
  Same-image A/B with identical prompts: step 51.7 → 49.5 ms, GPU idle 3.0 → 0.85 ms/step,
  English essay c1 42.6 → 45.1 tok/s, sparkDash prose c1 57–59 → 60–61, prose c4 120 → 124,
  structured 116 → 121; rows bit-identical in check mode.
- `--sleep-on-idle` on (RustyAiLab's suggestion in Mia #16): idle head scheduler CPU 47 % → 14 %,
  workers 5 %; first response after idle and decode unchanged.
- `docs/upstream-watch.md`: re-checked every pinned piece (SGLang dsv4.1 head, #39187/#39704,
  kernels, image, b12x, checkpoint); the SM121 candidate-indexer blocker still holds, pins stay.
- **Faster boot: `adapter/fast_load.py`, gate `DSV41_FAST_LOAD=1`, on in production.** This
  rank's checkpoint tensors are read by a 16-thread `pread` pool into pinned host memory instead
  of being page-faulted through the loader's mmap at 0.5 GB/s; the model's async copies are paced
  to a byte budget; the DSpark draft load opens the 3 shards that hold `mtp.*` instead of 48.
  Engine start 343–354 s → 111–129 s, values bitwise identical, decode/prefill/needle unchanged.
  Trade-off: the KV pool is 3–13 % smaller (6.71–7.27 M vs 7.47–7.82 M tokens) and varies more
  between boots. Profile, dead ends and memory accounting: `docs/fast-load.md`.
- Production decode/prefill rows in the README re-measured on the final boot (fast load + Engram
  prefetch): prose c1 61.0 / c16 292.9, code c1 113.3 / c16 882.4, structured 124.1.
- Merged from Saolence: #5 (worker containers mount the same NCCL as the head, worker preflight
  actually runs), #6 (anchored rsync excludes in `build`, `BUILD_DOCKERFILE`, `BUILD_ARGS`,
  `PIP_INDEX` mirror knob), #2 (second-plane addressing for the ring, docs). `PIP_INDEX` also
  added to `Dockerfile.canary-roce`. Noted that torch loads the pip NCCL through its RPATH, so
  only `NCCL_OVERLAY_PIP=1` replaces it.
- New: `scripts/loadprof.sh` (py-spy + diskstats sampler for the load phase),
  `tests/test_fast_load_pacing.py` (in every image build), `tests/test_fast_load_checkpoint.py`.

## 2026-09-17

- **Production image = `Dockerfile.canary-roce`**: upstream `dsv4.1` branch at `f80c91a4b` +
  RoCEnante overlay (rhys101's SG17, adapted to TP4, both rails, 2 MiB route) + all adapters.
  Prose c1 57.0 / c16 290.7, code c1 107.0 / c16 860.8, structured 115.4.
- **Prefill TP split** (rhys101's SG18) combined with the sglang#39187 backport in
  `adapter/indexer_chunked_v3.py`: sparkDash 128k 4364, 262k 3893; 985k needle in 585 s with
  5 GiB head low-water.
- Quality gate: 75-task paired evaluation, production 72/75 vs base 71/75 (p = 1.000);
  `scripts/qeval.py`.
- `--enable-cache-report` on; `SGLANG_RUST_BUILD_MODE=never` (the branch's cargo probe can hang
  the head before the HTTP server starts).
- Tested and not adopted: sglang#39704 on the pinned branch (±2 %), newer `dsv4.1` heads on
  SM12x (candidate indexer needs the DeepGEMM paged path), chunk 8192, split threshold 16k.
- Merged from Saolence: #1, switchless four-Spark ring as an opt-in (`NCCL_SWITCHLESS_RING_ONLY`,
  `NFS_SHARE=0`), 44 offline tests; the ring decode table is a same-stack reference, not a
  speedup.
- Repository published: fork of MiaAI-Lab's recipe with history, tuned TP4 profile (EP2, Engram
  row cache, 16 slots, chunk 4096 + sglang#39187 backport, shared-expert K pad, optional canary
  image); verified from a fresh clone including the 985k-token needle.

## 2026-09-16

- Maintenance window (`scripts/window-20260916.sh`, `docs/window-20260916.md`):
  `MAX_RUNNING_REQUESTS` 8 → 16 (adds the c16 tier), sglang#39187 backport with
  `CHUNKED_PREFILL_SIZE` 1024 → 4096 and the adaptive chunk sizer off, upstream `dsv4.1`
  canary image with folded sampling forced (`SGLANG_DSPARK_FOLDED_SAMPLING=2`).

## Before the fork

- MiaAI-Lab's `DeepSeek-v4.1-Flash-DGX-Sparks` up to 2026-09-15 (#18: thinking alias, output
  cap, loop abort, DSpark k=3). See `docs/README-upstream.md` and the Credits section.
