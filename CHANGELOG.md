# Changelog

Newest first. Each entry says what changed in the production stack and what was measured; the
raw results live under `docs/results/`.

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
