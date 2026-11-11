# Changelog

## v2.7.4 (candidate, release gate pending)

v2.7.3 plus lossless changes in the production profile (`DSV41_DENSE_TILE_TABLE=1 DSV41_PREFILL_MOE_FOLD=1
DSV41_WO_A_CFG_TABLE=1` appended
to `EXTRA_CONTAINER_ENV`). The image stack (SGLang tree, FlashInfer 0.7.2, NCCL 2.30.7 with #2393), the KV pool pin
(8,541,952 tokens at fraction 0.85), the launch line, `--swa-prefix-tails 384` and every v2.7.3 key are unchanged; the
image's `runtime/b12x_next` gains one kernel mode (below). Each change was measured on the four Sparks against the
v2.7.3 profile in the same boot (in-boot A/B with two control arms, the official 1017 checkpoint); the release gate's
numbers are TBD.

- **Decode: the narrow dense GEMMs on the measured b12x tile (`DSV41_DENSE_TILE_TABLE=1`, `adapter/dense_tile_table.py`).**
  b12x's default plan runs every N x 5120 MXFP8 weight on the M-independent (64, 64) tile. A single-GB10 sweep of every
  tile / tile_k / stage-cap config against the production plan (two activations per cell, whole-output byte compare)
  found all 24 configs bitwise and the (16, 64) tile at M <= 16, (32, 64) at 17..32 rows about 2x faster per call with
  warm L2 for `qa_kv` (512 x 5120 per rank, 17.4 -> 11.1 us at M 6), the shared-expert FC1 (1152 x 5120, 29.2 -> 13.1 us)
  and the draft `wkv` (1536 x 5120, 37.9 -> 15.7 us). `adapter/mxfp8_b12x.py` dispatches those keys to the same b12x
  runner with an explicit tactic; the first eager call of every (shape, M) runs the serving call and the tile on the same
  inputs and keeps the tile only if the output bytes are equal on every rank (MIN over TP); a default plan other than the
  measured one, a mismatch or a disagreement keeps the serving call for that key, and an unchecked key inside a graph
  capture takes the serving call. Measured (tier-10 window S2, 8 rounds, A / X / A'): c1 greedy step **-0.132 ms**
  paired per prompt (CI95 -0.265 to -0.0003); speed c1 +0.08 %, c1 sampled +1.25 %, c4 +0.62 %, c8 +0.33 %, c16 -0.99 %
  (c8 / c16: the table is inert above 32 rows; reported only); head memory 13.5 GiB at READY. Exactness: qeval 75 / 75
  T = 0 outputs equal to both controls (0 noisy tasks), 50 c1 greedy texts on the stable prompts equal. Prefill rows and
  every other shape are untouched.
- **Prefill: the shared-expert add folded into the top-k sum (`DSV41_PREFILL_MOE_FOLD=1`, `adapter/prefill_moe_fold.py`).**
  A prefill MoE call (eager, >= 2048 rows, b12x_next deterministic routed experts, shared experts first, routed scale
  folded into top-k) wrote the routed top-k sum into a new tensor and added it to the shared experts' output in a
  separate pass (168 MB in, 84 MB out per 8192-row layer). b12x_next's top-k sum kernel gains an accumulate-into-output
  mode (`scripts/b12x_next-topk-sum-accumulate.patch`: `out = bf16(float(out) + float(bf16(sum)))`, `add.rn.f32`
  without ftz, torch's `out.add_(routed)` bit for bit; its own op, the stock op's schema unchanged; `SOURCE_PATCH`
  a76ad82d137d9c5f), compiled and zero-checked at load per geometry, and the add site returns that tensor. The first call
  of every plan capacity runs both ways, compares bit for bit and agrees over TP before any fold (that call returns the
  stock result); a difference, an exception, drifted engine sources (sha256 pins), a missing accumulate variant or a
  broken return chain keep the stock add. Measured (tier-13 window P, 4 rounds, carried on review): cold 131K first
  token **-0.96 %** in the exact phase (19.67 s, n 6, against 19.86 s for the controls, n 12; every X sample below every
  A sample), servebench cold 131K +0.81 / +1.08 / +0.83 % in three rounds (the fourth drifted with A), 32K +0.86 %;
  memory flat (low-water -0.003 GiB), decode c1 -0.49 % / c4 +0.27 % (inside the controls' own 2.0 / 1.4 % A / A).
  Boot check bit-identical at capacities 8192 and 4096 on all four ranks; T = 0 states inside the stock code's own
  (the one new 131K state was also produced by an arm without the fold).
- **Decode: `wo_a`'s launch table (`DSV41_WO_A_CFG_TABLE=1`, `adapter/wo_a_cfg_table.py`).** An exact-M table of
  `wo_a`'s 8-bit twin GEMM launches from a single-GB10 sweep (tier-12o window B: every configuration built from the
  image's own kernel text, bytewise against production on two inputs, eager and graph replay): M = 6 -> `p:64:2:2:b`
  (production's 16x64 tile, grid, BK 128, SPLIT_K 8 and k order; 2 warps, 2 stages, one exp2 per 32x32 scale block:
  warm 28.14 -> 11.77 us, cold -2.45 us per call), M = 96 -> `mid:1:32:64:2:3` (production's split_k 1 / BM 32, 2 warps
  instead of 4: warm -19.18 us, cold -0.96 us). Every other M (M = 24 included) stays on production's launch.
  `_partial_w8` / `wo_a_mid` ask the table first; the production bodies move unchanged to `_partial_w8_stock` /
  `_wo_a_mid_stock`. A call takes the table only with the key on, qx off, no `_MID_CFG` drift, a passed load
  self-check (both launches on the first twinned layer's real bytes and two seeded inputs, a perturbed-exponent negative
  control that must differ, one TP MIN all-reduce for all entries) and a passed first eager call of its key (uint8
  compare, TP MIN); unchecked keys inside a capture take production's launch. In-boot A/B against the v2.7.3 profile
  (tier-15 window w): TBD. Image tests: `test_wo_a_cfg_table.py` (13; interpreter bitwise and, with
  `TRITON_INTERPRET=0`, the sm_121 PTX lowering).
- Receipts (every rank): `DSV41_DENSE_TILE_TABLE armed: 3 shapes ...` and one `DSV41 dense tile table N=... K=... M=...
  tile=...: ...` line per key; `[moe_b12x_next] accumulate top-k sum ready (capacity ..., zero rows -> zero output)`,
  `DSV41_PREFILL_MOE_FOLD installed: ...`, `DSV41_PREFILL_MOE_FOLD capacity <c> (M=<m>): fold vs stock add ... -> ON`
  per capacity, `DSV41_PREFILL_MOE_FOLD calls rank <r>: fold <n> check <c> stock <s> errors <e>` at the first fold and
  every 4096 calls; `[wo_a_cfg_table]` armed, load self-check and first-eager-call lines.
- Image tests: `test_dense_tile_table.py` (7: identical pair, mismatch incl. -0.0, plan drift, capture guard, M buckets,
  TP disagreement, gate off per call) and `test_prefill_moe_fold.py` (22: the fold's arithmetic against torch's bf16 add
  bit for bit over a stride of all bf16 pairs, the frame logic, the runtime patch on disk and its bookkeeping, the real
  SGLang MoE forward with a CPU stand-in for the b12x launches) join the `Dockerfile.canary-roce` chain; the
  engine-pin probe checks the three fold wrappers are installed and unwrap to the engine functions; the profile-order
  tests check the new tail.
- Not taken from the same test branches: megakernel stage 3 (the b12x dense B-prologue and the early quant trigger,
  closed), the tier-10 in-boot A/B graph sets (test only) and the prefill hc early-stats rider (closed: slower at 131K
  and a -6.6 GiB transient).
- Rollback, config only: unset `DSV41_DENSE_TILE_TABLE`, `DSV41_PREFILL_MOE_FOLD` and `DSV41_WO_A_CFG_TABLE` (= the v2.7.3
  profile; the adapters do nothing unset, and the patched b12x_next kernel is only selected under the fold key).

## 2026-11-10 v2.7.3

v2.7.2 plus two lossless prefill changes and one lossless decode change in the production profile
(`DSV41_PREFILL_SP_FP8_MOE=1 DSV41_PREFILL_GEMM_TACTIC=1`; `DSV41_L2_PREFETCH_WOA_MB=6 DSV41_L2_PREFETCH_WOB_MB=0`
(were 0 / 6) `DSV41_L2_PREFETCH_WOB2_MB=10 DSV41_WO_A_W8_EVICT_FIRST=1`: four keys appended to `EXTRA_CONTAINER_ENV`,
two values changed in place). The image stack (SGLang tree, FlashInfer 0.7.2, NCCL 2.30.7 with #2393), the KV pool pin
(8,541,952 tokens at fraction 0.85), the launch line, `--swa-prefix-tails 384` and every other v2.7.2 key are
unchanged. Measured on the four Sparks against the v2.7.2 profile in
the same boot (in-boot A/B with two control arms, the official 1017 checkpoint, 2026-11-10), then in the release gate
below (official 1017 checkpoint). In production on the four Sparks since 2026-11-10 18:00 (abliterated checkpoint).

- **Prefill: the MoE input is gathered as MXFP8 (`DSV41_PREFILL_SP_FP8_MOE=1`, `adapter/prefill_sp.py` stage 2b).**
  Prefill sequence parallel all-gathers each full-row layer's MoE input as bf16 rows, and b12x then quantizes them to
  MXFP8 inside the kernel. Stage 2b quantizes on each rank's quarter of the rows, gathers the MXFP8 bytes and UE8M0
  scales (half the bytes), runs the router and top-k on the shard and hands b12x the gathered input pre-quantized. The
  first chunk of every row count runs both paths and keeps the MXFP8 gather only if the routing, the gathered bytes
  (against b12x's own in-kernel quantization) and the MoE output are bit-identical on every rank (MIN over TP); a row
  with a FlashInfer scale-byte-0 block keeps the bf16 gather for that layer. Measured (five rounds, servebench cold
  prefill): 131K **+2.52 % [2.40, 2.64]** (A / A' 0.12 %), 32K +0.12 % (four of five rounds +0.4 to +1.4 %, one
  transient round -3.2 % shared with another arm); cold first token of a flushed 131K document -2.4 % and of a 32K one
  -2.2 % (all three X runs below all six control runs), about 30 ms per 8192-row chunk; trace: -1.7 ms of non-collective
  kernel time per full-row layer. Exactness: the 32K first token's top-20 logprobs bit-identical to the controls (3 of
  3); at 131K, where the controls themselves vary run to run, every X state is a control's or stock code's exact state
  or inside the controls' spread; the 64-token continuations equal a control's text. Head memory unchanged (11.98
  against 11.98 / 11.99 GiB). It existed since 2026-09 (off by default); it is in the production profile now.
- **Prefill: eight dense GEMM shapes on FlashInfer's CUTLASS SM120 MXFP8 tactic 5 (`DSV41_PREFILL_GEMM_TACTIC=1`,
  `adapter/prefill_gemm_tactic.py`).** The prefill dense MXFP8 GEMMs ran on FlashInfer's b12x kernel with its pinned
  default plan. A single-GB10 bench of the serving call against every alternative at the prefill row counts found
  FlashInfer's CUTLASS SM120 block-scaled MXFP8 GEMM, tactic 5 of `mm_mxfp8(backend="cutlass")` (CTA 128x128x128,
  swap_ab), returning the same bytes and faster: `wo_b` 1.28x and `q_b` 1.12x at 8192 rows, `wqkv_a` 1.33x and the
  shared FC1 1.37x at the row split's 4096-row blocks, the indexer `q_b` 1.10x. Only an exact (N, K, M) table entry
  switches (decode, verify and short extends never match). After the target model loads, every rank runs every table
  shape both ways on three synthetic operand sets quantized as production quantizes activations (gauss, mixed scales,
  mirrored K halves), compares the bf16 outputs byte for byte (tactic 5 twice), and the ranks agree in one MIN
  collective; a difference, an exception, FlashInfer sources other than the pinned 0.7.2 ones or no boot check keep
  the serving call. Measured on top of stage 2b (servebench cold prefill): 131K **+1.90 % [1.20, 2.60]**, 32K +1.79 %
  [0.45, 3.12] (A / A' 0.04 %); boot check 8 / 8 shapes on every rank (1.2 s); first tokens and the 131K continuations
  inside the controls' states; head memory +0.005 GiB.
- **Decode: a second `wo_b` L2 window at the `wo_a` call (`DSV41_L2_PREFETCH_WOA_MB=6 DSV41_L2_PREFETCH_WOB_MB=0
  DSV41_L2_PREFETCH_WOB2_MB=10 DSV41_WO_A_W8_EVICT_FIRST=1`, `adapter/l2_prefetch.py` v6 and `adapter/wo_a_w8.py`).**
  Since v2.7.1 the attention-core L2 window (after `wq_b`) warmed 6 MB of `wo_b` and left `wo_a` cold. The new layout
  warms 6 MB of `wo_a` there again and forks a second window ("W2") at the `wo_a` call itself, on the second side
  stream right before the `wo_a` kernel launches: it prefetches the `wo_b` bytes (weight, then scales) the first window
  did not take, up to 10 MB, while `wo_a` runs from L2 and DRAM is mostly idle; `wo_a`'s single-use FP8 twin loads
  carry an L2 evict-first policy (`createpolicy.fractional.L2::evict_first`, checked from the PTX). Cache hints only:
  values, kernels and their order are unchanged, outputs bit-identical. Measured in one boot against the v2.7.2 profile
  (tier-8 window M, two control arms, four rounds): c1 greedy step **-0.205 ms** (paired per prompt on the 25 prompts
  whose work is identical in every run, CI95 -0.302 to -0.108; the controls' A' - A -0.032 ms), c1 sampled -0.30 ms
  (CI95 -0.39 to -0.21), c4 **+1.09 %** (CI95 +0.16 to +2.01; per-step time -1.04 %); c8 / c16 per-step time -0.6 /
  -0.5 % (inside the noise; tok/s there carries ~1.3 % of T = 0 acceptance noise between identical controls). Trace,
  rank 0, c1: `wo_a` 44.6 -> 34.7 us per call, block span 109.4 -> 101.5 us per layer, the post-attention all-reduce
  not slower; `wo_b` 23.9 -> 25.2 us (this layout drops the 6 MB `wo_b` window of the attention core). qeval 75 / 75
  T = 0 outputs byte-identical to the controls'; head low-water during the speedbench 14.97 GiB (control 14.77). Evict-first alone was flat (+0.010
  ms, CI95 -0.10 to +0.13); it is kept because the measured layout carries it. Receipts: the bracket line names
  `W2 wo_b windows at the wo_a call (10.0 MB)`, every learned plan line ends ` wob2=<windows>/<MB> MB`, every capture
  line ` wob2=<forks>`.
- **`adapter/l2_prefetch.py`: every wrapper keeps `__wrapped__`** (the MXFP8 linear, the router GEMM, RoCE all-reduce /
  all-gather, the model / LM-head / DSpark draft forwards, the Engram row join and the `wo_a` recorder), so the
  adapters that hash-pin engine sources through `inspect.unwrap` (prefill_sp pins `DeepseekV4Model.forward`) see the
  stock source after the bracket. Behaviour unchanged; the tier-8 window measured the stage-1 layout on this code.
- **Receipts:** every rank logs its own stage-2b self-check per chunk row count (`... (M=<rows>) rank <r>: local ...
  -> agreed ON`) and its stage-2b call counts (`DSV41_PREFILL_SP fp8 MoE calls rank <r>: mxfp8 <n> bf16 <m> danger
  <d>`, first call and every 256); every rank logs its own GEMM-tactic boot-check count and its tactic call counts
  (`DSV41_PREFILL_GEMM_TACTIC calls rank <r>: tactic <n> ...`, first call and every 4096). Host prints only.
- **In-boot A/B registry:** `DSV41_PREFILL_SP_FP8_MOE` is a union gate read per MoE call and
  `DSV41_PREFILL_GEMM_TACTIC` a runtime key (`adapter/ab_variant.py`, test only); `prefill_sp`, `moe_b12x_next` and
  `prefill_gemm_tactic` are hashed sources.
- **Image tests:** the canary-roce image runs `tests/test_prefill_gemm_tactic.py`,
  `tests/test_prefill_sp_fp8_moe_ab.py`, `tests/test_l2_prefetch_wob2.py` (the W2 plan, the fork, and
  `test_every_wrapper_unwraps_to_engine`) and `tests/test_wo_a_w8_evict_first.py` (interpreter bitwise and the PTX hint
  placement) at build; `tests/test_adapter_engine_pins.py` checks that the tactic's `ModelRunner.load_model` hook, the
  l2_prefetch bracket and the `wo_a` recorder unwrap to the engine, and re-runs prefill_sp's drift guard after every
  adapter installed.
- Rollback, config only: unset `DSV41_PREFILL_SP_FP8_MOE`, `DSV41_PREFILL_GEMM_TACTIC`, `DSV41_L2_PREFETCH_WOB2_MB` and
  `DSV41_WO_A_W8_EVICT_FIRST` and set `DSV41_L2_PREFETCH_WOA_MB=0 DSV41_L2_PREFETCH_WOB_MB=6` (= the v2.7.2 profile on
  the same image); the decode change alone reverts with the last four.
- Open follow-up (not in this release): an attribution boot of `DSV41_L2_PREFETCH_WOA_MB=6 DSV41_L2_PREFETCH_WOB_MB=6`
  plus W2 on the `wo_b` remainder, which would keep the attention core's `wo_b` window that this layout drops (`wo_b`
  +1.4 us per call at c1), and of the layout without evict-first.
- **Credits:** the GEMM tactic is FlashInfer's CUTLASS SM120 block-scaled MXFP8 GEMM (built on NVIDIA CUTLASS); we
  only select it at measured shapes. The per-kernel comparison with Christopher Owen's spark-ds41f stack (its TileLang
  `mxfp8_gemm`, bitwise equal to b12x and a comparison arm of the bench) led us to measure a second kernel at the
  prefill shapes; no spark-ds41f code is included. Stage 2b is ours. The W2 window and the evict-first twin loads are
  ours (megakernel stage 1); that `wo_b` was worth warming at all came from v2.7.1's per-kernel comparison with
  Christopher Owen's spark-ds41f stack (credited there).

Release gate (run `run-20261110-114511-236731`, 2026-11-10, 362 min including the restore of production): v2.7.2 /
v2.7.3 / v2.7.2, three fresh boots in that order on the same four Sparks, official 1017 `1bae0e1` on all three, GPU
clock capped at 2200 MHz. The controls are production v2.7.2's image and profile; the v2.7.3 boot ran a gate image of
`9413b93` (production v2.7.2's image with this release's changed files added) with the six profile keys above. KV pool
8,541,952 tokens on all three boots. Verdict: **FAIL -> NO-RELEASE on one rule (exactness at 131K), accepted on review
as release-equivalent** (below); every other rule PASS (decode, sampled, load, decode step time, c1 step, cold prefill,
memory, qeval, idle CPU, idle first token, inter-token gaps, long-prompt decode, 32K exactness), 0 safety, 0 invalid,
restore PASS.
- **Prefill, cold** (median of three flushed prompts; v2.7.2 / v2.7.3 / v2.7.2): 64K 6,469 / 6,726 / 6,481 tok/s
  (x1.039 of the controls' mean), 131K 6,554 / 6,851 / 6,584 (x1.043), 1M (969K tokens) 5,431 / 5,697 / 5,545 (x1.038).
  A cold 388,000-token prompt (solo, median of three): first token 60.00 / 57.28 / 59.81 s (x1.048 / x1.044 the speed
  of the two controls). The flushed-document first tokens of the exactness probes: 32K 5.30 / 5.06 / 5.27 s, 131K
  20.84 / 19.87 / 20.76 s, 1M 188.0 / 181.4 / 186.9 s (x1.043 / x1.047 / x1.033 of the controls' mean speed).
- **Decode** (speedbench 0.3 full, mean of two runs; v2.7.2 / v2.7.3 / v2.7.2): greedy c1 all-prompt geomean 89.59 /
  90.41 / 90.00 tok/s (runs 89.35 / 89.83, 89.84 / 90.98, 89.33 / 90.66) and prose 66.68 / 67.22 / 66.71; c4 135.97 /
  136.65 / 135.66, c8 171.39 / 172.20 / 171.99 and c16 215.12 / 213.62 / 216.31 aggregate tok/s. Sampled (T 0.7, top-p
  0.95, one run per cell): c1 87.20 / 86.96 / 88.38 and prose 63.55 / 64.69 / 65.79 tok/s, c4 133.19 / 134.74 / 133.86.
  Speedbench load c16 228.29 / 226.80 / 227.57 aggregate tok/s (median TTFT 473 / 466 / 464 ms, p99 inter-token gap
  286 / 286 / 290 ms). v2.7.3 against the mean of the controls: greedy c1 x1.007, c4 x1.006, c8 x1.003, c16 x0.990;
  sampled c1 x0.990, c4 x1.009; load c16 x0.995; all inside the gate's bands. New rules in this gate: the decode step
  time at c8 / c16 (127.23 ms against 127.05 / 127.41, 195.14 against 194.26 / 194.72: at parity; the c16 tok/s follows
  the T = 0 text's acceptance) and the c1 step paired per prompt on identical work (-0.27 ms, CI95 -0.29 to -0.25).
- **Correctness.** qeval 71 / 71 / 71 with the same four failed tasks on every boot; all 75 T = 0 prompts inside the
  controls' bracket (75 stable, 0 outside). KLD against the controls: short 0.0054, long 0.0039 (limit 0.035). T = 1
  distribution check PASS (p 0.949, 12,000 samples). The 16 thinking-on reasoning prompts gave the same lengths on all
  three boots (mean 122.7 tokens). Needles 256K (248,175 tokens, 36.2 s) and 1M (968,669 tokens, 168.6 s) found.
- **Exactness** (new rule: flushed 32K, 131K and 1M documents; the first token's top-20 logprobs and a 64-token greedy
  continuation, three samples per boot). 32K: the first token identical in every run on all three boots, continuations
  on control states. 1M: inside the controls' spread. 131K: **the rule fired**: the six control samples gave one
  continuation, the three v2.7.3 samples two others, so the rule took the controls as stable and the v2.7.3 texts as
  different. The 131K continuation is not stable on the stock code: across this gate and three earlier test windows on
  the same code base, checkpoint and document, v2.7.2 gave five different continuations in 16 samples, because the DSA
  indexer's top-k returns its indices in a different order from call to call (prefill and every decode step), which
  moves near-tied logits; six equal control samples happen by chance about 6 % of the time. One v2.7.3 text (two
  samples) is one of the stock texts and leaves the commonest one at token 62 of 64, on a 0.13-nat near tie that the
  stock code also flips; the other is new, as about 3 in 16 stock samples are. Two of the three v2.7.3 first-token
  top-20 vectors are bit-identical to vectors the v2.7.2 code produced, which one changed bit anywhere in the prefill
  would rule out. Every rank logged all three changes' own checks passing (the GEMM tactic's boot check 8 / 8 shapes
  bit-identical and agreed, 0 errors; the MoE-input gather on for every chunk size with every self-check true; the same
  W2 windows on the four nodes). Reviewed as the engine's known T = 0 nondeterminism at 131K, not a numerical change,
  and accepted as release-equivalent; the machine verdict stays FAIL. From the next gate the 131K continuation is
  judged against the accumulated set of stock states (a document counts as stable only with one state in at least 12
  control samples) and the probe stores the full tokens.
- **Draft acceptance** (pooled over all prompts): c1 x1.001, c4 x1.004 of the lower control; c8 x1.004 and c16 x0.998
  reported only from this gate on (at concurrency the T = 0 text, and with it the acceptance, differs ~1.3 % between
  identical controls); qeval at T = 0 3.978 / 3.979 / 3.979 accepted tokens per verify step.
- **Long-prompt decode** (flushed 32K x 2 and 131K x 3 documents, each followed by a 512-token greedy continuation):
  decode 98.91 / 101.82 / 100.82 tok/s, 3.92 / 4.00 / 4.00 tokens per step.
- **Memory and the rest.** Head memory low-water over the common phases 8.50 / 8.43 / 8.46 GiB, during the cold 1M
  prompts 8.71 / 8.68 / 8.65 GiB; the issue-16 soak 308 / 308 requests ok and its companion load 230 requests, 0
  failed; idle first token median 193 / 191 / 194 ms; a short prompt beside a cold 131K prefill answered its first
  token in 0.33 / 0.29 / 0.33 s; no Xid on any boot. A deliberately cold boot of v2.7.3 (tuning caches off) before the
  arms: ready in 182 s, no Xid, core dump or fault line on any node.

## 2026-11-10 v2.7.2

v2.7.1 plus one lossless prefill change in the production profile (`DSV41_PREFILL_CED_SKIP=1`, one key appended to
`EXTRA_CONTAINER_ENV`). The image stack (SGLang tree, FlashInfer 0.7.2, NCCL 2.30.7 with #2393), the KV pool pin
(8,541,952 tokens at fraction 0.85), the launch line, `--swa-prefix-tails 384` and every v2.7.1 key are unchanged.
Measured on the four Sparks against v2.7.1 in the same boot (in-boot A/B with two control arms, the official 1017
checkpoint, 2026-11-09), then in the release gate below (official 1017 checkpoint). In production on the four Sparks
since 2026-11-10 01:02 (abliterated checkpoint).

- **Prefill: non-final chunks run the late layers on one row (`DSV41_PREFILL_CED_SKIP=1`,
  `adapter/prefill_ced_skip.py`).** The launch runs SGLang's decoder SWA bounded replay: every prefill chunk runs
  layers 0-19 over all its rows, layer 20 writes its compressed KV for all rows, and layers 20-39 then run over the
  chunk's last 128 rows (one sliding window), also for a chunk whose prompt goes on for thousands of rows. For a request
  with at least 128 prompt rows still to come after the chunk, that late-layer tail is now the chunk's last row only,
  its window floored at its own position. Nothing reads the rows it no longer writes: the next chunk's tail is floored at
  its own start minus one window, which lies at or past the skipped chunk's end; the final chunk's tail and every decode
  step read only the last window of the prompt, which the skip never touches; a later prefix hit re-prefills one full
  window (#41303), and the SWA cone replay floors its window at its start. The chunk that ends a prompt, the chunk
  before a final chunk of fewer than 128 rows, bypass prompts and any request whose final length is unknown keep the
  stock tail byte for byte. Final prompt lengths are recorded by request id when `ForwardBatch.init_new` builds a target
  extend (the eager runner hands the model a `dataclasses.replace()` copy of the batch, which keeps `rids` and drops
  extra attributes). The late-layer tail costs 86-93 ms per 8192-row chunk in the v2.7 trace, the one-row pass ~25 ms.
  Measured (four rounds, cold prefill): 131K **+6.18 % [5.89, 6.48]** (rounds 5.93 / 6.16 / 6.35 / 6.30; A / A'
  -0.21 %), 32K +3.65 %, 1M +7.39 / +5.82 %; qeval wall time unchanged (its prompts fit in one chunk). Exactness: qeval
  75 / 75 T = 0 outputs equal to both controls'; 64-token greedy continuations of a 32K and a 131K document equal on
  every arm; the 32K first token's top-20 logprobs bit-identical (the controls are bitwise stable there); at 131K and
  1M, where the controls themselves vary run to run, the skip lands on a control's exact state in 2 of 5 samples. Head
  memory low-water during the 32K / 131K prefill blocks 12.68 against 12.69 GiB, during the 1M prefills 8.44 against
  8.65 / 8.62 GiB. 0 Xid, 0 tracebacks.
  The window's own verdict text was BLOCKED by a leak check that counted skip receipts inside the controls' request
  windows: a judge attribution overlap (each control window was widened by 5 s and covered the start of the next
  request, an X request), disproved by the skip counter (the cumulative skip count at every printed receipt equals the
  X-only count of eligible chunks; one skipped control chunk would shift it).
- **In-boot A/B registry:** `DSV41_PREFILL_CED_SKIP` is an in-boot A/B key (`adapter/ab_variant.py`, test only).
- **Image tests:** the canary-roce image runs both CED CPU suites at build (`tests/test_prefill_ced_skip.py`, and
  `tests/test_prefill_ced_skip_engine.py`, which drives a 131K prompt chunk by chunk through a real `ForwardBatch` and
  the engine's tail builder); CI compiles them.
- Rollback, config only: unset `DSV41_PREFILL_CED_SKIP` (= the v2.7.1 profile on the same image).
- **Credits:** a port of **Christopher Owen**'s [spark-ds41f](https://github.com/christopherowen/spark-ds41f) CED skip
  (vLLM patches 0229-0255, 0255 "keep one decoder row of unread nonfinal chunks", and 0526-0527, Apache-2.0) to
  SGLang's decoder SWA bounded replay; the adaptation and its proof of which rows are read are described in the
  adapter's header.

Release gate (run `run-20261109-191418-2004513`, 2026-11-09/10, 342 min including the restore of production): v2.7.1 /
v2.7.2 / v2.7.1, three fresh boots in that order on the same four Sparks, official 1017 `1bae0e1` on all three, GPU
clock capped at 2200 MHz. The controls are production v2.7.1's image and profile; the v2.7.2 boot ran a gate image of
`1993630` (production v2.7.1's image with this release's changed files added) with one profile difference,
`DSV41_PREFILL_CED_SKIP=1`. KV pool 8,541,952 tokens on all three boots. Verdict: **PASS -> RELEASE**, every rule
(decode, sampled, load, cold prefill, memory, qeval, idle CPU, idle first token, inter-token gaps, long-prompt decode),
0 safety, 0 kill, 0 invalid, restore PASS.
- **Prefill, cold** (median of three flushed prompts; v2.7.1 / v2.7.2 / v2.7.1): 64K 6,264 / 6,488 / 6,395 tok/s
  (x1.025 of the controls' mean), 131K 6,379 / 6,611 / 6,393 (x1.035), 1M (968K tokens) 5,233 / 5,485 / 5,247 (x1.047).
  A cold 388,000-token prompt (solo, median of three): first token 62.78 / 59.72 / 62.46 s (x1.051 / x1.046 the speed of
  the two controls). Smaller than the in-boot A/B above (+6.2 % at 131K).
- **Decode** (speedbench 0.3 full, mean of two runs; v2.7.1 / v2.7.2 / v2.7.1): greedy c1 all-prompt geomean 90.09 /
  89.91 / 89.86 tok/s (runs 89.82 / 90.37, 89.46 / 90.37, 89.06 / 90.66) and prose 66.74 / 66.64 / 66.54; c4 135.72 /
  135.30 / 135.81, c8 167.96 / 172.37 / 171.39 and c16 216.59 / 215.56 / 216.03 aggregate tok/s. Sampled (T 0.7, top-p
  0.95, one run per cell): c1 87.84 / 87.35 / 87.47 and prose 62.50 / 64.62 / 64.06 tok/s, c4 133.36 / 131.92 / 134.59.
  Speedbench load c16 226.33 / 228.62 / 228.52 aggregate tok/s (median TTFT 468 / 468 / 466 ms, p99 inter-token gap
  296 / 282 / 290 ms). v2.7.2 against the mean of the controls: greedy c1 x0.999, c4 x0.997, c8 x1.016, c16 x0.997;
  sampled c1 x0.997, c4 x0.985; load c16 x1.005; all inside the gate's bands (the skip touches only non-final prefill
  chunks). Median first token at c1 166-167 ms on every boot.
- **Correctness.** qeval 71 / 71 / 71 with the same four failed tasks on every boot; all 75 T = 0 prompts inside the
  controls' bracket (75 stable, 0 outside). KLD against the controls: short 0.0057, long 0.0037 (limit 0.035). T = 1
  distribution check PASS (p 0.994, 12,000 samples). The 16 thinking-on reasoning prompts gave the same lengths on all
  three boots (mean 122.7 tokens). Needles 256K (248,153 tokens, 38.3 s) and 1M (968,936 tokens, 177.2 s) found. An
  evicted 302,048-token resume point replayed one 8,192-token chunk on every boot (first token 2.19 / 2.07 / 2.17 s).
- **Draft acceptance.** The rule now judges the pooled ratio (all tokens over all steps, both runs), as decided after
  the v2.7.1 gate: c1 x1.001, c4 x0.997, c8 x1.009, c16 x0.999 of the lower control; qeval at T = 0 3.979 accepted
  tokens per verify step on all three boots.
- **Long-prompt decode** (new rule: flushed 32K x 2 and 131K x 3 documents of natural text, each followed by a 512-token
  greedy continuation with the server's DSpark counters): decode 92.32 / 92.87 / 92.86 tok/s, 3.59 / 3.64 / 3.61 tokens
  per step.
- **Memory and the rest.** Head memory low-water 8.41 GiB (controls 8.48 / 8.54), during the cold 1M prompts 8.56 GiB
  (8.62 / 8.54; new rule: within 0.25 GiB of the lower control); the issue-16 soak 313 / 313 requests ok and its
  companion load 226 requests, 0 failed; idle first token median 194 ms (190 / 193); a short prompt beside a cold 131K
  prefill answered its first token in 0.33 s (0.37 / 0.36); every node logged the CED skip armed and its forward
  receipts, with no unknown-length warning; no Xid on any boot. A deliberately cold boot of v2.7.2 (tuning caches off)
  before the arms: ready in 171 s, no Xid, core dump or fault line on any node.

## 2026-11-09 v2.7.1

v2.7 plus five lossless changes in the production profile (eight keys of `EXTRA_CONTAINER_ENV`), a wider prefill
bypass, and two default-off levers. The image stack (SGLang tree, FlashInfer 0.7.2, NCCL 2.30.7 with #2393), the KV pool pin (8,541,952 tokens at
fraction 0.85), the launch line and `--swa-prefix-tails 384` are v2.7's. Every change was measured on the four Sparks
against production v2.7 in the same boot (in-boot A/B with two control arms, randomized round order, the official 1017
checkpoint, 2026-11-08/09), then together in the release gate below (official 1017 checkpoint). In production on
the four Sparks since 2026-11-09 14:48 (abliterated checkpoint).

- **Prefill: row split of the K = 5120 dense GEMMs (`DSV41_PREFILL_GEMM_ROW_SPLIT=4096
  DSV41_PREFILL_GEMM_ROW_SPLIT_N=1792,1152`, `adapter/prefill_gemm_split.py`, hooked from `adapter/mxfp8_b12x.py`).**
  At an 8192-row prefill chunk the MXFP8 activation of `wqkv_a` (5120 -> 1792) and of the shared-expert FC1
  (5120 -> 1152) is 42 MB, larger than GB10's L2, and the persistent b12x kernel re-reads it from DRAM for every N tile
  (51 TFLOP/s in the v2.7 trace against 132 at 4096 rows). With the split such a GEMM runs as two launches of the same
  kernel on 4096-row blocks, each writing its rows of the one output tensor (no concatenation copy); the activation is
  quantized once, a pre-quantized SP input is sliced on its 128-row scale tiles. The first eager call of every
  (N, K, M, pre-quantized, dtype) shape runs the stock GEMM and the split on the same input, compares the outputs bit for
  bit and the TP ranks agree, so a shape is split on every rank or on none (`DSV41_PREFILL_GEMM_ROW_SPLIT N=1792 ...
  -> SPLIT` in the head log); a difference, an exception or a CUDA-graph capture keeps the stock call. Decode, DSpark
  verify and short extends never split (M below 8192). Measured (four rounds): cold prefill **+3.77 % [3.60, 3.94]** at
  131K, +3.1 % at 32K, +3.5 % at 1M; the head's memory low-water and the decode stall of streams during a long prefill
  unchanged; the boot check bit-identical for both widths on every rank. 2048-row blocks measured the same as 4096
  (+0.05 points).
- **Decode: the attention-core L2 window warms `wo_b` instead of `wo_a` (`DSV41_L2_PREFETCH_WOA_MB=0
  DSV41_L2_PREFETCH_WOB_MB=6`, `adapter/l2_prefetch.py`, no code change).** `wo_b` was the one main-stream dense weight
  still ~60 % cold in serving (51 us per call against 15.5 us warm). With a `wo_a` budget of 0 the window opened before
  the attention core takes 6 MB of `wo_b` instead; `wo_a` goes cold and gives part of the gain back. The plan moves
  budget without adding any (835.1 MB of prefetch per forward, as before). Measured (six rounds, c1): decode step
  -0.40 ms, **+1.10 % [0.86, 1.36]** (prose +1.09 %), faster than both control arms in every round; tokens per step
  identical; qeval 75 / 75 T = 0 outputs equal to the controls'; c4 -0.34 % [-0.84, 0.16] (inside the noise). The dense
  b12x GEMM group falls from 7.03 to 5.94 ms per step in the trace.
- **Admission: extends without host syncs (`DSV41_EXTEND_NOSYNC=1`, `adapter/extend_nosync.py`) and MoE launches
  without the dispatcher (`DSV41_MOE_LAUNCH_PREPARED=plan,direct`, was `plan`; `adapter/moe_launch_prepared.py`).**
  Every extend (a new prompt, a chunk, a DSpark admission) built its late-layer tail metadata (decoder SWA bounded
  replay) and the Engram hasher's extend map with a pageable host-to-device copy and a `repeat_interleave` without
  `output_size`: two host syncs per extend that, under the overlap schedule, stall the CPU behind queued GPU work. Now
  the lengths go through a pinned buffer and the output size comes from the host-known lengths (the stock call runs
  when they are missing or disagree); bit-identical. In a sync audit of c16 traffic all 19
  audited extends on the head synced (11 behind queued GPU work); with the change no sync site was left on any of the
  four nodes (`DSV41_EXTEND_SYNC_AUDIT=1`, `adapter/extend_sync_audit.py`, test only).
  `plan,direct` also calls the MoE op's Python body directly in eager mode instead of through the PyTorch dispatcher
  (same body, same arguments). Measured together (six rounds): c16 per-stream decode **+3.38 % [1.38, 5.39]**, c4
  +2.98 %, c1 first token **-40.6 ms**; `plan,direct` alone: first token -38.7 ms, c16 +0.9 % (not significant). Greedy
  outputs within the controls' own T = 0 spread on every prompt, qeval bracket held.
- **Sampled verify: commit reuse and early draft probabilities (`DSV41_SAMPLED_COMMIT_REUSE=1`,
  `DSV41_SAMPLED_DRAFT_PROBS_EARLY=1`; `adapter/sampled_commit_reuse.py`, `adapter/sampled_draft_probs_early.py`).**
  On a sampled step (T > 0, a grammar, a logit bias) the target-verify graph already computes the draft-KV commit
  projection but stores nothing, and the worker then recomputed it eagerly (main_proj, its all-gather, norm, quant, the
  wkv GEMM). The graph now copies the projection into a persistent buffer (~2 us) and the eager commit stores from it
  when the replayed graph's batch equals the real one. And the accept's softmax of the draft logits is launched on a side
  stream right after the proposal, overlapping the verify graph, and used when its inputs are exactly the accept's.
  Same kernels on the same inputs, no random draw on either path: bitwise (`check` recomputes and counts differing bits;
  0 in testing). Greedy steps are untouched. Measured (T 1.0 / top-p 0.95, prose, c1): step **-0.251 ms [-0.356,
  -0.13]**.
- **Multi-user: a wider bypass (`DSV41_PREFILL_BYPASS=8192`, was 256; `adapter/prefill_bypass.py`, no code change).** A
  request with up to 8192 fresh tokens that arrives during a long cold prefill is now admitted beside it (in a solo pass
  of its own, as before; at most one solo pass between two chunks of the long prompt) instead of only requests of
  <= 256 tokens. Measured with Christopher Owen's servebench on production v2.7: a 500-token prompt beside a cold 131K
  prefill answers in 1.3-1.7 s instead of 26-28 s; the long prompt takes 6-10 % longer while such requests arrive.
  In production since 2026-11-08; the release gate runs it on every arm. The one-chunk limit follows Christopher
  Owen's spark-ds41f prefill yield (a waiting prompt that fits in one chunk takes the next step).
- **Default off, not recommended yet:** `DSV41_PREFILL_KEEP_CACHE=1` (`adapter/prefill_keep_cache.py`): the caching
  allocator keeps its blocks between the chunks of one long prefill instead of `empty_cache()` after every chunk, with
  padded indexer widths and a memory-neutrality guard. In its first measurement it never engaged on a solo prefill
  (released at every idle check) and cost 0.45 GiB where decode interleaved; fixed since (held across idle checks within
  one prompt, no keep once decode interleaves), not re-measured. `DSV41_CERT_HEAD_MASK_MAX_M=<rows>`
  (`adapter/cert_head.py`, not measured on its own): the certified head's masked mode is disarmed above that many
  verify rows, so greedy steps there run its full (exact) mode.
- **Image tests:** the canary-roce image runs the CPU suites of all of the above at build; CI compiles them.
- Rollback, config only: unset the seven new keys and set `DSV41_MOE_LAUNCH_PREPARED=plan` and
  `DSV41_PREFILL_BYPASS=256`: the v2.7 profile on the same image (the new adapters do nothing unset).
- Ideas and implementation ours. The cold `wo_b` was found by comparing our per-kernel times with those of
  Christopher Owen's spark-ds41f on the same fleet (its dense MXFP8 GEMMs run with L2-warm weights).

Release gate (run `run-20261109-085149-628062`, 2026-11-09, 341 min including the restore of production): v2.7 /
v2.7.1 / v2.7, three fresh boots in that order on the same four Sparks, official 1017 `1bae0e1` on all three, GPU
clock capped at 2200 MHz. The controls are production v2.7's image and profile; every boot ran
`DSV41_PREFILL_BYPASS=8192`, so the bypass is not part of the comparison. The v2.7.1 boot ran a gate image of `9cee7cf`
(production v2.7's image with this release's changed files added). KV pool 8,541,952 tokens on all three boots.
Verdict: every speed, prefill, memory, qeval, idle and latency rule PASS, 0 safety, 0 invalid; the verdict text is
**FAIL on one rule, draft acceptance at c4** (below), accepted on review as release-equivalent.
- **Decode** (speedbench 0.3 full, mean of two runs; v2.7 / v2.7.1 / v2.7): greedy c1 all-prompt geomean 89.03 /
  89.79 / 88.83 tok/s (runs 88.35 / 89.70, 89.52 / 90.07, 88.64 / 89.02) and prose 65.83 / 66.68 / 65.91; c4 131.31 /
  135.20 / 132.19, c8 166.19 / 171.81 / 166.84 and c16 209.61 / 216.17 / 207.59 aggregate tok/s. Sampled (T 0.7,
  top-p 0.95, one run per cell): c1 84.35 / 87.10 / 85.30 and prose 62.56 / 64.77 / 62.73 tok/s, c4 128.70 / 133.61 /
  129.82. Speedbench load c16 216.04 / 229.51 / 215.97 aggregate tok/s (median TTFT 539 / 466 / 533 ms, p99
  inter-token gap 355 / 283 / 354 ms). v2.7.1 against the mean of the controls: greedy c1 x1.010, c4 x1.026, c8 x1.032,
  c16 x1.036; sampled c1 x1.027, c4 x1.034; load c16 x1.063. Median first token at c1 166 ms against 207 ms, c4 253
  against 316-321 ms, c16 512 against 564-567 ms.
- **Prefill, cold** (median of three flushed prompts): 64K 6,028 / 6,308 / 6,045 tok/s (x1.045 of the controls'
  mean), 131K 6,096 / 6,400 / 6,134 (x1.047), 1M (968K tokens) 5,022 / 5,253 / 5,067 (x1.041).
- **Correctness.** qeval 71 / 71 / 71 with the same four failed tasks on every boot; all 75 T = 0 prompts inside the
  controls' bracket (75 stable). KLD against the controls: short 0.0069, long 0.0061 (limit 0.035). T = 1 distribution
  check PASS (p 0.994, 12,000 samples). The 16 thinking-on reasoning prompts gave the same lengths on all three boots
  (mean 122.7 tokens). Needles 256K (248,158 tokens, 39.3 s) and 1M (968,449 tokens, 185.9 s) found.
- **Draft acceptance.** qeval at T = 0: 3.979 accepted tokens per verify step on all three boots. The speedbench rule
  (the median over the 73 prompts of tokens per stream step, median of two runs, must stay above x0.98 of the lower
  control) failed at c4: 3.091 against 3.168 / 3.250. With 255 decoded tokens per prompt that median moves in steps of
  255/n, about 1.2 % per stream step, and the same rule fails a control at c16 (x0.976 with a control in the
  release's seat). Pooled over all prompts and both runs (all tokens over all steps) the acceptance is equal: c4 2.899
  against 2.898 / 2.931, c1 x1.001, c8 x0.999, c16 x1.003 of the lower control; on the 45 prompts whose output was
  byte-identical in all six runs, tokens per step are identical (3.033). At c4 the greedy text of the remaining prompts
  depends on which requests share a batch, which the faster admission shifts (first token 253 against 316 ms). The
  rule is to be replaced by the pooled ratio in the next gate.
- **Memory and the rest.** Head memory low-water 8.32 GiB (controls 8.34 / 8.36); the issue-16 soak 315 / 315 requests
  ok and its companion load 225 requests, 0 failed; idle first token 189.6 ms (187.0 / 187.2); every node logged the
  eight new keys armed and the row split's first-call check passed for both widths on every rank; no Xid on any boot.
  A deliberately cold boot of v2.7.1 (tuning caches off) before the arms: ready in 171 s, 391, no Xid, core dump or
  fault line on any node.

## 2026-11-08 v2.7

v2.6.5 plus FlashInfer 0.7.2 with DeepSeek-V4.1's own KV layouts on GB10, and the three changes of the v2.6.6
candidate (the SWA prompt-tail keep, NCCL 2.30.7 with the NVIDIA/nccl #2393 fence, the `NCCL_OVERLAY_PIP` launcher
fix). v2.6.6 was not released on its own; its changes ship here. In production on the four Sparks since 2026-11-08
06:52.

The production profile in `.env.tp4.example` replaces v2.6.5's v41n keys (`DSV41_KV_LAYOUT=v41c DSV41_V41C_ARM=1
DSV41_V41C_READER=native`), `DSV41_SWA_NATIVE_DECODE=1` and `DSV41_PREFILL_NATIVE_EXTRA=1` with `DSV41_KV_LAYOUT=v41
SGLANG_SM120_V41_PRECISION=bf16`, and adds `DSV41_SWA_PROMPT_TAIL_KEEP=1024`. The KV pool pin (8,541,952 tokens at
`MEM_FRACTION_STATIC=HEAD_MEM_FRACTION_STATIC=0.85`) and `--swa-prefix-tails 384` are unchanged.

- **FlashInfer 0.7.2 and the V4.1 KV layouts (`DSV41_KV_LAYOUT=v41 SGLANG_SM120_V41_PRECISION=bf16`).** The SWA cache
  (and the draft's paged SWA) holds `KVLayout.V41` (528 B per token, all 512 values e4m3 in groups of 32 with ue8m0
  scales, RoPE included, as DeepSeek's V4.1 reference stores the window K), the ratio-1/2 caches `V41_FP4` (288 B, as
  v41n), read in place by FlashInfer 0.7.2's SM120 DSv4.1 sparse-MLA kernels on the full-BF16 route (bf16 Q and P,
  exact on-chip dequantization). FlashInfer 0.7.2 is hash-pinned (`files/flashinfer-0.7.2.requirements`, git
  `d84335d8`) and the #7676 fence now applies to its copy of the b12x dense GEMM. This is a numerics change against
  v2.6.5 (the window keys' RoPE part is stored in FP8, as in the reference): T = 0 outputs differ from v2.6.5's on
  some prompts, inside the KLD bound (release gate below). The three FlashInfer-0.6.18-only modules (v41c / v41n /
  swan and the K8 / PCE extra-cache module) are no longer built; their knobs refuse at boot on FlashInfer >= 0.7. Pool
  sizing prices the 528-byte rows (912.25 B per full token as v2.6.5; the SWA term x 528 / 584, ~0.4 GiB less per
  rank). Cost: cold prefill from 64K to ~400K context is ~1.7 % slower (FlashInfer 0.7.2's BF16 sparse-MLA prefill
  kernel; measured below); decode speed is unchanged. Credits: sglang#45851 (05yuki), FlashInfer #5197 / #6616
  (lucifer1004), #7676 (Thireus), luz-deepseek-flash-all-in-one (luxingcom). Rollback: the v2.6.5 image and profile.
- **Fixed in testing: a cold-boot fault (no in-process sparse-MLA calibration).** FlashInfer 0.7 calibrates its SM120
  sparse-MLA kernels whenever an autotune pass meets a request with no profile on disk, also in SGLang's draft
  autotune pass, after the target-verify CUDA graphs are captured. In testing, every cold boot on the four Sparks then
  faulted at the first decode: a captured graph wrote to unmapped memory on every rank (Xid 31 / 13; the faulting
  kernel a cuBLAS bf16 batched GEMM; debug runs `run-20261107-105507-2650424` and `run-20261107-134634-3060892`),
  while every boot that found the calibration on disk served. `boot.py` now adds `sparse_mla_sm120` to
  `--flashinfer-autotune-skip-ops` and points `FLASHINFER_AUTOTUNE_DIR` at the calibration shipped in the image
  (`files/fi-autotune/sparse_mla_sm120_cpb.json`: FlashInfer 0.7.2's own calibration of the four V4.1 TP4 request
  shapes on a GB10, read-only, the same plans on every rank). A missing calibration refuses the boot;
  `DSV41_V41_CALIBRATION=inprocess` restores FlashInfer's behaviour for diagnostics. A deliberately cold boot of the
  release (tuning caches off) before its gate: ready in 242 s, no Xid, core dump or fault line on any node, the frozen
  calibration and the BF16 reader logged on every node.
- **SWA prompt tail in the production profile: `DSV41_SWA_PROMPT_TAIL_KEEP=1024`** (`adapter/swa_prompt_tail.py`; the
  hold `DSV41_SWA_PROMPT_TAIL_HOLD` stays at its default 8192). A prompt-end insert used to keep one page (256 tokens)
  of live SWA below the prompt's last full page, so a revisit whose shared prefix ends earlier (a document or a chat
  turn with a new question) found no live window to end its match on: it re-prefilled from 0 on its 2nd visit and hit
  only from the 3rd. Now the prompt end keeps 1024 tokens of live SWA. Port of Eric Jinho Jang's (dealign.ai)
  prompt-tail floor
  ([jjang-ai/dealignai-deepseek-v4.1-flash-1017-4x-dgx-spark](https://github.com/jjang-ai/dealignai-deepseek-v4.1-flash-1017-4x-dgx-spark)
  `c175335` / `a9f87fe`, AGPL-3.0-or-later), redesigned: their floor also stopped every decode-time SWA free once the
  tree owned the prompt end (a request held N + its whole output in SWA); here decode frees stay stock then, and a
  still request-owned prompt end is held for at most 8192 tokens; chunked inserts (and so the cone replay's first
  insert) are stock. Screen on the four Sparks (`run-20261107-140633-3102469`, A / TK / A' on v2.6.5): token-exact
  revisits with a new 192-token question cached 2048 of 2112 shared tokens (97.0 %, first token 0.24 s instead of 0.53
  s) at 2k and 6144 of 6208 (98.97 %, 0.25 s instead of 1.16 s) at 6k where v2.6.5 cached 0; a 6k chat turn 2 97.3 %
  cached; c1 x1.005 and c4 x1.011 of the slower control; qeval 71 / 71 / 71 with no task outside the controls'
  bracket; head low-water 6.35 GiB. The release gate (below) repeated the revisits: 2k / 6k first token 0.24 / 0.25 s
  against 0.53 / 1.14-1.16 s on the two v2.6.5 controls, which cached 0. Cost: each cached prompt end holds up to 1280
  SWA slots instead of 256 (the 384 prefix tails' headroom is shared with them). Outputs and the pool are unchanged;
  the key is volatile for `DSV41_AUTOTUNE_KEEP` (no re-tune). Rollback, config only: unset it.
- **NCCL 2.30.7 with NVIDIA/nccl #2393 in the canary-roce image** (no version bump). An `nccl-builder` stage rebuilds
  the base image's NCCL version (v2.30.7-1, `73cf112`) for sm_121 with Stanislav Bardyuk's backport of [NVIDIA/nccl
  #2393](https://github.com/NVIDIA/nccl/pull/2393) (`patches/nccl-2393-aarch64-cts-fence.patch`, until now only in the
  switchless-ring build): one acquire fence in `ncclIbIsend`, so a weakly ordered CPU cannot read a peer's CTS `nreqs`
  before the index check that guards it. The image replaces the wheel's `libnccl.so.2` (the path torch resolves) and
  checks the version, the fence and the library against the builder's receipt (`scripts/nccl_receipt.py`,
  `/opt/dsv41/files/nccl-2393.json`). Stage the source with `scripts/fetch-nccl.sh`. A/B on the four Sparks
  (`run-20261107-080134-2137538`, A / X / R / A', R = the same rebuild without the patch): all-reduce bus bandwidth
  x0.9998 of stock at the worst size from 8 to 256 MiB, reduction results bit-equal at every size, cold 64K x1.021 and
  1M x1.0017, no stable qeval item changed and the one item that had differed in the first run returned the production
  output on 5 of 5 replays. A newer NCCL release was reported to transfer ~15 % slower on this hardware, so the
  version stays 2.30.7 and only the fence is added.
- **Launcher fix: `NCCL_OVERLAY_PIP=1` on a switched fabric.** The head honoured the overlay only with
  `NCCL_SWITCHLESS_RING_ONLY=1` and otherwise put `NCCL_HOST_DIR` on `LD_LIBRARY_PATH`, which torch ignores (it
  resolves `libnccl.so.2` through its RPATH), while every worker mounted the library over the pip NCCL: the head and
  the workers ran different NCCL builds. The head now mounts NCCL exactly as the workers do, and `serve` checks the
  NCCL switches with the ring off too. With the overlay off the launch arguments are unchanged.
- **Dropped:** the working-set-preserving prefill trim (`DSV41_PREFILL_TRIM`, from the same jjang-ai commits) was
  screened in the same window and not adopted: the head's lowest free memory fell to 9.02 GiB at 64K and 4.82 GiB at
  388K against 11.63-12.01 and 10.30-10.70 GiB on the controls, for +3 % prefill measured at 64K only. The code is not
  in this release.

Release gate (run `run-20261108-010847-479202`, 2026-11-08, 334 min including the restore of production): v2.6.5 /
v2.7 / v2.6.5, three fresh boots in that order on the same four Sparks, abliterated 1017 `517efab`, GPU clock capped
at 2200 MHz. The controls are production v2.6.5 as it serves; the v2.7 boot ran the release's production profile on a
gate image of `1303d2f` (built on the v2.6.5 image with FlashInfer 0.7.2, this release's files and the #2393 NCCL
library added). KV pool 8,541,952 tokens on all three boots. Verdict: every speed, prefill, memory, qeval, idle and
latency rule PASS, 0 safety, 0 invalid; the verdict text is **FAIL on one rule, reasoning length** (below), accepted
on review as release-equivalent.
- **Decode** (speedbench 0.3 full, mean of two runs; v2.6.5 / v2.7 / v2.6.5): greedy c1 all-prompt geomean 89.56 /
  89.92 / 89.45 tok/s (runs 88.91 / 90.20, 89.40 / 90.43, 88.97 / 89.93) and prose 65.89 / 65.97 / 65.96; c4 132.51 /
  132.08 / 132.16 and c16 208.39 / 208.12 / 209.13 aggregate tok/s. Sampled (T 0.7, top-p 0.95, one run per cell): c1
  86.77 / 87.18 / 87.29 and prose 63.99 / 63.36 / 62.58 tok/s, c4 128.69 / 128.74 / 130.46. Speedbench load c16 210.93
  / 216.55 / 214.90 aggregate tok/s (median TTFT 545 / 542 / 537 ms). v2.7 against the two controls: greedy c1 x1.004
  / x1.005, c4 x0.997 / x0.999, c16 x0.999 / x0.995, load c16 x1.027 / x1.008. Per category (greedy c1) the text
  changes with the numerics, and so does the acceptance on it: code 108.8 against 112.6 / 112.7 tok/s, structured
  110.2 against 107.3 / 107.4, agentic 106.7 against 103.1 / 104.2, multilingual 72.1 against 70.6 / 70.6.
- **Draft acceptance** (accepted tokens per verify step, v2.7 against the lower control): qeval x1.006, speedbench c1
  x1.046 (3.33 against 3.25 / 3.19), c4 x1.020, c16 x1.029.
- **Prefill, cold** (median of three flushed prompts): 64K 6,145 / 6,026 / 6,165 tok/s (x0.981 / x0.977), 1M (968K
  tokens) 5,180 / 5,151 / 5,312 tok/s (x0.994 / x0.970); a cold 388K-token prompt's first token 64.4 / 65.6 / 64.0 s.
  The steady-state screen before the gate (`run-20261107-234605-295462`, v2.6.5 / v2.7 / v2.6.5, chunks measured
  between bursts) puts the cost at x0.983 of the slower control at 64K and at 388K (control spread 0.2 % / 0.05 %),
  every 40K-token stretch of the 388K prompt between x0.982 and x0.987; its profile puts it in the attention kernel
  (FlashInfer 0.7.2's BF16 sparse-MLA prefill kernel; net +1.5-1.7 % busy time per chunk after v2.6.5's separate
  reader copies are gone). In the kernel screens the V4.1 prefill attention is faster than v2.6.5's from ~380K context
  on (x0.92 / x0.99 of its time per call on the ratio-1 / ratio-2 layers at 376K, x0.85 / x0.93 at 768K), so a cold 1M
  prompt is close to parity. The gate's cold-prefill bound for this release was x0.97 (0.98 before), set after that
  screen; decode and load stayed at x0.995.
- **Correctness.** qeval 71 / 71 / 71 with the same four failed tasks on every boot; at T = 0 the v2.7 text differs
  from the controls' on 26 of the 74 prompts on which the controls agree (expected from the numerics change). KLD
  against v2.6.5: short 0.0111, long 0.0064 (control vs control 0.0061 / 0.0026; limit 0.035). T = 1 distribution
  check PASS (12,000 samples). Needles 256K (248,129 tokens, 41.4 s) and 1M (969,022 tokens, 187.9 s) found. Evicted
  resume point (302,048-token prompt, 800 pressure prompts, then the document plus 2,048 tokens): every boot replayed
  one 8,192-row chunk with 0 reverts, first token 2.19 / 2.22 / 2.21 s, and the v2.7 output (16 greedy tokens) equals
  both controls'.
- **Revisit with a new question** (the prompt tail): 2k revisit cached 0 / 2,048 / 0 tokens, first token 0.537 / 0.244
  / 0.534 s; 6k 0 / 6,144 / 0, 1.135 / 0.247 / 1.157 s; a 6k chat turn 2 cached 97.5 / 97.3 / 97.7 % on all three.
- **Memory and load.** The head's lowest MemAvailable over the boot 6.47 / 8.73 / 6.95 GiB; NVRM out-of-memory log
  lines 12 / 0 / 13 (the controls' in the Engram warm-up, the cold prefills and the soak). Soak (issue #16
  reproduction plus a c2 companion load): 321 / 317 / 320 requests with 0 failures, companion 48.69 / 48.55 / 48.70
  tok/s. A short request beside a 131,072-token prefill answered in 0.457 / 0.469 / 0.431 s; median first token after
  60 s idle 194.4 / 194.1 / 192.9 ms; idle RDMA proxy CPU summed over the four nodes 1.70 / 1.78 / 1.50 % of a core.
  Every scheduler of the v2.7 boot loaded the #2393 NCCL 2.30.7 library, and every rank logged the #7676 fence as
  present.
- **The rule that failed: reasoning length.** The gate holds the mean and median completion tokens and the mean
  reasoning characters of 16 thinking-on T = 0 reasoning prompts within +-3 % of the controls (a guard against less
  thinking). v2.7 reasoned longer: mean 139.4 against 133.4 tokens (+4.5 %), median 124.5 against 103.5, reasoning
  characters 514 against 454. The two controls are byte-identical, so the band had no control spread to measure noise
  against; per prompt v2.7 was longer on 6, shorter on 9 and equal on 1, and one prompt (+337 tokens) carries the mean
  (without it v2.7 is 12 % shorter). All 16 answers are correct and identical on the three boots, none truncated. The
  review accepted the result as release-equivalent: the direction is more reasoning, not less, and every other rule
  passed.

## 2026-11-07 v2.6.5

v2.6.4 plus three changes. The production profile in `.env.tp4.example` adds `DSV41_KV_LAYOUT=v41c
DSV41_V41C_ARM=1 DSV41_V41C_READER=native` ("v41n": the compressed KV records in SGLang's V4.1 FP4 layout, read in
place) with the KV pool pinned at `MAX_TOTAL_TOKENS=8541952` at `MEM_FRACTION_STATIC=HEAD_MEM_FRACTION_STATIC=0.85`, and `DSV41_TOKENIZER_HEAP=trim2` (the head's tokenizer
process returns its free heap). The fix for [#66](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/66)
(named loop-abort exit, broken tool calls in the history) is on by default. All new keys are volatile for
`DSV41_AUTOTUNE_KEEP` (no re-tune when toggled; the pool size is out of the fingerprint too). The full KV pool is
8,541,952 tokens: +69 % against v2.6.2's ~5.06M and x2.3-2.5 against v2.6.4's 3.44-3.70M, eight 1M-token contexts
instead of three. The price is host memory on the head: its lowest MemAvailable was 5.72 GiB in the memory ladder's
stress cells (eight 1M-token prompts resident at once) against 8.87-9.32 GiB at 5,130,752 / fraction 0.78. Rollback
is config only (the v2.6.4 values, below).

- **KV pool 3.44-3.70M -> 8,541,952 tokens (x2.3), measured.** Memory ladder on the four Sparks, first
  `~/ds-gate/mlad-v264np-20261106/run-20261106-090824-539338` (2026-11-06; production's fraction 0.78, tails 384,
  trim2 in every boot): two v2.6.4 control boots (pools 3,700,480 / 3,447,040) and two boots of this release's code at
  the pin (5,130,752 granted on both). With four 1M-token prompts resident at once (controls: three) and the long-prompt
  soak, the head's lowest MemAvailable was 9.32 / 8.87 GiB (controls 9.28 / 10.25), NVRM 0, swap 0, every 1M needle
  retrieved. Then the bigger rungs at fraction 0.85 (at 0.78 v41n's budget is 6,098,432 / 6,194,944 tokens, so a larger
  pin would not bind), `~/ds-gate/mlad-v264nx-20261106/run-20261106-173315-3341279`: 8,000,000 SAFE (head low-water
  5.96 GiB, 7 / 7 1M needles at once, NVRM 0) and **8,541,952 SAFE** (head low-water 5.72 GiB, 8 / 8 1M needles resident
  at once, a cold 1M prefill and the soak, NVRM 2: one line each on two workers during the 8 x 1M cell; speed against
  the faster reference control c1 0.998 / c4 0.998 / c16 0.994 / stress 1.018 / cold 1M 1.008): verdict X-SAFE-POOL
  8,541,952. The fraction 0.85 leaves the 8192-token prefill chunk less spare memory than 0.78 did; that run's 1M
  prefills passed at it. The first window's verdict was INCONCLUSIVE on control-vs-control drift only (a slow control, a different control pool); the candidate rules
  all passed and the review accepted it (`diagnostics/ds-mem-audit-20261029/v264nx-20261106/REVIEW-DECISION-v264np-20261106.md`
  in the workspace).
- **Speed, measured** (same run, c1 / c4 / c16 tok/s): candidates 77.09 / 131.89 / 211.6 and 77.17 / 133.88 /
  214.73, controls 77.01 / 131.71 / 212.18 and 77.20 / 127.6 / 205.24; the worst candidate cell against the faster
  control is c16 0.997. The capacity A/Bs of 2026-11-03..05 (12 boots, tails 128) put v41n's real cost at c16
  -0.4..-0.8 %, c1 / c4 / prose within +-0.1 %, cold 1M prefill +1.6 %.
- **Exactness.** v41n's readers rebuild the V4 operand bytes: on one GB10 decode, prefill and swan are bit-equal to
  V4 (38 / 38 layers); in TP4 serving the KLD against the control is inside control-vs-control (0.0031-0.0053 vs
  0.0073 / 0.0076) and qeval 71 / 71 / 71 (capacity A/Bs). The cone replay copies the V4.1 FP4 pages and masks their
  replay writes (`tests/test_swa_cone_v41c.py`); on the fleet the evicted-resume cell replayed one 8192-row chunk on all
  four nodes in every ladder boot (TTFT 2.07-2.13 s at 302K tokens). In the release gate (below) the replayed
  resume's 16 output tokens equal a v2.6.2 control's token for token.
- **Tokenizer heap, measured.** In the trim A/B (`~/ds-gate/trim-20261106/run-20261106-023911-3456984`, 30-minute
  long-prompt soak; verdict INVALID on a pool mismatch and a receipt bug, fixed since): launch_server 1.0-3.1 GiB with
  trim2 against 3.9-4.3 GiB at the controls' soak end, 19 trims returning 27.8 GiB, 0 of 74 stable qeval tasks
  changed, long-prompt count, companion and c1 / c4 speed within 1 %. In the ladder's four boots: 16-18 trims each,
  17.7-20.1 GiB returned. Cost: a busy trim runs on the HTTP event loop, so under back-to-back long prompts token
  delivery pauses up to 0.69-0.90 s about once per 2 minutes (the GPUs keep decoding).
- **Rollback, config only:** unset `DSV41_KV_LAYOUT`, `DSV41_V41C_ARM`, `DSV41_V41C_READER` and set
  `MAX_TOTAL_TOKENS=8000000` and the fraction 0.78 (v2.6.4); unset `DSV41_TOKENIZER_HEAP` on its own; `DSV41_LOOP_NAMED_EXIT=0` /
  `DSV41_HISTORY_REPAIR=0` restore v2.6.4's loop exit / the HTTP 400.
- **Credits:** v41n's in-place readers extend FlashInfer's SM120 sparse-MLA kernels (NVIDIA, BSD-3-Clause) and write
  through SGLang's own V41_FP4 layout (Apache-2.0); the design idea of copying raw FP4 records asynchronously and
  dequantizing across many warps is Christopher Owen's ([christopherowen](https://github.com/christopherowen),
  spark-ds41f). The #66 fix is Alexander Busse's ([GanyX19](https://github.com/GanyX19)) MiaAI-Lab#55, reported here by
  [slidonnici](https://github.com/slidonnici) (details below).

Release gate: PASS (run `run-20261106-214411-640735`, 2026-11-06 / 07, 277 min including the restore of production;
0 KILL, 0 INVALID). v2.6.2 / v2.6.5 / v2.6.2, three fresh boots in that order, abliterated 1017 `517efab`; the
controls are production v2.6.2 as it serves (tails 128, no pin, fraction 0.78). The v2.6.5 boot ran the v2.6.4 gate
image with this release's files overlaid (the module `00505fe850b7` unchanged), launched with `--swa-prefix-tails
384`, `MAX_TOTAL_TOKENS=8541952` and the fraction 0.85; the #66 switches at their defaults. Speedbench 0.3 full,
median of two runs: greedy c1 all-prompt geomean 88.91 / 89.91 / 88.77 and prose 65.49 / 66.11 / 65.34 tok/s, c4
131.39 / 132.08 / 131.33 aggregate tok/s; sampled (T 0.7, top-p 0.95, one run per cell) c1 84.85 / 85.77 / 85.00 and
prose 62.26 / 62.58 / 62.73 tok/s, c4 127.62 / 129.24 / 127.99 aggregate tok/s; speedbench load c16 213.65 / 217.44 /
216.81 aggregate tok/s. Cold 64K prefill 6182 / 6198 / 6210 tok/s (median of three flushed prompts); a cold
388K-token prompt's first token 63.5 / 64.1 / 63.3 s (median of three); a short request beside a 131,072-token
prefill 0.44 / 0.45 / 0.43 s. qeval 71 / 71 / 71; at T = 0 the v2.6.5 outputs equal the controls' on all 74 of the
75 prompts on which the two controls agree. KLD long 0.0049 and short 0.00459 (control vs control 0.00485 / 0.00531;
PASS); T = 1 distribution check PASS (12,000 samples); 1M needle (968,707 tokens) found in 186.9 s. Median first
token after 60 s idle 188.8 / 190.8 / 189.5 ms; idle RDMA proxy CPU summed over the four nodes 9.45 / 2.13 / 9.25 % of
a core (v2.6.2 has no idle park). Every rank of the v2.6.5 boot logged the v41n pool (912.25 B per full token,
8,541,952 tokens) and its native reader, SWA thin, the cone replay, the decode yield, the #66 named exit and the
history repair as installed, and trim2 was armed in the head's `launch_server` (10 trims, 12,101 MiB returned, longest
493 ms), with no fallback and no error; the FlashInfer #7676 fence was present on every rank of the v2.6.5 boot.
- **KV pool:** 4,316,160 / 8,541,952 / 4,552,192 tokens (the controls size their pool from the free memory at boot:
  4,997,632 / 5,101,824 in the v2.6.4 gate).
- **Memory:** the head's lowest MemAvailable over the boot 8.77 / 7.15 / 10.22 GiB (the judge's floors: 3.5 GiB KILL,
  4.0 GiB MARGINAL); at the soak's 15th minute the schedulers held 7.77 / 7.82 / 7.77 GiB and the tokenizer process
  4.12 / 1.63 / 3.13 GiB; NVRM out-of-memory lines 0 / 0 / 0.
- **Evicted resume point:** a 302,048-token prompt of natural text (this repository's own files), first the document,
  then 800 distinct 1K-token prompts that pushed its SWA window out (800 of 800 OK, 149.9 s), then the document plus
  2,048 tokens. Every node of the v2.6.5 boot replayed one 8,192-row chunk from token 291,328, with 0 reverts: first
  token 2.21 s, 291,328 tokens cached. The controls answered the same request with a full prefill in 53.3 / 50.8 s.
  The v2.6.5 output (16 greedy tokens) equals the second control's token for token; the two controls differ from each
  other after 9 tokens.
- **Soak** (issue #16 reproduction, seeded document order, plus a c2 companion load): 268 / 307 / 261 requests with 0
  failures, companion 39.57 / 49.28 / 38.92 tok/s (v2.6.5 keeps the soak's documents cached where the controls
  re-prefill them), companion token-gap p99 2447 / 1712 / 2445 ms.

### v41n (`DSV41_KV_LAYOUT=v41c`, `DSV41_V41C_READER=native`)

- **v41n on v2.6.4, off by default** (`DSV41_KV_LAYOUT=v41c DSV41_V41C_READER=native`): the ratio-1 / ratio-2
  compressed KV pools hold SGLang's own V4.1 FP4 records (288 B instead of the 584-byte V4 slot; `adapter/v41c_pool.py`)
  and the sparse-MLA readers convert them in place (`runtime/flashinfer_v41c`, DSV41_V41C_READER=native). Port of
  `perf/v41n-cap-atk-20261103` @ `15c3e33` (the capacity A/B branch: v2.6.3 + `perf/v41n-native-reader-20261029`
  `8531d87` + the autotune_keep fixes `59c719a` / `15c3e33`): `adapter/v41c_pool.py`, the sitecustomize hooks (pool,
  configurator, backend pins + layer tags), the canary's two v41c routing blocks, the `b12x_next` reference writer
  arithmetic (`runtime/b12x_next/v41c-provenance.json`, test only), `adapter/autotune_keep.py` (the v41c / v41n keys
  volatile; `--max-total-tokens` out of the fingerprint; `.launch` kept in the raw-argv form an older adapter reads,
  `.launch-pool` beside it) and the v41c / v41n tests. `runtime/flashinfer_v41c` is v2.6.4's directory unchanged
  (module tag `00505fe850b7`: v2.6.4's swan module already carries the v41c / v41n readers), so the module and .so
  of a v2.6.4 build serve both. With the keys unset nothing of this is imported or routed: the finder list, the
  canary (minus its v41c blocks) and sitecustomize (minus its v41c branches) are v2.6.4's (tests below).
- **Autotune.** The v41c / v41n keys are volatile for `DSV41_AUTOTUNE_KEEP`: they change the KV pool layout and the
  attention readers, never a tuned GEMM / MoE shape; the native reader never tunes and resolves the stock decode-dsv4
  tactic for each shape from the kept table (`native_tactic`: the runner keeps the stock class name), so a v41n boot
  must reuse the cache a V4 boot of the same launch tuned (a re-tune would change the MoE tactics and so the numerics
  it is compared on: the 2026-11-03 capacity A/B, where every arm re-tuned, was invalid for that reason).
- **SWA cone replay x v41c.** The cone copies every full-page-addressed pool buffer page for page and redirects the
  replay rows' compressor writes to the padding slot. Under v41c the ratio-1 / ratio-2 buffers are V4.1 FP4 pages
  (73,728 / 36,864 B per full page, data and scale rows in one row), and the prefill latent write of those pools is
  `_low_ratio_write_group`'s V41_FP4 branch with the same `slots`. `tests/test_swa_cone_v41c.py` (real SGLang pool
  construction with the v41c patch, the real `_low_ratio_compress_torch` / `_low_ratio_write_group` on it, CPU writers
  of the same page ABI at the CUDA leaves): the cone's buffer list equals SGLang's own `get_contiguous_buf_infos` on
  v41c and V4; the replay rows' FP4 latent and index-key slots are 0 and the copied pages keep their bytes; without
  the mask the same writes overwrite them; both adapters pin the same engine text; all hooks install together.
- Tests: `test_v41c_pool.py` (base sitecustomize = v2.6.4's), `test_v41c_reader.py`, `test_v41n_native.py`,
  `test_v41n_page.py`, `test_v41c_fast.py`, `test_v41_kv_numerics.py`, `test_autotune_keep.py` (cross-adapter reuse),
  `test_prefill_native_extra.py`, `test_v41c_offsets_64bit.py` (from `fix/v41n-cap-kill-20261105` `4fa4f70`: FP4
  record addressing stays 64-bit past 2 GiB), `test_swa_cone_v41c.py`; the last runs in the canary-roce build, as
  `test_v41c_pool.py` does in both canary builds.

### Tokenizer heap trim (`DSV41_TOKENIZER_HEAP=trim2`)

- **New, off by default: `DSV41_TOKENIZER_HEAP=trim2`** (`adapter/heap_trim2.py`). The head's tokenizer process
  (`launch_server`) keeps the memory it used for long prompts: with several 200K-900K-token prompts in flight its
  glibc heap grows by GiBs of free but resident pages (v2.6.4 on the four Sparks: 1.9 -> 5.1 / 7.1 GiB in a 15-minute
  long-prompt soak, still rising; the schedulers flat). trim2 returns it with `malloc_trim(0)`: after >= 256 MiB of
  new heap, at most once per 120 s, in a gap of the tokenizer (1.5 s without a prompt being rendered or encoded) or,
  when prompts arrive back to back, 30 s later at the next end of a prompt's tokenization (on that thread, so no prompt
  is being encoded during the call). The v2.6.3 candidate's `trim` (every 2 s, reverted) released the same pages over
  and over and cost long prompts 17 % of their throughput; trim2's interval bounds that cost. Outputs are unchanged
  (allocation only); the key and its knobs are volatile for `DSV41_AUTOTUNE_KEEP`. Tests:
  `tests/test_tokenizer_heap_trim2.py` (CI and the canary-roce image). On the four Sparks (2026-11-06, one A / X / A'
  window, 30-minute soak): launch_server 1.0-3.1 GiB with trim2 against 3.9-4.3 GiB at the controls' soak end, 19
  trims returning 27.8 GiB, long-prompt count, companion tok/s and c1 / c4 speed within 1 % of the controls, 0 of 74
  stable qeval tasks changed; the longest trim held the HTTP event loop 0.79 s (busy trims run between two prompts on
  the thread that just finished a tokenization). The `hooks` receipt is now cumulative and printed again whenever a
  later OpenAI serving class is hooked, so the last line names every hooked method (the first one was printed before
  `serving_chat` was imported and named only the base).

### Loop-abort exit and history repair (#66; on main since 540bf07)

A loop-aborted request now ends as a named error instead of a normal stop, and a request whose history holds a
broken tool call is no longer rejected with HTTP 400 ([#66](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/66)).
Ported from upstream: [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks#55](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/55)
(upstream commit `bdf657c`, issue [#54](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/54)).
Requests the loop detector does not stop are unchanged, token for token; the detector's thresholds and what it keeps
are unchanged.

- **Fix: named exit for the loop abort** (`adapter/loop_abort.py`, `DSV41_LOOP_NAMED_EXIT`, default 1). The detector
  used to finish a looping request with `finish_reason=stop` / `matched_stop=repetition`, which a client cannot tell
  from a finished answer: an agent client (OpenCode) took a reasoning-only step that ended this way as the end of its
  turn and sat idle, and a tool call cut mid-arguments looked like a complete one. A hit now finishes the request with
  SGLang's `FINISH_ABORT` (HTTP 500 status, error type `loop_abort`) and a message that names the cause, e.g.
  `loop_abort: DSV41 repetition (art=ngram, period=32, output_tokens=1102, kept=823)`. Streaming clients get an error
  event after the tokens already sent; non-streaming clients get HTTP 500 with that message. On the Anthropic
  Messages endpoint the stream ends with an `api_error` event carrying the message; a non-streaming Messages request
  gets HTTP 500 with SGLang's generic 5xx text. The log line gains `art=`, `period=` and `exit=`.
  `DSV41_LOOP_NAMED_EXIT=0` restores the stop exit. The old exit had one more defect that goes away with it: SGLang's
  detokenizer trims a string `matched_stop` at its first occurrence in the finished output, so an aborted answer that
  contained the word "repetition" was also cut there.
- **Fix: broken tool calls in the request history** (`adapter/history_repair.py`, `DSV41_HISTORY_REPAIR`, default 1).
  An agent client keeps a truncated tool call in its history next to its own "invalid JSON, try again" tool result;
  SGLang's `parse_tool_call_arguments` rejected that history with HTTP 400, which agent clients do not retry. Tool-call
  arguments that are not a JSON object are now passed to the template as `{"_unparseable_arguments": <raw text>}`
  and logged as `DSV41 history repair: len=... reason=json|not_object`, so the model sees its own broken call. Valid
  arguments go through SGLang's parser unchanged. `DSV41_HISTORY_REPAIR=0` restores the 400.
- `start.sh` passes both switches to every container (head and workers), like the other `DSV41_LOOP_*` keys, so a
  value set in `.env.tp4` takes effect; both are documented in `.env.tp4.example` / `.env.example`.
- `scripts/ds_gate.py tscan` and `scripts/prefix_scan.py` count the named exit (an HTTP error whose message starts
  with `loop_abort:`) as a loop / degenerate output, as they counted `matched_stop=repetition`, instead of as a
  request error.
- **Tests:** upstream's `tests/test_loop_abort.py` (15, three new) and `tests/test_history_repair.py` (6), in the
  build of all three images. New `tests/test_loop_exit_passthrough.py` (`Dockerfile.canary-roce` build, against the
  image's own SGLang tree): the detector decides hit / no hit and the kept length exactly as v2.6.3's on a fuzz
  corpus; a decode loop with speculative accept lengths, EOS and the length cap gives identical tokens, finish step,
  finish reason and `finished_len` with no hook, the named exit, `DSV41_LOOP_NAMED_EXIT=0` and `DSV41_LOOP_ABORT=0`
  for every request the detector does not stop; for stopped requests the two exits keep the same tokens; valid tool
  histories encode to byte-identical prompts through the pinned `encoding_dsv41` with and without the repair; and
  the serving layer still has the `FINISH_ABORT` signature, the streaming error branch for an `HTTPStatus` abort and
  the non-streaming HTTP 500 that the named exit relies on.
- **Credits:** Alexander Busse ([GanyX19](https://github.com/GanyX19)) found the problem on a TP4 agent workload and
  wrote the fix (MiaAI-Lab#54, #55, merged upstream by MiaAI-Lab as `bdf657c`); `adapter/history_repair.py`,
  `tests/test_history_repair.py` and the `loop_abort.py` / `test_loop_abort.py` changes are his, unchanged.
  [slidonnici](https://github.com/slidonnici) reported the stall on this repository with OpenCode (#66).

## 2026-11-05 v2.6.4

Four switches added to the production profile in `.env.tp4.example`. `DSV41_SWA_NATIVE_DECODE=1` ("swan"): the
decode attention reads the sliding-window (SWA) cache in place instead of copying it into smaller pages first. Outputs
are bit-identical; greedy c1 decode +1.4 / +1.9 % in two in-boot A/Bs on the four Sparks. `DSV41_DECODE_YIELD=33`:
while one prompt prefills in chunks, the other running streams keep about 40 % of their decode rate instead of 8 %
(measured on the four Sparks), and that prompt's first token comes x1.47 later. `DSV41_SWA_THIN=1` with
`--swa-prefix-tails 384` (was 128): idle agent sessions keep their prefix-cache resume points (resumed 300-330K
sessions 3.9-6.5 s instead of 61-64 s on the four Sparks), for a full KV pool of 3.44-3.75M instead of 5.0-5.16M
tokens.
`DSV41_SWA_CONE_REPLAY=1` with `DSV41_SWA_CONE_TOKENS=8192`: a resume point that is evicted anyway costs a
one-chunk replay (1.6-2.1 s) instead of a whole-context re-prefill (22-68 s at 130-400K tokens). Unset all four
(and set `--swa-prefix-tails 128`) to return to v2.6.3; each can be unset alone. All the new keys are volatile for
`DSV41_AUTOTUNE_KEEP` (toggling them reuses the tuned MoE tactics, see below); the tails change is a launch argument,
so the first boot after it re-tunes once.

- **KV pool: ~5.1M -> ~3.4-3.75M tokens** (`--swa-prefix-tails 384`). The SWA pool is a fixed cost subtracted from the
  KV budget before the full pool is sized, so 98,304 more SWA slots (~23.9 KB each per rank, +2.2 GiB) cost ~1.4M
  full-pool tokens: measured 5.06-5.16M -> 3.60-3.75M on the four Sparks, and 4,997,632 / 5,101,824 -> 3,444,224
  in the release gate (below; no KV pin, each boot sized its own pool). The static pool and the head's
  MemAvailable are unchanged (low-water 12.69 / 12.50 GiB with it vs 12.71 / 12.64 without). Three 1M-token
  contexts still fit, four no longer do; the production peak in flight over a full day was 1.91M. **Rollback is
  config only:** unset `DSV41_SWA_THIN` and set `--swa-prefix-tails 128` (256 is the middle step, ~4.4M tokens).
- **New in the production profile: SWA pages read in place by the decode** (`DSV41_SWA_NATIVE_DECODE=1`). Every
  decode attention call (<= 64 query rows: the DSpark verify and draft forwards up to c4) used to copy each touched
  256-token SWA page into four 64-token pages first (`_split_kv_pages_to_64` in `runtime/flash_mla_sm120.canary.py`:
  mask fill + page mark + page split, 43 times per c1 step), because FlashInfer's decode-dsv4 kernel is instantiated
  for 64-token pages only. Its address math is generic in the page size, so the same kernel instantiated at 256-token
  pages (`runtime/flashinfer_v41c/csrc/swan_decode_dsv4.cu`, FlashInfer's launcher and stock kernel body) reads the
  SWA pool in place and loads exactly the bytes the split copy held: same shared-memory rows, same tactic (the stock
  call's AutoTuner entry), same output bits. Calls it does not serve (other head counts / top-k, > 64 rows, AutoTuner
  tuning mode) keep the split, counted and logged once per reason; the module's own default stays off, the profile
  turns it on. The module (`runtime/flashinfer_v41c`, module tag `00505fe850b7`) is built at image build next to the
  K8 module (`dsv41_v41c build` / `verify`, .so sha256 recorded and checked at load); it also compiles the V4.1 FP4
  readers of the same directory, which have no caller in this tree (about 25 s of nvcc in total). Offline:
  `tests/test_swa_native_decode.py` (load rule on the native view == on the canary's own Triton split, launcher text,
  routing, canary block, the profile line and the autotune key), `tests/v41n_nvcc_check.sh` (sm_121a: the 64-token
  twin of the swan kernel has the stock kernel's SASS; the 256-token kernel differs only in the page constant).
  - **One GB10** (2026-11-02, 38 per-layer SWA pools, CUDA graphs): output bits equal to the split path in every
    cell (short / 64K / 256K windows, c1 / c4, 16 and 64 heads, top-k 128 / 192, after replay and in-place
    mutation); per 38-layer step -164 to -276 us, 8 of 8 rounds faster in every cell.
  - **Four Sparks, TP4** (v2.6.3 image + this module, production profile, abliterated 1017 `517efab`, KV pool pinned
    at 4,194,304; one boot with both variants' CUDA graphs captured and the switch toggled between speedbench full
    runs in A B B A A B order): two A/Bs on 2026-11-02 / 11-03. Greedy c1 prose + code +1.36 % and +1.86 % (A's own
    run-to-run spread 1.36 % / 1.59 %; in the second, means of three runs: prose 65.5 -> 66.9, code 112.1 -> 114.0
    tok/s), greedy c4 -0.06 % and +0.85 %, sampled c1 -1.95 % and +0.79 % (the sampled cells' spread between A runs
    is up to 3 %). T = 0 outputs: 76 of 77 prompts (qeval 75 + 16K + 300K cold) gave the same text in both A runs,
    and swan's text equals A's on all 76.
- **New in the production profile (share 33): running streams keep a decode share while a long prompt prefills**
  (`DSV41_DECODE_YIELD=33`, `adapter/decode_yield.py`; the module is off when the variable is unset).
  - **Before.** Between two 8192-token chunks of a long prompt (~1.35-1.7 s each) the engine runs only the
    `--prefill-decode-interval` decode steps (2 in the production profile, ~0.06-0.18 s). In the 2026-11-03/04
    production log every long prefill with streams running shows 20 chunks per 40-step `Decode batch` line, and the
    other agents' gen throughput falls from 100-160 to 3-28 tok/s for 0.5-2 minutes.
  - **Now.** After a non-final chunk beside running streams, the next prefill is deferred until the streams have had
    decode time worth `pct / (100 - pct)` of that chunk's time (50: as much decode as chunk time). Both times come from
    an integer cost model fitted to the production log (8192-token chunk 1.3545 s + 0.487 us per prefix token; decode
    step 26.3 ms + 15.2 ms per running request), never a clock. The deferral is the engine's own
    `_should_defer_prefill` path that `--prefill-decode-interval` already uses between chunks in production. The yield
    ends at once when no stream is left or the chunked prompt is gone, is never shorter than the configured interval
    and never longer than 128 steps. Nothing is armed after a final chunk, a solo short pass, a chunk with no stream
    running (a lone long prompt prefills exactly as before), or under DP attention / dLLM.
  - **Four Sparks, TP4 (measured 2026-11-05, run `run-20261105-064842-138505`, verdict PASS on every pre-registered
    rule):** fresh boots A / X / A' on the v2.6.2 image plus this adapter, production profile, abliterated 1017
    `517efab`; X at share 50 with the test-only rid arms, so one boot measured shares 0, 33 and 50 (the loop guard
    `DSV41_LOOP_ABORT=0` in all three boots: the workload's random-token streams would otherwise trip it). Two streams
    decoding on 300K contexts (T 0) while a cold 500K prompt prefilled: the streams' decode rate during the prefill,
    relative to their rate in the 20 s before it, **0.080 stock -> 0.393 at share 33** (0.524 at 50); the 500K
    prompt's time to first token **x1.47 at 33** (x1.94 at 50). Switch off (A, A') = stock scheduling. Speedbench
    full with X (share 50) vs the mean of A and A': c1 prose +0.99 %, c1 code +0.76 %, c4 prose +0.72 %, c4 code
    +0.81 % (no cost outside long prefills). The T 0 identity rule (16K, 300K cold) and the long prompt's
    first-output-ids rule passed. Share 33 is the production compromise between the streams' rate and the long prompt's first token.
  - **Trade-off (model, 2 streams beside a 500K cold prompt):** share 50 gives the streams ~7x their decode steps during
    the prefill (0.51 instead of 0.07 of the time) and makes that prompt's first token ~1.87x later; share 33: ~4.7x
    and ~1.40x. Streams that finish sooner hand the time back.
  - **Exactness / rank invariance:** only when the next chunk runs changes; chunk boundaries, chunk batches and decode
    batches are the engine's own. Every input is replicated scheduler state (the scheduled batch, the running batch
    size, the chunk position, the request id) and integer constants, so all TP ranks take the same decision.
    `DSV41_DECODE_YIELD_RID_ARMS=1` (test only) lets a long prompt's request id (`dyield-s<NN>-...`) pick the share,
    for in-boot A/B; never set it in serving.
  - **Tests:** `tests/test_decode_yield.py` drives the pinned `get_next_batch_to_run`, `_should_defer_prefill` and
    `_arm_prefill_decode_interval` (lifted from the staged SGLang source) through a simulated event loop: off and
    share 0 are the stock trace round for round, the long prompt's chunks are unchanged, the share is reached, a lone
    prompt / final chunk / solo pass arm nothing, a yield ends when the streams finish or the prompt is aborted, four
    simulated ranks with different clocks and environments produce one trace, pins, drift refusal, env parsing; the
    v2.6.4 profile (`Profile`: share 33 last on the production line and only there, volatile keys, module default off,
    both canary Dockerfiles run the test on the image's own scheduler source).
  - **Evaluated and not used:** a smaller chunk while streams run (each chunk carries a ~0.33 s fixed cost, derived from
    v2.4's 4096 -> 8192 change, and other chunk boundaries change the long prompt's numerics); `--enable-mixed-chunk`
    (with DSpark the running rows degrade to a 1-token plain decode through the extend path, eager, while the decode
    CUDA graphs and speculative steps are what make them fast); a fixed larger `--prefill-decode-interval` (also
    delays every short prompt queued behind any prefill, and gives a share that varies 2x with the batch size).
  - **Credits:** the deferral mechanism is SGLang's `--prefill-decode-interval`, with **Shangming Cai**'s #44001 gate
    already backported. **Christopher Owen**'s prefill yield (spark-ds41f, vLLM 0545: a long prompt's chunk step goes
    to short prompts and held first tokens) was the starting point of the comparison; it addresses short prompts,
    which our solo short pass covers, rather than the cadence of running streams.
- **New in the production profile: superseded SWA checkpoints are evicted first** (`DSV41_SWA_THIN`, `adapter/swa_thin.py`; off when
  unset). A cached prefix is reusable only at a node whose last sliding window (128 tokens) still holds SWA slots, and
  the SWA pool is small (cap mode: request cap 35,328 + `--swa-prefix-tails` x 384 slots). Every 8192-token prefill
  chunk leaves a 512-slot checkpoint and every agent turn ~512 more, so in stock LRU order an idle session loses its
  resume point after ~70-100K slots of other traffic, and its next turn re-prefills the whole context although its
  full KV is still cached (production 2026-11-04: 8 re-prefills of 170-680K tokens in 6 h, 32-129 s each).
  `1`: a checkpoint its own request has moved past goes to the evict-first end of the SWA LRU: chunk boundaries a
  chunked prefill has passed (a 950K prefill no longer evicts every other document's resume point: the issue16 soak
  lost the 200K and 400K documents after every cold phase, 33 / 67 s each), and the prefix-match node of a request
  that needed no chunked step (an agent turn's previous resume point, now replaced by the turn's own prompt-end and
  output checkpoints). `chunk`: chunk boundaries only. Locked nodes, branch points and a prompt's last chunk boundary
  keep their place; a match on a demoted checkpoint refreshes it as before. Only the LRU order changes: kernels, pool
  sizes and outputs are untouched. Engine source sha256-pinned (refuses to install on drift); no-op with radix
  sessions, HiCache or Mamba. Receipt `DSV41_SWA_THIN receipt: mode= chunk_steps= prompt_steps= demoted= errors=` on
  the first hooked insert and every 1024th. `tests/test_swa_thin.py` (real `UnifiedRadixCache` on CPU, also run in
  the canary-roce image build): stock loses the two older documents under SWA pressure and the switch keeps both; in
  an agent-session pattern the idle session's resume point survives 264 evicted SWA tokens instead of 96; without
  pressure every match and both allocators equal stock.
  - **In the production profile since v2.6.4, with `--swa-prefix-tails 384`.** Four Sparks, TP4, v2.6.2 image +
    this adapter, production profile, abliterated 1017 `517efab`, fresh boots A / X / A' (X = `DSV41_SWA_THIN=1` +
    tails 384). Window 1 (`run-20261105-005746-3350673`), the owner's pattern (two long sessions of 330K / 300K
    tokens idle for 3-14 minutes, a sub-agent turn every second, two new 500K sessions): whole-session re-prefills
    A 2 / A' 2 / X 0; the resumed sessions' first token 62.3 / 64.1 s (A), 61.2 / 63.0 s (A') -> 3.9 / 6.5 s (X),
    cached 338,944 / 326,912 tokens; the other agents' sub-agent turns peaked at 64.4 / 62.2 s during the controls'
    re-prefills and never in X; median sub-agent turn 3.62 / 3.66 -> 3.03 s; uncached prefill 1.44M -> 0.77M tokens.
    Window 2 (`run-20261105-040948-3893907`, six boots, three speedbench runs per cell): mean(X, X') vs mean(A, A')
    c1 prose +0.17 %, c1 code +0.22 %, c4 prose +0.36 %, c4 code +0.78 %; every X run inside the controls' range.
    Neither window passed its pre-registered rules to the letter: window 1 read a c4 code KILL from one run per cell
    (a step-time shift only in X's last ~45 s, coinciding with head timer jobs; not reproduced in six later X runs),
    window 2 is INVALID because its two controls sat 3.2 % apart on c4 prose (head apt / clock-cap jobs during A').
    The review (2026-11-05) judged the whole evidence: ship. T 0 outputs: no candidate-specific difference (the
    controls already disagree with each other on the cold items); head memory unchanged; no guard, Xid or engine
    events. Model of the owner's pattern: whole-session misses 1.41-4.11/h stock -> 0.00-0.01/h.
  - **Known limit:** a request that forks a session from an older turn boundary (an edit or rewind, 2-4 % of
    production requests) falls back further under SWA pressure than stock, since superseded checkpoints go first.
- **New in the production profile: a lost SWA resume point costs a replay of one chunk instead of a whole-context re-prefill**
  (`DSV41_SWA_CONE_REPLAY`, `adapter/swa_cone.py`; off when unset). When a request's prefix match stops early only
  because the last sliding window at its resume point F was evicted while the full KV (ratio-1/2 latents and index
  keys) is still resident, it matches on full KV at s = F - N instead (N = `DSV41_SWA_CONE_TOKENS`, default 8192,
  minimum 3072) and prefills [s, L): the replay rows get the resident latents / index keys copied into their fresh
  pages before the forward, their own latent / index-key writes go to the padding slot, and their window is floored
  at s. Rows from F on depend only on resident KV and on rows >= F - 127 x 20, so they compute what a prefix hit at
  F computes (same class as any hit off a chunk boundary: other chunk shape, top-k order noise). The first pass must
  be a chunk that covers the replay and ends before the prompt end, else the stock match is kept (a whole-prompt
  pass could finish at its first token and insert the replay rows' approximate SWA); the first insert re-points the
  request to the tree's resident pages, and no SWA from inside the replay reaches the tree. Replays only when it
  saves >= `DSV41_SWA_CONE_MIN_SAVE` (8192) rows. Each replay logs one `DSV41_SWA_CONE plan` and one `copy` line;
  `DSV41_SWA_CONE_TEST=1` is test only. `tests/test_swa_cone.py`: the length rule on a toy model of the same
  dependency structure (exact at 127 x 20 rows, not one row fewer, not without the write mask), and on the real
  `UnifiedRadixCache` (CPU, also in the canary-roce image build) the whole turn: plan, page copy, write mask, floor,
  dedup, no slot leaks, resume, the finish-at-prefill path the chunked-pass rule closes, and the import hook next to
  `DSV41_SWA_THIN`, the prefill bypass and the decode yield.
  - **In the production profile since v2.6.4 (`DSV41_SWA_CONE_TOKENS=8192`).** Four Sparks, TP4, v2.6.2 image + this
    adapter at N = 8192 with the test switch (forced eviction before the match, a stock re-prefill reference by
    request id), one boot. Window `run-20261105-082118-415058`: 16 chat prompts of 130-400K tokens, per prompt a
    cold run, three stock re-prefills (ref), two plain prefix hits (warm) and three cone replays, 64 tokens each.
    Cone vs re-prefill distance relative to warm vs re-prefill: G = 1.013 (90 % CI 0.76-1.34, bound 2.0), 0
    confident token flips; geometric-mean symmetric KL of the top-20 lists: cone-warm 0.040 < warm-ref 0.052 <
    cone-ref 0.060 (ref-ref 0.010); a cone arm reproduced a warm arm token for token on 6 prompts. Path: 48 plan
    and 48 copy lines, 0 reverts, after every cone arm the tree held exactly 768 evictable SWA tokens (no replay-row
    SWA adopted). First token: cone 1.63-2.07 s, re-prefill 21.8-68.3 s, plain hit 0.31-0.50 s. The window is
    INVALID by the letter of one rule (median positions compared: the re-prefill references themselves diverge
    after a median of 17 tokens); the first window (`run-20261105-033337-3800238`, raw-text prompts that mostly
    ended at token 0) was INVALID on its A/A ceiling, with 20 plans, 20 copies, 0 reverts. The review (2026-11-05)
    judged the whole evidence: the cone is in the exactness class of a prefix hit; ship on by default with
    N = 8192. Its fleet windows ran without `DSV41_SWA_THIN` and with tails 128; the release gate ran them
    together (below: one forced evicted resume, replayed on every node).
  - **Residual risks (availability, not exactness):** an invariant violation raises (all ranks stop) instead of
    falling back; next to a co-batched prefill, or with a chunk smaller than N, the replay reverts to the stock
    re-prefill; an engine pin drift makes it a logged no-op.
- **`adapter/autotune_keep.py`: `DSV41_SWA_NATIVE_DECODE` is volatile.** Swan never tunes (tuning mode takes the
  stock split) and runs the tactic the stock call resolves for the same shape, so the tuned shapes and kernels are
  the same with it on or off; without this, switching it on would change the launch fingerprint and re-draw every
  fused-MoE tactic on the next boot. **`DSV41_DECODE_YIELD` and `DSV41_DECODE_YIELD_RID_ARMS` are volatile** too:
  the yield only decides when the next prefill chunk runs, in the scheduler, after boot; no GEMM shape, kernel or
  tuning pass depends on it. **`DSV41_SWA_THIN`, `DSV41_SWA_CONE_REPLAY`, `DSV41_SWA_CONE_TOKENS`,
  `DSV41_SWA_CONE_MIN_SAVE` and `DSV41_SWA_CONE_TEST` are volatile:** they act on the prefix-cache tree and on replays
  of a cached prefix after boot; the autotune pass has no cached prefix, so no replay runs while tuning.
- Credits (SWA thin, cone replay): built on SGLang's unified radix cache (SWA LRU, tombstones) and the `dsv4.1`
  branch's decoder bounded replay, whose `swa_replay_start` window floor the replay reuses; the eviction rule, the
  replay, the page copy and the write mask are ours.
- Credits (swan): FlashInfer's SM120 sparse-MLA decode kernel (Gabriel Wu and the contributors of flashinfer-ai/flashinfer
  #3395 and later), whose page-size-generic address math is what lets the stock kernel body read 256-token pages; the
  256-token instantiation, its routing and the tests are ours.

Release gate: PASS (run `run-20261105-170445-1455369`, 2026-11-05, 274 min including the restore of production; 0
KILL, 0 INVALID). v2.6.3 / v2.6.4 / v2.6.3, three fresh boots in that order, production profile, abliterated 1017
`517efab`; the v2.6.4 boot ran the v2.6.3 gate image with this release's files overlaid and the swan module built in
the image, launched with `--swa-prefix-tails 384`. No KV pin (earlier gates pinned 4,194,304): each boot sized its own
pool, so the v2.6.4 boot ran with the pool it ships with. Speedbench 0.3 full, median of two runs: greedy c1
all-prompt geomean 89.34 / 89.09 / 88.73 and prose 65.62 / 65.95 / 65.34 tok/s, c4 131.75 / 132.62 / 131.76
aggregate tok/s; sampled (T 0.7, top-p 0.95, one run per cell) c1 87.30 / 85.72 / 86.01 and prose 62.76 / 61.65 /
62.52 tok/s, c4 129.03 / 129.53 / 129.37 aggregate tok/s; speedbench load c16 215.47 / 216.16 / 216.65 aggregate
tok/s (the full c16 run was not repeated). Cold 64K prefill 6147 / 6137 / 6166 tok/s (median of three flushed
prompts); a cold 388K-token prompt's first token 64.0 / 63.6 / 63.4 s (median of three); a short request beside a
131,072-token prefill 0.43 / 0.42 / 0.47 s. qeval 71 / 71 / 71; at T = 0 the v2.6.4 outputs equal the controls' on
all 74 of the 75 prompts on which the two controls agree. KLD long 0.00677 and short 0.0045 (PASS); T = 1
distribution check PASS (12,000 samples); 1M needle (969,071 tokens) found in 195 s. Idle RDMA proxy CPU summed over
the four nodes 1.75 / 1.63 / 2.68 % of a core; median first token after 60 s idle 189.8 / 190.7 / 192.9 ms. Every
rank of the v2.6.4 boot logged the swan module (`00505fe850b7`, no fallback), SWA thin (mode all, 0 errors), the cone
replay (N 8192) and the decode yield (share 33) as installed; the controls logged none of them. The FlashInfer #7676
fence was present on every rank of all three boots.
- **KV pool:** 4,997,632 / 3,444,224 / 5,101,824 tokens. The 384 prefix tails reserve 182,784 SWA slots instead of
  84,480 (a fixed 4.08 instead of 1.88 GB in the head's pool sizing), so the full pool is 1,605,504 tokens below the
  controls' mean: the price of keeping idle sessions' resume points. Rollback is config only: unset `DSV41_SWA_THIN`
  and set `--swa-prefix-tails 128`.
- **Evicted resume point** (v2.6.4 boot): a 300,000-token document (prefill 49.7 s), then 800 distinct 1K-token
  prompts that pushed its SWA window out (800 of 800 OK, 150.5 s), then the document plus 2,048 tokens. Every node
  planned and copied one cone replay of 8,192 rows from token 291,328, with 0 reverts; first token 2.07 s with
  291,328 of 302,048 prompt tokens cached, 0 errors.
- **Soak** (issue #16 reproduction, seeded document order, plus a c2 companion load, 12 timed minutes): 256 / 318 /
  263 requests with 0 failures, companion 39.07 / 48.96 / 39.11 tok/s (v2.6.4 keeps the soak's documents cached where
  the controls re-prefill them, so its soak does less prefill), head MemAvailable low-water 9.55 / 8.37 / 8.73 GiB.

## 2026-11-02 v2.6.3

Two switches added to the production profile in `.env.tp4.example`: `DSV41_BLOCK_VERIFY_FOLD=1` (sampled
requests decode a little faster: speedbench sampled c1 +0.8 / +1.4 % and c4 +0.5 / +0.7 % against the two v2.6.2
boots of the release gate; the accepted tokens are bit-identical to the torch path for the same random draws) and
`DSV41_ROCE_IDLE_PARK=1` (the RDMA proxy threads stop spinning while the engine is idle: idle proxy CPU 9.6 % ->
1.75 % of a core over the four nodes). Unset both to return to v2.6.2. Greedy outputs are unchanged by either. The
`canary-roce` image also carries one correctness fix with no switch: FlashInfer PR #7676's proxy fence in the b12x
dense MXFP8 GEMM. The tokenizer heap trim (`DSV41_TOKENIZER_HEAP=trim`) of the first v2.6.3 candidate is not in this
release: in a production soak it cut long-prompt (200K-950K tokens) throughput by 17 % (it holds glibc's arena locks
while the tokenizer works); an idle-gated version is deferred to v2.6.4.

- **Fix (correctness): a proxy fence in the b12x SM120/SM121 dense MXFP8 GEMM**, the kernel every MXFP8 dense linear
  runs (`adapter/mxfp8_b12x.py`). The kernel refills a pipeline stage with TMA as soon as `producer_acquire` returns,
  with nothing ordering the MMA warps' `ldmatrix` reads of that stage before the TMA write; under concurrent GPU work a
  tile can be computed from the next pipeline lap's data. Found and fixed upstream by Thireus
  ([flashinfer#7675](https://github.com/flashinfer-ai/flashinfer/issues/7675),
  [flashinfer#7676](https://github.com/flashinfer-ai/flashinfer/pull/7676), commit `c1d1214`): on an RTX 5090,
  1,537 of 17,920 CUDA-graph replays on DeepSeek-V4.1 shapes were wrong with a second stream busy, 0 with the fence.
  Our decode graphs run dense GEMMs beside other work by design (the shared expert on SGLang's `alt_stream` next to
  the routed MoE, the attention multi-stream overlap, the L2-prefetch branch). The image applies PR #7676's line to
  FlashInfer 0.6.18's file at build time (`runtime/flashinfer_fi7676/apply_fi7676.py`: sha256 pinned before and after,
  any other file fails the build; provenance in `runtime/flashinfer_fi7676/PROVENANCE.json`), and every rank logs
  `DSV41 b12x dense GEMM fence (FlashInfer #7676): present|ABSENT ...` at boot. Outputs of an undisturbed GEMM are
  unchanged (the fence orders memory, not arithmetic). Tests: `tests/test_fi7676_fence.py` (also in the image build).
  - **GB10** (2026-11-02, one DGX Spark, FlashInfer 0.6.18 of the v2.6.2 image, production's pinned tactic, nine
    TP4 dense shapes at decode M 6-96 and prefill M 128 / 2048, every replay compared byte for byte with the quiet
    production call): the race reproduces on GB10. Stock kernel: 26 of 2,304 eager prefill calls and 2 of 23,040
    graph replays with a second stream busy were wrong; with the fence, 0 wrong in every mode, and quiet outputs
    byte-identical to stock. Quiet speed with the fence x1.0005 of stock. The routed MoE (b12x_next W4A8, which
    fills shared memory without TMA) under the same noise: 0 wrong replays.
- **New in the production profile: block verification for sampled rows in three kernels**
  (`DSV41_BLOCK_VERIFY_FOLD=1`, `adapter/block_verify_fold.py`). Sampled requests run their acceptance after the
  verify CUDA graph, eagerly; `block_accept_probs` (block verification, since 2026-09-23) issued about 55 small torch launches
  per step for it. The fold computes the per-element and per-request values in three Triton kernels, with torch's
  FP32 operation order, IEEE division and no FMA contraction, and leaves the three vocabulary-wide reductions (the
  acceptance sums S, the residual mass and the correction argmax) to the same torch operations on the same tensors.
  The accepted length and the correction token are therefore bit-identical to the torch path for the same random
  draws, and the draws are unchanged. Greedy requests never take this path. `DSV41_BLOCK_VERIFY_FOLD=check` runs
  both and counts mismatches on device; every mode logs per-path call receipts every
  `DSV41_BLOCK_VERIFY_FOLD_LOG_EVERY` calls (default 2000) and a terminal receipt (with the check counters) at each
  busy -> idle transition of the scheduler, so no call goes unreported (`tests/test_block_verify_fold_receipts.py`).
  - **Tests:** `tests/test_block_verify_fold.py` (Triton interpreter, also in the image build): 75 checks against
    the torch path, including the intermediate S, h, mass, residual and keys, eta exactly at the threshold and one
    ulp either side, ties, NaN / inf keys, a zero residual mass and the full 129,280-token vocabulary.
    `tests/gpu_test_block_verify_fold.py`: the same cases on the GPU, 12,000 random production-shaped rows and
    CUDA-graph replays.
  - **Fleet** (2026-10-29, abliterated 1017): GPU unit test 0 differences, 103 vs 328 us at bs 1; in-boot A/B on
    sampled traffic (T 0.7, top_p 0.95, thinking on) with the fold off / on / check: check mode 0 mismatches,
    T > 0 distribution check PASS, c1 decode step -0.52 % (95 % CI -0.90..-0.11), c4 -0.8 % (CI includes 0).
  - **Credits:** block verification is Sun et al. (ICLR 2025), in this profile since 2026-09-23
    (`adapter/block_verify.py`); the three-kernel fold is our own.
- **New in the production profile: idle parking for the RoCE proxy threads** (`DSV41_ROCE_IDLE_PARK=1`). Each TP rank's
  RDMA proxy thread spins `ROCE_IDLE_SPINS` polls after its last doorbell and then polls once per
  `nanosleep(20 us)` for as long as the engine is idle, about 10^4 wakeups a second per rank. With the
  gate on it blocks on a futex instead, and only while the scheduler itself is idle:
  `Scheduler.on_idle` marks idle (before `--sleep-on-idle`'s poll), `process_input_requests` with
  requests and `run_batch` (before every forward, so before its first collective) wake it with one
  futex wake. The proxy parks after `DSV41_ROCE_IDLE_PARK_MS` (default 1000) of marked idle with no
  doorbell and never with RDMA writes in flight; a decode never parks (`run_batch` clears the mark,
  and `on_idle` runs only without a batch). A parked proxy re-checks its doorbell every
  `DSV41_ROCE_IDLE_PARK_POLL_MS` (default 20), so a missed wake costs at most that, never
  correctness: the existing doorbell catch-up posts every pending op in order. Rank-safe: every
  rank wakes its own proxy before its own kernels, and a collective needs every rank's kernel.
  Needs the v2.6 transport (`DSV41_ROCE_TWOSHOT=1`); with `b12x.comm.roce` or `roce_ring` the
  engine refuses to start. Gate unset: nothing is imported, and the proxy builds with the v2.6
  command and cache name; the additions are seven `#if ROCE_IDLE_PARK` blocks plus the new
  `_roce_idle_park.h`, and without the define the compiled `.text`/`.rodata`/`.data`/`.dynsym` are
  identical to v2.6.1's. Receipts: `DSV41_ROCE_IDLE_PARK: armed`, `DSV41_ROCE_IDLE_PARK rank=R
  proxy_tid=T`, and per idle period `DSV41_ROCE_IDLE_PARK_STATS` (parks, wakes, backstop polls,
  late doorbells, wake latency, wake-to-first-collective lead). Tests: `tests/test_roce_idle_park.py`
  (gate-off identity, adapter and scheduler hooks, SGLang's event loops), with
  `tests/roce_idle_park/park_unit.c` (the state machine, a lost-wakeup race of 20,000 rounds) and
  `park_stress.c` (four real proxy loops over an in-process fake RDMA, ranks waking in random order,
  wakes skipped on one rank).
  Credits: idea: idle RDMA proxy parking (uchkw), independent implementation.
  Fleet A / X / A' (2026-10-29, official 1017, KV pinned at 4,194,304): idle proxy CPU summed over the four
  ranks 9.5 / 1.6 / 9.6 % of a core (6x lower), parks 20 and late doorbells 0 on every rank; speedbench load
  c1 / c4 / c16 x0.998 / x1.002 / x1.020; first token after 60 s idle 190.2 / 193.5 / 192.4 ms; qeval 71 / 71 / 71,
  KLD within noise, head memory and NVRM unchanged.
- **Docs: NVIDIA's NVFP4 checkpoints described correctly**
  ([#64](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/64)). The README and
  `docs/history.md` said they change the routed experts only. Their routed experts are the official MXFP4 values
  re-encoded losslessly with twice the scale bytes (+15.8 GiB in the file, about +4 GiB per rank on the GPU); the
  smaller file (-76 GiB for `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`) comes from the two Engram tables re-quantized
  from FP8 to FP4, which is lossy, and this engine serves Engram rows from NVMe, so it saves no GPU memory. Not used.

Release gate (v2.6.2 vs v2.6.3 vs v2.6.2, fresh boots in that order, production profile, abliterated 1017 `517efab`,
KV pinned at 4,194,304, 2026-11-02; the v2.6.3 boot ran the v2.6.2 image with this release's files overlaid and the
fence applied by the release's own `apply_fi7676.py`): speedbench full, median of two runs, greedy c1 all-prompt
geomean 89.73 / 89.77 / 89.94 and prose 65.94 / 65.93 / 65.88 tok/s, c4 132.58 / 132.87 / 132.38 and c16
211.77 / 211.81 / 210.85 aggregate tok/s; sampled (T 0.7, top-p 0.95, seed 1) c1 86.31 / 87.03 / 85.87 and prose
63.26 / 63.84 / 62.10 tok/s, c4 129.50 / 130.18 / 129.28 aggregate tok/s; cold prefill 1M 5287 / 5299 / 5307 tok/s;
idle RDMA proxy CPU summed over the four nodes 9.6 % of a core on v2.6.2, 1.75 % on v2.6.3; qeval 71 / 71 / 71, KLD
within the controls' noise, 1M needle found in all three; the fence logged as present on every rank of the v2.6.3
boot.
Long-session soak (separate window, 2026-11-02, five fresh boots v2.6.2 / v2.6.3 / v2.6.3 with the idle park off /
v2.6.3 / v2.6.2; issue #16 reproduction for 30 min plus a c2 speedbench load, the same seeded document order on every
boot): issue #16 requests 868 / 883 / 885 / 883 / 881 with 0 failures, companion 46.84 / 45.88 / 46.45 / 46.12 / 46.21
tok/s, head MemAvailable low-water 10.28 / 10.49 / 10.62 / 10.53 / 10.53 GiB. The release gate's own soak could not
judge memory (its two v2.6.2 boots differed by 2.9 GiB): its document order was drawn from the clock, so each boot
re-prefilled a different set of evicted documents; the soak window above fixes the order.

## 2026-10-28 v2.6.2

A correctness fix in the routed-MoE kernel, a memory fix in prefill attention and a latency fix for short requests
that arrive during a long prefill. The production profile in `.env.tp4.example` adds
`DSV41_PREFILL_NATIVE_EXTRA=1` and `DSV41_PREFILL_BYPASS_SOLO=1`; nothing else in it changes.

- **Fixed: a CTA shared-memory race in the routed-MoE dynamic kernel** (`runtime/b12x_next`,
  `moe/_shared/kernels/dynamic.py`, the producer loop). The CTA leader stores each batch claim in one shared word
  (`ctrl[7]`). A barrier orders that store before every thread reads it, but nothing ordered those reads before the
  next iteration's store, and the loop's back edge has no barrier. A warp that reads its claim late can therefore see
  the next batch's claim, and tokens are then skipped or routed twice, or the loop stops early.
  - **The fix** is a backport of **Martin Vit**'s b12x commit `d7937d7f` (2026-10-06, after our pinned `a7d7d29b`),
    applied verbatim: one CTA barrier right after the claim read (`scripts/b12x_next-claim-barrier.patch`). b12x is now
    developed in flashinfer-ai/flashinfer under `flashinfer/experimental/b12x`. We found the race independently in an
    offline racecheck and SASS analysis before finding the upstream commit.
  - **Not carried:** the same commit also fixes the deterministic shared-activation gathers of the monolithic in-kernel
    FC1 path, which our specializations do not compile (their external phase-1 kernel already divides by top-k).
  - **Code generation** at sm_121a: registers, spills and the FC1 / FC2 kernels are unchanged; the dynamic kernel gains
    one `BAR.SYNC` per producer iteration.
  - **Tests:** `tests/test_b12x_next_claim_barrier.py` holds an exhaustive interleaving model (the stock loop has a
    counterexample with two warps; the barrier closes it) and pins the source. `SOURCE_PATCH` changes from
    da662ccc7e2372b5 to 5555496ebd40f308.
  - **Fleet A / X / A'** (2026-10-27): every T = 0 output of 75 qeval prompts (top-20 logprobs and greedy, c1, three
    passes per boot) equals a v2.6 output, and load c1 / c4 / c16 run at 0.999 / 1.002 / 1.011 of the slower v2.6 boot.
- **New in the production profile: prefill reads the extra cache's native pages** (`DSV41_PREFILL_NATIVE_EXTRA=1`;
  `runtime/flashinfer_pce/`, read by `runtime/flash_mla_sm120.canary.py`).
  - **Before.** Stock FlashInfer instantiates the SM120 sparse-MLA dual-cache prefill only for extra pages of 64 / 2
    tokens. The adapter therefore copied the whole compressed extra pool into a persistent 64-token split buffer, about
    3 GB per rank at the v2.6 pool, allocated after the KV pool is sized, and ran a page-mark and split pass on every
    prefill call.
  - **Now.** A separate module reads SGLang's own 128 / 256-token pages in place. FlashInfer's `prefill_kernel.cuh` is
    generic in the page size, so the kernels are the stock templates; only the dispatch is opened to 128 / 256 and the
    symbols are renamed. The module is generated at image build from the image's own `sparse_mla_sm120` sources: the
    build refuses unless the whole source closure equals `APPROVED_CLOSURE.json` (FlashInfer 0.6.18, git 69ff11fc,
    23 files). It is then compiled for sm_121a in the build and served with its sha256 pinned (no JIT at boot). The SWA
    source is still split, and 64 / 2-token extra pages keep the stock module.
  - **Tests:** `tests/test_prefill_native_extra.py` (CPU), `tests/pce_nvcc_check.sh` (20 kernels, registers and spills
    equal to the stock 64-token twins) and `tests/gpu_test_prefill_native_extra.py` (OFF vs ON bit-identical on
    42 eager and edge cases and 30 CUDA-graph replays of the wide-verify route).
  - **Fleet A / X / A'** (2026-10-28; the coordinator's ruling over two kit faults in the receipt):
    - every T = 0 output (lp-c1 and greedy-c1) equals a v2.6 output, and the module served the 128- and 256-token
      pages;
    - **persistent memory allocated after pool sizing drops by 2.45-2.76 GiB per node**;
    - **cold 1M prefill takes 194.4 s against 205.3 / 201.7 s on v2.6** (-5.2 % against the first boot); 64k and 388K
      are within noise;
    - c16 load is 1.007 x the slower v2.6 boot;
    - no NVRM `NV_ERR_NO_MEMORY` line.
  - **Credits:** the SM120 sparse-MLA kernels are FlashInfer's: **Gabriel Wu** (flashinfer-ai/flashinfer #3395 and the
    later refactors), with fixes from **Ka-Hyun Nam** (#4732, issue #3700) and the other contributors to
    `sparse_mla_sm120`. Origin and changes are in `runtime/flashinfer_pce/PROVENANCE.json` and NOTICE.

- **New in the production profile: a short request that arrives during a long prefill runs in a pass of its own**
  (`DSV41_PREFILL_BYPASS_SOLO=1`, on top of the existing `DSV41_PREFILL_BYPASS=256`; `adapter/prefill_bypass.py`).
  - **Before.** The bypass admitted a short request next to the long prompt's in-flight 8192-token chunk, so the short
    request's first token waited for that whole chunk pass: about 1.9 s at 93K-128K.
  - **Now.** The short request runs in a prefill pass of its own (one per pass, a batch-size-1 extend as on an idle
    server; a second one takes the next solo pass). The long prompt is parked for that pass (the engine's parked-chunk
    state: empty extend range, nothing stashed, no in-flight chunk counted, the batch built without it) and resumes
    alone in the next pass with the same chunk, so its chunks, boundaries and numerics are those of an idle run. Solo
    and chunk passes alternate, so the long prompt is never starved. A final chunk is parked like any other, with the
    bypass admission contract kept beside it (the cumulative fresh-token debit, no partial admission: a short request
    that would need chunking is skipped). The interval cadence is unchanged.
  - **Rank-invariant:** every decision reads replicated scheduler state only (the parked flag lives on the request), so
    all ranks build the same batch. Each solo pass and each resume logs one `DSV41_SOLO_RECEIPT` JSON line (long rid,
    prefix / total, the batch's rids and ranges), the same on every rank. Same pinned `_get_new_batch_prefill_raw` /
    `PrefillAdder` sources and sha pins as the bypass; with the flag unset or 0 the patched function is v2.6.1's byte
    for byte.
  - **Tests:** `tests/test_prefill_bypass_solo.py` (run in both canary image builds): a pass simulator over the lifted
    pinned scheduler (long prompt plus short arrivals, final chunk, continuous arrivals, rank invariance, page-256
    final-chunk admission regressions, receipts), the off path against a v2.6.1 fixture, and refusal on anchor drift.
  - **Fleet A / X / A'** (2026-10-28, v2.6 / v2.6 + solo / v2.6, fresh boots, KV pool pinned, abliterated checkpoint
    on all three):
    - **time to first token of a short request sent 1 s into a cold prefill: 0.42 s against 1.89-1.92 s** (93K and
      128K, median of three each);
    - on every judged run the long prompt's output tokens and the short reply equal their idle runs, and every T = 0
      qeval output (top-20 logprobs and greedy, c1) equals one of the two v2.6 boots';
    - cold prefill alone is unchanged (93K: 14.54 s against 14.65 / 14.57 s; 388K: 65.03 s against 64.93 /
      64.81 s); the short request's total time, the long prompt's time and the decode gaps of a stream running beside
      them stayed within 1.10 x, 1.03 x + 0.5 s and 1.25 x of the slower v2.6 boot;
    - the receipts are the same sequence on all four ranks; no NVRM `NV_ERR_NO_MEMORY` line.
  - **Credits:** our own design on top of our bypass. The idea draws on **Christopher Owen**'s late-admission and
    prefill-interleaving work in [spark-ds41f](https://github.com/christopherowen/spark-ds41f), and the measurement
    follows the `mixed_probe` shape of **James Meadlock**'s J-M Recipes (a short streaming chat sent 1 s after a cold
    long prompt).

<!-- v2.6.2 release gate: filled in from the receipts before the PR (diagnostics RUNBOOK). -->
Release gate (v2.6 vs v2.6.2 vs v2.6, fresh boots, production profile, official 1017, KV pinned at 4,194,304, 2026-10-29): speedbench full c1 88.53 / 88.38 / 88.37, c4 131.17 / 130.72 / 130.12, c16 207.67 / 207.62 / 206.03 tok/s; cold prefill 64K 6029 / 6054 / 6033 and 1M 4932 / 5310 / 4940 tok/s (+7.7 %); short-reply TTFT during a 93K / 128K prefill 1.90 / 0.43 / 1.90 s and 1.90 / 0.42 / 1.89 s; 388K prefill 65.3 / 63.3 / 64.9 s; qeval 71 / 71 / 71 with no answer outside the controls, KLD within noise, 1M needle found in all three; lowest head MemAvailable 4.2 / 9.7 / 7.0 GiB and NVRM out-of-memory log lines 155 / 1 / 133 over the boot.

## 2026-10-27 v2.6.1

- Fixed: building `Dockerfile.canary-roce` failed in the image test battery at
  `tests/test_roce_ring.py` (`ImageOverlay.test_overlay_selects_package`). The test still asserted
  v2.5's two-way package selection in `pynccl.py`, which `runtime/roce_twoshot_adapt.py` turns into a
  three-way one in v2.6. The test now asserts the v2.6 selection and the refusal of
  `DSV41_ROCE_RING=1` together with `DSV41_ROCE_TWOSHOT=1`. No runtime change. Reported with the fix by
  **Saolence** in #59.

## 2026-10-27 v2.6

Production profile = v2.5 plus the two-shot all-reduce for wide steps and two prefill host-overhead
cuts, on an image that also carries the block-verification threshold fix. `.env.tp4.example`'s
"Production profile" line adds `DSV41_ROCE_TWOSHOT=1 DSV41_ROCE_TWOSHOT_ROWS=18
B12X_ROCE_PIPE_BLOCKS=4 B12X_ROCE_PIPE_CHUNK_BYTES=65536 DSV41_MOE_LAUNCH_PREPARED=plan
DSV41_INDEXER_NOSYNC=1`; every other setting is unchanged.

Fleet A / B / A' on 2026-10-26: four fresh boots in order (production, the two-shot switches only,
the full set, production again), host memory compacted before each, abliterated 1017 `517efab`,
GPU clock cap 2200 MHz, speedbench 0.3 (`6733653`), thinking off, first run after each boot
discarded. A difference counts only beyond both production boots by more than their own
difference (floor 0.5 %).

| metric | production (2 boots) | two-shot only | full set |
|---|---:|---:|---:|
| c1 all-prompt geomean tok/s (2 full runs per boot) | 88.91 / 88.93 | 88.97 | 88.56 (noise) |
| c1 prose geomean tok/s | 65.34 / 65.59 | 65.33 | 65.27 (noise) |
| c4 block tok/s | 131.56 / 131.30 | 131.64 | 131.74 (noise) |
| c16 block tok/s | 201.33 / 201.20 | 208.66 (+3.7 %) | 211.02 (+4.9 %) |
| cold prefill 4k tok/s | 4,555 / 4,586 | 4,613 | 5,309 (+16.2 %) |
| cold prefill 64k tok/s | 5,779 / 5,773 | 5,803 | 5,970 (+3.4 %) |
| TTFT, c1 full median | 213.7 / 215.6 ms | 216.3 ms | 206.4 ms (-3.9 %) |
| TTFT, load c1 median | 215.5 / 217.5 ms | 218.9 ms | 206.7 ms (-4.5 %) |
| TTFT, load c16 median | 564.1 / 561.8 ms | 547.6 ms (-2.7 %) | 539.3 ms (-4.2 %) |

Exactness: 75 qeval prompts at T = 0 with top-20 logprobs and cold (salted) prefills on every
boot; both new arms byte-identical to production on all 75, and the two production boots identical
to each other.

Release gate on the full set (one more fresh boot, same weights and protocol): speedbench full c1
all-prompt geomean 88.84 and 89.80 (prose 65.04 / 66.14, code 111.57 / 112.84), median TTFT 206 /
208 ms, 0 of 73 failed; `scripts/qeval.py` 71 / 75 (code 24/25, format 7/7, json 6/8, math 13/14,
prose 5/5, reason 16/16), the same score and split as v2.5; T > 0 scan 0 / 35; needles PASS at
124k (19.8 s) and 388k tokens (66.6 s); cold 388K prefill 65.2 s (5,770 tok/s; 69.0 s on v2.5);
`/v1/messages` with the image placeholder token in text and with thinking enabled without a
budget both answered; issue #16 accelerated reproducer 15 minutes, 410 requests, 0 failures. A
T > 0 distribution check (production boot, 12,000 samples, 3 tokens, T 1.0): the second token's
counts match the verify row's own probabilities (chi-square p 0.61).

- Two-shot RoCE all-reduce for wide steps (`DSV41_ROCE_TWOSHOT=1`, `DSV41_ROCE_TWOSHOT_ROWS=18`,
  `B12X_ROCE_PIPE_BLOCKS=4`, `B12X_ROCE_PIPE_CHUNK_BYTES=65536`; unset = one-shot only, as
  before). At c16 the TP all-reduce moves 983 KB and the one-shot sends three full copies per rank
  (163-175 us per call, ~14 ms per step on the wire). The pipelined two-shot of Luke Alonso's
  flashinfer#6915 sends each rank's quarter to its owner, which sums the four partials in FP32 in
  rank order exactly as the one-shot does, then gathers the reduced rows back: half the bytes,
  bit-identical output. All-reduces of 18 rows or more take it; smaller ones stay one-shot. Ported
  as a separate package `runtime/b12x/b12x/comm/roce_twoshot` (the `roce` package is byte for byte
  untouched; `runtime/roce_twoshot_adapt.py` selects the copy in pynccl.py when the switch is set),
  routed per call by row count through `adapter/roce_twoshot.py` (an in-boot A/B key). A two-shot
  op after the 32-bit sequence wrap fails closed, and every kernel of the copy poisons the runtime
  at sequence 0 instead of touching a flag or payload. Tests: `tests/test_roce_twoshot.py`
  (protocol model bit-identical to the one-shot, ctrl/ABI agreement, the unchanged production
  hashes, the sequence-wrap guard); the proxy builds with `-Werror` and every kernel compiles for
  sm_121a without spills. Fleet: c16 +3.7 % alone, c1 / c4 within noise, T = 0 byte-identical.
  Not with `DSV41_ROCE_RING=1`. Credits: Luke Alonso (flashinfer-ai/flashinfer#6915, the two-shot
  all-reduce), on rhys101's RoCEnante path and local-inference-lab/b12x's RoCEnante transport.

- Prepared eager MoE launch (`DSV41_MOE_LAUNCH_PREPARED=plan`, `adapter/moe_launch_prepared.py`).
  Eager (prefill) calls of b12x_next's `tp_moe_dynamic_launch` repeated the host-side
  quant/routing/materialization/tile decisions and the kernel lookup on every call: 1.45-1.77 ms of
  host time per call on a c1 trace, 40 calls in a host-paced ~200 ms short prefill. `plan` makes
  those decisions once per launch contract (every scalar argument, scratch size, device,
  pre-quantized state, the programmatic overrides) and mirrors b12x main's `_launch_dynamic_flat`
  (`a7d7d29`, source sha256-pinned; refuses to arm on drift). The barrier fill, route planner,
  tensor pointers and current stream stay per call; a plan is dropped on any `B12X_NEXT_*`
  environment write and whenever the kernel cache no longer maps it to the same compiled kernel.
  Same kernels, same arguments, same order. Decode runs in CUDA graphs and is untouched. Test:
  `tests/test_moe_launch_prepared.py` (stock and prepared bodies against recording stubs of every
  side effect over 262 production contracts). `plan,direct` (skip the PyTorch dispatcher) and the
  `DSV41_MOE_LAUNCH_TIMING` hooks stay test only. Credits: local-inference-lab/b12x (the launch
  path it mirrors).

- Prefill indexer slot ids without host syncs (`DSV41_INDEXER_NOSYNC=1`, with
  `DSV41_INDEXER_CHUNKED=1`, `adapter/indexer_chunked_v3.py`). The eager prefill indexer of the
  eight index-source layers indexed `req_to_token` with a 0-dim CUDA tensor per request (an
  `.item()`: one stream sync per request and layer) and copied the request starts from pageable
  memory (another sync). It now builds the compressed-slot ids with one device gather and pinned
  non-blocking copies. Integer ops on the same elements in the same order: bit-identical. GPU check
  (`tests/gpu_test_indexer_nosync.py`): 300 random batches equal to the stock loop, PyTorch's sync
  detector raises nothing (it catches the stock loop), host time behind a queued 2 ms kernel 2.3 ms
  -> 0.08 ms, 2-4 MiB extra device memory; top-level syncs per c1 extend on every rank >= 10 -> <= 2.
  CPU test: `tests/test_indexer_nosync.py`. Credits: kpham-sgl (sglang#39187, the chunked indexer
  backport this edits).

- Block verification threshold arithmetic (`adapter/block_verify.py`, sampled rows only, always on). The
  acceptance threshold h_i = S_i / (S_i + 1 - w_i) was evaluated as `(S + 1) - w` in FP32, which
  rounds S against 1: with w = 1 - 2^-24 and S = 3e-8 it gave 0.503 for 0.335, with a clipped
  w = 1 and S = 1.4 * 2^-24 it gave 0.70 for 1.0, and in general it moved h by up to
  2^-24 / (S + 1 - w) relative. It is now `S + (1 - w)`; `1 - w` is exact in FP32 for w >= 0.5.
  New `tests/test_block_verify_numerics.py` checks every threshold against an FP64 oracle on the
  same FP32 probabilities (straddling draws, a grid of 1 - w and S, V = 129280 production-shaped
  rows with margins 2-30, ties, -inf, one-hot drafts, p == q, truncated blocks, bf16 inputs) and
  fails on the previous arithmetic. The old association moved the effective (clipped) threshold by
  up to ~0.5 on those rows; its raw value was unbounded. Greedy selection is protected structurally:
  an all-greedy batch never calls this code, and in a mixed batch it runs on every row while
  SelectMixedAccept keeps the greedy rows' own AcceptGreedy result (greedy includes top_k <= 1, also
  at temperature > 0). The draws consumed per call are unchanged. Affected: sampled rows of
  sampled requests, at a verification step whose draw lands between the old and the new threshold.
  On synthetic production-shaped rows (fixed drafts, offline) that was at most 7.4e-8 total
  variation of the accepted length, and at most 9.7e-8 accepted-length mismatch under the
  implementation's shared-uniform coupling, per step; these are synthetic observations, not a
  bound over production histories. This is a correctness fix, not
  a speed change, and it does not touch the FP32 rounding of the probabilities themselves
  (SoftmaxTemp), which dominates the remaining error for confident tokens. Credit: Barron,
  Gnanapragasam and Tschannen's softmax fix as applied in Christopher Owen's spark-ds41f #584.

- `stop.sh`: the shared NFS exporter it keeps (`dsv41-nfs`, or a reused `vllm-fn-nfs`/`glm53-nfs`) is no longer
  reported as "still running", the `kept:` line names the exporter it found, and the banner counts the head plus the
  configured workers instead of a fixed 3x. Report only; cleanup unchanged. Tests in `tests/test_stop_nfs_filter.py`;
  checked live on four Sparks. Credits: SvangenStudios
  ([#57](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/57)).

- Ring mesh (`scripts/ring_mesh/plan.py`, `hairpin.sh`): the generated `show | grep -q ... || add` checks
  read the whole output (`grep ... >/dev/null`), so under `set -o pipefail` a show command that still had output when
  `grep -q` exited no longer died of SIGPIPE and re-added an existing rule ("Filter already exists", mesh-up aborted).
  Test `tests/test_ring_mesh_pipefail.py` (stub `tc`/`ip`, output larger than a pipe buffer), now in CI. Credits:
  joeym82956 ([#54](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/54)).

## 2026-10-23

- Engram NVMe keepalive (`adapter/row_store.cpp`, `DSV41_ENGRAM_KEEPALIVE_MS`, default 25, 0 = off).
  Under Linux APST the GB10's Samsung drive enters a low-power state after ~100 ms without commands
  and the next read pays ~10 ms to wake it (bursts of 200 random 4 KiB O_DIRECT reads on 96 threads:
  150 ms gaps -> median per-burst max 9.7 ms; 40-80 ms gaps -> 0.8-0.9 ms). A c16 decode step is
  ~145 ms and issues its Engram reads in one burst, so every step paid the wake-up: the pool measured
  2.0-2.4 ms per read at c16 against ~0.5 ms at c1. While lookups arrived in the last 2 s, a
  background thread now reads one pseudo-random 4 KiB page of the packed shard every 25 ms (own
  descriptor and buffer; served rows untouched); an idle server still lets the drive sleep. Fleet
  A/B/A (v2.5 image, production profile, abliterated 1017, three fresh boots): sparkDash c16 prose
  342.4 / 342.8 vs 323.5 off, code 476.3 / 477.3 vs 456.1; c4 +0.4-1.7 %; c1 unchanged. Rank-0
  trace at c16: step 149.2 -> 139.7 / 141.7 ms, layer-1 Engram join 10.5 -> 4.0 ms after the hasher.
  The thread pins the library (no unload under it), uses a fork-excluded buffer, and a failure to
  start it leaves serving as before. Test: `tests/test_row_store_keepalive.py`. Full release gate
  on the final code (abliterated, production profile): sparkDash c1 prose / code / structured / json
  89.0 / 145.2 / 160.1 / 131.4, c16 prose 355.7, code 479.1; speedbench all 87.6 (85.9 before);
  needles 124k/388k PASS, cold 388K 69.0 s, qeval 71/75 (same split as v2.5), T > 0 scan
  0/35, KL long/short 0.0075 / 0.0126 PASS, issue #16 soak 1,018 requests in 30 min, 0 failures,
  no CUDA error on any rank.

- Two weights profiles in `.env.tp4.example`: the official `deepseek-ai/DeepSeek-V4.1-Flash-1017`
  (`1bae0e1`, now the default `HF_REPO`/`HF_REVISION`, `MODEL_DIR` and NFS names) and the
  abliterated `mrfakename/DeepSeek-V4.1-Flash-1017-abliterated` (`517efab`); the original V4.1
  release stays as a commented option. Measured on the v2.5 image and production profile, same
  protocol: official prose / code / structured / json c1 91.0 / 140.3 / 159.7 / 130.5, speedbench
  all 86.1; abliterated 88.6 / 144.9 / 158.4 / 128.1, 85.9. README now leads with the official
  checkpoint and lists both.

- `DSV41_WO_A_INPUT_FP8` measured on the fleet (v2.5 image, production profile, same boot protocol,
  flag on vs off): sparkDash prose / code c1 85.4 / 140.3 vs 88.6 / 144.9, speedbench all 85.1 vs
  85.9; qeval 71/75, T>0 scan 0/35 and the KL panels pass either way; the GPU unit test passes with
  the hardware and the emulated e4m3 rounding. No quality difference our evals can see, ~3 % slower
  decode: it stays off by default.

## 2026-10-22 v2.5

Production profile = the v2.4 lever set again, now on an image with the issue #16 backport, the
logprob decode hang fixed and the Linux staging repaired. Image `Dockerfile.canary-roce` at
`f9a1de4`, checkpoint DeepSeek-V4.1-Flash-1017 (abliterated `517efab`), one fresh boot:
sparkDash prose / code / structured / json c1 88.6 / 144.9 / 158.4 / 128.1 (conservative fallback
profile on the same checkpoint: 84.8 / 138.8 / 153.8 / 126.0), speedbench all 85.9 (83.1), cold
387K prefill 68.2 s (77.0 s); KL panels against the stable 1017 image PASS (0.0062 / 0.0061),
qeval 71/75, T>0 scan 0/35. The accelerated issue #16 reproducer passed on two cold boots
(1,445 and 813 requests on 190K-900K cached prefixes, 0 failures, no CUDA error or Xid; see
`docs/issue-16-swa-crash.md`).


- `scripts/fetch-sglang-canary.sh` failed on Linux: GNU patch rejects a hunk whose leading and
  trailing context differ in length unless it sits at the file's start or end, and the #44751,
  #45663 accept-output, #45518 and #45726 integration patches had such hunks, so staging stopped at
  #44751 on a DGX Spark (BSD patch on macOS accepted them, which is where the images were staged).
  The four patches are regenerated with symmetric context from the staged trees; both GNU and BSD
  patch now produce byte-identical files, and the pins follow. CI stages the tree on Ubuntu and
  macOS.

- Backport Divy (divyvasal) and Claude Opus 5.5's open SGLang #44923 (head
  `6e0355d9`): the sampling info keeps a CPU greedy flag per row, and
  `filter_batch` recomputes `is_all_greedy` / `is_any_greedy` from the rows
  that remain. Before, a
  merge could only clear `is_all_greedy`, so once one sampled request had
  joined a greedy decode batch the flag stayed false after it finished, and
  DSpark verify kept taking the eager accept path instead of the folded one
  (`fold_eligible` requires `sampling_info.is_all_greedy`) until the whole
  batch drained. Merges build a new list, so `copy_for_forward()` copies are
  never changed in place; an info built without per-row flags keeps the old
  behaviour. The flags come from request parameters and the scheduler's keep
  indices, so every TP rank makes the same decision. Upstream reports outputs
  unaffected (the stale flag only costs step time); the remaining greedy rows
  go back to the path an all-greedy batch takes anyway. Not yet measured on
  GB10; the fleet check is that a greedy stream returns to the folded path
  after a concurrent sampled request ends.
- Not ported, reviewed against this pin (details in `docs/upstream-watch.md`):
  ormandj's SGLang #42861 (exp2-free FP4 indexer key decode) changes only the
  Hopper Triton scorer, which GB10 never runs here (SM12x decode uses
  DeepGEMM `fp4_paged_mqa_logits`); ormandj's #42864 (vectorized mHC split-K
  reduce, 160 decode slices) is gated to SM120 upstream, needs the newer
  `HcState` model plumbing, and would break the exact-match self-check and
  source pins of `adapter/hc_fused.py`; omkaark's #45086 (no row-count
  specialization of the mHC kernels) edits two kernels whose source hashes
  `adapter/hc_fused.py` pins.
- **`adapter/wo_a_input_fp8.py`, `DSV41_WO_A_INPUT_FP8=1`, off by default (to be A/B-tested on the
  fleet).** Numerics parity with DeepSeek's reference: since revision
  [`4dd75a8b`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/4dd75a8b448d95ae8e284d5c5a1e4b8808bd583b) of
  `inference/model.py` (2026-10-12, also shipped with DeepSeek-V4.1-Flash-1017) the reference rounds the
  attention output `o` to FP8 and back before the bf16 `wo_a` einsum, in the target attention and the
  DSpark draft attention: blocks of 32 values along each group's 4096 features, power-of-two (ue8m0)
  scales, e4m3 round-to-nearest-even. Both checkpoints use 32x32 weight blocks, so the engine
  dequantizes `wo_a` to bf16 and every serving path (stock GEMV / small-batch / MXFP8-epilogue / bmm /
  einsum, `wo_a_w8`'s twin kernels, the draft's fast kernel) fed `o` unrounded; the stock
  `wo_a_bf16_small_batch_mxfp8` quantizes `wo_a`'s output for `wo_b`, not its input. With the flag the
  bf16 dispatch rounds `o` in place first (one Triton kernel), and `wo_a_w8`'s DROP/MID routes round
  the loaded tile in registers (their PTX is unchanged with the flag off). Rank-local rounding is the
  reference's: no block straddles a head, group or TP rank. `tests/test_wo_a_input_fp8.py` checks it
  bit for bit against a transcription of the reference kernel and an independent e4m3 grid rounding.
  No speed or quality effect is claimed; nothing is measured on GB10 yet. Credits: DeepSeek (the
  reference change), 05yuki ([SGLang #45185](https://github.com/sgl-project/sglang/pull/45185), the
  same step in SGLang, open) and christopherowen
  ([spark-ds41f](https://github.com/christopherowen/spark-ds41f), which adopted it in its vLLM recipe).

- Drop the SGLang #45251 grouped static-verify indexer backport: it hung every TP rank in decode
  on a greedy `/generate` with `return_logprob` (the 18,000-token code text of the long KL panel),
  every time, while chat traffic passed. Bisected on the fleet over the 26 backport steps with
  thin overlays of the fix image: 24 steps answer in 7 s, adding #45251 hangs, all steps without
  it answer (prefill SP on, long and short KL panels against stable-1017: continuation KL 0.0067
  and 0.0073, PASS). It had never been measured on GB10.

- Fix `DSV41_PREFILL_SP=1` on the current canary images. The adapter hash-pins
  the engine functions it copies or wraps; the #42901 layer-20 tail (PR #24)
  and #45301 (PR #48) changed two of them, so the adapter refused at import and
  the head died about 20 s into boot. The prefill-SP layer loop now follows
  #42901: the last KV-source layer gathers every row and writes full-prompt
  compressed KV before the tail, as the stock loop does (checked against the
  stock loop: exact mode identical, fast mode identical to P0). The SP region
  is now layers 0-19. A new image-build test imports the model with every
  adapter flag of `.env.tp4.example` on, so a pinned function that drifts fails
  the build instead of the boot.
- The canary images point rustup at an unreachable dist server, so the
  `cargo --version` probe of the image preprocessor fails at once instead of
  holding a rank before torch distributed init (one rank waited 7 minutes).
- `stop.sh` keeps the shared `dsv41-nfs` exporter. Removing it left the worker
  NFS volumes without a server, and the next stop hung on them.
- The image builds again: two source verifiers followed the #45457 DSV4.1
  encode wrapper and the #45726 layout order, and the packaging tests' inputs
  are copied into the image.
- README: current results (sparkDash and speedbench) of this image on the
  conservative profile; the v2.4 table moved to `docs/history.md`.

- Backport maocheng23 and Claude Opus 5.5's merged SGLang #43448
  (`42228038`). When chunked prefill is disabled or unset, V4.1 MHC kernel
  prewarm now derives its token buckets from `max_prefill_tokens` instead of a
  non-positive chunk size. This fixes startup/weight-load failure for valid
  `CHUNKED_PREFILL_SIZE=-1`, `0` and unset configurations without changing the
  default positive-chunk profile. Extracted tests cover all four cases.
- Backport JiaShuai0209 and Liangsheng Yin's merged SGLang #40424
  (`720857de`) to the pinned V4.1 paths. An eager DP-attention extend now
  slices low-ratio source inputs to the hoisted live-row prefix before request
  indexing, so padding cannot create uninitialized request ids and a CUDA
  device assert. Padded idle Engram batches return correctly shaped zero hashes
  before request-table access instead of failing the extend-mode assertion.
  Extracted production-statement tests cover padded, unpadded and idle cases.
- Adapt Liangsheng Yin's merged SGLang #45301 (`117247ea`) to the pinned
  V4.1 attention class split. TP-padded query storage now grows only before a
  decoder-tail CUDA graph captures it; smaller row buckets reuse slices of the
  same allocation, and a shape/dtype expansion after capture fails closed.
  This prevents recorded graph nodes from retaining a freed buffer address and
  the reported MMU fault after cache flush. Source and CPU behavior falsifiers
  cover pointer reuse, capture state, row/head growth refusal and packaging.
- Adapt Justin Tong's merged SGLang #44751 (`a038a0dc`) to the pinned native
  V4.1 encoder. Images embedded in tool results are collected after results are
  sorted into tool-call order, list-valued tool content is preserved, and a
  tool-result-only user message stays in the current assistant turn instead of
  dropping its reasoning. The local placeholder sanitizer still works on
  private copies, so caller-owned request objects remain unchanged. Focused
  tests cover all four accepted message shapes, image order, reasoning
  retention/removal, immutability, zero-fuzz application and image packaging.
- Port three bounded RoCEnante fixes from FujitsuPolycom/sparkring into the
  active switchless-ring overlay. Kernel timeout state now has a flag separate
  from the uint32 sequence, so a failure at sequence zero remains visible after
  wrap; host diagnostics render sequence/epoch values unsigned. Proxy counters
  sampled from Python are C atomics, and the PTX/poll-budget contracts are
  documented precisely. CuTe kernel cache versions advance to 3. CPU simulation
  covers both collectives, both path counts and the uint32 boundary cases; no
  transport scheduling, model math or claimed TPS changes.
- Adapt Yangmin Li's second SGLang #45450 commit (`1f8e4f3e`): the static
  DSpark draft runner now captures its device-only attention-metadata prep in
  the existing metadata glue graph. The port rebinds `out_cache_loc` to the
  runner's pointer-stable registry buffer and caches each batch-size extend
  length on device, eliminating the two stale/freed-pointer hazards that made
  the older adapter experiment unsafe. It fails closed for ragged verify,
  request windows, context parallelism, online C128 MTP, TRT-LLM and every
  backend that does not explicitly opt in. Upstream reports this as a small
  default-on seam optimization; no local GB10 TPS claim is made before A/B.
- Adapt hnyls2002's merged SGLang #45586 and #45699 idle-store fixes. An idle
  forward now returns before reading stale or unset attention metadata: with an
  encoder-replay request window it activates a cached idle layout of the right
  row count; otherwise it writes the translated dummy SWA slot. Upstream's
  original encoder-replay/DSpark repro crashed at 197/200 requests and completed
  200/200 after the fix. Non-idle routing is unchanged.
- Vendor alphabetc1's exact merged SGLang #45726 Triton kernel and adapt only
  its CUDA call site to the pinned request-window layout. Two kernels replace
  the eager pointwise construction of query indices/lengths/commit masks and
  history metadata after the existing request-group scans; CPU and HIP keep
  the reference path. Upstream's isolated GB300 metadata benchmark improved
  from roughly 0.89 to 0.41 ms per call at 4K and 32K rows. This is not an
  end-to-end or GB10 performance claim before GPU parity and timing.
- Adapt hnyls2002's merged SGLang #45579 to the pinned request-window
  implementation: each prefill layout computes the source row for every SWA
  history slot once, carries it through graph replay refreshes, and all layers
  reuse that tensor instead of rebuilding it. The row formula and zero-row
  fallback are unchanged. Upstream observed this bookkeeping in 1.7-4.8% of
  scheduler-thread time on GB300 prefill, but did not isolate the end-to-end
  gain; this remains a source-qualified prefill optimization pending GB10 A/B.
- Refine #45762 with potoior's SGLang #45804: only the native DeepSeek
  `dsv4`/`dsv41`/`dsv32` encoders inherit `SGLANG_DEFAULT_THINKING` on the
  response-parsing side. Jinja-driven models no longer receive that unrelated
  default, and the Responses API now mirrors Chat. Explicit request toggles
  remain authoritative in both APIs. The follow-up is applied after #45762 so
  the exact scope correction is reviewable and both intermediate and final
  semantics are tested.
- Backport Yangmin Li's merged SGLang #45630 commit (`741aea0a`): startup
  autotuning may now select its measured FlashInfer CuTeDSL MXFP8 tactic for
  block-FP8 linears at decode and verify row counts, instead of forcing the
  backend heuristic below 4,096 rows. Deterministic inference still pins the
  heuristic. Upstream measured about 2% shorter V4.1 decode steps at 16-64
  requests per rank on GB300 and no change at smaller batches; no local GB10
  or C1 claim is made before an A/B boot.
- Backport liweizheng's SGLang #45518: clamp FlashInfer's fresh DSpark
  softmax output in place to the valid `1.0` probability ceiling. FlashInfer
  can overshoot by roughly 7e-6 on near-one-hot rows; the accept kernel treats
  such values as invalid and rejects the draft. The zero-fuzz patch preserves
  the existing allocation path and has source, production-function and image
  packaging checks. Upstream measured acceptance recovery from 1.75 to 6.08
  at temperature 0.3; no local GB10 performance claim is made yet.
- Adapt hantenglong's SGLang #45762 so DeepSeek V4/V4.1 response parsing
  follows `SGLANG_DEFAULT_THINKING` when a request omits the `thinking`
  kwarg. This matches the native encoder, keeps an explicit `thinking: false`
  authoritative, and prevents a default-on reasoning block from leaking into
  ordinary `content`. Other `enable_thinking` model families remain opt-in.
- Adapt the width-alignment commit from BJ Wang's SGLang #45794 to the pinned
  Torch V4.1 prefill indexer. CUDA BF16 score widths of at least 1,024 rows are
  padded once per request to a multiple of 64, the chunk budget uses that real
  width, and scores are sliced back to the logical context before any mask or
  top-k selection. Upstream H200 gains motivate a GB10 canary only; local GPU
  parity and end-to-end timing remain required.
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
