# Changelog

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
Release gate (v2.6 vs v2.6.2 vs v2.6, fresh boots, production profile): PENDING.

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
